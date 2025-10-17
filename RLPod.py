import pandas as pd
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
import os
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
INPUT_CSV = "smartpod_data.csv"
OUTPUT_CSV = "smartpod_outputs.csv"
MODEL_PATH_ACTOR = "smartpod_actor.weights.h5"
MODEL_PATH_CRITIC = "smartpod_critic.weights.h5"

STATE_SIZE = 4
ACTION_SIZE = 2
MEMORY_SIZE = 3000
BATCH_SIZE = 32
GAMMA = 0.95
TAU = 0.05
LR_ACTOR = 0.001
LR_CRITIC = 0.002
SMOOTH_FACTOR = 0.15

FLOW_RANGE = (0.05, 0.2)
TURBIDITY_RANGE = (1.0, 5.0)
UV_RANGE = (0.0, 100.0)
PATHOGEN_RANGE = (0.0, 1.0)

# =========================
# HELPERS
# =========================
def normalize_state(row):
    return np.array([
        (row["flow_rate"] - FLOW_RANGE[0]) / (FLOW_RANGE[1] - FLOW_RANGE[0]),
        (row["turbidity"] - TURBIDITY_RANGE[0]) / (TURBIDITY_RANGE[1] - TURBIDITY_RANGE[0]),
        (row["pathogen_proxy"] - PATHOGEN_RANGE[0]) / (PATHOGEN_RANGE[1] - PATHOGEN_RANGE[0]),
        row["current_uv"] / UV_RANGE[1]
    ], dtype=np.float32)

# =========================
# OU Noise
# =========================
class OUActionNoise:
    def __init__(self, mean, std_dev=0.15, theta=0.15, dt=1e-2):
        self.theta = theta
        self.mean = np.array(mean, dtype=np.float32)
        self.std_dev = std_dev
        self.dt = dt
        self.reset()
    def __call__(self):
        x = self.x_prev + self.theta*(self.mean - self.x_prev)*self.dt + \
            self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        self.x_prev = x.astype(np.float32)
        return self.x_prev
    def reset(self):
        self.x_prev = np.zeros_like(self.mean, dtype=np.float32)

# =========================
# DDPG AGENT
# =========================
class DDPGAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.target_actor = self.build_actor()
        self.target_critic = self.build_critic()
        self.update_target(self.target_actor, self.actor, tau=1.0)
        self.update_target(self.target_critic, self.critic, tau=1.0)
        self.noise = OUActionNoise(mean=np.zeros(action_size), std_dev=0.15)
        self.prev_action = np.zeros(action_size, dtype=np.float32)

    def build_actor(self):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(self.state_size,)),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(LR_ACTOR))
        return model

    def build_critic(self):
        state_input = Input(shape=(self.state_size,))
        action_input = Input(shape=(self.action_size,))
        x = Concatenate()([state_input, action_input])
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        out = Dense(1, activation='linear')(x)
        model = Model([state_input, action_input], out)
        model.compile(optimizer=Adam(LR_CRITIC), loss='mse')
        return model

    def update_target(self, target, source, tau):
        new_weights = [tau*sw + (1-tau)*tw for sw, tw in zip(source.get_weights(), target.get_weights())]
        target.set_weights(new_weights)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state_tf = tf.convert_to_tensor(state.reshape(1,-1), dtype=tf.float32)
        action = self.actor(state_tf).numpy()[0]
        action = SMOOTH_FACTOR*self.prev_action + (1-SMOOTH_FACTOR)*action
        action += self.noise()
        action = np.clip(action, 0, 1)
        self.prev_action = action
        return action

    def replay(self):
        if len(self.memory) < max(BATCH_SIZE, 128):
            return
        minibatch = np.random.choice(len(self.memory), BATCH_SIZE, replace=False)
        for idx in minibatch:
            state, action, reward, next_state, done = self.memory[idx]
            state_tf = tf.convert_to_tensor(state.reshape(1,-1), dtype=tf.float32)
            next_state_tf = tf.convert_to_tensor(next_state.reshape(1,-1), dtype=tf.float32)
            target_action = self.target_actor(next_state_tf)
            target_q = self.target_critic([next_state_tf, target_action])
            y = reward + (1 - done) * GAMMA * target_q
            self.critic.fit([state.reshape(1,-1), action.reshape(1,-1)], y.numpy().reshape(-1,1), verbose=0)
            with tf.GradientTape() as tape:
                pred_action = self.actor(state_tf)
                q_val = self.critic([state_tf, pred_action])
                loss = -tf.reduce_mean(q_val)
            grads = tape.gradient(loss, self.actor.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
        self.update_target(self.target_actor, self.actor, TAU)
        self.update_target(self.target_critic, self.critic, TAU)

    def save(self):
        self.actor.save_weights(MODEL_PATH_ACTOR)
        self.critic.save_weights(MODEL_PATH_CRITIC)

    def load(self):
        if os.path.exists(MODEL_PATH_ACTOR):
            self.actor.load_weights(MODEL_PATH_ACTOR)
        if os.path.exists(MODEL_PATH_CRITIC):
            self.critic.load_weights(MODEL_PATH_CRITIC)

# =========================
# Base UV & vibration
# =========================
def base_uv_vibration(row):
    # Calculate preliminary UV based on pathogen and turbidity
    uv_base = 20.0 + 60.0 * row["pathogen_proxy"] + 10.0 * ((row["turbidity"]-1)/4) - 0.5 * row["current_uv"]
    
    # Only allow UV to go to 0 if pathogen is extremely low
    if row["pathogen_proxy"] < 0.05:   # threshold for negligible pathogen
        uv_pct = 0.0
    else:
        uv_pct = max(10.0, uv_base)   # enforce a minimum UV intensity
    uv_pct = np.clip(uv_pct, 0.0, 100.0)

    # Vibration base remains the same
    vib_base = max(0.5, 2.5 * row["pathogen_proxy"])
    vib_hz = np.clip(vib_base, 0.5, 5.0)

    return vib_hz, uv_pct

# =========================
# Reward function for moving pod
# =========================
def reward_function_moving(state, action):
    """
    Calculates reward for a single water segment.
    state: normalized [flow, turbidity, pathogen, current_uv]
    action: normalized [vibration, UV intensity]
    """
    flow = state[0]
    turbidity = state[1]
    pathogen = state[2]

    # Instantaneous pathogen kill
    uv_effect = pathogen * (1 - np.exp(-action[1] * (1 - turbidity) / (flow + 0.01)))
    vib_effect = action[0] * 0.1
    pathogen_kill = min(uv_effect + vib_effect, pathogen)

    # Energy cost
    uv_energy = action[1] * 0.02
    vib_energy = action[0] * 0.01

    reward = pathogen_kill - (uv_energy + vib_energy)
    return reward

# =========================
# MAIN LOOP
# =========================
if __name__ == "__main__":
    agent = DDPGAgent(STATE_SIZE, ACTION_SIZE)
    agent.load()
    df = pd.read_csv(INPUT_CSV).reset_index(drop=True)
    output_actions = []

    REPLAY_INTERVAL = 50  # replay every 50 timesteps
    
    for idx, row in df.iterrows():
        # Normalize current water segment
        state = normalize_state(row)

        # Base action estimation (formula-based initialization)
        vib_base, uv_base = base_uv_vibration(row)

        # RL agent chooses delta adjustments
        delta_action = agent.act(state)
        delta_vib = (delta_action[0] - 0.5) * 4.0
        delta_uv = (delta_action[1] - 0.5) * 50.0

        # Clip actions to safe range
        vibration = np.clip(vib_base + delta_vib, 0.5, 5.0)
        uv_intensity = np.clip(uv_base + delta_uv, 0.0, 100.0)

        # Normalized action for RL agent
        action_norm = np.array([vibration / 5.0, uv_intensity / 100.0], dtype=np.float32)

        # Directly compute reward
        reward = reward_function_moving(state, action_norm)
        next_state = state  # next state is independent

        output_actions.append({
            "timestep": idx+1,
            "vibration_frequency_Hz": vibration,
            "uv_intensity_percent": uv_intensity,
            "reward": reward
        })

        print(f"Step {idx+1}: Vib={vibration:.2f} Hz, UV={uv_intensity:.1f} %, Reward={reward:.3f}")

        # Store experience and update agent
        agent.remember(state, action_norm, reward, state, done=False)  # state reused because next state is unknown
        if idx % REPLAY_INTERVAL == 0:
            agent.replay()

    # Save actions CSV
    pd.DataFrame(output_actions).to_csv(OUTPUT_CSV, index=False)
    agent.save()
    print(f"\n💾 Actions saved: {OUTPUT_CSV}")
    print("✅ DDPG simulation loop finished.")

    # Plot reward progression
    rewards = [x["reward"] for x in output_actions]
    plt.plot(rewards)
    plt.title("Reward progression over moving water segments")
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.show()
