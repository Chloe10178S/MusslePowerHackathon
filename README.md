# Mussel Power

**Team Members:** Chloe Bendenoun, Christian Friis, Catarina Lorenzo  
**Project Title:** Mussel Power

---

## Project Summary

### The Problem

Many water bodies are contaminated with pathogens and harmful algae. We are stuck between two flawed approaches:

- **Natural mussels** filter water but accumulate pathogens without killing them, making them unsuitable for safe purification. Moreover, mussels themselves are fragile:  
  - They bioaccumulate toxins and pathogens, eventually dying and releasing pollutants back into the water  
  - They depend on specific ecosystems; introducing them artificially risks upsetting local biodiversity  
  - They cannot survive in highly polluted or warm waters, limiting practical use 
- **Existing mechanical or chemical solutions** are often:
  - Energy-intensive
  - Not adaptive
  - Ineffective under variable water conditions (flow, turbidity, pathogen density)

**Challenge:** Design a system that actively kills pathogens while remaining energy-efficient and adaptive, even as the pod moves through different parts of a water body.

### Our Solution

**Mussel Power** is our answer. We started with **biomimicry**, taking direct inspiration from the mussel's brilliant natural design but fixing its fatal flaws.

Our solution is an end-to-end system with three core parts:

- **The Pod**: This is the physical heart of our system, inspired by the gill structure of mussels. It uses an internal cilia-like motion to actively draw in water with minimal energy consumption. As the water flows through, integrated UV-C light eliminates bacteria, algae, and other contaminants in real-time. Each unit is modular and designed to work in clusters, making the system customizable and scalable for any body of water, from a small pond to an urban waterway.
- **The App**: A real-time dashboard that acts as the mission control. It allows operators to monitor the Water Safety Score, track energy usage, and manage every pod in the field. Crucially, this allows our pods to act as decentralized data-gathering points, bringing real-time water quality data to rural or remote communities that have never had access to it before.
- **The AI**: This is what makes our product unique and safe, robust, and highly efficient. Instead of a fixed "on/off" setting, our Reinforcement Learning AI gives each pod adaptive intelligence.
  - It is safe from the moment it's deployed, using a **formula-based initialization** to ensure effective purification even before the AI has fully learned.
  - It is robust, using its **dual control** over UV intensity and cilia vibration to handle even extreme changes in water turbidity or flow.
  - Most importantly, it is **energy-efficient**, continuously learning and calculating the absolute minimum energy required to keep the water safe. This isn't just automation; it's a self-optimizing system that makes purification radically more efficient.

---

## How AI is Used

The **core intelligence** of Mussel Power is driven by **Reinforcement Learning (RL)** — a branch of AI that allows the system to *learn by interacting with its environment*.  

Instead of running on fixed, pre-programmed settings, each mussel pod uses an **AI agent** that **observes** real-time water quality data and **decides** how to operate most efficiently.

###  1. Decision-Making in Real Time
The AI agent receives continuous sensor inputs such as:
- Water **flow rate**
- **Turbidity** (clarity of the water)
- Current **UV intensity**
- **Pathogen proxy** (estimated contamination level)

Based on this, it decides:
- How much to **vibrate** the internal channels (affects water residence time).
- How much **UV power** to use (affects pathogen kill rate and energy consumption).

This allows the pod to **self-regulate** and stay efficient even as water conditions change — something static systems can’t do.

###  2. Learning Through Reward Signals
The RL agent is trained with a **reward function**:

`Reward = Pathogen Kill Effect - Energy Cost`

- If the AI kills more pathogens using less energy → **positive reward**   
- If it uses too much energy with little benefit → **negative reward** 

Over time, it learns **optimal control strategies** without needing human tuning.

###  3. Continuous Adaptation
Unlike traditional control systems:
- It **adapts dynamically** to unexpected situations (e.g., sudden turbidity spikes).
- It **balances trade-offs** between energy cost and cleaning performance.
- It can be **transferred to multiple pods**, creating intelligent colonies of filtration units.

###  4. Algorithm Used
The prototype uses **Deep Deterministic Policy Gradient (DDPG)**:
- A **model-free, off-policy** RL algorithm.
- Well-suited for **continuous control problems** (like vibration frequency and UV intensity).
- Combines **actor-critic architecture** for stability and precision.

## Link to App

https://aqua-pure-ai-649a37d1.base44.app/
