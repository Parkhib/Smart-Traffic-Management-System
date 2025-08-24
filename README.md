# 🚦 Smart Traffic Management System using Reinforcement Learning  

## 📌 Overview  
This project implements an **AI-powered Smart Traffic Management System** that uses **Reinforcement Learning (RL)** to optimize traffic signal timings in real-time.  

Instead of fixed timers, the system learns to minimize **vehicle waiting time** and **congestion** by dynamically switching signals based on live traffic density.  

Built using **Google Colab** with **Python, OpenAI Gym, PyTorch, and RL algorithms (Q-learning & DQN)**.  

---

## 🎯 Problem Statement  
Traditional traffic lights work on **fixed timers** that don’t adapt to real-time traffic conditions.  
This leads to:  
- Long waiting times 🚗🚕🚙  
- Fuel wastage ⛽  
- Increased congestion 🚧  

**Goal**: Train an RL agent that observes traffic queues and decides the optimal green signal direction to reduce congestion.  

---

## ⚙️ Tech Stack  
- **Python 3.9+**  
- **Google Colab (for training)**  
- **Libraries:**  
  - `gym` → Custom traffic simulation environment  
  - `numpy` → State & reward calculations  
  - `torch` → Deep Q-Network (DQN) implementation  
  - `matplotlib` → Visualizations  

---

## 🏗 Project Architecture  
smart_traffic_rl/
│── traffic_env.py # Custom Gym environment
│── q_learning.py # Q-learning baseline
│── dqn.py # Deep Q-Network training
│── utils.py # Replay buffer, plotting
│── app.py # (Optional) API wrapper
│── results/ # Training graphs & saved models
│── README.md # Project documentation
│── smart_traffic_rl.ipynb # Colab notebook


---

## 🚀 Implementation  

### 🔹 1. Environment (OpenAI Gym)  
- State → Queue lengths `[North, South, East, West]`  
- Actions → `0 = NS Green` | `1 = EW Green`  
- Reward → Negative of total waiting cars (minimize congestion)  

### 🔹 2. Algorithms Implemented  
1. **Fixed Timer Baseline** (for comparison)  
2. **Q-Learning** (tabular RL)  
3. **Deep Q-Network (DQN)** using PyTorch  

### 🔹 3. Training  
- Episodes: `5000+`  
- Exploration: ε-greedy strategy  
- Replay Buffer + Target Network for stability  

---

## 📊 Results  

### ✅ Average Waiting Time Comparison  
| Method        | Avg. Waiting Time (cars/step) | Improvement |
|---------------|-------------------------------|-------------|
| Fixed Timer   | ~14                           | - |
| Q-Learning    | ~10                           | ~28% |
| DQN (PyTorch) | ~9                            | ~35% |

📉 RL-based control significantly **reduced congestion vs fixed-timer** signals.  

*(Graphs available in `/results`)*  

---

