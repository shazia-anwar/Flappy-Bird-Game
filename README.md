# 🐦 Flappy Bird AI using Deep Q-Learning (DQN)

An AI agent that learns to play Flappy Bird using Reinforcement Learning and Deep Q-Networks (DQN).
The agent improves over time by interacting with the environment and learning optimal actions through trial and error.

---

## 🎮 Gameplay Demo

![Flappy Bird AI Gameplay](flappybird.gif)

---

## 🚀 Project Overview

This project demonstrates how an agent can learn to play Flappy Bird from scratch using Deep Reinforcement Learning techniques.

* Implements Deep Q-Network (DQN)
* Uses Experience Replay for stable learning
* Applies Epsilon-Greedy strategy for exploration vs exploitation
* Neural network approximates Q-values for decision making

---

## 🧠 How It Works

1. The agent observes the current game state
2. Chooses an action (jump / no jump)
3. Receives a reward or penalty
4. Stores the experience in replay memory
5. Learns by sampling past experiences to improve future decisions

---

## 🏗️ Project Structure

```
DQN FOR FLAPPY BIRD GAME/
│
├── Agent.py                # Main training loop & agent logic
├── dqn.py                  # Deep Q-Network model
├── Experience_Replay.py    # Replay buffer implementation
├── Flappy_Bird_Game.py    # Replay buffer implementation
├── flappybird.gif          # Gameplay demo (used in README)
├── parameters.yaml         # Hyperparameters configuration
├── README.md               # Project documentation
├── requirements.txt        # Project dependencies
├── .gitignore              # Files ignored by Git
│
├── runs/                   # Training logs (ignored)
├── rl_env/                 # Virtual environment (ignored)
└── __pycache__/            # Python cache (ignored)
```

---

## ⚙️ Installation

Clone the repository and install dependencies:

```
git clone https://github.com/shazia-anwar/Flappy-Bird-Game
pip install -r requirements.txt
```

---

## ▶️ How to Run

```
python Agent.py flappybirdv0 --train
python Agent.py flappybirdv0
```

---

## 📊 Features

* Deep Q-Network (DQN) implementation
* Experience Replay Buffer
* Epsilon Decay Strategy
* Configurable hyperparameters via YAML
* Training logs for performance tracking

---

## 🧪 Hyperparameters

All training configurations can be modified in:

```
parameters.yaml
```

---

## 📈 Results

* Initially performs random actions
* Gradually learns to survive longer
* Improves decision-making over time
* Learns to avoid obstacles effectively
* After training for around 10,124 episodes, the agent achieves significantly improved performance, demonstrating stable and intelligent gameplay behavior.

---

## 🛠️ Tech Stack

* Python 🐍
* PyTorch 🔥
* NumPy
* YAML

---

## 📌 Future Improvements

* Implement Double DQN for better stability
* Add visualization of training metrics
* Optimize reward function
* Deploy trained agent as an interactive demo

---

## 👨‍💻 Author

### Shazia Anwar (Shaz)
Aspiring AI & Machine Learning Developer

🔗 GitHub: https://github.com/shazia-anwar

---

If you find this project helpful, feel free to ⭐ the repository.

