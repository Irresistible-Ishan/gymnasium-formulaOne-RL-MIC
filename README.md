# Gymnasium Formula One RL - MIC

🔗 **Notebook Created:** [View on Google Colab](https://colab.research.google.com/drive/10kHvsWxw-_rbfyNSrgrAwgFsyhKOoXcd?usp=sharing)

This repository contains the materials and codebase created to host the **MIC - AIML workshop (Pre TechnoVIT 2026)**. The primary goal of this project and the workshop is to teach the fundamentals and advanced concepts of Reinforcement Learning (RL) through a Formula 1 racing simulation.

## Workshop Objectives & Learnings

During the development of this project and the workshop sessions, we focus on:

- **Reinforcement Learning (RL) Fundamentals:** Understanding the core concepts of agents, environments, states, actions, and rewards.
- **Proximal Policy Optimization (PPO):** Deep diving into modern policy gradient methods and how PPO provides stable and efficient learning.
- **Drawbacks of Old RL Methods:** Discussing why traditional methods (like Q-Learning or basic Policy Gradients) struggle with continuous action spaces, suffer from training instability, and how modern algorithms like PPO overcome these issues.
- **OpenAI Gymnasium Simulation Environment:** Learning how to build, customize, and wrap environments using the standard `gymnasium` API. *(Note: I personally learned how to build and integrate custom environments with Gymnasium while creating this material for the workshop, and I am still continuously learning and improving it!)*

## Repository Contents

- **`F1ENV.ipynb`**: The main Jupyter Notebook (also accessible on Colab) containing the complete F1 racing environment built with Gymnasium, including the PPO training loops and visualizations. [View on Colab](https://colab.research.google.com/drive/10kHvsWxw-_rbfyNSrgrAwgFsyhKOoXcd?usp=sharing)
- **`envdoc.md`**: Technical specification and rationale for the F1 RL Environment. It details the Kinematic Physics Engine, the Single-Track Bicycle Model, and the modular reward functions.
- **Test Scripts & Iterations (`f1v1.py`, `f1v2visual.py`, `f1v3lotsvisuals.py`, `learningenv.py`, `nogym.py`, `blackjack.py`)**: Various iterative scripts showcasing the evolution of the environment from basic logic (without Gymnasium) to a fully visual and Gym-compliant simulation.
- **Presentations & Plans**: Includes the workshop execution plan (`Formula1_Technical_Execution_Plan_Polished.pdf`) and event brochures (`FormulA1_Brochure-2.pdf`).

## The Physics Engine

The custom environment relies on a **Kinematic Single-Track Bicycle Model**. This model strikes a perfect balance between performance and realism—it efficiently computes millions of steps required for PPO training while enforcing real-world racing physics like the Friction Circle (lateral acceleration limits). The core racing principle remains: **you must slow down to turn!**

---
*Created for the MIC AIML Workshop, Pre TechnoVIT 2026.*