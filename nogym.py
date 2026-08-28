import random

# ==========================================
# 1. THE ENVIRONMENT SETUP
# ==========================================
NUM_STATES = 5     # Positions: 0, 1, 2, 3, 4
GOAL_STATE = 4     # Target location
ACTIONS = [0, 1]   # 0 = Left, 1 = Right

# ==========================================
# 2. AGENT MEMORY (Q-Table)
# ==========================================
# Rows = States (0-4), Columns = Actions (0, 1)
# Initially, the agent knows nothing (all values 0.0)
q_table = [[0.0, 0.0] for _ in range(NUM_STATES)]

# ==========================================
# 3. HYPERPARAMETERS
# ==========================================
learning_rate = 0.5   # How quickly it overwrites old knowledge (alpha)
discount_factor = 0.9 # How much it cares about future rewards (gamma)
epsilon = 0.2         # 20% chance of random action (exploration)
episodes = 200      # How many practice runs to do

# ==========================================
# 4. TRAINING LOOP
# ==========================================
for episode in range(episodes):
    state = 0  # Reset to starting position

    while state != GOAL_STATE:
        # A. CHOOSE ACTION (Epsilon-Greedy Strategy)
        if random.random() < epsilon:
            action = random.choice(ACTIONS)  # Explore: Try something random
        else:
            # Exploit: Pick the action with highest expected reward
            action = 0 if q_table[state][0] > q_table[state][1] else 1

        # B. ENVIRONMENT STEP (State Transition)
        if action == 0:
            next_state = max(0, state - 1)  # Move Left (can't go below 0)
        else:
            next_state = min(GOAL_STATE, state + 1)  # Move Right

        # C. REWARD FUNCTION
        if next_state == GOAL_STATE:
            reward = 10.0   # Big reward for reaching goal
        else:
            reward = -1.0   # Small penalty per step to encourage speed

        # D. UPDATE KNOWLEDGE (Bellman Equation)
        best_future_q = max(q_table[next_state])
        old_q = q_table[state][action]
        
        # New Q = Old Q + LR * (Reward + Discount * Best Future Q - Old Q)
        q_table[state][action] = old_q + learning_rate * (
            reward + discount_factor * best_future_q - old_q
        )

        # Move to next state
        state = next_state

# ==========================================
# 5. INSPECT WHAT THE AGENT LEARNED
# ==========================================
print("Learned Q-Table:")
print("State | Left Value | Right Value | Preferred Move")
print("-------------------------------------------------")
for s in range(NUM_STATES - 1):
    best_move = "RIGHT" if q_table[s][1] > q_table[s][0] else "LEFT"
    print(f"  {s}   |   {q_table[s][0]:6.2f}   |   {q_table[s][1]:6.2f}    |  ---> {best_move}")