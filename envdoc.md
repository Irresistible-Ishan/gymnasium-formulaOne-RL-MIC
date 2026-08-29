
```markdown
# F1 RL Environment: Technical Specification & Rationale
**Version:** 1.0  
**Scope:** Kinematic Physics Engine & Modular Reward Function  

---

## 1. System Constants & Configuration (`config.py`)

> ### Rationale: What & Why
> **What we picked:** A set of simplified scalar values representing vehicle dimensions, tire grip, and track limits.
> **How it works:** These act as the global "laws of physics" for the simulation. By changing `DRAG_COEFFICIENT` or `FRICTION_MU`, you essentially change the car from an F1 car to a heavy truck.
> **Why we picked it:** Hardcoding these in a central config file prevents "magic numbers" scattered across the codebase. For the workshop, these values are locked to participants, but your dev team can easily tweak them here to make the car feel "right" during testing.

```python
import numpy as np

# Physics Constants
WHEELBASE = 3.5           # meters (L) - Distance between front and rear axle
MAX_SPEED = 20.0          # m/s (v_max) - Artificial cap for simulation stability
MIN_SPEED = 0.0           # m/s - Car cannot reverse
DRAG_COEFFICIENT = 0.05   # (C_d) - Aerodynamic resistance
ROLLING_RESISTANCE = 0.1  # (C_r) - Tire friction against asphalt
FRICTION_MU = 1.2         # Tire grip coefficient (F1 tires are ~1.5-1.7, we use 1.2 for forgiving gameplay)
GRAVITY = 9.81            # m/s^2 (g)
DT = 0.1                  # Simulation time step (10 Hz update rate)

# Track Constants
TRACK_WIDTH = 10.0        # meters (W)
TOTAL_WAYPOINTS = 100     # Number of points defining the centerline spline

# Action Space Limits
MAX_STEERING_RAD = 0.6    # Max physical steering angle (~34 degrees)
MAX_ACCELERATION = 8.0    # Max throttle/brake force (m/s^2)
```

---

## 2. Physics Engine Specification (`physics.py`)

> ### Rationale: What & Why
> **What we picked:** The **Kinematic Single-Track Bicycle Model**. 
> **How it works:** Instead of simulating 4 individual wheels, it merges the front two wheels into one center front wheel, and the rear two into one center rear wheel, connected by a rigid `WHEELBASE`. It uses basic trigonometry to update the car's X/Y position and heading based on its current speed and steering angle.
> **Why we picked it:** This is the exact physics model used by the **F1TENTH** autonomous racing league. Full "Dynamic" models (calculating individual tire slip angles, suspension, and weight transfer) take ~5-10 milliseconds per step. PPO needs to run *millions* of steps in a few hours on Kaggle. The Kinematic model takes ~0.001 milliseconds per step, making it 5,000x faster, while still accurately capturing the most important F1 concept: **you must slow down to turn.**

### 2.1 State & Action Vectors
```python
# state = [x, y, theta, v]
# x, y: float -> Global 2D position
# theta: float -> Yaw/Heading in radians [-pi, pi]
# v: float -> Longitudinal velocity in m/s [0.0, MAX_SPEED]

# action = [steering_input, throttle_input]
# steering_input: float -> [-1.0, 1.0] mapped to physical radians
# throttle_input: float -> [-1.0, 1.0] mapped to physical m/s^2
```

### 2.2 Core Update Logic
```python
def step_physics(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    x, y, theta, v = state
    
    # 1. Map normalized agent inputs to physical units
    delta = np.clip(action[0], -1.0, 1.0) * MAX_STEERING_RAD
    a = np.clip(action[1], -1.0, 1.0) * MAX_ACCELERATION
    
    # 2. FRICTION CIRCLE ENFORCEMENT (See Section 2.3 below)
    if v > 0.1: 
        turning_radius = WHEELBASE / np.tan(delta) if np.abs(delta) > 1e-5 else float('inf')
        a_lateral = (v ** 2) / turning_radius
        max_lateral_grip = FRICTION_MU * GRAVITY
        
        if np.abs(a_lateral) > max_lateral_grip:
            max_delta = np.arctan(WHEELBASE / (v ** 2 / max_lateral_grip))
            delta = np.clip(delta, -max_delta, max_delta)

    # 3. Calculate resistive forces (Drag + Rolling)
    f_drag = DRAG_COEFFICIENT * (v ** 2)
    f_rolling = ROLLING_RESISTANCE * v
    net_acceleration = a - f_drag - f_rolling

    # 4. Kinematic Equations
    x_new = x + (v * np.cos(theta) * DT)
    y_new = y + (v * np.sin(theta) * DT)
    
    # Prevent yaw update if car is stationary
    if v > 0.1:
        theta_new = theta + ((v / WHEELBASE) * np.tan(delta) * DT)
    else:
        theta_new = theta
        
    v_new = np.clip(v + (net_acceleration * DT), MIN_SPEED, MAX_SPEED)

    return np.array([x_new, y_new, theta_new, v_new])
```

### 2.3 Deep Dive: The Friction Circle Implementation
> **What we picked:** A hard-clamp lateral acceleration limit.
> **How it works:** In reality, tires have a finite amount of grip. If you use 100% of your grip to turn, you have 0% grip left for braking. The code calculates the lateral G-force required to make the requested turn. If that force is greater than `FRICTION_MU * GRAVITY`, the code mathematically forces the steering angle to be smaller.
> **Why we picked it:** Without this, a kinematic model allows an agent to output `[1.0, 1.0]` (full turn, full gas) and the car will happily trace a perfect tight circle at max speed. By clamping the steering, we force the RL agent to learn the fundamental racing skill: **Trail Braking**. It must learn to reduce speed *before* the corner so it has the grip required to turn.

---

## 3. Reward Function Architecture (`formula1_env.py`)

> ### Rationale: What & Why
> **What we picked:** A **Weighted Additive Modular Reward** system.
> **How it works:** Instead of one massive, complex `if/else` block for rewards, we calculate 5 small, independent metrics (Progress, Speed, Centering, Steering, Heading), normalize them to similar scales (roughly -1 to 1), and multiply them by specific weights.
> **Why we picked it:** This is modeled directly after **AWS DeepRacer**, which proved this is the best reward structure for RL workshops. Because they are separated by weights, a participant can easily change the car's personality. Doubling the speed weight creates an aggressive, crash-prone car. Doubling the centering weight creates a slow, cautious car. It makes RL concepts visually obvious to beginners.

### 3.1 Master Equation & Baseline Weights
```python
# Default Baseline Weights (Participants will tune these)
W_PROGRESS = 10.0    # Highest priority: actually move forward
W_SPEED = 0.5        # Low priority: don't crash trying to go fast yet
W_CENTERING = 2.0    # Medium priority: stay on track
W_STEERING = -1.0    # Mild penalty: discourage jerky wheel movements
W_HEADING = 1.5      # Medium priority: point the car where it's going

def calculate_reward(state, action, info):
    # (Calculations below)
    total_reward = (W_PROGRESS * r_progress) + \
                   (W_SPEED * r_speed) + \
                   (W_CENTERING * r_centering) + \
                   (W_STEERING * r_steering) + \
                   (W_HEADING * r_heading)
                   
    if info['crashed']:
        total_reward = -100.0
    elif info['lap_complete']:
        total_reward = +100.0
        
    return total_reward
```

### 3.2 Component Implementations & Explanations

#### A. Progress Reward (`r_progress`)
> **How it works:** Measures how many track waypoints the car passed in the last step, divided by total waypoints.
> **Why:** This is the only "positive" driver of the car. Without it, the car would learn to just sit still to avoid steering/crashing penalties.
```python
r_progress = (current_waypoint_idx - previous_waypoint_idx) / TOTAL_WAYPOINTS
if r_progress < 0: r_progress = 0 # Prevent reward for driving backwards
```

#### B. Speed Reward (`r_speed`)
> **How it works:** Current speed divided by max speed.
> **Why:** Encourages the agent to keep its foot on the gas. Normalizing it ensures this reward doesn't dwarf the others when the car is going fast.
```python
r_speed = v / MAX_SPEED
```

#### C. Centering Reward (`r_centering`)
> **How it works:** Finds distance from the centerline, divides by half the track width, and squares the result. Subtracts from 1.
> **Why:** We square the result (`** 2`) because being 1m off center is bad, but being 4m off center (near the edge) is *exponentially* worse and should carry a much heavier penalty.
```python
normalized_distance = (2.0 * distance_from_center) / TRACK_WIDTH
r_centering = 1.0 - (normalized_distance ** 2)
```

#### D. Steering Smoothness Penalty (`r_steering`)
> **How it works:** Negative absolute value of the change in steering angle between this step and the last step.
> **Why:** In F1, violently spinning the steering wheel destroys tires and causes snap-oversteer. This penalty teaches the neural network to act like a smooth, professional driver rather than a jittery algorithm.
```python
r_steering = -abs(delta_current - delta_previous)
```

#### E. Heading Alignment (`r_heading`)
> **How it works:** Uses `cosine` of the angle difference between where the car is pointing, and where the next track waypoint is.
> **Why:** Using `math.cos()` for angle differences is an elegant RL trick. If the angle difference is 0 degrees (perfectly aligned), `cos(0) = 1.0` (max reward). If the angle difference is 90 degrees (sideways), `cos(90) = 0.0` (no reward). If the car is facing backwards, `cos(180) = -1.0` (penalty). It automatically scales smoothly without needing clunky `if/else` angle logic.
```python
r_heading = math.cos(theta - theta_track)
```

---

## 4. Anti-Hallucination Constraints (For AI Coding Agents)
1. **Libraries:** Use ONLY `numpy` and `math`. Do not use `scipy`, `shapely`, or `pygame` for logic.
2. **Determinism:** The environment must be 100% deterministic. No random noise in physics/rewards.
3. **NaN Prevention:** ALWAYS check `v > 0.1` before dividing by velocity or calculating yaw rates. A `ZeroDivisionError` or `NaN` value will instantly crash PPO training and ruin the workshop.
4. **Bounds Clipping:** All returned observations must be strictly clipped to the defined `gym.spaces.Box` bounds before returning in `step()`.
```