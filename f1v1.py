import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from stable_baselines3 import PPO

# ==========================================
# 1. ENVIRONMENT CONFIGURATION
# ==========================================
MAX_SPEED = 20.0
MAX_ACCEL = 10.0
MAX_TURN = 1.0       # Radians per second
DT = 0.1             # Simulation step time

# Circular Track Dimensions
INNER_RADIUS = 30.0
OUTER_RADIUS = 50.0
LIDAR_ANGLES = [-math.pi/2, -math.pi/4, 0, math.pi/4, math.pi/2] # 5 rays
LIDAR_MAX_RANGE = 50.0

class FastTrackEnv(gym.Env):
    """A minimal, high-speed kinematic RL racing environment."""
    
    def __init__(self):
        super(FastTrackEnv, self).__init__()
        
        # Action: [Steering (-1 to 1), Acceleration (-0.7 to 1.0)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -0.7]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )
        
        # Observation: [5 LiDAR distances, Speed]
        self.observation_space = spaces.Box(
            low=0.0, 
            high=max(LIDAR_MAX_RANGE, MAX_SPEED), 
            shape=(6,), 
            dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Start at the middle of the track (radius 40), on the right side, facing UP
        self.x = 40.0
        self.y = 0.0
        self.theta = math.pi / 2  # Facing "North" along the circle
        self.speed = 0.0
        
        # Tracking lap progress using angles
        self.last_angle = math.atan2(self.y, self.x)
        self.total_angle_traveled = 0.0
        
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        
        # 1. PARSE ACTIONS (Apply the 70% brake limit)
        steer_input = float(action[0])
        accel_input = np.clip(float(action[1]), -0.7, 1.0)
        
        # 2. APPLY PHYSICS
        # Stiff steering: Drops to 0.5 effectiveness at MAX_SPEED
        speed_factor = self.speed / MAX_SPEED
        steering_penalty = 1.0 - (speed_factor * 0.5)
        turn_rate = steer_input * MAX_TURN * steering_penalty
        
        # Update Speed & Heading
        self.speed = np.clip(self.speed + (accel_input * MAX_ACCEL * DT), 0.0, MAX_SPEED)
        if self.speed > 0.1:
            self.theta += turn_rate * DT
            
        # Update Position
        self.x += self.speed * math.cos(self.theta) * DT
        self.y += self.speed * math.sin(self.theta) * DT
        
        # 3. CALCULATE PROGRESS (Lap Tracking)
        current_angle = math.atan2(self.y, self.x)
        angle_diff = current_angle - self.last_angle
        # Normalize angle difference to handle the wrap-around at -pi/pi
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        self.total_angle_traveled += angle_diff
        self.last_angle = current_angle

        # 4. CHECK BOUNDARIES & REWARD
        dist_to_center = math.hypot(self.x, self.y)
        crashed = dist_to_center < INNER_RADIUS or dist_to_center > OUTER_RADIUS
        lap_complete = self.total_angle_traveled >= 2 * math.pi
        
        terminated = False
        reward = -0.1  # The constant time penalty
        
        if crashed:
            reward += -100.0
            terminated = True
        elif lap_complete:
            reward += 1000.0
            terminated = True
            
        # Failsafe: End if taking too long
        truncated = self.steps >= 1000 
        
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        """Calculates LiDAR rays using quadratic circle intersection math."""
        lidar_readings = []
        for angle_offset in LIDAR_ANGLES:
            ray_theta = self.theta + angle_offset
            dx = math.cos(ray_theta)
            dy = math.sin(ray_theta)
            
            # Quadratic formula components: a*t^2 + b*t + c = 0
            b = 2.0 * (self.x * dx + self.y * dy)
            
            min_t = LIDAR_MAX_RANGE
            
            # Check intersection with both INNER and OUTER walls
            for R in [INNER_RADIUS, OUTER_RADIUS]:
                c = (self.x**2 + self.y**2) - R**2
                discriminant = b**2 - 4*c
                
                if discriminant >= 0:
                    t1 = (-b + math.sqrt(discriminant)) / 2.0
                    t2 = (-b - math.sqrt(discriminant)) / 2.0
                    
                    if 0 < t1 < min_t: min_t = t1
                    if 0 < t2 < min_t: min_t = t2
            
            lidar_readings.append(min_t)
            
        lidar_readings.append(self.speed)
        return np.array(lidar_readings, dtype=np.float32)

# ==========================================
# 2. RUNNING THE TRAINING
# ==========================================
if __name__ == "__main__":
    print("Initializing Custom Environment...")
    env = FastTrackEnv()
    
    print("Building PPO Model...")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0005)
    
    print("Training for 50,000 steps...")
    model.learn(total_timesteps=50000)
    
    print("Training Complete! Testing the trained agent...")
    obs, info = env.reset()
    for _ in range(500):
        # The agent predicts the best action based on the LiDAR and Speed
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            if reward > 500:
                print("SUCCESS: The agent completed a full lap!")
            else:
                print("FAILED: The agent crashed.")
            break