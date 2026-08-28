import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import pygame
from stable_baselines3 import PPO

MAX_SPEED = 20.0
MAX_ACCEL = 10.0
MAX_TURN = 1.0       
DT = 0.1             
INNER_RADIUS = 35.0
OUTER_RADIUS = 40.0
LIDAR_ANGLES = [-math.pi/2, -math.pi/4, 0, math.pi/4, math.pi/2]
LIDAR_MAX_RANGE = 50.0

class FastTrackEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    def __init__(self, render_mode=None):
        super(FastTrackEnv, self).__init__()
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.action_space = spaces.Box(low=np.array([-1.0, -0.7]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=max(LIDAR_MAX_RANGE, MAX_SPEED), shape=(6,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x = 40.0
        self.y = 0.0
        self.theta = math.pi / 2  
        self.speed = 0.0
        self.last_angle = math.atan2(self.y, self.x)
        self.total_angle_traveled = 0.0
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        steer_input = float(action[0])
        accel_input = np.clip(float(action[1]), -0.7, 1.0)
        speed_factor = self.speed / MAX_SPEED
        steering_penalty = 1.0 - (speed_factor * 0.5)
        turn_rate = steer_input * MAX_TURN * steering_penalty
        self.speed = np.clip(self.speed + (accel_input * MAX_ACCEL * DT), 0.0, MAX_SPEED)
        if self.speed > 0.1:
            self.theta += turn_rate * DT
        self.x += self.speed * math.cos(self.theta) * DT
        self.y += self.speed * math.sin(self.theta) * DT
        
        current_angle = math.atan2(self.y, self.x)
        angle_diff = current_angle - self.last_angle
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        self.total_angle_traveled += angle_diff
        self.last_angle = current_angle

        dist_to_center = math.hypot(self.x, self.y)
        crashed = dist_to_center < INNER_RADIUS or dist_to_center > OUTER_RADIUS
        lap_complete = self.total_angle_traveled >= 2 * math.pi
        
        terminated = False
        
        # we are adding a little reward for speed herr ig
        #  + default negative reward wrt time
        reward = -0.1 + (self.speed * 0.05)  
        if crashed:
            reward += -100.0
            terminated = True
        elif lap_complete:
            reward += 1000.0
            terminated = True
        truncated = self.steps >= 1000 
        return self._get_obs(), reward, terminated, truncated, {}

    # the eyes of car
    def _get_obs(self):
        lidar_readings = []
        for angle_offset in LIDAR_ANGLES:
            ray_theta = self.theta + angle_offset
            dx, dy = math.cos(ray_theta), math.sin(ray_theta)
            b = 2.0 * (self.x * dx + self.y * dy)
            min_t = LIDAR_MAX_RANGE
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

    def render(self):
        if self.render_mode != "human": return
        SCALE = 6
        CENTER = (300, 300)
        if self.window is None:
            pygame.init()
            pygame.display.set_caption("F1 AI Showdown")
            self.window = pygame.display.set_mode((600, 600))
            self.clock = pygame.time.Clock()
        self.window.fill((30, 30, 30)) 
        pygame.draw.circle(self.window, (100, 100, 100), CENTER, OUTER_RADIUS * SCALE)
        pygame.draw.circle(self.window, (30, 30, 30), CENTER, INNER_RADIUS * SCALE)
        pygame.draw.circle(self.window, (255, 255, 255), CENTER, OUTER_RADIUS * SCALE, 2)
        pygame.draw.circle(self.window, (255, 255, 255), CENTER, INNER_RADIUS * SCALE, 2)
        car_screen_x = int(CENTER[0] + self.x * SCALE)
        car_screen_y = int(CENTER[1] + self.y * SCALE)
        pygame.draw.circle(self.window, (255, 50, 50), (car_screen_x, car_screen_y), 6)
        end_x = int(car_screen_x + math.cos(self.theta) * 15)
        end_y = int(car_screen_y + math.sin(self.theta) * 15)
        pygame.draw.line(self.window, (50, 255, 50), (car_screen_x, car_screen_y), (end_x, end_y), 3)
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(30)
    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None

if __name__ == "__main__":
    print("Initializing Background Environment for Training...")
    train_env = FastTrackEnv()
    model = PPO("MlpPolicy", train_env, verbose=1, learning_rate=0.0005)
    print("Training for 50,000 steps (This will take ~90 seconds)...")
    model.learn(total_timesteps=50000)
    print("Training Complete! Opening UI to test the agent...")
    test_env = FastTrackEnv(render_mode="human")
    obs, info = test_env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        test_env.render()
        if terminated or truncated:
            if reward > 500:
                print("SUCCESS: The agent completed a full lap!")
            else:
                print("FAILED: The agent crashed.")
            pygame.time.wait(1000) 
            break
    test_env.close()