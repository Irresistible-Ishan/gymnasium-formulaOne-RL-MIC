import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

# this code im trying to understand how the general parent super() 
# class for environment definition works in the gymnasium


class SimpleGrassEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.width, self.height = 600, 600
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=0, high=600, shape=(2,), dtype=np.float32)

        # Rock location
        self.rock_x, self.rock_y = 300, 300
        self.rock_size = 25

        pygame.init()
        self.window = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_x, self.agent_y = 100, 300
        self.facing = "RIGHT"
        return np.array([self.agent_x, self.agent_y], dtype=np.float32), {}

    def step(self, action):
        speed = 5

        if action == 0:
            self.agent_y -= speed
            self.facing = "UP"
        elif action == 1:
            self.agent_y += speed
            self.facing = "DOWN"
        elif action == 2:
            self.agent_x -= speed
            self.facing = "LEFT"
        elif action == 3:
            self.agent_x += speed
            self.facing = "RIGHT"
        elif action == 4:
            self.agent_y += speed
            self.agent_x -= speed
            self.facing = "TOP-LEFT"
        elif action == 5:
            self.agent_x += speed
            self.agent_y += speed
            self.facing = "TOP-RIGHT"
        elif action == 6:
            self.agent_y -= speed
            self.agent_x -= speed
            self.facing = "DOWN-LEFT"
        elif action == 7:
            self.agent_y -= speed
            self.agent_x += speed
            self.facing = "DOWN-RIGHT"

        line_x, line_y = self.agent_x, self.agent_y
        if self.facing == "UP":    line_y -= 100
        if self.facing == "DOWN":  line_y += 100
        if self.facing == "LEFT":  line_x -= 100
        if self.facing == "RIGHT": line_x += 100
        if self.facing == "TOP-LEFT": 
            line_x -= 100
            line_y += 100
        if self.facing == "TOP-RIGHT": 
            line_x += 100
            line_y += 100
        if self.facing == "DOWN-LEFT": 
            line_y -= 100
            line_x -= 100
        if self.facing == "DOWN-RIGHT": 
            line_y -= 100
            line_x += 100
        distance_to_rock = np.hypot(line_x - self.rock_x, line_y - self.rock_y)
        hit = False
        if distance_to_rock < self.rock_size:
            print("ahh someone is touching me !! ")
            hit = True
        obs = np.array([self.agent_x, self.agent_y], dtype=np.float32)
        reward = 1.0 if hit else 0.0
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {"hit": hit}

    def render(self):
        self.window.fill((34, 139, 34))
        pygame.draw.circle(self.window, (120, 120, 120), (self.rock_x, self.rock_y), self.rock_size)
        pygame.draw.circle(self.window, (255, 255, 0), (int(self.agent_x), int(self.agent_y)), 15)
        line_x, line_y = self.agent_x, self.agent_y
        if self.facing == "UP":    line_y -= 100
        if self.facing == "DOWN":  line_y += 100
        if self.facing == "LEFT":  line_x -= 100
        if self.facing == "RIGHT": line_x += 100
        if self.facing == "TOP-LEFT": 
            line_x -= 100
            line_y += 100
        if self.facing == "TOP-RIGHT": 
            line_x += 100
            line_y += 100
        if self.facing == "DOWN-LEFT": 
            line_y -= 100
            line_x -= 100
        if self.facing == "DOWN-RIGHT": 
            line_y -= 100
            line_x += 100
        pygame.draw.line(self.window, (255, 255, 255), (int(self.agent_x), int(self.agent_y)), (int(line_x), int(line_y)), 3)
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(30)

if __name__ == "__main__":
    env = SimpleGrassEnv()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        action = None
        if keys[pygame.K_w]: 
            action = 0
        if keys[pygame.K_s]: 
            action = 1
        if keys[pygame.K_a]: 
            action = 2
        if keys[pygame.K_d]: 
            action = 3
        if keys[pygame.K_q]: 
            action = 4
        if keys[pygame.K_r]: 
            action = 5
        if keys[pygame.K_z]: 
            action = 6
        if keys[pygame.K_c]: 
            action = 7
        if action is not None:
            env.step(action)
        env.render()

    pygame.quit()