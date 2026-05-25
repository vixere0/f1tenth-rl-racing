import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

class SimpleRacingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0.0, high=30.0, shape=(21,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        self.x = 8.5
        self.y = 0.0
        self.theta = np.pi / 2
        self.speed = 0.0
        self.steps = 0
        return self._get_obs(), {}

    def _on_track(self, x, y):
        outer = (x/11.0)**2 + (y/6.0)**2
        inner = (x/7.0)**2 + (y/3.0)**2
        return inner >= 1.0 and outer <= 1.0

    def _get_lidar(self):
        angles = np.linspace(-np.pi*0.75, np.pi*0.75, 20)
        ranges = []
        for angle in angles:
            ray_angle = self.theta + angle
            hit = 10.0
            for dist in np.arange(0.1, 10.0, 0.1):
                rx = self.x + dist * np.cos(ray_angle)
                ry = self.y + dist * np.sin(ray_angle)
                if not self._on_track(rx, ry):
                    hit = dist
                    break
            ranges.append(hit)
        return np.array(ranges, dtype=np.float32)

    def _get_obs(self):
        return np.append(self._get_lidar(), self.speed).astype(np.float32)

    def step(self, action):
        steering = float(np.clip(action[0], -1.0, 1.0)) * 0.35
        throttle = float(np.clip(action[1], -1.0, 1.0))
        self.speed = np.clip(self.speed + throttle * 0.08, 0.0, 3.0)
        self.theta += steering * self.speed * 0.1
        self.x += self.speed * np.cos(self.theta) * 0.1
        self.y += self.speed * np.sin(self.theta) * 0.1
        self.steps += 1
        obs = self._get_obs()
        min_dist = np.min(obs[:-1])
        crashed = not self._on_track(self.x, self.y)
        reward = self.speed * 2.0
        reward -= abs(steering) * 0.05
        reward -= max(0, 1.0 - min_dist) * 0.5
        reward -= 0.1
        if crashed:
            reward = -50.0
        terminated = crashed or self.steps > 3000
        return obs, reward, terminated, False, {}

# Modeli yükle ve test et
model = PPO.load("ppo_f1tenth_final")
env = SimpleRacingEnv()
obs, _ = env.reset()

total_reward = 0
speeds = []
crash_count = 0

for i in range(3000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, _, _ = env.step(action)
    total_reward += reward
    speeds.append(obs[-1])
    if terminated:
        crash_count += 1
        obs, _ = env.reset()

print(f"Total Reward: {total_reward:.2f}")
print(f"Average Speed: {np.mean(speeds):.2f} m/s")
print(f"Max Speed: {np.max(speeds):.2f} m/s")
print(f"Collisions: {crash_count}")
