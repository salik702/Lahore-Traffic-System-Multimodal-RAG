"""Multi-agent DQN for signal control. Torch is already a backend dependency via Ultralytics."""
from __future__ import annotations

import json
import math
import random
from collections import deque
from pathlib import Path

import numpy as np

OBS_SIZE = 16
N_ACTIONS = 4


def observation_vector(queues: list[float], waiting: list[float], occupancy: float, phase: int, elapsed: float, neighbor: float, n_phases: int) -> list[float]:
    q = (queues + [0, 0, 0, 0])[:4]
    w = (waiting + [0, 0, 0, 0])[:4]
    phase_oh = [1.0 if i == phase else 0.0 for i in range(4)]
    return [
        *[min(x / 20.0, 2.0) for x in q],
        *[min(x / 40.0, 2.0) for x in w],
        min(occupancy, 2.0),
        min(elapsed / 45.0, 2.0),
        min(neighbor / 20.0, 2.0),
        n_phases / 4.0,
        *phase_oh,
    ][:OBS_SIZE]


class DQNPolicy:
    def __init__(self, path: Path | None = None):
        self.available = False
        self.net = None
        self.n_actions = N_ACTIONS
        try:
            import torch
            from torch import nn
        except Exception:
            return
        self.torch = torch
        self.nn = nn
        self.net = nn.Sequential(
            nn.Linear(OBS_SIZE, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, N_ACTIONS),
        )
        self.target = nn.Sequential(
            nn.Linear(OBS_SIZE, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, N_ACTIONS),
        )
        self.target.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.available = True
        self.path = path
        if path and Path(path).is_file():
            self.load_weights(Path(path))

    def load_weights(self, path: Path) -> None:
        if not self.available:
            return
        state = self.torch.load(path, map_location="cpu", weights_only=True)
        self.net.load_state_dict(state)
        self.target.load_state_dict(state)

    def save(self, path: Path) -> None:
        if not self.available:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(self.net.state_dict(), path)

    @classmethod
    def load(cls, path: str | None):
        if not path:
            return None
        file = Path(path)
        if not file.is_file():
            return None
        policy = cls(file)
        return policy if policy.available else None

    def act(self, observation: list[float], n_phases: int, greedy: bool = False, epsilon: float = 0.0) -> int:
        n = max(1, min(n_phases, N_ACTIONS))
        if (not greedy and random.random() < epsilon) or not self.available or self.net is None:
            return random.randrange(n)
        with self.torch.no_grad():
            tensor = self.torch.tensor(observation, dtype=self.torch.float32).unsqueeze(0)
            q = self.net(tensor)[0, :n]
            return int(q.argmax().item())

    def learn(self, batch: list[tuple], gamma: float = 0.97) -> float | None:
        if not self.available or len(batch) < 16:
            return None
        obs, actions, rewards, next_obs, dones = zip(*batch)
        device_obs = self.torch.tensor(obs, dtype=self.torch.float32)
        acts = self.torch.tensor(actions, dtype=self.torch.int64).unsqueeze(1)
        rew = self.torch.tensor(rewards, dtype=self.torch.float32)
        nxt = self.torch.tensor(next_obs, dtype=self.torch.float32)
        done = self.torch.tensor(dones, dtype=self.torch.float32)
        q = self.net(device_obs).gather(1, acts).squeeze(1)
        with self.torch.no_grad():
            target = rew + gamma * (1 - done) * self.target(nxt).max(1).values
        loss = self.nn.functional.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())

    def sync_target(self) -> None:
        if self.available:
            self.target.load_state_dict(self.net.state_dict())


def train_policy(make_engine, episodes: int, steps: int, save_path: Path, seed: int = 7, resume: bool = True, on_episode=None) -> dict:
    policy = DQNPolicy(save_path if resume and save_path.is_file() else None)
    if not policy.available:
        return {"trained": False, "reason": "PyTorch unavailable; use the max-pressure baseline."}
    buffer: deque = deque(maxlen=8_000)
    history = []
    rng = random.Random(seed)
    for episode in range(episodes):
        engine = make_engine(seed + episode, "max_pressure")
        engine.bank.policy = policy
        engine.bank.mode = "rl"
        episode_reward = 0.0
        last_obs = engine.observations()
        epsilon = max(0.05, 0.6 * math.exp(-episode / max(8, episodes / 3)))
        for step in range(steps):
            policy.epsilon = epsilon
            reward = engine.step()
            episode_reward += reward
            next_obs = engine.observations()
            done = step == steps - 1
            for jid in engine.bank.agents:
                action = engine.bank.last_actions.get(jid, 0)
                buffer.append((last_obs[jid], action % N_ACTIONS, reward / max(1, len(engine.bank.agents)), next_obs[jid], float(done)))
            last_obs = next_obs
            if len(buffer) >= 64:
                batch = [buffer[i] for i in rng.sample(range(len(buffer)), 64)]
                policy.learn(batch)
        if episode % 4 == 0:
            policy.sync_target()
        metrics = engine.metrics()
        history.append({"episode": episode, "reward": round(episode_reward, 2), "wait": metrics["average_waiting_time"], "throughput": metrics["vehicles_completed"]})
        if on_episode:
            on_episode(episode + 1, episodes, history[-1])
    policy.epsilon = 0.0
    policy.save(save_path)
    meta = {"trained": True, "episodes": episodes, "steps": steps, "seed": seed, "path": str(save_path), "history": history}
    save_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
