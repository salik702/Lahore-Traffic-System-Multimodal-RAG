"""Fixed-time, max-pressure, and RL signal controllers with safety interlocks."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SignalAgent:
    junction_id: str
    phases: list[dict]
    phase: int = 0
    elapsed: float = 0.0
    in_yellow: bool = False
    yellow_elapsed: float = 0.0
    pending_phase: int | None = None
    failed: bool = False
    switches: int = 0

    def current_green(self) -> set[str]:
        if self.failed:
            return set()
        if self.in_yellow:
            return set()
        return set(self.phases[self.phase]["green"])

    def display(self) -> str:
        if self.failed:
            return "failed"
        if self.in_yellow:
            return "yellow"
        return "green"

    def tick_safety(self, dt: float, desired_phase: int, phase_scale: float = 1.0) -> None:
        if self.failed:
            self.elapsed += dt
            return
        phase = self.phases[self.phase]
        min_green = phase["min_green"] * phase_scale
        max_green = phase["max_green"] * phase_scale
        yellow = phase["yellow"]
        if self.in_yellow:
            self.yellow_elapsed += dt
            if self.yellow_elapsed >= yellow:
                self.phase = self.pending_phase if self.pending_phase is not None else (self.phase + 1) % len(self.phases)
                self.pending_phase = None
                self.in_yellow = False
                self.yellow_elapsed = 0.0
                self.elapsed = 0.0
                self.switches += 1
            return
        self.elapsed += dt
        target = desired_phase % len(self.phases)
        must_switch = self.elapsed >= max_green
        may_switch = self.elapsed >= min_green and target != self.phase
        if must_switch or may_switch:
            self.in_yellow = True
            self.yellow_elapsed = 0.0
            if must_switch and target == self.phase:
                target = (self.phase + 1) % len(self.phases)
            self.pending_phase = target


def fixed_time_action(agent: SignalAgent) -> int:
    return (agent.phase + 1) % len(agent.phases) if agent.elapsed >= agent.phases[agent.phase]["min_green"] else agent.phase


def max_pressure_action(agent: SignalAgent, queues: dict[str, int], neighbor_pressure: float = 0.0) -> int:
    best = agent.phase
    best_score = -1e9
    for phase in agent.phases:
        pressure = sum(queues.get(edge_id, 0) for edge_id in phase["green"]) + 0.15 * neighbor_pressure
        if phase["id"] == agent.phase:
            pressure += 0.4  # switching cost
        if pressure > best_score:
            best_score = pressure
            best = phase["id"]
    return best


@dataclass
class ControllerBank:
    mode: str
    agents: dict[str, SignalAgent] = field(default_factory=dict)
    policy: object | None = None
    last_actions: dict[str, int] = field(default_factory=dict)

    def step(self, dt: float, queues: dict[str, int], observations: dict[str, list[float]], neighbor_queues: dict[str, float], phase_scale: float) -> dict[str, int]:
        actions = {}
        selected_actions = {}
        for jid, agent in self.agents.items():
            if self.mode == "fixed_time":
                action = fixed_time_action(agent)
            elif self.mode == "rl" and self.policy is not None:
                action = self.policy.act(observations[jid], len(agent.phases), greedy=False, epsilon=getattr(self.policy, "epsilon", 0.0))
            else:
                action = max_pressure_action(agent, queues, neighbor_queues.get(jid, 0.0))
            agent.tick_safety(dt, action, phase_scale)
            actions[jid] = agent.phase
            selected_actions[jid] = action
        self.last_actions = selected_actions
        return actions
