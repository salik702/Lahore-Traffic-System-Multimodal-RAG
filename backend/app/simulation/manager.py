"""Digital Twin simulation manager coordinating engines, experiments, RL training, and API state."""
from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.simulation.compare import compare_metrics
from app.simulation.engine import Engine
from app.simulation.metrics import empty_metrics
from app.simulation.osm_network import load_or_build
from app.simulation.rl import DQNPolicy, train_policy
from app.simulation.scenarios import SCENARIOS, resolve_scenario
from app.simulation.store import ExperimentStore
from app.simulation.sumo_detect import detect_sumo
from app.simulation.sumo_engine import SumoLiveEngine

logger = logging.getLogger(__name__)


class TwinManager:
    """Coordinates lifecycle, live streaming, scenario comparisons, and RL training."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.network = load_or_build(settings)
        self.sumo_info = detect_sumo(settings)
        self.store = ExperimentStore(settings.database_path)
        self.model_path = settings.project_root / "data" / "simulation" / "kalma_chowk" / "rl_model.pt"

        # Active simulation state
        self.seed = 42
        self.controller_mode = "fixed_time"  # "fixed_time", "max_pressure", "rl"
        self.scenario_name = "normal"
        self.demand_scale = 1.0
        self.speed_multiplier = 1.0
        self.is_running = False
        self.engine_type = "native_osm"  # or "sumo_traci" if TraCI active

        # Engine instance
        policy_path = str(self.model_path) if self.model_path.is_file() else None
        self.engine = Engine(
            network=self.network,
            seed=self.seed,
            controller_mode=self.controller_mode,
            scenario_name=self.scenario_name,
            demand_scale=self.demand_scale,
            policy_path=policy_path,
        )

        self._loop_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._training_job: asyncio.Task | None = None
        self._training_status: dict = {
            "status": "idle",
            "episodes": 0,
            "current_episode": 0,
            "progress": 0.0,
            "rewards": [],
            "error": None,
        }

    @property
    def status(self) -> dict:
        return {
            "running": self.is_running,
            "engine": self.engine_type,
            "sumo": self.sumo_info,
            "controller_mode": self.controller_mode,
            "scenario": self.scenario_name,
            "demand_scale": self.demand_scale,
            "speed": self.speed_multiplier,
            "sim_time": round(self.engine.time, 1),
            "vehicles_active": len(self.engine.vehicles),
            "vehicles_completed": self.engine.completed,
            "rl_model_available": self.model_path.is_file(),
            "training": self._training_status,
        }

    def get_network(self) -> dict:
        return self.engine.network_view()

    def get_snapshot(self) -> dict:
        snapshot = self.engine.snapshot()
        snapshot.update({
            "running": self.is_running,
            "speed": self.speed_multiplier,
            "controller_mode": self.controller_mode,
            "scenario": self.scenario_name,
            "demand_scale": self.demand_scale,
            "engine_type": self.engine_type,
            "sumo_detected": self.sumo_info.get("installed", False),
        })
        return snapshot

    def get_metrics(self) -> dict:
        return self.engine.metrics()

    def get_risk_events(self, limit: int = 50) -> list[dict]:
        return self.engine.risk_events[-limit:]

    async def start(self) -> dict:
        async with self._lock:
            if not self.is_running:
                self.is_running = True
                if self._loop_task is None or self._loop_task.done():
                    self._loop_task = asyncio.create_task(self._run_loop())
        return self.status

    async def pause(self) -> dict:
        async with self._lock:
            self.is_running = False
        return self.status

    async def resume(self) -> dict:
        return await self.start()

    async def reset(
        self,
        seed: int | None = None,
        controller_mode: str | None = None,
        scenario: str | None = None,
        demand_scale: float | None = None,
    ) -> dict:
        async with self._lock:
            if seed is not None:
                self.seed = seed
            if controller_mode is not None:
                self.controller_mode = controller_mode
            if scenario is not None:
                self.scenario_name = scenario
            if demand_scale is not None:
                self.demand_scale = demand_scale

            policy_path = str(self.model_path) if self.model_path.is_file() else None
            self.engine.policy_path = policy_path
            self.engine.reset(
                seed=self.seed,
                mode=self.controller_mode,
                scenario=self.scenario_name,
                demand_scale=self.demand_scale,
            )
        return self.status

    def set_speed(self, speed: float) -> dict:
        self.speed_multiplier = max(0.25, min(float(speed), 10.0))
        return self.status

    def set_demand(self, demand: float) -> dict:
        self.demand_scale = max(0.2, min(float(demand), 4.0))
        self.engine.demand_scale = self.demand_scale
        return self.status

    def set_mode(self, mode: str) -> dict:
        if mode in {"fixed_time", "max_pressure", "rl"}:
            self.controller_mode = mode
            self.engine.controller_mode = mode
            policy = DQNPolicy.load(str(self.model_path)) if mode == "rl" else None
            effective_mode = mode if (mode != "rl" or policy) else "max_pressure"
            self.engine.bank.mode = effective_mode
            self.engine.bank.policy = policy
        return self.status

    def set_scenario(self, scenario: str) -> dict:
        if scenario in SCENARIOS:
            self.scenario_name = scenario
            # Selecting a named scenario should apply its preset arrival rate.
            # The demand slider can then override it explicitly afterwards.
            self.demand_scale = float(SCENARIOS[scenario]["demand_scale"])
            self.engine.reset(
                seed=self.seed,
                mode=self.controller_mode,
                scenario=scenario,
                demand_scale=self.demand_scale,
            )
        return self.status

    async def step(self) -> dict:
        async with self._lock:
            self.engine.step()
        return self.get_snapshot()

    async def _run_loop(self) -> None:
        try:
            while True:
                if not self.is_running:
                    await asyncio.sleep(0.15)
                    continue

                # Sleep adjusted for speed multiplier
                sleep_sec = max(0.015, (self.engine.dt / self.speed_multiplier))
                await asyncio.sleep(sleep_sec)

                async with self._lock:
                    if self.is_running:
                        self.engine.step()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Simulation loop error")
            self.is_running = False

    def list_scenarios(self) -> list[dict]:
        return list(SCENARIOS.values())

    async def run_scenario_experiment(
        self,
        scenario: str,
        controller: str = "fixed_time",
        demand_scale: float = 1.0,
        duration: float = 120.0,
        seed: int = 42,
    ) -> dict:
        """Run an isolated headless simulation experiment and persist its results."""
        def _execute():
            policy_path = str(self.model_path) if (controller == "rl" and self.model_path.is_file()) else None
            exp_engine = Engine(
                network=deepcopy(self.network),
                seed=seed,
                controller_mode=controller,
                scenario_name=scenario,
                demand_scale=demand_scale,
                policy_path=policy_path,
            )
            metrics = exp_engine.run(duration)
            config = {
                "scenario": scenario,
                "controller": controller,
                "demand_scale": demand_scale,
                "duration": duration,
                "seed": seed,
            }
            exp_id = self.store.save_experiment(
                kind="scenario_run",
                controller=controller,
                scenario=scenario,
                seed=seed,
                demand_scale=demand_scale,
                duration=duration,
                engine="native_osm",
                config=config,
                metrics=metrics,
                notes=f"What-if scenario run: {scenario}",
            )
            return {
                "experiment_id": exp_id,
                "scenario": scenario,
                "controller": controller,
                "metrics": metrics,
                "config": config,
                "provenance": "simulated",
            }

        return await asyncio.to_thread(_execute)

    async def run_comparison(
        self,
        scenario: str = "normal",
        demand_scale: float = 1.0,
        duration: float = 120.0,
        seed: int = 42,
    ) -> dict:
        """Run identical baseline (fixed_time) and treatment (AI / RL) simulations."""
        def _execute():
            # 1. Baseline: fixed_time
            base_engine = Engine(
                network=deepcopy(self.network),
                seed=seed,
                controller_mode="fixed_time",
                scenario_name=scenario,
                demand_scale=demand_scale,
            )
            base_metrics = base_engine.run(duration)

            # 2. Treatment: RL (or max_pressure if no weights)
            treatment_mode = "rl" if self.model_path.is_file() else "max_pressure"
            policy_path = str(self.model_path) if treatment_mode == "rl" else None
            treat_engine = Engine(
                network=deepcopy(self.network),
                seed=seed,
                controller_mode=treatment_mode,
                scenario_name=scenario,
                demand_scale=demand_scale,
                policy_path=policy_path,
            )
            treat_metrics = treat_engine.run(duration)

            # Calculate metrics delta
            comparison = compare_metrics(base_metrics, treat_metrics)
            config = {
                "scenario": scenario,
                "baseline_controller": "fixed_time",
                "treatment_controller": treatment_mode,
                "demand_scale": demand_scale,
                "duration": duration,
                "seed": seed,
            }
            exp_id = self.store.save_experiment(
                kind="comparison",
                controller=f"fixed_time_vs_{treatment_mode}",
                scenario=scenario,
                seed=seed,
                demand_scale=demand_scale,
                duration=duration,
                engine="native_osm",
                config=config,
                metrics=treat_metrics,
                comparison=comparison,
                notes=f"Comparison: Fixed-Time vs {treatment_mode.upper()}",
            )
            return {
                "experiment_id": exp_id,
                "baseline_metrics": base_metrics,
                "treatment_metrics": treat_metrics,
                "comparison": comparison,
                "config": config,
            }

        return await asyncio.to_thread(_execute)

    def list_experiments(self, limit: int = 40) -> list[dict]:
        return self.store.list_experiments(limit)

    def get_experiment(self, experiment_id: str) -> dict | None:
        return self.store.get(experiment_id)

    async def start_training(self, episodes: int = 5, episode_duration: float = 60.0) -> dict:
        """Start RL model training in a non-blocking background task."""
        episodes = max(1, min(int(episodes), 100))
        episode_duration = max(10.0, min(float(episode_duration), 3600.0))
        if self._training_job and not self._training_job.done():
            return {"status": "already_training", "training": self._training_status}

        self._training_status = {
            "status": "running",
            "episodes": episodes,
            "current_episode": 0,
            "progress": 0.0,
            "rewards": [],
            "error": None,
        }
        self._training_job = asyncio.create_task(
            self._train_worker(episodes, episode_duration)
        )
        return {"status": "started", "training": self._training_status}

    async def _train_worker(self, episodes: int, episode_duration: float) -> None:
        def _train():
            def make_engine(seed: int, mode: str):
                return Engine(network=deepcopy(self.network), seed=seed, controller_mode=mode, scenario_name="normal", demand_scale=1.0)

            def update_progress(current: int, total: int, record: dict) -> None:
                self._training_status["current_episode"] = current
                self._training_status["progress"] = round(current / total * 100, 1)
                self._training_status["rewards"].append(record["reward"])

            result = train_policy(
                make_engine, episodes, max(1, int(episode_duration / self.engine.dt)),
                self.model_path, seed=self.seed, resume=True, on_episode=update_progress,
            )
            if not result.get("trained"):
                raise RuntimeError(result.get("reason", "RL training could not start."))
            policy = DQNPolicy.load(str(self.model_path))
            if policy is None:
                raise RuntimeError("Training completed but its saved model could not be loaded.")
            self.engine.policy_path = str(self.model_path)
            self.engine.bank.policy = policy
            self.store.save_training(
                status="completed",
                payload=result,
                episodes=episodes,
            )
            return [item["reward"] for item in result["history"]]

        try:
            rewards = await asyncio.to_thread(_train)
            self._training_status["status"] = "completed"
            self._training_status["rewards"] = rewards
        except Exception as exc:
            logger.exception("RL training failed")
            self._training_status["status"] = "failed"
            self._training_status["error"] = str(exc)

    def get_simulation_context_for_rag(self) -> str:
        """Provide structured digital-twin context to Ground the RAG Chat Assistant."""
        snap = self.get_snapshot()
        metrics = snap["metrics"]
        signals = snap["signals"]
        hotspots = snap["risk_events"]

        longest_queue_sig = max(
            signals,
            key=lambda s: max((inc["queue"] for inc in s["incoming"]), default=0),
            default=None,
        ) if signals else None

        q_info = "None"
        if longest_queue_sig:
            max_q = max((inc["queue"] for inc in longest_queue_sig["incoming"]), default=0)
            q_info = f"{longest_queue_sig['name']} (Max Queue: {max_q} vehicles, Phase: {longest_queue_sig['display']})"

        context_lines = [
            "### LAHORE TRAFFIC DIGITAL TWIN SIMULATION OBSERVATIONS",
            f"- **Map / Network**: {self.network.get('area', 'Kalma Chowk, Lahore')}",
            f"- **Active Scenario**: {self.scenario_name} (Traffic Demand Scale: {self.demand_scale}x)",
            f"- **Signal Control Mode**: {self.controller_mode} (AI RL Available: {self.model_path.is_file()})",
            f"- **Congestion Level**: {metrics.get('congestion_level', 'unknown').upper()}",
            f"- **Average Waiting Time**: {metrics.get('average_waiting_time', 0)} seconds",
            f"- **Average Travel Time**: {metrics.get('average_travel_time', 0)} seconds",
            f"- **Average Queue Length**: {metrics.get('average_queue_length', 0)} vehicles",
            f"- **Max Queue Length**: {metrics.get('max_queue_length', 0)} vehicles",
            f"- **Active Vehicles**: {metrics.get('total_vehicles', 0)} | **Completed**: {metrics.get('vehicles_completed', 0)}",
            f"- **Throughput**: {metrics.get('throughput_per_hour', 0)} vehicles/hour",
            f"- **Critical Intersection Bottleneck**: {q_info}",
            f"- **Simulated Risk Events**: {metrics.get('risk_events', 0)} near-miss/braking events (Simulated Collisions: {metrics.get('simulated_collisions', 0)})",
            "- **Note**: These metrics reflect model simulations on the imported Lahore OpenStreetMap road graph, distinct from official historic accident records.",
        ]
        return "\n".join(context_lines)
