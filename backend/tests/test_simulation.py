"""Automated test suite for Lahore Traffic Digital Twin simulation."""
from pathlib import Path
import pytest
from app.core.config import get_settings
from app.simulation.compare import compare_metrics
from app.simulation.controllers import SignalAgent, fixed_time_action, max_pressure_action
from app.simulation.engine import Engine
from app.simulation.manager import TwinManager
from app.simulation.metrics import congestion_level, empty_metrics, percentage_change
from app.simulation.osm_network import load_or_build, synthetic_network
from app.simulation.risk import following_risk
from app.simulation.rl import DQNPolicy, observation_vector
from app.simulation.scenarios import SCENARIOS, resolve_scenario
from app.simulation.store import ExperimentStore
from app.simulation.sumo_detect import detect_sumo


def test_sumo_detection():
    settings = get_settings()
    info = detect_sumo(settings)
    assert isinstance(info, dict)
    assert "installed" in info
    assert info["installed"] is True
    assert info["sumo_binary"] is not None
    assert "sumo.exe" in str(info["sumo_binary"]).lower()


def test_osm_network_loading():
    settings = get_settings()
    net = load_or_build(settings)
    assert isinstance(net, dict)
    assert len(net["nodes"]) > 0
    assert len(net["edges"]) > 0
    assert len(net["junctions"]) > 0
    assert any(j.get("controllable") for j in net["junctions"].values())
    # Verify bounds
    bounds = net["bounds"]
    assert bounds["min_lat"] < bounds["max_lat"]
    assert bounds["min_lon"] < bounds["max_lon"]


def test_percentage_change_and_metrics():
    # Regular change
    assert percentage_change(120, 100) == 20.0
    assert percentage_change(80, 100) == -20.0
    # Zero handling
    assert percentage_change(0, 0) is None
    assert percentage_change(50, 0) is None

    # Congestion level categorization
    assert congestion_level(1.0, 0.1, 5.0) == "low"
    assert congestion_level(15.0, 0.9, 50.0) in {"high", "severe"}


def test_accident_risk_kinematics():
    # Normal safe car following
    safe = following_risk(leader_speed=12.0, follower_speed=12.0, gap=15.0, follower_decel=0.0)
    assert safe is None

    # Near miss: high closing speed with small gap
    near_miss = following_risk(leader_speed=2.0, follower_speed=12.0, gap=4.0, follower_decel=-2.0)
    assert near_miss is not None
    assert near_miss["kind"] in {"near_miss", "harsh_braking"}
    assert near_miss["ttc"] is not None
    assert near_miss["ttc"] < 1.5

    # Collision overlap
    collision = following_risk(leader_speed=0.0, follower_speed=5.0, gap=0.1, follower_decel=-5.0)
    assert collision is not None
    assert collision["kind"] == "simulated_collision"


def test_signal_agent_safety_interlocks():
    phases = [
        {"id": 0, "green": ["e1", "e2"], "min_green": 10.0, "max_green": 30.0, "yellow": 3.0},
        {"id": 1, "green": ["e3", "e4"], "min_green": 10.0, "max_green": 30.0, "yellow": 3.0},
    ]
    agent = SignalAgent(junction_id="j1", phases=phases)

    # Initial state
    assert agent.phase == 0
    assert agent.current_green() == {"e1", "e2"}
    assert agent.in_yellow is False

    # Cannot switch before min_green
    agent.tick_safety(dt=5.0, desired_phase=1)
    assert agent.phase == 0
    assert agent.in_yellow is False

    # At elapsed >= min_green, desired switch enters yellow
    agent.tick_safety(dt=6.0, desired_phase=1)
    assert agent.in_yellow is True
    # During yellow, green set is empty (safe clearance interval)
    assert agent.current_green() == set()

    # After yellow duration, switch completes
    agent.tick_safety(dt=3.5, desired_phase=1)
    assert agent.in_yellow is False
    assert agent.phase == 1
    assert agent.current_green() == {"e3", "e4"}


def test_simulation_engine_run():
    net = synthetic_network()
    engine = Engine(network=net, seed=42, controller_mode="fixed_time", scenario_name="normal")
    assert engine.time == 0.0

    # Step forward 20 seconds
    for _ in range(40):  # 40 steps * 0.5s = 20s
        engine.step()

    assert engine.time >= 20.0
    snap = engine.snapshot()
    assert "vehicles" in snap
    assert "signals" in snap
    assert "metrics" in snap
    metrics = snap["metrics"]
    assert metrics["sim_time"] >= 20.0
    assert metrics["congestion_level"] in {"low", "moderate", "high", "severe"}


def test_scenario_resolution():
    net = synthetic_network()
    normal = resolve_scenario("normal", net)
    assert normal["demand_scale"] == 1.0

    peak = resolve_scenario("peak", net)
    assert peak["demand_scale"] > normal["demand_scale"]

    closed = resolve_scenario("closed_road", net)
    assert len(closed["closed_edges"]) > 0


def test_comparison_logic():
    base_metrics = empty_metrics()
    base_metrics.update({"average_waiting_time": 25.0, "throughput_per_hour": 600.0, "vehicles_completed": 40})

    treat_metrics = empty_metrics()
    treat_metrics.update({"average_waiting_time": 18.0, "throughput_per_hour": 720.0, "vehicles_completed": 48})

    res = compare_metrics(base_metrics, treat_metrics)
    assert "rows" in res
    assert "claim" in res
    assert res["claim"] == "ai_lower_waiting_time"
    wait_row = next(r for r in res["rows"] if r["metric"] == "average_waiting_time")
    assert wait_row["percent_change"] < 0  # waiting time decreased


def test_rl_dqn_policy():
    policy = DQNPolicy()
    assert policy.available is True
    obs = observation_vector([5, 2, 0, 0], [10, 4, 0, 0], 0.3, 0, 15.0, 3.0, 2)
    assert len(obs) == 16
    action = policy.act(obs, n_phases=2, greedy=True)
    assert action in {0, 1}


def test_twin_manager_lifecycle():
    settings = get_settings()
    twin = TwinManager(settings)
    assert twin.status["running"] is False

    # Snapshot
    snap = twin.get_snapshot()
    assert "signals" in snap
    assert "metrics" in snap

    # RAG Context
    rag_ctx = twin.get_simulation_context_for_rag()
    assert "LAHORE TRAFFIC DIGITAL TWIN" in rag_ctx
    assert "Kalma Chowk" in rag_ctx
