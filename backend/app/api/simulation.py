"""API routes for Lahore Traffic Digital Twin simulation."""
from __future__ import annotations

import asyncio
from typing import Any, Literal
from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

router = APIRouter(prefix="/simulation", tags=["simulation"])


class ControlRequest(BaseModel):
    action: Literal["start", "pause", "resume", "reset", "step"]
    seed: int | None = Field(None, ge=0, le=2**31 - 1)
    scenario: str | None = Field(None, pattern=r"^(normal|peak|high_traffic|motorcycle_heavy|closed_road|blocked_lane|incident|demand_increase|signal_failure|longer_green)$")
    controller_mode: Literal["fixed_time", "max_pressure", "rl"] | None = None
    demand_scale: float | None = Field(None, ge=0.2, le=4.0)


class ConfigRequest(BaseModel):
    speed: float | None = Field(None, ge=0.25, le=10.0)
    demand_scale: float | None = Field(None, ge=0.2, le=4.0)
    controller_mode: Literal["fixed_time", "max_pressure", "rl"] | None = None
    scenario: str | None = Field(None, pattern=r"^(normal|peak|high_traffic|motorcycle_heavy|closed_road|blocked_lane|incident|demand_increase|signal_failure|longer_green)$")
    seed: int | None = Field(None, ge=0, le=2**31 - 1)


class ScenarioExperimentRequest(BaseModel):
    scenario: str = Field(pattern=r"^(normal|peak|high_traffic|motorcycle_heavy|closed_road|blocked_lane|incident|demand_increase|signal_failure|longer_green)$")
    controller: Literal["fixed_time", "max_pressure", "rl"] = "fixed_time"
    demand_scale: float = Field(1.0, ge=0.2, le=4.0)
    duration: float = Field(120.0, ge=5.0, le=3600.0)
    seed: int = Field(42, ge=0, le=2**31 - 1)


class CompareExperimentRequest(BaseModel):
    scenario: str = Field("normal", pattern=r"^(normal|peak|high_traffic|motorcycle_heavy|closed_road|blocked_lane|incident|demand_increase|signal_failure|longer_green)$")
    demand_scale: float = Field(1.0, ge=0.2, le=4.0)
    duration: float = Field(120.0, ge=5.0, le=3600.0)
    seed: int = Field(42, ge=0, le=2**31 - 1)


class TrainingRequest(BaseModel):
    episodes: int = Field(5, ge=1, le=100)
    episode_duration: float = Field(60.0, ge=10.0, le=3600.0)


@router.get("/status")
async def get_status(request: Request):
    twin = request.app.state.twin
    return twin.status


@router.get("/network")
async def get_network(request: Request):
    twin = request.app.state.twin
    return twin.get_network()


@router.get("/snapshot")
async def get_snapshot(request: Request):
    twin = request.app.state.twin
    return twin.get_snapshot()


@router.get("/metrics")
async def get_metrics(request: Request):
    twin = request.app.state.twin
    return twin.get_metrics()


@router.get("/scenarios")
async def get_scenarios(request: Request):
    twin = request.app.state.twin
    return {"scenarios": twin.list_scenarios()}


@router.post("/control")
async def control_simulation(payload: ControlRequest, request: Request):
    twin = request.app.state.twin
    action = payload.action.lower()
    if action == "start":
        return await twin.start()
    elif action == "pause":
        return await twin.pause()
    elif action == "resume":
        return await twin.resume()
    elif action == "reset":
        return await twin.reset(
            seed=payload.seed,
            controller_mode=payload.controller_mode,
            scenario=payload.scenario,
            demand_scale=payload.demand_scale,
        )
    elif action == "step":
        return await twin.step()
    else:
        raise HTTPException(400, f"Unsupported control action: {action}")


@router.post("/config")
async def update_config(payload: ConfigRequest, request: Request):
    twin = request.app.state.twin
    if payload.speed is not None:
        twin.set_speed(payload.speed)
    if payload.demand_scale is not None:
        twin.set_demand(payload.demand_scale)
    if payload.controller_mode is not None:
        twin.set_mode(payload.controller_mode)
    if payload.scenario is not None:
        twin.set_scenario(payload.scenario)
    if payload.seed is not None:
        await twin.reset(seed=payload.seed)
    return twin.status


@router.post("/experiment/scenario")
async def run_scenario_experiment(payload: ScenarioExperimentRequest, request: Request):
    twin = request.app.state.twin
    return await twin.run_scenario_experiment(
        scenario=payload.scenario,
        controller=payload.controller,
        demand_scale=payload.demand_scale,
        duration=payload.duration,
        seed=payload.seed,
    )


@router.post("/experiment/compare")
async def run_compare_experiment(payload: CompareExperimentRequest, request: Request):
    twin = request.app.state.twin
    return await twin.run_comparison(
        scenario=payload.scenario,
        demand_scale=payload.demand_scale,
        duration=payload.duration,
        seed=payload.seed,
    )


@router.get("/experiments")
async def list_experiments(request: Request, limit: int = Query(40, ge=1, le=100)):
    twin = request.app.state.twin
    return {"experiments": twin.list_experiments(limit)}


@router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str, request: Request):
    twin = request.app.state.twin
    exp = twin.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(404, "Experiment not found")
    return exp


@router.post("/training/start")
async def start_training(payload: TrainingRequest, request: Request):
    twin = request.app.state.twin
    return await twin.start_training(payload.episodes, payload.episode_duration)


@router.get("/training/status")
async def get_training_status(request: Request):
    twin = request.app.state.twin
    return twin.status["training"]


@router.get("/risks")
async def get_risk_events(request: Request, limit: int = Query(50, ge=1, le=200)):
    twin = request.app.state.twin
    return {"risk_events": twin.get_risk_events(limit)}


@router.get("/context")
async def get_simulation_context(request: Request):
    twin = request.app.state.twin
    return {"context": twin.get_simulation_context_for_rag()}


@router.websocket("/ws")
async def simulation_websocket(websocket: WebSocket):
    await websocket.accept()
    twin = websocket.app.state.twin
    try:
        while True:
            snapshot = twin.get_snapshot()
            await websocket.send_json(snapshot)
            # Stream at ~5 Hz (every 200ms)
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
