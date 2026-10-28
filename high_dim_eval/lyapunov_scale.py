from __future__ import annotations

import numpy as np

from coupled_lorenz63 import CoupledLorenz63Config, UndirectedGraph, coupled_lorenz63_rhs


def _rk4_step(
    state: np.ndarray,
    dt: float,
    graph: UndirectedGraph,
    system_config: CoupledLorenz63Config,
) -> np.ndarray:
    k1 = coupled_lorenz63_rhs(0.0, state, graph, system_config)
    k2 = coupled_lorenz63_rhs(0.0, state + 0.5 * dt * k1, graph, system_config)
    k3 = coupled_lorenz63_rhs(0.0, state + 0.5 * dt * k2, graph, system_config)
    k4 = coupled_lorenz63_rhs(0.0, state + dt * k3, graph, system_config)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def estimate_lambda_max(
    graph: UndirectedGraph,
    init_state: np.ndarray,
    system_config: CoupledLorenz63Config,
    warmup_steps: int = 1_024,
    estimate_steps: int = 4_096,
    base_dt: float = 1e-3,
    perturbation: float = 1e-8,
    random_seed: int = 0,
) -> float:
    if base_dt <= 0.0:
        raise ValueError("base_dt must be positive.")
    if perturbation <= 0.0:
        raise ValueError("perturbation must be positive.")
    if warmup_steps < 0 or estimate_steps <= 0:
        raise ValueError("warmup_steps must be >= 0 and estimate_steps > 0.")

    y_ref = init_state.astype(np.float64, copy=True)
    for _ in range(warmup_steps):
        y_ref = _rk4_step(y_ref, base_dt, graph, system_config)

    rng = np.random.default_rng(random_seed)
    direction = rng.normal(size=y_ref.shape[0])
    direction /= np.linalg.norm(direction)
    y_pert = y_ref + perturbation * direction

    log_growth_sum = 0.0
    for _ in range(estimate_steps):
        y_ref = _rk4_step(y_ref, base_dt, graph, system_config)
        y_pert = _rk4_step(y_pert, base_dt, graph, system_config)

        delta = y_pert - y_ref
        dist = np.linalg.norm(delta)
        if not np.isfinite(dist) or dist <= 0.0:
            raise RuntimeError("Lyapunov estimation diverged or collapsed (invalid perturbation norm).")

        log_growth_sum += float(np.log(dist / perturbation))
        y_pert = y_ref + perturbation * (delta / dist)

    return log_growth_sum / (estimate_steps * base_dt)


def derive_dt_for_steps_per_lyapunov(lambda_max: float, steps_per_lyapunov: int = 128) -> float:
    if lambda_max <= 0.0:
        raise ValueError(f"lambda_max must be > 0, got {lambda_max}.")
    if steps_per_lyapunov <= 0:
        raise ValueError("steps_per_lyapunov must be > 0.")
    return 1.0 / (float(steps_per_lyapunov) * lambda_max)
