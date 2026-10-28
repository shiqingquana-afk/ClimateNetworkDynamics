from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoupledLorenz63Config:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    kappa: float = 0.5


@dataclass(frozen=True)
class UndirectedGraph:
    num_nodes: int
    edge_u: np.ndarray
    edge_v: np.ndarray
    degrees: np.ndarray

    @property
    def num_edges(self) -> int:
        return int(self.edge_u.shape[0])


def _is_connected(num_nodes: int, edge_u: np.ndarray, edge_v: np.ndarray) -> bool:
    if num_nodes <= 1:
        return True
    if edge_u.size == 0:
        return False
    neighbors: list[list[int]] = [[] for _ in range(num_nodes)]
    for u, v in zip(edge_u.tolist(), edge_v.tolist()):
        neighbors[u].append(v)
        neighbors[v].append(u)

    visited = np.zeros(num_nodes, dtype=bool)
    stack = [0]
    visited[0] = True
    while stack:
        node = stack.pop()
        for nxt in neighbors[node]:
            if not visited[nxt]:
                visited[nxt] = True
                stack.append(nxt)
    return bool(np.all(visited))


def _sample_er_edges(n_nodes: int, p: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    upper = np.triu(rng.random((n_nodes, n_nodes)) < p, k=1)
    edge_u, edge_v = np.where(upper)
    return edge_u.astype(np.int64), edge_v.astype(np.int64)


def _sample_connected_graph_with_target_density(
    n_nodes: int,
    p: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    # fallback used when rejection sampling is impractical:
    # build a random spanning tree, then add random edges to match expected ER edge count.
    perm = rng.permutation(n_nodes)
    tree_u = perm[:-1]
    tree_v = perm[1:]

    total_pairs = n_nodes * (n_nodes - 1) // 2
    target_edges = int(round(p * total_pairs))
    tree_edges = n_nodes - 1
    remaining_pairs = total_pairs - tree_edges
    extra_needed = max(target_edges - tree_edges, 0)
    q = 0.0 if remaining_pairs == 0 else min(1.0, extra_needed / remaining_pairs)

    upper = np.triu(rng.random((n_nodes, n_nodes)) < q, k=1)
    upper[tree_u, tree_v] = False
    upper[tree_v, tree_u] = False

    extra_u, extra_v = np.where(upper)
    edge_u = np.concatenate([tree_u.astype(np.int64), extra_u.astype(np.int64)])
    edge_v = np.concatenate([tree_v.astype(np.int64), extra_v.astype(np.int64)])
    return edge_u, edge_v


def generate_connected_er_graph(
    n_nodes: int,
    p: float,
    seed: int,
    max_attempts: int = 10_000,
) -> UndirectedGraph:
    if n_nodes <= 1:
        raise ValueError("n_nodes must be > 1.")
    if not (0.0 < p <= 1.0):
        raise ValueError("p must be in (0, 1].")

    base_rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        # keep deterministic stream for a given seed
        sub_seed = int(base_rng.integers(0, np.iinfo(np.int32).max))
        rng = np.random.default_rng(sub_seed)
        edge_u, edge_v = _sample_er_edges(n_nodes=n_nodes, p=p, rng=rng)
        if edge_u.size == 0:
            continue
        degrees = np.bincount(
            np.concatenate([edge_u, edge_v]),
            minlength=n_nodes,
        ).astype(np.float64)
        if np.any(degrees == 0):
            continue
        if _is_connected(n_nodes, edge_u, edge_v):
            return UndirectedGraph(
                num_nodes=n_nodes,
                edge_u=edge_u,
                edge_v=edge_v,
                degrees=degrees,
            )
    edge_u, edge_v = _sample_connected_graph_with_target_density(
        n_nodes=n_nodes,
        p=p,
        rng=base_rng,
    )
    degrees = np.bincount(
        np.concatenate([edge_u, edge_v]),
        minlength=n_nodes,
    ).astype(np.float64)
    return UndirectedGraph(
        num_nodes=n_nodes,
        edge_u=edge_u,
        edge_v=edge_v,
        degrees=degrees,
    )


def _neighbor_sum(edge_u: np.ndarray, edge_v: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    np.add.at(out, edge_u, x[edge_v])
    np.add.at(out, edge_v, x[edge_u])
    return out


def coupled_lorenz63_rhs(
    t: float,
    state: np.ndarray,
    graph: UndirectedGraph,
    config: CoupledLorenz63Config,
) -> np.ndarray:
    del t
    n = graph.num_nodes
    if state.shape[0] != 3 * n:
        raise ValueError(f"State dimension must be 3N={3 * n}, got {state.shape[0]}.")
    x = state[:n]
    y = state[n : 2 * n]
    z = state[2 * n :]

    neighbor_sum = _neighbor_sum(graph.edge_u, graph.edge_v, x)
    coupling = config.kappa * (neighbor_sum - graph.degrees * x) / graph.degrees

    dx = config.sigma * (y - x) + coupling
    dy = x * (config.rho - z) - y
    dz = x * y - config.beta * z

    return np.concatenate([dx, dy, dz], axis=0)


def rk4_integrate(
    init_state: np.ndarray,
    n_steps: int,
    dt: float,
    graph: UndirectedGraph,
    config: CoupledLorenz63Config,
) -> np.ndarray:
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    dim = init_state.shape[0]
    traj = np.empty((dim, n_steps), dtype=np.float64)
    y = init_state.astype(np.float64, copy=True)
    traj[:, 0] = y
    t = 0.0
    for i in range(1, n_steps):
        k1 = coupled_lorenz63_rhs(t, y, graph, config)
        k2 = coupled_lorenz63_rhs(t + 0.5 * dt, y + 0.5 * dt * k1, graph, config)
        k3 = coupled_lorenz63_rhs(t + 0.5 * dt, y + 0.5 * dt * k2, graph, config)
        k4 = coupled_lorenz63_rhs(t + dt, y + dt * k3, graph, config)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        traj[:, i] = y
        t += dt
    return traj


def sample_initial_state(num_nodes: int, rng: np.random.Generator) -> np.ndarray:
    # close to Lorenz63's common chaotic regime but with per-node perturbation
    x = rng.normal(loc=1.0, scale=2.0, size=num_nodes)
    y = rng.normal(loc=1.0, scale=2.0, size=num_nodes)
    z = rng.normal(loc=20.0, scale=3.0, size=num_nodes)
    return np.concatenate([x, y, z], axis=0).astype(np.float64)
