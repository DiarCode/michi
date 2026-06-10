"""Hierarchical forecast reconciliation (MinT/OLS) for station→route→district→city aggregation."""

import numpy as np


def build_summing_matrix(
    station_district: dict[str, str],
    station_routes: dict[str, list[str]],
) -> tuple[np.ndarray, list[str]]:
    """Build the summing matrix S for hierarchical reconciliation.

    Bottom level: individual stations
    Aggregate levels: routes, districts, total

    Returns:
        S: (n_bottom + n_agg, n_bottom) summing matrix
        labels: list of series labels
    """
    stations = sorted(station_district.keys())
    station_idx = {s: i for i, s in enumerate(stations)}
    n_bottom = len(stations)

    # Collect all routes and districts
    all_routes = sorted(set(r for rs in station_routes.values() for r in rs))
    all_districts = sorted(set(station_district.values()))

    route_idx = {r: i for i, r in enumerate(all_routes)}
    district_idx = {d: i for i, d in enumerate(all_districts)}

    n_agg = len(all_routes) + len(all_districts) + 1  # routes + districts + total
    n_total = n_bottom + n_agg

    S = np.zeros((n_total, n_bottom), dtype=np.float32)

    # Bottom level: identity
    for i in range(n_bottom):
        S[i, i] = 1.0

    # Route aggregates
    offset = n_bottom
    for rid, rname in enumerate(all_routes):
        for s in stations:
            if rname in station_routes.get(s, []):
                S[offset + rid, station_idx[s]] = 1.0

    # District aggregates
    offset2 = offset + len(all_routes)
    for did, dname in enumerate(all_districts):
        for s in stations:
            if station_district.get(s) == dname:
                S[offset2 + did, station_idx[s]] = 1.0

    # Total aggregate
    S[-1, :] = 1.0

    labels = stations + [f"route_{r}" for r in all_routes] + [f"district_{d}" for d in all_districts] + ["total"]
    return S, labels


def reconcile_mint(y_hat: np.ndarray, S: np.ndarray, W_diag: np.ndarray) -> np.ndarray:
    """MinT reconciliation using diagonal weight matrix.

    Args:
        y_hat: (n_series, H) base forecasts for all series
        S: (n_series, n_bottom) summing matrix
        W_diag: (n_series,) diagonal weights (inverse variance)

    Returns:
        Reconciled forecasts (n_series, H)
    """
    W_inv = np.diag(1.0 / (W_diag + 1e-8)).astype(np.float64)
    S64 = S.astype(np.float64)
    middle = S64.T @ W_inv @ S64
    middle_inv = np.linalg.pinv(middle)
    P = S64 @ middle_inv @ S64.T @ W_inv
    return ((P @ y_hat.astype(np.float64)).T).astype(np.float32).T


def reconcile_ols(y_hat: np.ndarray, S: np.ndarray) -> np.ndarray:
    """OLS reconciliation (simplest method).

    Args:
        y_hat: (n_series, H) base forecasts
        S: (n_series, n_bottom) summing matrix

    Returns:
        Reconciled forecasts (n_series, H)
    """
    S64 = S.astype(np.float64)
    P = S64 @ np.linalg.pinv(S64.T @ S64) @ S64.T
    return ((P @ y_hat.astype(np.float64)).T).astype(np.float32).T


def coherence_error(y: np.ndarray, S: np.ndarray, bottom_dim: int) -> float:
    """Compute coherence error between bottom-level and aggregate forecasts."""
    yb = y[..., :bottom_dim]
    implied = (S @ yb.T).T
    num = np.linalg.norm((y - implied).reshape(-1))
    den = np.linalg.norm(y.reshape(-1)) + 1e-8
    return float(num / den)
