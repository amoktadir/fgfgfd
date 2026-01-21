import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ARM (Apriori)
from mlxtend.frequent_patterns import apriori, association_rules

# MI
from sklearn.metrics import mutual_info_score

# Reliability optimization (Eq. 9) via LP
import pulp

# Optional network viz
import networkx as nx


# =========================
# Core math (paper-aligned)
# =========================
def sigmoid(x: np.ndarray, lam: float) -> np.ndarray:
    # Eq. (2)
    return 1.0 / (1.0 + np.exp(-lam * x))


def fcm_state_update(A: np.ndarray, W: np.ndarray, lam: float) -> np.ndarray:
    # Eq. (1) with convention W[i,j] = i -> j
    incoming = A @ W
    return sigmoid(incoming + A, lam=lam)


def hebbian_update(W: np.ndarray, A: np.ndarray, eta: float, mask: np.ndarray) -> np.ndarray:
    # Eq. (3) implemented elementwise for W[i,j] = i->j
    W_new = W.copy()
    A_i = A.reshape(-1, 1)
    A_j = A.reshape(1, -1)
    delta = eta * (A_j) * (A_i - (A_j * W_new))
    W_new = W_new + delta
    W_new = np.clip(W_new, 0.0, 1.0)
    W_new *= mask
    return W_new


def steady_reached(A_prev: np.ndarray, A_next: np.ndarray, tol: float = 0.001) -> bool:
    return np.max(np.abs(A_next - A_prev)) <= tol


# =========================
# Phase 1: ARM + MI topology
# =========================
def compute_mi_matrix(binary_df: pd.DataFrame) -> pd.DataFrame:
    cols = list(binary_df.columns)
    n = len(cols)
    mi = np.zeros((n, n), dtype=float)
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            if i == j:
                mi[i, j] = 0.0
            else:
                mi[i, j] = mutual_info_score(binary_df[ci], binary_df[cj])
    return pd.DataFrame(mi, index=cols, columns=cols)


def mine_association_rules(binary_df: pd.DataFrame, min_support: float, min_conf: float) -> pd.DataFrame:
    frequent = apriori(binary_df.astype(bool), min_support=min_support, use_colnames=True)
    if frequent.empty:
        return pd.DataFrame()
    rules = association_rules(frequent, metric="confidence", min_threshold=min_conf)
    if rules.empty:
        return pd.DataFrame()
    rules = rules[(rules["antecedents"].apply(len) == 1) & (rules["consequents"].apply(len) == 1)].copy()
    return rules


def hfacs_filter_edges(edges, factor_levels: dict, forbid_same_level=True):
    filtered = []
    for u, v, meta in edges:
        if u not in factor_levels or v not in factor_levels:
            continue
        if forbid_same_level and factor_levels[u] == factor_levels[v]:
            continue
        filtered.append((u, v, meta))
    return filtered


def build_edges_from_arm_mi(
    rules_df: pd.DataFrame,
    mi_df: pd.DataFrame,
    mi_top_k_pct: float,
    lift_feedback_threshold: float,
    factor_levels: dict,
):
    cols = list(mi_df.columns)

    # MI top-k (undirected -> add both directions as candidates)
    mi_values = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            mi_values.append((cols[i], cols[j], float(mi_df.iat[i, j])))
    mi_values.sort(key=lambda x: x[2], reverse=True)
    k = max(1, int(len(mi_values) * mi_top_k_pct))
    mi_top = mi_values[:k]

    edges = []

    # ARM directed rules
    if not rules_df.empty:
        for _, r in rules_df.iterrows():
            a = next(iter(r["antecedents"]))
            c = next(iter(r["consequents"]))
            edges.append((a, c, {
                "source": "ARM",
                "lift": float(r["lift"]),
                "confidence": float(r["confidence"]),
                "support": float(r["support"])
            }))

    # MI candidates (both directions)
    for a, b, val in mi_top:
        edges.append((a, b, {"source": "MI", "mi": float(val)}))
        edges.append((b, a, {"source": "MI", "mi": float(val)}))

    # Feedback edges (ARM lift threshold)
    if not rules_df.empty:
        for _, r in rules_df.iterrows():
            if float(r["lift"]) >= lift_feedback_threshold:
                a = next(iter(r["antecedents"]))
                c = next(iter(r["consequents"]))
                edges.append((a, c, {
                    "source": "ARM_feedback",
                    "lift": float(r["lift"]),
                    "confidence": float(r["confidence"]),
                    "support": float(r["support"])
                }))

    # HFACS filter: no same-level edges
    edges = hfacs_filter_edges(edges, factor_levels, forbid_same_level=True)

    # Deduplicate: keep best per (u,v) (prefer ARM higher lift > MI)
    best = {}

    def score(meta):
        src = meta.get("source", "")
        if src.startswith("ARM"):
            return 1000.0 + meta.get("lift", 0.0)
        if src == "MI":
            return 100.0 + meta.get("mi", 0.0)
        return 0.0

    for u, v, meta in edges:
        key = (u, v)
        if key not in best or score(meta) > score(best[key]):
            best[key] = meta

    return [(u, v, best[(u, v)]) for (u, v) in best.keys()]


def edges_to_adjacency(edges, nodes):
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)), dtype=int)
    meta_map = {}
    for u, v, meta in edges:
        if u in idx and v in idx and u != v:
            A[idx[u], idx[v]] = 1
            meta_map[(u, v)] = meta
    return A, meta_map


# =========================
# Phase 2: Reliability optimization (Eq. 6-9)
# =========================
def reliability_optimize_group_weights(expert_mats, mask, epsilon: float):
    m = len(expert_mats)
    n = expert_mats[0].shape[0]
    N = int(np.sum(mask))
    if N == 0:
        raise ValueError("No edges (N=0). Topology is empty; relax ARM/MI thresholds or provide adjacency.")

    prob = pulp.LpProblem("reliability_optimization", pulp.LpMaximize)

    w_opt, w_grp = {}, {}
    abs_cons, abs_dist = {}, {}

    for k in range(n):
        for h in range(n):
            if mask[k, h] == 1:
                w_grp[(k, h)] = pulp.LpVariable(f"wgrp_{k}_{h}", lowBound=0.0, upBound=1.0)
                for i in range(m):
                    w_opt[(i, k, h)] = pulp.LpVariable(f"wopt_{i}_{k}_{h}", lowBound=0.0, upBound=1.0)
                    abs_cons[(i, k, h)] = pulp.LpVariable(f"abscons_{i}_{k}_{h}", lowBound=0.0)
                    abs_dist[(i, k, h)] = pulp.LpVariable(f"absdist_{i}_{k}_{h}", lowBound=0.0)

    # group mean: w_grp = avg w_opt
    for k in range(n):
        for h in range(n):
            if mask[k, h] == 1:
                prob += w_grp[(k, h)] == (1.0 / m) * pulp.lpSum(w_opt[(i, k, h)] for i in range(m))

    # epsilon deviation |w_opt - w_grp| <= epsilon
    for k in range(n):
        for h in range(n):
            if mask[k, h] == 1:
                for i in range(m):
                    prob += w_opt[(i, k, h)] - w_grp[(k, h)] <= epsilon
                    prob += w_grp[(k, h)] - w_opt[(i, k, h)] <= epsilon

    # absolute linearizations
    for k in range(n):
        for h in range(n):
            if mask[k, h] == 1:
                for i in range(m):
                    prob += abs_cons[(i, k, h)] >= w_grp[(k, h)] - w_opt[(i, k, h)]
                    prob += abs_cons[(i, k, h)] >= w_opt[(i, k, h)] - w_grp[(k, h)]
                    w_init = float(expert_mats[i][k, h])
                    prob += abs_dist[(i, k, h)] >= w_opt[(i, k, h)] - w_init
                    prob += abs_dist[(i, k, h)] >= w_init - w_opt[(i, k, h)]

    avg_abs_cons = (1.0 / (m * N)) * pulp.lpSum(abs_cons.values())
    avg_abs_dist = (1.0 / (m * N)) * pulp.lpSum(abs_dist.values())
    gamma_expr = 1.0 - avg_abs_cons - avg_abs_dist
    prob += gamma_expr

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"Optimization failed: {pulp.LpStatus[prob.status]}")

    w_group = np.zeros((n, n), dtype=float)
    for k in range(n):
        for h in range(n):
            if mask[k, h] == 1:
                w_group[k, h] = float(pulp.value(w_grp[(k, h)]))

    CL = 1.0 - float(pulp.value(avg_abs_cons))
    D = float(pulp.value(avg_abs_dist))
    gamma = float(pulp.value(gamma_expr))
    return w_group, gamma, CL, D


# =========================
# Phase 3: λ selection + simulation
# =========================
def choose_lambda_grid(
    A0: np.ndarray,
    W0: np.ndarray,
    mask: np.ndarray,
    lambdas=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    TR: int = 10,
    Tmax: int = 50,
    eta: float = 0.05,
    tol: float = 0.001,
):
    results = []
    for lam in lambdas:
        A = A0.copy()
        W = W0.copy()
        hist = [A.copy()]
        tsteady = None

        for t in range(Tmax):
            A_next = fcm_state_update(A, W, lam)
            hist.append(A_next.copy())
            if steady_reached(A, A_next, tol=tol):
                tsteady = t + 1
                break
            W = hebbian_update(W, A_next, eta=eta, mask=mask)
            A = A_next

        hist = np.array(hist)
        if tsteady is None:
            tsteady = len(hist) - 1

        start = min(tsteady, len(hist) - 1)
        end = min(start + TR, len(hist))
        window = hist[start:end]
        if window.shape[0] < 2:
            var = float(np.var(hist[-min(TR, len(hist)):], axis=0).mean())
        else:
            var = float(np.var(window, axis=0).mean())

        results.append({"lambda": lam, "tsteady": tsteady, "var": var})

    df = pd.DataFrame(results)
    tmin, tmax = df["tsteady"].min(), df["tsteady"].max()
    vmin, vmax = df["var"].min(), df["var"].max()
    df["Nt"] = 0.0 if tmax == tmin else (df["tsteady"] - tmin) / (tmax - tmin)
    df["NVar"] = 0.0 if vmax == vmin else (df["var"] - vmin) / (vmax - vmin)
    df["Cscore"] = (df["Nt"] + df["NVar"]) / 2.0
    best_row = df.sort_values("Cscore", ascending=True).iloc[0]
    return float(best_row["lambda"]), df


def run_full_fcm(
    A0: np.ndarray,
    W0: np.ndarray,
    mask: np.ndarray,
    lam: float,
    eta: float = 0.05,
    Tmax: int = 200,
    tol: float = 0.001,
):
    A = A0.copy()
    W = W0.copy()
    A_hist = [A.copy()]
    W_hist = [W.copy()]

    for _ in range(Tmax):
        A_next = fcm_state_update(A, W, lam)
        A_hist.append(A_next.copy())
        if steady_reached(A, A_next, tol=tol):
            A = A_next
            break
        W = hebbian_update(W, A_next, eta=eta, mask=mask)
        W_hist.append(W.copy())
        A = A_next

    return np.array(A_hist), np.array(W_hist), A.copy(), W.copy()


# =========================
# Input parsing helpers
# =========================
def parse_levels_text(text: str) -> dict:
    """
    Expect CSV-like lines: factor,level
    Example:
      DE,L1
      EV,L2
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Bad levels line: '{line}'. Use: factor,level")
        rows.append(parts)
    return {f: lvl for f, lvl in rows}


def parse_binary_matrix(text: str) -> pd.DataFrame:
    """
    Parse pasted CSV text into df (supports header row).
    Must be 0/1 (or values coerced via >=0.5).
    """
    df = pd.read_csv(io.StringIO(text))
    df.columns = [str(c).strip() for c in df.columns]
    df = df.applymap(lambda x: 1 if float(x) >= 0.5 else 0)
    return df


def parse_expert_matrices(text: str, nodes: list[str]) -> list[np.ndarray]:
    """
    Experts separated by a line containing: ---
    Each expert matrix is CSV with header row and n columns. Must match 'nodes' columns.
    """
    blocks = [b.strip() for b in text.split("\n---\n") if b.strip()]
    mats = []
    for bi, block in enumerate(blocks, start=1):
        df = pd.read_csv(io.StringIO(block))
        df.columns = [str(c).strip() for c in df.columns]
        # If first column looks like an index column with factor names, set it as index
        # We'll try: if df first col values match nodes
        first_col = df.columns[0]
        if df[first_col].astype(str).isin(nodes).all():
            df = df.set_index(first_col)
        df = df.reindex(index=nodes, columns=nodes).fillna(0.0)
        mat = np.clip(df.to_numpy(dtype=float), 0.0, 1.0)
        mats.append(mat)
    return mats


def parse_adjacency_override(text: str, nodes: list[str]) -> np.ndarray:
    """
    Optional pasted adjacency CSV with header row; values 0/1.
    """
    df = pd.read_csv(io.StringIO(text))
    df.columns = [str(c).strip() for c in df.columns]
    first_col = df.columns[0]
    if df[first_col].astype(str).isin(nodes).all():
        df = df.set_index(first_col)
    df = df.reindex(index=nodes, columns=nodes).fillna(0)
    A = (df.to_numpy(dtype=float) > 0.5).astype(int)
    np.fill_diagonal(A, 0)
    return A


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Enhanced FCM (Direct Input)", layout="wide")
st.title("Enhanced FCM (ARM + MI + Reliability Optimization) — Direct Input")

with st.expander("Paste format examples (click to expand)", expanded=False):
    st.markdown(
        """
**1) Factor levels (factor,level)**  
