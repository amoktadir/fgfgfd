import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from itertools import combinations

# ARM (Apriori)
from mlxtend.frequent_patterns import apriori, association_rules

# MI
from sklearn.metrics import mutual_info_score

# Optimization (reliability model Eq. 9) via LP
import pulp

# Optional network plot
import networkx as nx


# =========================
# Core math (paper-aligned)
# =========================
def sigmoid(x: np.ndarray, lam: float) -> np.ndarray:
    # Eq. (2): f(x) = 1/(1+exp(-λx))
    return 1.0 / (1.0 + np.exp(-lam * x))


def fcm_state_update(A: np.ndarray, W: np.ndarray, lam: float) -> np.ndarray:
    # Eq. (1): A_i(t+1) = f( sum_{j!=i} w_ji(t) A_j(t) + A_i(t) )
    # Note: if you store W[i,j] = influence i->j, then incoming to i is column i.
    # We'll compute incoming = A @ W  (gives each target's total input from all sources),
    # then add self term A, then apply sigmoid.
    incoming = A @ W
    return sigmoid(incoming + A, lam=lam)


def hebbian_update(W: np.ndarray, A: np.ndarray, eta: float, mask: np.ndarray) -> np.ndarray:
    # Eq. (3): w_ij(t+1) = w_ij(t) + η * A_j(t) * ( A_i(t) - A_j(t) * w_ij(t) )
    # Using convention W[i,j] = i -> j (source i, target j)
    # The paper’s Eq. (3) is written w_ij updates using A_i and A_j; we implement elementwise.
    # Ensure only existing edges (mask=1) can update; others stay 0.
    W_new = W.copy()
    A_i = A.reshape(-1, 1)       # sources
    A_j = A.reshape(1, -1)       # targets
    # For W[i,j] (i->j), use A_j (target) as "A_j" multiplier? Paper uses A_j * (A_i - A_j*w_ij).
    # To stay aligned with their form, interpret i as "i", j as "j" directly:
    delta = eta * (A_j) * (A_i - (A_j * W_new))
    W_new = W_new + delta
    # Keep within [0,1] as used in their expert scale for weights; and preserve topology
    W_new = np.clip(W_new, 0.0, 1.0)
    W_new *= mask
    return W_new


def steady_reached(A_prev: np.ndarray, A_next: np.ndarray, tol: float = 0.001) -> bool:
    # paper termination example: |A_i(t+1)-A_i(t)| <= 0.001 for all i
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


def mine_association_rules(binary_df: pd.DataFrame, min_support: float, min_conf: float):
    # binary_df must be 0/1
    frequent = apriori(binary_df.astype(bool), min_support=min_support, use_colnames=True)
    if frequent.empty:
        return pd.DataFrame()
    rules = association_rules(frequent, metric="confidence", min_threshold=min_conf)
    if rules.empty:
        return pd.DataFrame()
    # Keep only 2-item rules (single antecedent -> single consequent) for edge extraction like paper
    rules = rules[(rules["antecedents"].apply(len) == 1) & (rules["consequents"].apply(len) == 1)].copy()
    return rules


def hfacs_filter_edges(edges, factor_levels: dict, forbid_same_level=True):
    # Remark 3: prohibit same-level edges; allow cross-level
    # (paper notes this reduces redundancy and improves interpretability)
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
    # MI top-k edges (undirected association, paper uses as potential positive; we’ll propose both directions)
    mi_values = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            mi_values.append((cols[i], cols[j], mi_df.iat[i, j]))
    mi_values.sort(key=lambda x: x[2], reverse=True)
    k = max(1, int(len(mi_values) * mi_top_k_pct))
    mi_top = mi_values[:k]

    edges = []

    # ARM edges: antecedent -> consequent as directed positive causal candidates
    if not rules_df.empty:
        for _, r in rules_df.iterrows():
            a = next(iter(r["antecedents"]))
            c = next(iter(r["consequents"]))
            edges.append((a, c, {"source": "ARM", "lift": float(r["lift"]), "confidence": float(r["confidence"]), "support": float(r["support"])}))

    # MI edges: treat as candidate association; direction unknown -> add both directions as candidates
    for a, b, val in mi_top:
        edges.append((a, b, {"source": "MI", "mi": float(val)}))
        edges.append((b, a, {"source": "MI", "mi": float(val)}))

    # Feedback loops: paper screens ARM rules with lift > 1.5 to allow bottom-up feedback (Remark / Step 3)
    feedback = []
    if not rules_df.empty:
        for _, r in rules_df.iterrows():
            if float(r["lift"]) >= lift_feedback_threshold:
                a = next(iter(r["antecedents"]))
                c = next(iter(r["consequents"]))
                # Keep as candidate "feedback" (we can label; user can later decide to keep)
                feedback.append((a, c, {"source": "ARM_feedback", "lift": float(r["lift"]), "confidence": float(r["confidence"]), "support": float(r["support"])}))

    # Apply HFACS level constraint (no same-level)
    edges = hfacs_filter_edges(edges, factor_levels, forbid_same_level=True)
    feedback = hfacs_filter_edges(feedback, factor_levels, forbid_same_level=True)

    # Deduplicate by (u,v) keeping best meta (prefer ARM > MI, and higher lift)
    best = {}
    def score(meta):
        if meta.get("source", "").startswith("ARM"):
            return 1000.0 + meta.get("lift", 0.0)
        if meta.get("source") == "MI":
            return 100.0 + meta.get("mi", 0.0)
        return 0.0

    for u, v, meta in edges + feedback:
        key = (u, v)
        if key not in best or score(meta) > score(best[key]):
            best[key] = meta

    out = [(u, v, best[(u, v)]) for (u, v) in best.keys()]
    return out


def edges_to_adjacency(edges, nodes):
    idx = {n:i for i,n in enumerate(nodes)}
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
    """
    expert_mats: list of (n,n) arrays (initial w_{kh,i})
    mask: (n,n) 0/1 adjacency, non-edges must remain 0
    epsilon: allowed deviation |w_{kh,i} - w_{kh}| <= ε
    Returns:
      w_group (n,n), w_opt_list (list of n,n), gamma, CL, D
    """
    m = len(expert_mats)
    n = expert_mats[0].shape[0]
    # N = number of non-zero elements in causal weight matrix (edges)
    N = int(np.sum(mask))

    # LP
    prob = pulp.LpProblem("reliability_optimization", pulp.LpMaximize)

    # Decision variables: w_opt[i,k,h] for each expert i and each edge (k,h)
    w_opt = {}
    # Group w_group[k,h]
    w_grp = {}

    # Abs vars for consensus and distortion
    abs_cons = {}
    abs_dist = {}

    for k in range(n):
        for h in range(n):
            if mask[k, h] == 1:
                w_grp[(k, h)] = pulp.LpVariable(f"wgrp_{k}_{h}", lowBound=0.0, upBound=1.0, cat="Continuous")
                for i in range(m):
                    w_opt[(i, k, h)] = pulp.LpVariable(f"wopt_{i}_{k}_{h}", lowBound=0.0, upBound=1.0, cat="Continuous")
                    abs_cons[(i, k, h)] = pulp.LpVariable(f"abscons_{i}_{k}_{h}", lowBound=0.0, cat="Continuous")
                    abs_dist[(i, k, h)] = pulp.LpVariable(f"absdist_{i}_{k}_{h}", lowBound=0.0, cat="Continuous")
            else:
                # Non-edges fixed at 0 conceptually (no variables needed)
                pass

    # Constraints:
    # (a) group mean: w_grp = avg_i w_opt
    for k in range(n):
        for h in range(n):
            if mask[k, h] == 1:
                prob += w_grp[(k, h)] == (1.0 / m) * pulp.lpSum(w_opt[(i, k, h)] for i in range(m))

    # (b) epsilon deviation: |w_opt - w_grp| <= epsilon
    for k in range(n):
        for h in range(n):
            if mask[k, h] == 1:
                for i in range(m):
                    prob += w_opt[(i, k, h)] - w_grp[(k, h)] <= epsilon
                    prob += w_grp[(k, h)] - w_opt[(i, k, h)] <= epsilon

    # (c) absolute linearization:
    # abs_cons >= |w_grp - w_opt|
    # abs_dist >= |w_opt - w_init|
    for k in range(n):
        for h in range(n):
            if mask[k, h] == 1:
                for i in range(m):
                    prob += abs_cons[(i, k, h)] >= w_grp[(k, h)] - w_opt[(i, k, h)]
                    prob += abs_cons[(i, k, h)] >= w_opt[(i, k, h)] - w_grp[(k, h)]
                    w_init = float(expert_mats[i][k, h])
                    prob += abs_dist[(i, k, h)] >= w_opt[(i, k, h)] - w_init
                    prob += abs_dist[(i, k, h)] >= w_init - w_opt[(i, k, h)]

    # Eq. (6): CL = 1 - (1/(mN)) * sum_i sum_edges |w_grp - w_opt|
    # Eq. (7): D  = (1/(mN)) * sum_i sum_edges |w_opt - w_init|
    # Eq. (8): gamma = CL - D = 1 - avg_abs_cons - avg_abs_dist
    avg_abs_cons = (1.0 / (m * N)) * pulp.lpSum(abs_cons.values())
    avg_abs_dist = (1.0 / (m * N)) * pulp.lpSum(abs_dist.values())
    gamma_expr = 1.0 - avg_abs_cons - avg_abs_dist

    prob += gamma_expr

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"Optimization failed: {pulp.LpStatus[prob.status]}")

    # Extract
    w_group = np.zeros((n, n), dtype=float)
    w_opt_list = [np.zeros((n, n), dtype=float) for _ in range(m)]

    for k in range(n):
        for h in range(n):
            if mask[k, h] == 1:
                w_group[k, h] = float(pulp.value(w_grp[(k, h)]))
                for i in range(m):
                    w_opt_list[i][k, h] = float(pulp.value(w_opt[(i, k, h)]))

    CL = 1.0 - float(pulp.value(avg_abs_cons))
    D = float(pulp.value(avg_abs_dist))
    gamma = float(pulp.value(gamma_expr))
    return w_group, w_opt_list, gamma, CL, D


# =========================
# Phase 3: λ selection (Table 1 style) + simulation
# =========================
def choose_lambda_grid(
    A0: np.ndarray,
    W0: np.ndarray,
    mask: np.ndarray,
    lambdas=(0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
    TR: int = 10,
    Tmax: int = 50,
    eta: float = 0.05,
    tol: float = 0.001,
):
    """
    Implements a practical version of Table 1:
    For each λ:
      iterate FCM+Hebbian until steady or Tmax,
      record tsteady,
      then compute Var over TR steps after steady (or last TR states if not steady),
      normalize tsteady and Var across λ, compute Cscore = (NVar + Nt)/2, choose min.
    """
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
            tsteady = len(hist) - 1  # max iters reached

        # variance over TR window after tsteady (or last TR available)
        start = min(tsteady, len(hist) - 1)
        end = min(start + TR, len(hist))
        window = hist[start:end]
        if window.shape[0] < 2:
            var = float(np.var(hist[-min(TR, len(hist)):], axis=0).mean())
        else:
            var = float(np.var(window, axis=0).mean())
        results.append({"lambda": lam, "tsteady": tsteady, "var": var})

    df = pd.DataFrame(results)

    # normalize
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
# Streamlit UI
# =========================
st.set_page_config(page_title="Enhanced FCM (ARM+MI+Reliability) - Paper Implementation", layout="wide")
st.title("Enhanced Fuzzy Cognitive Map (FCM) – Paper Method Implementation")

st.markdown(
    """
This app implements the method proposed in the attached paper: **ARM + MI + expert elicitation**, then
**reliability-optimized group integration** of causal weights, **adaptive λ selection**, **dynamic simulation with nonlinear Hebbian learning**, and **risk priority ranking**.
"""
)

with st.expander("Paper alignment (what this app implements)", expanded=False):
    st.write(
        "- Phase 1: Topology construction via ARM + MI + HFACS constraints + (optional) manual expert edits. "
        "ARM/MI + feedback lift threshold are in the paper’s construction stage. "
        "\n- Phase 2: Initial state A(0) from factor frequency; group weight matrix via reliability optimization (Eq. 6–9). "
        "\n- Phase 3: FCM recurrence (Eq. 1) with sigmoid (Eq. 2); dynamic weight learning via nonlinear Hebbian (Eq. 3); λ selected by grid scoring (Table 1 style). "
        "\n- Phase 4: Prioritization by steady-state values."
    )


# -------------------------
# Sidebar inputs
# -------------------------
with st.sidebar:
    st.header("Inputs")

    st.subheader("A) HFACS factor level mapping (required)")
    st.caption("Upload a CSV with columns: factor, level (e.g., L1/L2/L3/L4). Used to enforce 'no same-level edges'.")
    levels_file = st.file_uploader("Upload factor levels CSV", type=["csv"], key="levels")

    st.subheader("B) Accident report factor matrix (required)")
    st.caption("Upload a binary matrix CSV: rows=reports, columns=factors, values in {0,1}.")
    data_file = st.file_uploader("Upload binary HFACS matrix CSV", type=["csv"], key="data")

    st.subheader("C) ARM parameters")
    min_support = st.number_input("Min support", min_value=0.0001, max_value=0.5, value=0.01, step=0.001)
    min_conf = st.number_input("Min confidence", min_value=0.01, max_value=0.99, value=0.30, step=0.01)
    lift_min = st.number_input("Lift (keep rules lift >=)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    st.subheader("D) MI parameters")
    mi_top_k_pct = st.slider("Top-K % (MI)", min_value=1, max_value=50, value=10, step=1) / 100.0

    st.subheader("E) Feedback mining (ARM lift threshold)")
    lift_feedback_threshold = st.number_input("Feedback lift threshold", min_value=1.0, max_value=10.0, value=1.5, step=0.1)

    st.subheader("F) Expert weights (optional but recommended)")
    st.caption("Upload one or more expert weight matrices (CSV, n×n, matching factor order).")
    expert_files = st.file_uploader("Upload expert matrices (multiple)", type=["csv"], accept_multiple_files=True, key="experts")

    st.subheader("G) Reliability optimization")
    epsilon = st.number_input("ε (allowed deviation)", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    st.subheader("H) Simulation settings")
    eta = st.number_input("η (Hebbian learning rate)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    Tmax = st.number_input("Tmax (max iterations)", min_value=10, max_value=2000, value=200, step=10)
    tol = st.number_input("Steady tolerance (ΔA max)", min_value=1e-6, max_value=1e-1, value=0.001, step=0.0005, format="%.6f")

    st.subheader("I) λ selection grid (Table 1 style)")
    TR = st.number_input("TR (variance window length)", min_value=2, max_value=200, value=10, step=1)
    Tmax_lambda = st.number_input("Tmax for λ search", min_value=10, max_value=500, value=50, step=5)

    st.subheader("Run")
    run_button = st.button("Run full pipeline", type="primary")


# -------------------------
# Load required files
# -------------------------
if not levels_file or not data_file:
    st.info("Upload (A) factor levels CSV and (B) binary HFACS matrix CSV to proceed.")
    st.stop()

levels_df = pd.read_csv(levels_file)
required_cols = {"factor", "level"}
if not required_cols.issubset(set(levels_df.columns)):
    st.error("Factor levels CSV must have columns: factor, level")
    st.stop()

factor_levels = dict(zip(levels_df["factor"].astype(str), levels_df["level"].astype(str)))

binary_df = pd.read_csv(data_file)
# Ensure factor columns are strings
binary_df.columns = [str(c) for c in binary_df.columns]
# Coerce to 0/1
binary_df = binary_df.applymap(lambda x: 1 if float(x) >= 0.5 else 0)

# Align nodes to intersection of provided levels and data
nodes = [c for c in binary_df.columns if c in factor_levels]
missing_levels = [c for c in binary_df.columns if c not in factor_levels]
if missing_levels:
    st.warning(f"{len(missing_levels)} factors exist in the data but not in the levels mapping; they will be ignored: {missing_levels[:10]}{'...' if len(missing_levels)>10 else ''}")
binary_df = binary_df[nodes].copy()

if len(nodes) < 2:
    st.error("Need at least 2 factors after aligning levels mapping and data columns.")
    st.stop()

# Initial state vector A(0) from frequency proportion Eq. (5)
# A0_i = n(H_i) / n(ar)
A0 = (binary_df.sum(axis=0) / len(binary_df)).to_numpy(dtype=float)

# -------------------------
# Show inputs overview
# -------------------------
st.subheader("Input overview")
c1, c2, c3 = st.columns([1.1, 1.0, 0.9])
with c1:
    st.write("**Binary HFACS matrix (preview)**")
    st.dataframe(binary_df.head(10), use_container_width=True)
with c2:
    st.write("**Initial state A(0) (Eq. 5)**")
    A0_df = pd.DataFrame({"factor": nodes, "A0": A0})
    st.dataframe(A0_df.sort_values("A0", ascending=False), use_container_width=True, height=320)
with c3:
    st.write("**HFACS levels (preview)**")
    st.dataframe(levels_df[levels_df["factor"].astype(str).isin(nodes)].head(15), use_container_width=True, height=320)


# -------------------------
# Run pipeline
# -------------------------
if not run_button:
    st.stop()

# Phase 1: ARM + MI topology
st.header("Phase 1 — FCM topology construction (ARM + MI + HFACS constraints)")

rules_df = mine_association_rules(binary_df, min_support=min_support, min_conf=min_conf)
if not rules_df.empty:
    rules_df = rules_df[rules_df["lift"] >= lift_min].copy()

mi_df = compute_mi_matrix(binary_df)

edges = build_edges_from_arm_mi(
    rules_df=rules_df,
    mi_df=mi_df,
    mi_top_k_pct=mi_top_k_pct,
    lift_feedback_threshold=lift_feedback_threshold,
    factor_levels=factor_levels,
)

adj_mask, meta_map = edges_to_adjacency(edges, nodes)
mask = adj_mask.astype(float)

st.write(f"**Nodes:** {len(nodes)}  |  **Candidate directed edges after HFACS constraint:** {int(mask.sum())}")

topo_col1, topo_col2 = st.columns([1.2, 0.8])
with topo_col1:
    st.write("**Candidate edges (preview)**")
    edge_rows = []
    for (u, v), meta in meta_map.items():
        row = {"from": u, "to": v, **meta}
        edge_rows.append(row)
    edge_df = pd.DataFrame(edge_rows)
    if edge_df.empty:
        st.warning("No edges found. Consider lowering min_support/min_conf or increasing MI top-k.")
    else:
        st.dataframe(edge_df.sort_values(by=edge_df.columns.tolist()[2], ascending=False).head(50), use_container_width=True)
with topo_col2:
    st.write("**Adjacency mask (1=edge)**")
    st.dataframe(pd.DataFrame(adj_mask, index=nodes, columns=nodes), use_container_width=True, height=420)

# Optional: let user manually edit edges by uploading an adjacency CSV
st.subheader("Optional expert elicitation edit (upload revised adjacency)")
st.caption("Paper includes expert revision (delete spurious/add missing links). Upload an adjacency CSV (n×n) to override the mined topology if desired.")
adj_override = st.file_uploader("Upload adjacency override CSV (optional)", type=["csv"], key="adj_override")
if adj_override is not None:
    adj_user = pd.read_csv(adj_override, index_col=0)
    # try align
    adj_user = adj_user.reindex(index=nodes, columns=nodes).fillna(0)
    adj_mask = (adj_user.to_numpy(dtype=float) > 0.5).astype(int)
    mask = adj_mask.astype(float)
    st.success("Adjacency overridden by uploaded file.")
    st.write(f"**Edges after override:** {int(mask.sum())}")
    st.dataframe(pd.DataFrame(adj_mask, index=nodes, columns=nodes), use_container_width=True)

# Phase 2: Expert matrices + reliability optimization
st.header("Phase 2 — Initial weights via reliability-optimized group integration (Eq. 6–9)")

if not expert_files:
    st.warning("No expert matrices uploaded. The paper’s method uses expert elicitation + reliability optimization; "
               "to proceed, we will initialize weights with a simple heuristic (scaled adjacency).")
    # Heuristic: set all edges to 0.5
    W0 = 0.5 * mask
    gamma = CL = D = np.nan
else:
    # Read expert matrices
    expert_mats = []
    for f in expert_files:
        df = pd.read_csv(f, index_col=0)
        df = df.reindex(index=nodes, columns=nodes).fillna(0.0)
        mat = df.to_numpy(dtype=float)
        # enforce [0,1] and topology (non-edges forced to 0)
        mat = np.clip(mat, 0.0, 1.0) * mask
        expert_mats.append(mat)

    try:
        W0, wopt_list, gamma, CL, D = reliability_optimize_group_weights(expert_mats, mask=mask, epsilon=float(epsilon))
    except Exception as e:
        st.error(f"Reliability optimization failed: {e}")
        st.stop()

    st.success("Reliability-optimized group weight matrix computed.")

stat1, stat2, stat3 = st.columns(3)
stat1.metric("Edges (N)", int(mask.sum()))
stat2.metric("Consensus CL (Eq. 6)", "—" if np.isnan(CL) else f"{CL:.4f}")
stat3.metric("Reliability γ (Eq. 8)", "—" if np.isnan(gamma) else f"{gamma:.4f}")

wcol1, wcol2 = st.columns([1.2, 0.8])
with wcol1:
    st.write("**Initial group weight matrix W (after reliability optimization)**")
    st.dataframe(pd.DataFrame(W0, index=nodes, columns=nodes), use_container_width=True, height=420)
with wcol2:
    st.write("**Weight summary (edges only)**")
    vals = W0[mask == 1]
    summ = pd.DataFrame({
        "count": [len(vals)],
        "mean": [float(np.mean(vals)) if len(vals) else 0.0],
        "std": [float(np.std(vals)) if len(vals) else 0.0],
        "min": [float(np.min(vals)) if len(vals) else 0.0],
        "max": [float(np.max(vals)) if len(vals) else 0.0],
    })
    st.dataframe(summ, use_container_width=True)

# Phase 3: λ selection + dynamic simulation
st.header("Phase 3 — Adaptive λ selection + dynamic simulation (Eq. 1–3, Table 1)")

best_lam, lam_df = choose_lambda_grid(
    A0=A0, W0=W0, mask=mask,
    TR=int(TR), Tmax=int(Tmax_lambda),
    eta=float(eta), tol=float(tol)
)

st.write("**λ grid search results (lower Cscore is better):**")
st.dataframe(lam_df.sort_values("Cscore"), use_container_width=True)
st.success(f"Selected λ = {best_lam:.2f}")

A_hist, W_hist, A_steady, W_steady = run_full_fcm(
    A0=A0, W0=W0, mask=mask,
    lam=best_lam, eta=float(eta),
    Tmax=int(Tmax), tol=float(tol)
)

st.write(f"Iterations run: **{len(A_hist)-1}**  |  Steady reached: **{'Yes' if len(A_hist)-1 < Tmax else 'Maybe/No (hit Tmax)'}**")

# Phase 4: Prioritization
st.header("Phase 4 — Priority ranking by steady-state values")
rank_df = pd.DataFrame({"factor": nodes, "steady_value": A_steady})
rank_df = rank_df.sort_values("steady_value", ascending=False).reset_index(drop=True)
rank_df["rank"] = np.arange(1, len(rank_df) + 1)

cA, cB = st.columns([1.0, 1.0])
with cA:
    st.write("**Priority ranking (steady state)**")
    st.dataframe(rank_df, use_container_width=True, height=520)
with cB:
    st.write("**Steady-state plot (sorted)**")
    fig = plt.figure()
    plt.plot(rank_df["steady_value"].to_numpy())
    plt.xlabel("Rank (1 = highest)")
    plt.ylabel("Steady-state value")
    plt.tight_layout()
    st.pyplot(fig)

st.subheader("Trajectory plot (all factors)")
traj_df = pd.DataFrame(A_hist, columns=nodes)
traj_df.insert(0, "t", np.arange(len(traj_df)))

fig = plt.figure()
for n in nodes:
    plt.plot(traj_df["t"], traj_df[n], label=n)
plt.xlabel("Iteration (t)")
plt.ylabel("Activation")
plt.ylim(0, 1)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
plt.tight_layout()
st.pyplot(fig)

# Optional network visualization
st.subheader("FCM network visualization (topology)")
G = nx.DiGraph()
for i, u in enumerate(nodes):
    G.add_node(u)
for i, u in enumerate(nodes):
    for j, v in enumerate(nodes):
        if mask[i, j] == 1:
            G.add_edge(u, v, weight=float(W_steady[i, j]))

pos = nx.spring_layout(G, seed=42, k=1.0 / math.sqrt(len(nodes)))
fig = plt.figure()
nx.draw_networkx_nodes(G, pos, node_size=450)
nx.draw_networkx_labels(G, pos, font_size=8)
nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", width=1.0)
plt.axis("off")
plt.tight_layout()
st.pyplot(fig)

# Downloads
st.header("Downloads")
out_rank_csv = rank_df.to_csv(index=False).encode("utf-8")
out_traj_csv = traj_df.to_csv(index=False).encode("utf-8")
out_Wsteady_csv = pd.DataFrame(W_steady, index=nodes, columns=nodes).to_csv().encode("utf-8")
out_mask_csv = pd.DataFrame(adj_mask, index=nodes, columns=nodes).to_csv().encode("utf-8")
out_lam_csv = lam_df.to_csv(index=False).encode("utf-8")

st.download_button("Download ranking CSV", data=out_rank_csv, file_name="ranking_steady_state.csv", mime="text/csv")
st.download_button("Download trajectory CSV", data=out_traj_csv, file_name="trajectory.csv", mime="text/csv")
st.download_button("Download steady weight matrix CSV", data=out_Wsteady_csv, file_name="W_steady.csv", mime="text/csv")
st.download_button("Download adjacency mask CSV", data=out_mask_csv, file_name="adjacency_mask.csv", mime="text/csv")
st.download_button("Download λ search table CSV", data=out_lam_csv, file_name="lambda_search.csv", mime="text/csv")

st.caption(
    "Implementation notes: topology = ARM + MI + HFACS no-same-level constraint; initial state from frequency; "
    "group weights via reliability optimization; sigmoid activation and nonlinear Hebbian weight learning; λ chosen by grid scoring."
)
