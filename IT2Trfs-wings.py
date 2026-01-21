import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.metrics import mutual_info_score
import pulp
import networkx as nx


# =========================
# Core math (paper-aligned)
# =========================
def sigmoid(x: np.ndarray, lam: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-lam * x))


def fcm_state_update(A: np.ndarray, W: np.ndarray, lam: float) -> np.ndarray:
    # Eq (1) using W[i,j] = i -> j
    incoming = A @ W
    return sigmoid(incoming + A, lam=lam)


def hebbian_update(W: np.ndarray, A: np.ndarray, eta: float, mask: np.ndarray) -> np.ndarray:
    # Eq (3) implemented elementwise for W[i,j] = i->j
    W_new = W.copy()
    A_i = A.reshape(-1, 1)   # sources
    A_j = A.reshape(1, -1)   # targets
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


def hfacs_filter_edges(edges, factor_levels: dict, forbid_same_level: bool = True):
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

    # MI top-k (undirected -> add both directions)
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
                "support": float(r["support"]),
            }))

    # MI candidates
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
                    "support": float(r["support"]),
                }))

    # HFACS filter
    edges = hfacs_filter_edges(edges, factor_levels, forbid_same_level=True)

    # Deduplicate by (u,v), prefer ARM with higher lift
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
# Phase 2: Reliability optimization
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

    # Group mean
    for k in range(n):
        for h in range(n):
            if mask[k, h] == 1:
                prob += w_grp[(k, h)] == (1.0 / m) * pulp.lpSum(w_opt[(i, k, h)] for i in range(m))

    # Epsilon deviation
    for k in range(n):
        for h in range(n):
            if mask[k, h] == 1:
                for i in range(m):
                    prob += w_opt[(i, k, h)] - w_grp[(k, h)] <= epsilon
                    prob += w_grp[(k, h)] - w_opt[(i, k, h)] <= epsilon

    # Absolute linearization
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
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if (not line) or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            raise ValueError("Bad levels line: " + line + "  (use factor,level)")
        rows.append(parts)
    return {f: lvl for f, lvl in rows}


def parse_binary_matrix(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text))
    df.columns = [str(c).strip() for c in df.columns]
    df = df.applymap(lambda x: 1 if float(x) >= 0.5 else 0)
    return df


def parse_expert_matrices(text: str, nodes):
    blocks = [b.strip() for b in text.split("\n---\n") if b.strip()]
    mats = []
    for block in blocks:
        df = pd.read_csv(io.StringIO(block))
        df.columns = [str(c).strip() for c in df.columns]
        first_col = df.columns[0]
        if df[first_col].astype(str).isin(nodes).all():
            df = df.set_index(first_col)
        df = df.reindex(index=nodes, columns=nodes).fillna(0.0)
        mats.append(np.clip(df.to_numpy(dtype=float), 0.0, 1.0))
    return mats


def parse_adjacency_override(text: str, nodes):
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
# UI
# =========================
st.set_page_config(page_title="Enhanced FCM (Direct Input)", layout="wide")
st.title("Enhanced FCM (ARM + MI + Reliability) — Direct Input")

help_lines = [
    "Paste formats:",
    "1) Levels: factor,level  (one per line)",
    "2) Binary matrix: CSV with header row; 0/1 values",
    "3) Expert matrices (optional): CSV; separate experts with a line containing ---",
    "4) Adjacency override (optional): 0/1 matrix as CSV",
]
with st.expander("Paste format help", expanded=False):
    st.text("\n".join(help_lines))

with st.sidebar:
    st.header("Direct Inputs")
    levels_text = st.text_area("A) Factor levels (factor,level)", height=160)
    data_text = st.text_area("B) Binary HFACS matrix CSV", height=220)

    st.subheader("ARM")
    min_support = st.number_input("Min support", min_value=0.0001, max_value=0.5, value=0.01, step=0.001)
    min_conf = st.number_input("Min confidence", min_value=0.01, max_value=0.99, value=0.30, step=0.01)
    lift_min = st.number_input("Lift keep (>=)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    st.subheader("MI")
    mi_top_k_pct = st.slider("Top-K % (MI)", min_value=1, max_value=50, value=10, step=1) / 100.0

    st.subheader("Feedback mining")
    lift_feedback_threshold = st.number_input("Feedback lift threshold", min_value=1.0, max_value=10.0, value=1.5, step=0.1)

    st.subheader("Optional adjacency override (paste CSV)")
    adjacency_override_text = st.text_area("Adjacency override", height=140)

    st.subheader("Optional expert matrices (paste)")
    expert_text = st.text_area("Expert matrices (separate experts with ---)", height=220)

    st.subheader("Reliability optimization")
    epsilon = st.number_input("ε (allowed deviation)", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    st.subheader("Simulation")
    eta = st.number_input("η (Hebbian learning rate)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    Tmax = st.number_input("Tmax", min_value=10, max_value=2000, value=200, step=10)
    tol = st.number_input("Tolerance ΔA", min_value=1e-6, max_value=1e-1, value=0.001, step=0.0005, format="%.6f")

    st.subheader("λ selection")
    TR = st.number_input("TR", min_value=2, max_value=200, value=10, step=1)
    Tmax_lambda = st.number_input("Tmax (λ search)", min_value=10, max_value=500, value=50, step=5)

    run_button = st.button("Run full pipeline", type="primary")

if not run_button:
    st.info("Paste levels + binary matrix, then click Run.")
    st.stop()

# Parse levels and matrix
try:
    factor_levels = parse_levels_text(levels_text)
except Exception as e:
    st.error("Levels parse error: " + str(e))
    st.stop()

try:
    binary_df = parse_binary_matrix(data_text)
except Exception as e:
    st.error("Binary matrix parse error: " + str(e))
    st.stop()

nodes = [c for c in binary_df.columns if c in factor_levels]
missing_levels = [c for c in binary_df.columns if c not in factor_levels]
if missing_levels:
    st.warning("Columns missing in levels map (ignored): " + ", ".join(missing_levels))

binary_df = binary_df[nodes].copy()
if len(nodes) < 2:
    st.error("Need at least 2 factors after aligning with factor levels.")
    st.stop()

A0 = (binary_df.sum(axis=0) / len(binary_df)).to_numpy(dtype=float)

st.subheader("Input overview")
c1, c2 = st.columns([1.2, 0.8])
with c1:
    st.write("Binary matrix preview")
    st.dataframe(binary_df.head(10), use_container_width=True)
with c2:
    st.write("Initial state A(0)")
    st.dataframe(pd.DataFrame({"factor": nodes, "A0": A0}).sort_values("A0", ascending=False), use_container_width=True, height=320)

# Phase 1
st.header("Phase 1 — Topology (ARM + MI + HFACS constraint)")
rules_df = mine_association_rules(binary_df, min_support=float(min_support), min_conf=float(min_conf))
if not rules_df.empty:
    rules_df = rules_df[rules_df["lift"] >= float(lift_min)].copy()
mi_df = compute_mi_matrix(binary_df)

edges = build_edges_from_arm_mi(
    rules_df=rules_df,
    mi_df=mi_df,
    mi_top_k_pct=float(mi_top_k_pct),
    lift_feedback_threshold=float(lift_feedback_threshold),
    factor_levels=factor_levels,
)

adj_mask, meta_map = edges_to_adjacency(edges, nodes)

if adjacency_override_text.strip():
    try:
        adj_mask = parse_adjacency_override(adjacency_override_text, nodes)
        st.success("Adjacency overridden.")
    except Exception as e:
        st.error("Adjacency override error: " + str(e))
        st.stop()

mask = adj_mask.astype(float)
N_edges = int(mask.sum())
st.write("Nodes: " + str(len(nodes)) + " | Directed edges (N): " + str(N_edges))

if N_edges == 0:
    st.error("No edges found. Lower ARM thresholds / increase MI top-k / paste adjacency override.")
    st.stop()

st.dataframe(pd.DataFrame(adj_mask, index=nodes, columns=nodes), use_container_width=True, height=420)

# Phase 2
st.header("Phase 2 — Initial weights (reliability optimization)")
if not expert_text.strip():
    st.warning("No expert matrices pasted. Using heuristic W0 = 0.5 * adjacency.")
    W0 = 0.5 * mask
    gamma = CL = D = np.nan
else:
    try:
        expert_mats = parse_expert_matrices(expert_text, nodes)
    except Exception as e:
        st.error("Expert parse error: " + str(e))
        st.stop()

    expert_mats = [np.clip(m, 0.0, 1.0) * mask for m in expert_mats]
    try:
        W0, gamma, CL, D = reliability_optimize_group_weights(expert_mats, mask=mask, epsilon=float(epsilon))
    except Exception as e:
        st.error("Reliability optimization failed: " + str(e))
        st.stop()

    st.success("Reliability-optimized group weights computed.")

m1, m2, m3 = st.columns(3)
m1.metric("Edges (N)", N_edges)
m2.metric("Consensus CL", "—" if np.isnan(CL) else f"{CL:.4f}")
m3.metric("Reliability γ", "—" if np.isnan(gamma) else f"{gamma:.4f}")

st.write("Initial group weight matrix W0")
st.dataframe(pd.DataFrame(W0, index=nodes, columns=nodes), use_container_width=True, height=420)

# Phase 3
st.header("Phase 3 — λ selection + simulation")
best_lam, lam_df = choose_lambda_grid(
    A0=A0, W0=W0, mask=mask,
    TR=int(TR), Tmax=int(Tmax_lambda),
    eta=float(eta), tol=float(tol),
)
st.dataframe(lam_df.sort_values("Cscore"), use_container_width=True)
st.success("Selected λ = " + f"{best_lam:.2f}")

A_hist, W_hist, A_steady, W_steady = run_full_fcm(
    A0=A0, W0=W0, mask=mask,
    lam=best_lam, eta=float(eta),
    Tmax=int(Tmax), tol=float(tol),
)

st.write("Iterations run: " + str(len(A_hist) - 1))

# Phase 4
st.header("Phase 4 — Ranking (steady state)")
rank_df = pd.DataFrame({"factor": nodes, "steady_value": A_steady})
rank_df = rank_df.sort_values("steady_value", ascending=False).reset_index(drop=True)
rank_df["rank"] = np.arange(1, len(rank_df) + 1)
st.dataframe(rank_df, use_container_width=True, height=520)

# Trajectory
st.subheader("Trajectory plot")
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

# Network plot
st.subheader("FCM network visualization")
G = nx.DiGraph()
for u in nodes:
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
st.download_button("Download steady weights CSV", data=out_Wsteady_csv, file_name="W_steady.csv", mime="text/csv")
st.download_button("Download adjacency CSV", data=out_mask_csv, file_name="adjacency_mask.csv", mime="text/csv")
st.download_button("Download λ search CSV", data=out_lam_csv, file_name="lambda_search.csv", mime="text/csv")
