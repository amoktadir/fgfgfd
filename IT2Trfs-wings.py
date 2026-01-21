import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="FCM Simulator", layout="wide")

# -----------------------------
# FCM functions
# -----------------------------
def sigmoid(x: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Sigmoid activation to keep states in [0,1]."""
    return 1.0 / (1.0 + np.exp(-lam * x))

def run_fcm(W: np.ndarray, A0: np.ndarray, steps: int = 20, lam: float = 1.0, clamp=None) -> np.ndarray:
    """
    Common FCM update rule:
      A(t+1) = f( A(t) + A(t) @ W )
    """
    W = np.asarray(W, dtype=float)
    A = np.asarray(A0, dtype=float).copy()

    n = W.shape[0]
    if W.shape != (n, n):
        raise ValueError("W must be a square matrix (n x n).")
    if A.shape != (n,):
        raise ValueError("A0 must be a vector of length n.")

    clamp = clamp or {}
    hist = [A.copy()]

    for _ in range(steps):
        influence = A @ W
        A_next = sigmoid(A + influence, lam=lam)

        for idx, val in clamp.items():
            A_next[int(idx)] = float(val)

        A = A_next
        hist.append(A.copy())

    return np.vstack(hist)

def parse_csv_matrix(file_bytes: bytes) -> pd.DataFrame:
    """Read CSV into a DataFrame (supports header/no-header)."""
    df = pd.read_csv(io.BytesIO(file_bytes))
    # If the CSV has unnamed columns, treat it as no-header matrix
    if any(str(c).startswith("Unnamed") for c in df.columns):
        df = pd.read_csv(io.BytesIO(file_bytes), header=None)
    return df

# -----------------------------
# UI
# -----------------------------
st.title("FCM (Fuzzy Cognitive Map) Simulator")

with st.sidebar:
    st.header("Inputs")

    mode = st.radio("Weight Matrix Input", ["Example matrix", "Upload CSV matrix"])

    if mode == "Upload CSV matrix":
        up = st.file_uploader("Upload W as CSV (n×n)", type=["csv"])
        if up is None:
            st.info("Upload a CSV to continue, or switch to Example matrix.")
            st.stop()
        W_df = parse_csv_matrix(up.getvalue())
    else:
        # Example 5x5
        W_df = pd.DataFrame(
            np.array([
                [ 0.0,  0.7,  0.0, -0.4,  0.0],
                [ 0.0,  0.0,  0.6,  0.0,  0.0],
                [ 0.0,  0.0,  0.0,  0.8, -0.2],
                [ 0.0,  0.0,  0.0,  0.0,  0.5],
                [ 0.3,  0.0,  0.0,  0.0,  0.0],
            ], dtype=float),
            columns=[f"C{i+1}" for i in range(5)],
            index=[f"C{i+1}" for i in range(5)],
        )

    # Ensure square matrix
    if W_df.shape[0] != W_df.shape[1]:
        st.error(f"Matrix must be square (n×n). Current shape: {W_df.shape}")
        st.stop()

    n = W_df.shape[0]

    # Concept names
    st.subheader("Concept labels (optional)")
    default_names = list(W_df.columns) if all(isinstance(c, str) for c in W_df.columns) else [f"C{i+1}" for i in range(n)]
    concept_names_text = st.text_area(
        "One per line (must equal n concepts). Leave empty to use matrix column names.",
        value="\n".join(default_names),
        height=140
    )
    concept_names = [x.strip() for x in concept_names_text.splitlines() if x.strip()]
    if len(concept_names) != n:
        st.warning(f"Expected {n} concept names, got {len(concept_names)}. Using defaults.")
        concept_names = [f"C{i+1}" for i in range(n)]

    # Initial state
    st.subheader("Initial activation A0")
    init_mode = st.radio("A0 input", ["Sliders", "Paste comma-separated"], horizontal=False)

    if init_mode == "Sliders":
        A0 = np.array([st.slider(f"{concept_names[i]}", 0.0, 1.0, 0.1, 0.01) for i in range(n)], dtype=float)
    else:
        A0_text = st.text_input("A0 (comma-separated, length n)", value=",".join(["0.1"] * n))
        try:
            A0_vals = [float(x.strip()) for x in A0_text.split(",")]
            if len(A0_vals) != n:
                st.error(f"A0 must have length {n}.")
                st.stop()
            A0 = np.array(A0_vals, dtype=float)
        except Exception:
            st.error("Could not parse A0. Please enter numbers separated by commas.")
            st.stop()

    st.subheader("Simulation settings")
    steps = st.number_input("Steps", min_value=1, max_value=500, value=25, step=1)
    lam = st.number_input("Sigmoid λ (steepness)", min_value=0.1, max_value=50.0, value=2.0, step=0.1)

    st.subheader("Scenario / Clamping (optional)")
    clamp_help = (
        "Clamp keeps selected concepts fixed during simulation.\n"
        "Enter one per line: index,value  (index is 1..n)\n"
        "Example:\n"
        "1,1.0\n"
        "4,0.0"
    )
    clamp_text = st.text_area("Clamp entries", value="", help=clamp_help, height=120)

    def parse_clamp(text: str):
        clamp = {}
        if not text.strip():
            return clamp
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                raise ValueError(f"Bad clamp line: '{line}'. Use index,value")
            idx = int(parts[0]) - 1
            val = float(parts[1])
            if idx < 0 or idx >= n:
                raise ValueError(f"Clamp index out of range: {idx+1}")
            clamp[idx] = val
        return clamp

    try:
        clamp = parse_clamp(clamp_text)
    except Exception as e:
        st.error(str(e))
        st.stop()

    run_btn = st.button("Run FCM", type="primary")

# Main area
col1, col2 = st.columns([1.15, 1.0], gap="large")

with col1:
    st.subheader("Weight matrix W (Ci → Cj)")
    W_show = W_df.copy()
    W_show.index = concept_names
    W_show.columns = concept_names
    st.dataframe(W_show, use_container_width=True)

with col2:
    st.subheader("Current configuration")
    st.write(f"Concepts: **{n}**")
    st.write(f"Steps: **{steps}**, λ: **{lam}**")
    if clamp:
        pretty = ", ".join([f"{concept_names[i]}={v}" for i, v in clamp.items()])
        st.write(f"Clamp: **{pretty}**")
    else:
        st.write("Clamp: **None**")

if not run_btn:
    st.info("Set inputs on the left, then click **Run FCM**.")
    st.stop()

# Run simulation
W = W_df.to_numpy(dtype=float)
try:
    history = run_fcm(W, A0, steps=int(steps), lam=float(lam), clamp=clamp)
except Exception as e:
    st.error(f"Simulation error: {e}")
    st.stop()

hist_df = pd.DataFrame(history, columns=concept_names)
hist_df.insert(0, "t", np.arange(len(hist_df)))

st.subheader("Results")

cA, cB = st.columns([1.2, 0.8], gap="large")

with cA:
    st.markdown("### Trajectory (table)")
    st.dataframe(hist_df, use_container_width=True)

with cB:
    st.markdown("### Final state")
    final = hist_df.iloc[-1].drop("t")
    st.dataframe(final.to_frame("Final").round(4), use_container_width=True)

st.markdown("### Trajectory plot")
fig = plt.figure()
for name in concept_names:
    plt.plot(hist_df["t"], hist_df[name], label=name)
plt.xlabel("Iteration (t)")
plt.ylabel("Activation")
plt.ylim(0, 1)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
st.pyplot(fig)

# Download outputs
csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download results as CSV",
    data=csv_bytes,
    file_name="fcm_results.csv",
    mime="text/csv"
)

st.caption("FCM update rule used: A(t+1) = sigmoid( A(t) + A(t)·W ), with optional clamping.")
