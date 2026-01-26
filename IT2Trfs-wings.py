import json
import math
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px
import streamlit as st


# ---------------------------
# Safe expression evaluation
# ---------------------------

_ALLOWED_FUNCS = {
    # basic math
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,

    # math module
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "pi": math.pi,
}

_NAME_RE = re.compile(r"\b[A-Za-z_]\w*\b")

def extract_names(expr: str) -> List[str]:
    """Extract candidate identifiers (variables/functions) from an expression."""
    if not isinstance(expr, str):
        return []
    names = _NAME_RE.findall(expr)
    # Remove Python keywords-ish and allowed functions/constants
    blocked = {"and", "or", "not", "if", "else", "True", "False", "None"}
    return [n for n in names if n not in blocked and n not in _ALLOWED_FUNCS]

def safe_eval(expr: str, env: Dict[str, float]) -> float:
    """
    Safe-ish eval: only math ops + approved functions + variables in env.
    Disallows __, import, attribute access, indexing, lambda, etc.
    """
    if expr is None or str(expr).strip() == "":
        return 0.0
    expr = str(expr)

    bad_tokens = ["__", "import", "lambda", "[", "]", "{", "}", "=", ";", "class", "def", "for", "while", "try", "except", "open", "exec", "eval", "globals", "locals", "."]
    if any(t in expr for t in bad_tokens):
        raise ValueError("Expression contains disallowed tokens.")

    local_env = dict(_ALLOWED_FUNCS)
    local_env.update(env)

    # Only allow names that exist in local_env (functions/constants/variables)
    for name in extract_names(expr):
        if name not in local_env:
            raise NameError(f"Unknown name: {name}")

    return float(eval(expr, {"__builtins__": {}}, local_env))


# ---------------------------
# Model structure
# ---------------------------

@dataclass
class Model:
    stocks: pd.DataFrame       # columns: name, initial, doc, unit
    flows: pd.DataFrame        # columns: name, equation, doc, unit
    aux: pd.DataFrame          # columns: name, equation/value, doc, unit, kind (param/aux)
    stock_flow_map: pd.DataFrame  # columns: stock, inflows (csv), outflows (csv)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stocks": self.stocks.to_dict(orient="records"),
            "flows": self.flows.to_dict(orient="records"),
            "aux": self.aux.to_dict(orient="records"),
            "stock_flow_map": self.stock_flow_map.to_dict(orient="records"),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Model":
        def df(records, cols):
            out = pd.DataFrame(records)
            for c in cols:
                if c not in out.columns:
                    out[c] = ""
            return out[cols]

        return Model(
            stocks=df(d.get("stocks", []), ["name", "initial", "doc", "unit"]),
            flows=df(d.get("flows", []), ["name", "equation", "doc", "unit"]),
            aux=df(d.get("aux", []), ["name", "equation", "doc", "unit", "kind"]),
            stock_flow_map=df(d.get("stock_flow_map", []), ["stock", "inflows", "outflows"]),
        )


def default_model() -> Model:
    # Simple inventory model (classic SD example)
    stocks = pd.DataFrame([
        {"name": "Inventory", "initial": 100.0, "doc": "Stock of items", "unit": "items"},
    ])
    flows = pd.DataFrame([
        {"name": "Production", "equation": "Target_Inventory_Adjustment", "doc": "Items produced per time", "unit": "items/day"},
        {"name": "Sales", "equation": "Demand", "doc": "Items sold per time", "unit": "items/day"},
    ])
    aux = pd.DataFrame([
        {"name": "Demand", "equation": "10", "doc": "Baseline demand", "unit": "items/day", "kind": "param"},
        {"name": "Desired_Inventory", "equation": "150", "doc": "Inventory target", "unit": "items", "kind": "param"},
        {"name": "Adjustment_Time", "equation": "7", "doc": "Time to correct gap", "unit": "day", "kind": "param"},
        {"name": "Target_Inventory_Adjustment", "equation": "(Desired_Inventory - Inventory) / Adjustment_Time + Demand", "doc": "Production decision rule", "unit": "items/day", "kind": "aux"},
    ])
    stock_flow_map = pd.DataFrame([
        {"stock": "Inventory", "inflows": "Production", "outflows": "Sales"},
    ])
    return Model(stocks, flows, aux, stock_flow_map)


# ---------------------------
# Simulation
# ---------------------------

def split_csv_names(s: str) -> List[str]:
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def build_dependency_graph(model: Model) -> nx.DiGraph:
    g = nx.DiGraph()
    # nodes
    for df, col in [(model.stocks, "name"), (model.flows, "name"), (model.aux, "name")]:
        for n in df[col].astype(str).tolist():
            if n:
                g.add_node(n)

    # edges: var -> depends_on
    def add_edges(from_name: str, expr: str):
        for dep in extract_names(expr):
            if dep in g.nodes:
                g.add_edge(dep, from_name)  # dep -> var (direction for topo sort)

    for _, row in model.flows.iterrows():
        add_edges(str(row["name"]), str(row["equation"]))

    for _, row in model.aux.iterrows():
        add_edges(str(row["name"]), str(row["equation"]))

    # Stocks can appear in equations but they do not have equations themselves.
    return g

def topo_order_nonstocks(model: Model) -> List[str]:
    g = build_dependency_graph(model)
    nonstocks = set(model.flows["name"].astype(str)) | set(model.aux["name"].astype(str))
    # Subgraph over nonstocks + their dependencies (stocks might be dependencies)
    # We'll topo sort full graph and filter to nonstocks
    try:
        order = list(nx.topological_sort(g))
        return [n for n in order if n in nonstocks]
    except Exception:
        # cycles exist
        return [n for n in nonstocks if n]  # fallback

def simulate(model: Model, t0: float, t1: float, dt: float, scenario_overrides: Dict[str, float], solver: str = "Euler") -> pd.DataFrame:
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if t1 <= t0:
        raise ValueError("t1 must be > t0")

    times = np.arange(t0, t1 + 1e-12, dt)
    stocks = {str(r["name"]): float(r["initial"]) for _, r in model.stocks.iterrows() if str(r["name"]).strip()}

    # Prepare aux/params equations
    aux_rows = model.aux.copy()
    aux_rows["name"] = aux_rows["name"].astype(str)
    aux_rows["equation"] = aux_rows["equation"].astype(str)
    aux_kind = {r["name"]: str(r.get("kind", "aux") or "aux") for _, r in aux_rows.iterrows()}

    # flow equations
    flow_eq = {str(r["name"]): str(r["equation"]) for _, r in model.flows.iterrows() if str(r["name"]).strip()}

    # stock inflow/outflow mapping
    sfm = model.stock_flow_map.copy()
    sfm["stock"] = sfm["stock"].astype(str)
    sfm["inflows"] = sfm["inflows"].astype(str)
    sfm["outflows"] = sfm["outflows"].astype(str)

    order = topo_order_nonstocks(model)

    records = []

    def compute_all(env_base: Dict[str, float]) -> Dict[str, float]:
        """
        Compute aux/params and flows into env using:
        - topological order if possible
        - light fixed-point iteration if cycles exist
        """
        env = dict(env_base)

        # Apply scenario overrides to params (and allow overriding any variable)
        env.update({k: float(v) for k, v in scenario_overrides.items()})

        # Seed params/aux if equation is numeric and not overridden
        for _, r in aux_rows.iterrows():
            nm = r["name"]
            if nm in env:  # overridden
                continue
            eq = r["equation"]
            try:
                # if it's a plain number, eval quickly
                env[nm] = float(eq)
            except Exception:
                pass

        # First pass: compute in topo-ish order
        for nm in order:
            if nm in stocks:
                continue
            # if overridden, keep it
            if nm in scenario_overrides:
                env[nm] = float(scenario_overrides[nm])
                continue
            if nm in flow_eq:
                env[nm] = safe_eval(flow_eq[nm], env)
            else:
                # aux/param
                row = aux_rows[aux_rows["name"] == nm]
                if not row.empty:
                    eq = row.iloc[0]["equation"]
                    # params still computed unless overridden
                    env[nm] = safe_eval(eq, env)

        # Fixed-point iteration for any remaining undefined non-stocks / cycles
        nonstocks = set(flow_eq.keys()) | set(aux_rows["name"].tolist())
        unresolved = [n for n in nonstocks if n not in env and n not in scenario_overrides]
        if unresolved:
            # Try iterative updates for a few rounds
            for _ in range(15):
                changed = False
                for nm in list(nonstocks):
                    if nm in scenario_overrides:
                        env[nm] = float(scenario_overrides[nm])
                        continue
                    try:
                        if nm in flow_eq:
                            val = safe_eval(flow_eq[nm], env)
                        else:
                            row = aux_rows[aux_rows["name"] == nm]
                            if row.empty:
                                continue
                            val = safe_eval(str(row.iloc[0]["equation"]), env)
                        if (nm not in env) or (abs(env[nm] - val) > 1e-10):
                            env[nm] = val
                            changed = True
                    except Exception:
                        continue
                if not changed:
                    break

        return env

    def stock_derivatives(env: Dict[str, float], stock_vals: Dict[str, float]) -> Dict[str, float]:
        # env has current computed variables; stock_vals provides current stock values
        local_env = dict(env)
        local_env.update(stock_vals)

        # Recompute everything with updated stock values
        full_env = compute_all(local_env)

        deriv = {s: 0.0 for s in stock_vals}
        for _, r in sfm.iterrows():
            sname = r["stock"]
            if sname not in deriv:
                continue
            infl = split_csv_names(r["inflows"])
            outf = split_csv_names(r["outflows"])
            din = sum(float(full_env.get(f, 0.0)) for f in infl)
            dout = sum(float(full_env.get(f, 0.0)) for f in outf)
            deriv[sname] = din - dout
        return deriv, full_env

    # Simulation loop
    for t in times:
        base_env = dict(stocks)  # stocks visible in equations
        env0 = compute_all(base_env)

        # Save record (include stocks + computed vars)
        row = {"time": float(t)}
        row.update({k: float(v) for k, v in stocks.items()})
        # include aux + flows
        for nm in list(env0.keys()):
            if nm not in row:
                try:
                    row[nm] = float(env0[nm])
                except Exception:
                    pass
        records.append(row)

        # advance
        if t >= times[-1] - 1e-12:
            break

        if solver == "RK4":
            k1, _ = stock_derivatives(env0, stocks)
            s2 = {s: stocks[s] + 0.5 * dt * k1[s] for s in stocks}
            k2, _ = stock_derivatives(env0, s2)
            s3 = {s: stocks[s] + 0.5 * dt * k2[s] for s in stocks}
            k3, _ = stock_derivatives(env0, s3)
            s4 = {s: stocks[s] + dt * k3[s] for s in stocks}
            k4, _ = stock_derivatives(env0, s4)
            for s in stocks:
                stocks[s] = stocks[s] + (dt / 6.0) * (k1[s] + 2*k2[s] + 2*k3[s] + k4[s]
                )
        else:
            deriv, _ = stock_derivatives(env0, stocks)
            for s in stocks:
                stocks[s] = stocks[s] + dt * deriv[s]

        # basic non-negativity clamp (optional)
        for s in stocks:
            if np.isnan(stocks[s]) or np.isinf(stocks[s]):
                raise ValueError(f"Stock '{s}' became invalid (NaN/Inf). Check equations.")
    df = pd.DataFrame(records)
    return df


# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Mini Vensim DSS (Streamlit)", layout="wide")
st.title("Mini Vensim-style DSS (System Dynamics) — Streamlit")

if "model" not in st.session_state:
    st.session_state.model = default_model().to_dict()

if "scenarios" not in st.session_state:
    st.session_state.scenarios = {
        "Base": {},  # name -> overrides dict
    }

if "last_run" not in st.session_state:
    st.session_state.last_run = None

model = Model.from_dict(st.session_state.model)

with st.sidebar:
    st.header("Simulation Settings")
    t0 = st.number_input("Start time (t0)", value=0.0, step=1.0)
    t1 = st.number_input("End time (t1)", value=100.0, step=1.0)
    dt = st.number_input("Time step (dt)", value=1.0, min_value=1e-6, step=0.5, format="%.6f")
    solver = st.selectbox("Solver", ["Euler", "RK4"], index=1)

    st.divider()
    st.subheader("Model Import/Export")
    export_str = json.dumps(model.to_dict(), indent=2)
    st.download_button("Download Model JSON", export_str, file_name="model.json", mime="application/json")

    uploaded = st.file_uploader("Import Model JSON", type=["json"])
    if uploaded is not None:
        try:
            d = json.loads(uploaded.read().decode("utf-8"))
            _m = Model.from_dict(d)  # validate structure
            st.session_state.model = _m.to_dict()
            st.success("Model imported.")
            st.rerun()
        except Exception as e:
            st.error(f"Import failed: {e}")

tabs = st.tabs(["1) Model Builder", "2) Scenarios", "3) Run & Charts", "4) Sensitivity", "5) Diagnostics"])

# ---- 1) Model Builder
with tabs[0]:
    st.subheader("Model Builder (Stocks / Flows / Auxiliaries & Parameters)")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Stocks")
        stocks_edit = st.data_editor(
            model.stocks,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn(required=True),
                "initial": st.column_config.NumberColumn(required=True),
                "doc": st.column_config.TextColumn(),
                "unit": st.column_config.TextColumn(),
            },
        )

        st.markdown("### Stock ↔ Flow Mapping")
        st.caption("For each stock: comma-separated inflows and outflows by flow name.")
        sfm_edit = st.data_editor(
            model.stock_flow_map,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "stock": st.column_config.TextColumn(required=True),
                "inflows": st.column_config.TextColumn(),
                "outflows": st.column_config.TextColumn(),
            },
        )

    with c2:
        st.markdown("### Flows")
        flows_edit = st.data_editor(
            model.flows,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn(required=True),
                "equation": st.column_config.TextColumn(required=True, help="Use variables by name, e.g., (Desired - Inventory)/T"),
                "doc": st.column_config.TextColumn(),
                "unit": st.column_config.TextColumn(),
            },
        )

        st.markdown("### Auxiliaries / Parameters")
        aux_edit = st.data_editor(
            model.aux,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn(required=True),
                "equation": st.column_config.TextColumn(required=True, help="Constants like 10 or formulas like A*B"),
                "kind": st.column_config.SelectboxColumn(options=["param", "aux"], required=True),
                "doc": st.column_config.TextColumn(),
                "unit": st.column_config.TextColumn(),
            },
        )

    if st.button("Save Model", type="primary"):
        # Basic cleanup
        def clean(df, keycol):
            df = df.copy()
            df[keycol] = df[keycol].astype(str).str.strip()
            df = df[df[keycol] != ""]
            return df.reset_index(drop=True)

        stocks_edit = clean(stocks_edit, "name")
        flows_edit = clean(flows_edit, "name")
        aux_edit = clean(aux_edit, "name")
        sfm_edit = clean(sfm_edit, "stock")

        # Ensure numeric initial
        if "initial" in stocks_edit.columns:
            stocks_edit["initial"] = pd.to_numeric(stocks_edit["initial"], errors="coerce").fillna(0.0)

        st.session_state.model = Model(stocks_edit, flows_edit, aux_edit, sfm_edit).to_dict()
        st.success("Model saved.")
        st.rerun()

    st.markdown("---")
    st.markdown("**Equation tips**")
    st.markdown(
        "- Use names exactly as written (case-sensitive).\n"
        "- Allowed functions: `min, max, abs, round, exp, log, log10, sqrt, sin, cos, tan, pi`.\n"
        "- Avoid dots `.` in names (use underscores)."
    )

# ---- 2) Scenarios
with tabs[1]:
    st.subheader("Scenarios (Decision Support)")

    scenario_names = list(st.session_state.scenarios.keys())
    sel = st.selectbox("Select scenario", scenario_names, index=0)

    left, right = st.columns([1, 1])
    with left:
        new_name = st.text_input("Create new scenario", value="")
        if st.button("Add Scenario"):
            nm = new_name.strip()
            if nm and nm not in st.session_state.scenarios:
                st.session_state.scenarios[nm] = {}
                st.success(f"Added scenario: {nm}")
                st.rerun()
            else:
                st.warning("Enter a unique scenario name.")

    with right:
        if sel != "Base":
            if st.button("Delete selected scenario"):
                st.session_state.scenarios.pop(sel, None)
                st.success(f"Deleted scenario: {sel}")
                st.rerun()

    st.markdown("### Overrides (typically parameters)")
    # Show params list for convenience
    params = Model.from_dict(st.session_state.model).aux
    params = params[params["kind"].astype(str) == "param"].copy()
    param_names = params["name"].astype(str).tolist()

    overrides = st.session_state.scenarios.get(sel, {})
    rows = []
    for p in param_names:
        base_eq = params.loc[params["name"] == p, "equation"].iloc[0] if (params["name"] == p).any() else ""
        try:
            base_val = float(base_eq)
        except Exception:
            base_val = None
        rows.append({"param": p, "base": base_val, "override": overrides.get(p, None)})

    ov_df = pd.DataFrame(rows)
    ov_edit = st.data_editor(
        ov_df,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "param": st.column_config.TextColumn(disabled=True),
            "base": st.column_config.NumberColumn(disabled=True),
            "override": st.column_config.NumberColumn(help="Leave blank to use base value."),
        },
    )

    if st.button("Save Overrides", type="primary"):
        new_over = {}
        for _, r in ov_edit.iterrows():
            if pd.notna(r["override"]):
                new_over[str(r["param"])] = float(r["override"])
        st.session_state.scenarios[sel] = new_over
        st.success("Overrides saved.")
        st.rerun()

# ---- 3) Run & Charts
with tabs[2]:
    st.subheader("Run & Charts")
    model = Model.from_dict(st.session_state.model)

    c1, c2 = st.columns([1, 1])
    with c1:
        run_scenarios = st.multiselect(
            "Scenarios to run",
            list(st.session_state.scenarios.keys()),
            default=["Base"],
        )

    with c2:
        show_vars = st.multiselect(
            "Variables to plot",
            ["(All Stocks)"] + sorted(
                list(set(model.stocks["name"].astype(str).tolist()
                         + model.flows["name"].astype(str).tolist()
                         + model.aux["name"].astype(str).tolist()))
            ),
            default=["(All Stocks)"],
        )

    if st.button("Run Simulation", type="primary", disabled=(len(run_scenarios) == 0)):
        outputs = {}
        for sc in run_scenarios:
            overrides = st.session_state.scenarios.get(sc, {})
            try:
                df = simulate(model, t0, t1, dt, overrides, solver=solver)
                outputs[sc] = df
            except Exception as e:
                st.error(f"Scenario '{sc}' failed: {e}")
                outputs[sc] = None

        st.session_state.last_run = outputs

    outputs = st.session_state.last_run
    if outputs:
        # Build long dataframe for plotting
        long_frames = []
        for sc, df in outputs.items():
            if df is None or df.empty:
                continue
            df2 = df.copy()
            df2["scenario"] = sc
            long_frames.append(df2)
        if long_frames:
            all_df = pd.concat(long_frames, ignore_index=True)

            # Determine plotted variables
            if "(All Stocks)" in show_vars:
                plot_vars = model.stocks["name"].astype(str).tolist()
            else:
                plot_vars = [v for v in show_vars if v in all_df.columns]

            if plot_vars:
                plot_long = all_df.melt(id_vars=["time", "scenario"], value_vars=plot_vars,
                                        var_name="variable", value_name="value")
                fig = px.line(plot_long, x="time", y="value", color="scenario", line_group="scenario",
                              facet_row="variable", title="Simulation Results")
                fig.update_layout(height=min(1200, 260 * max(1, len(plot_vars))))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Data (latest run)")
            st.dataframe(all_df, use_container_width=True)

            st.download_button(
                "Download results CSV",
                all_df.to_csv(index=False),
                file_name="simulation_results.csv",
                mime="text/csv",
            )

# ---- 4) Sensitivity
with tabs[3]:
    st.subheader("Sensitivity (1-parameter sweep)")

    model = Model.from_dict(st.session_state.model)
    params = model.aux[model.aux["kind"].astype(str) == "param"].copy()
    param_names = params["name"].astype(str).tolist()

    if not param_names:
        st.info("No parameters found. Mark aux rows as kind='param' to enable sensitivity.")
    else:
        p = st.selectbox("Parameter", param_names)
        base_eq = params.loc[params["name"] == p, "equation"].iloc[0]
        try:
            base_val = float(base_eq)
        except Exception:
            base_val = 1.0

        colA, colB, colC = st.columns(3)
        with colA:
            lo = st.number_input("Low", value=float(base_val) * 0.5)
        with colB:
            hi = st.number_input("High", value=float(base_val) * 1.5)
        with colC:
            n = st.number_input("Steps", value=7, min_value=3, max_value=51, step=2)

        metric_var = st.selectbox(
            "Output variable (to plot)",
            sorted(list(set(model.stocks["name"].astype(str).tolist()
                            + model.flows["name"].astype(str).tolist()
                            + model.aux["name"].astype(str).tolist()))),
            index=0,
        )

        if st.button("Run Sensitivity"):
            vals = np.linspace(lo, hi, int(n))
            frames = []
            for v in vals:
                overrides = {p: float(v)}
                df = simulate(model, t0, t1, dt, overrides, solver=solver)
                if metric_var not in df.columns:
                    continue
                frames.append(pd.DataFrame({
                    "time": df["time"],
                    "value": df[metric_var],
                    "param_value": v
                }))
            if frames:
                sen = pd.concat(frames, ignore_index=True)
                fig = px.line(sen, x="time", y="value", color="param_value",
                              title=f"Sensitivity: {metric_var} vs time (sweep {p})")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(sen, use_container_width=True)

# ---- 5) Diagnostics
with tabs[4]:
    st.subheader("Diagnostics")
    model = Model.from_dict(st.session_state.model)

    # show dependency graph warnings
    g = build_dependency_graph(model)

    st.markdown("### Basic checks")
    # Check mapping names exist
    stock_names = set(model.stocks["name"].astype(str))
    flow_names = set(model.flows["name"].astype(str))
    aux_names = set(model.aux["name"].astype(str))

    issues = []

    for _, r in model.stock_flow_map.iterrows():
        s = str(r["stock"]).strip()
        if s and s not in stock_names:
            issues.append(f"Mapping references unknown stock: '{s}'")

        for f in split_csv_names(r.get("inflows", "")):
            if f not in flow_names:
                issues.append(f"Mapping for stock '{s}' references unknown inflow: '{f}'")
        for f in split_csv_names(r.get("outflows", "")):
            if f not in flow_names:
                issues.append(f"Mapping for stock '{s}' references unknown outflow: '{f}'")

    # Check equations for unknown names
    known = stock_names | flow_names | aux_names | set(_ALLOWED_FUNCS.keys())
    def check_expr(label: str, expr: str):
        for nm in extract_names(expr):
            if nm not in known:
                issues.append(f"{label}: unknown name '{nm}' in equation: {expr}")

    for _, r in model.flows.iterrows():
        check_expr(f"Flow '{r['name']}'", str(r["equation"]))
    for _, r in model.aux.iterrows():
        check_expr(f"Aux/Param '{r['name']}'", str(r["equation"]))

    # Cycle detection
    try:
        cycles = list(nx.simple_cycles(g))
    except Exception:
        cycles = []
    if cycles:
        issues.append(f"Dependency cycles detected (may still run with iteration): {cycles[:3]}{' ...' if len(cycles) > 3 else ''}")

    if issues:
        st.error("Issues found:")
        for it in issues:
            st.write("•", it)
    else:
        st.success("No obvious issues detected.")

    st.markdown("### Dependency graph (text)")
    st.caption("Edges show dependency direction: A → B means B depends on A.")
    edges = list(g.edges())
    st.write(pd.DataFrame(edges, columns=["depends_on", "variable"]))
