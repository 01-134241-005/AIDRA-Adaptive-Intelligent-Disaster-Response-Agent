
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import io

st.set_page_config(
    page_title="AIDRA — AI Disaster Response",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0f1117; color: #e0e0e0; }
section[data-testid="stSidebar"] { background: #161b27; border-right: 1px solid #2a2f3e; }
.metric-card {
    background: linear-gradient(135deg,#1e2535,#252d42);
    border: 1px solid #2e3855; border-radius: 12px;
    padding: 18px 22px; margin: 6px 0; text-align: center;
}
.metric-card .val { font-size: 2rem; font-weight: 700; color: #60a5fa; }
.metric-card .lbl { font-size: 0.78rem; color: #94a3b8; margin-top: 2px; }
.sev-critical { background:#7f1d1d; color:#fca5a5; border-radius:6px; padding:2px 9px; font-size:.8rem; font-weight:600; }
.sev-moderate { background:#78350f; color:#fcd34d; border-radius:6px; padding:2px 9px; font-size:.8rem; font-weight:600; }
.sev-minor    { background:#14532d; color:#86efac; border-radius:6px; padding:2px 9px; font-size:.8rem; font-weight:600; }
.dec-box  { background:#1a2235; border-left:3px solid #60a5fa; border-radius:6px; padding:8px 14px; margin:4px 0; font-size:.82rem; font-family:monospace; }
.evt-box  { background:#1a2a1e; border-left:3px solid #4ade80; border-radius:6px; padding:8px 14px; margin:4px 0; font-size:.82rem; font-family:monospace; }
.winner   { background:linear-gradient(90deg,#1e3a1e,#14532d); border:1px solid #4ade80; border-radius:10px; padding:10px 18px; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚑 AIDRA Dashboard")
    st.markdown("**AIC-201 · Sem 5-A · Group Project**")
    st.markdown("---")
    st.markdown("**System Configuration**")
    st.markdown("- Grid: `5 × 5`")
    st.markdown("- Ambulances: `2`")
    st.markdown("- Capacity: `2 victims each`")
    st.markdown("- Rescue teams: `1`")
    st.markdown("- Victims: `5` (2C / 2M / 1Mi)")
    st.markdown("---")
    run_btn = st.button("▶ Run Full Simulation", use_container_width=True, type="primary")
    if st.button("🔄 Reset", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
    st.markdown("---")
    st.markdown("**Legend**")
    st.markdown("🟦 S = Start/Base · ⬛ X = Blocked")
    st.markdown("🟧 R = Hazard (+3 cost) · 🟩 H = Hospital")
    st.markdown("🔴 Critical · 🟡 Moderate · 🟢 Minor")

# ── Run Pipeline ─────────────────────────────────────────────────────────────
if run_btn or "results" not in st.session_state:
    with st.spinner("Running AIDRA AI pipeline…"):
        from pipeline import run_simulation
        st.session_state.results = run_simulation()

R = st.session_state.get("results")

# ── Tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5, t6 = st.tabs([
    "🗺 Grid & Victims",
    "🤖 ML Analysis",
    "🔍 Algorithm Comparison",
    "🧩 CSP & Fuzzy",
    "📋 Decision Log",
    "📊 KPI Dashboard",
])

# ═══════════════════════════════════════════════════════
# TAB 1 — GRID & VICTIMS
# ═══════════════════════════════════════════════════════
with t1:
    if R is None:
        st.info("Click **▶ Run Full Simulation** in the sidebar.")
    else:
        grid    = R["grid"]
        victims = R["victims"]
        victim_locs = {loc: sev for loc, sev in victims}

        CELL_STYLES = {
            'S': ("background:#1e3a5f;color:#93c5fd;font-weight:700;", "🏠"),
            'H': ("background:#14532d;color:#86efac;font-weight:700;", "🏥"),
            'X': ("background:#1a1a1a;color:#374151;",                 "⛔"),
            'R': ("background:#7c2d12;color:#fdba74;",                 "⚠️"),
            '.': ("background:#1e2535;color:#94a3b8;",                 ""),
        }
        SEV_ICONS = {"Critical": "🔴", "Moderate": "🟡", "Minor": "🟢"}

        # Build HTML table
        html = '<table style="border-collapse:collapse;width:100%;max-width:480px;margin:auto">'
        for ri, row in enumerate(grid):
            html += "<tr>"
            for ci, cell in enumerate(row):
                style, icon = CELL_STYLES.get(cell, ("", ""))
                sev = victim_locs.get((ri, ci))
                if sev:
                    style = "background:#2d1b69;color:#c4b5fd;font-weight:700;"
                    icon  = SEV_ICONS[sev]
                html += (
                    f'<td style="width:80px;height:72px;text-align:center;'
                    f'border:1px solid #2e3855;border-radius:6px;font-size:1.5rem;{style}">'
                    f'{icon}<br>'
                    f'<span style="font-size:.6rem;opacity:.7">({ri},{ci})</span>'
                    f'</td>'
                )
            html += "</tr>"
        html += "</table>"

        col_g, col_v = st.columns([1, 1])
        with col_g:
            st.markdown("### 🗺 Disaster Grid")
            st.markdown(html, unsafe_allow_html=True)

        with col_v:
            st.markdown("### 🧍 Victim Roster")
            order = R["prioritized_order"]
            for rank, (loc, sev, score) in enumerate(order, 1):
                badge_cls = f"sev-{sev.lower()}"
                st.markdown(
                    f'<div class="metric-card">'
                    f'<b>#{rank} — {loc}</b>&nbsp;&nbsp;'
                    f'<span class="{badge_cls}">{sev}</span><br>'
                    f'<span style="font-size:.8rem;color:#94a3b8">Urgency score: {score}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ═══════════════════════════════════════════════════════
# TAB 2 — ML ANALYSIS
# ═══════════════════════════════════════════════════════
with t2:
    if R is None:
        st.info("Run simulation first.")
    else:
        ml   = R["ml_results"]
        best = ml["best_model_name"]
        classes = ["Critical", "Moderate", "Minor"]

        st.markdown("### 🤖 ML Model Comparison (KNN vs Naive Bayes)")
        c1, c2 = st.columns(2)

        def render_model_card(col, name, rep, is_best):
            with col:
                border = "#4ade80" if is_best else "#2e3855"
                st.markdown(
                    f'<div style="border:2px solid {border};border-radius:12px;padding:16px;'
                    f'background:#1e2535;margin-bottom:12px">'
                    f'<h4 style="margin:0;color:#e0e0e0">{name}'
                    + (' &nbsp;✅ <span style="color:#4ade80;font-size:.8rem">BEST</span>' if is_best else '')
                    + f'</h4></div>',
                    unsafe_allow_html=True,
                )
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy",  f"{rep['accuracy']:.3f}")
                m2.metric("Precision", f"{rep['macro_precision']:.3f}")
                m3.metric("Recall",    f"{rep['macro_recall']:.3f}")
                m4.metric("F1 (macro)",f"{rep['macro_f1']:.3f}")

                st.markdown("**Per-Class Metrics**")
                rows = []
                for c in classes:
                    pc = rep["per_class"][c]
                    rows.append({
                        "Class": c,
                        "Precision": f"{pc['precision']:.3f}",
                        "Recall":    f"{pc['recall']:.3f}",
                        "F1":        f"{pc['f1']:.3f}",
                    })
                st.table(rows)

                st.markdown("**Confusion Matrix**")
                cm   = rep["confusion_matrix"]
                fig, ax = plt.subplots(figsize=(4, 3))
                mat  = np.array([[cm[a][p] for p in classes] for a in classes])
                im   = ax.imshow(mat, cmap="Blues")
                ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
                ax.set_xticklabels([c[:3] for c in classes], color="white", fontsize=8)
                ax.set_yticklabels([c[:3] for c in classes], color="white", fontsize=8)
                ax.set_xlabel("Predicted", color="white", fontsize=8)
                ax.set_ylabel("Actual",    color="white", fontsize=8)
                ax.set_title(name, color="white", fontsize=9)
                for i in range(len(classes)):
                    for j in range(len(classes)):
                        ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                                color="white" if mat[i, j] < mat.max()/2 else "black", fontsize=10)
                fig.patch.set_facecolor("#1e2535")
                ax.set_facecolor("#1e2535")
                buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight", dpi=100); buf.seek(0)
                st.image(buf); plt.close(fig)

        render_model_card(c1, "KNN (k=3)",    ml["knn"], best == "KNN")
        render_model_card(c2, "Naive Bayes",  ml["nb"],  best == "NaiveBayes")

# ═══════════════════════════════════════════════════════
# TAB 3 — ALGORITHM COMPARISON
# ═══════════════════════════════════════════════════════
with t3:
    if R is None:
        st.info("Run simulation first.")
    else:
        st.markdown("### 🔍 Search Algorithm Comparison")
        algos = ["BFS", "A*", "Greedy", "Hill", "DFS"]

        for vi, vr in enumerate(R["victim_results"]):
            with st.expander(f"Victim {vi+1}  {vr['loc']}  —  {vr['severity']}", expanded=(vi == 0)):
                ac = vr["algo_comparison"]
                if not ac:
                    st.write("No paths found (unreachable).")
                    continue
                rows = []
                for a in algos:
                    d = ac.get(a, {})
                    rows.append({
                        "Algorithm": a,
                        "Path Len":  d.get("length", "—"),
                        "Time Cost": d.get("time",   "—"),
                        "Risk":      d.get("risk",   "—"),
                        "Nodes Exp": d.get("nodes_expanded", "—"),
                        "Max Front": d.get("max_frontier",   "—"),
                        "Chosen":    "✅" if a == vr["chosen_algorithm"] else "",
                    })
                st.table(rows)

                # Mini bar chart
                times = [ac.get(a, {}).get("time") or 0 for a in algos]
                risks = [ac.get(a, {}).get("risk") or 0 for a in algos]
                fig, axes = plt.subplots(1, 2, figsize=(8, 2.5))
                fig.patch.set_facecolor("#1e2535")
                colors = ["#4ade80" if a == vr["chosen_algorithm"] else "#60a5fa" for a in algos]
                for ax, vals, title, col in [
                    (axes[0], times, "Time Cost", "#60a5fa"),
                    (axes[1], risks, "Risk Exposure", "#f87171"),
                ]:
                    ax.bar(algos, vals, color=colors)
                    ax.set_title(title, color="white", fontsize=9)
                    ax.tick_params(colors="white", labelsize=7)
                    ax.set_facecolor("#1e2535")
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#2e3855")
                plt.tight_layout()
                buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight", dpi=100); buf.seek(0)
                st.image(buf); plt.close(fig)

                st.markdown(f"**Decision:** `{vr['reason']}`  →  **{vr['chosen_algorithm']}** selected")

# ═══════════════════════════════════════════════════════
# TAB 4 — CSP & FUZZY
# ═══════════════════════════════════════════════════════
with t4:
    if R is None:
        st.info("Run simulation first.")
    else:
        st.markdown("### 🧩 CSP Allocation")
        ca, cb = st.columns(2)
        with ca:
            alloc = R["csp_allocation"]
            victims_map = {loc: sev for loc, sev in R["victims"]}
            for amb_id, locs in alloc.items():
                st.markdown(
                    f'<div class="metric-card"><b>Ambulance A{amb_id}</b><br>'
                    + "".join(
                        f'<span class="sev-{victims_map.get(l, "minor").lower()}">'
                        f'{l} ({victims_map.get(l, "?")})</span>&nbsp;'
                        for l in locs
                    )
                    + f'<br><span style="font-size:.75rem;color:#94a3b8">'
                      f'{len(locs)} victim(s) assigned</span></div>',
                    unsafe_allow_html=True,
                )
        with cb:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="val">{R["csp_backtracks"]}</div>'
                f'<div class="lbl">CSP Backtrack Count</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="val">MRV</div>'
                f'<div class="lbl">Heuristic used (Minimum Remaining Values)</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="val">FC</div>'
                f'<div class="lbl">Forward Checking applied</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### 🌫 Fuzzy Logic Priority Scores")
        fuzzy_data = [(f"V{i+1} {vr['loc']}", vr["fuzzy_priority"], vr["severity"])
                      for i, vr in enumerate(R["victim_results"])]
        labels = [f[0] for f in fuzzy_data]
        scores = [f[1] for f in fuzzy_data]
        sev_colors = {"Critical": "#f87171", "Moderate": "#fbbf24", "Minor": "#4ade80"}
        bar_colors = [sev_colors.get(f[2], "#60a5fa") for f in fuzzy_data]

        fig, ax = plt.subplots(figsize=(8, 3))
        bars = ax.barh(labels, scores, color=bar_colors)
        ax.axvline(x=7, color="white", linestyle="--", linewidth=1, label="SAFE threshold (7)")
        ax.axvline(x=4, color="#fbbf24", linestyle="--", linewidth=1, label="BALANCED threshold (4)")
        ax.set_xlim(0, 10)
        ax.set_xlabel("Fuzzy Priority (0–10)", color="white")
        ax.set_title("Fuzzy Priority per Victim", color="white")
        ax.tick_params(colors="white")
        ax.set_facecolor("#1e2535")
        fig.patch.set_facecolor("#1e2535")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2e3855")
        legend = ax.legend(facecolor="#1e2535", labelcolor="white", fontsize=8)
        buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight", dpi=100); buf.seek(0)
        st.image(buf); plt.close(fig)

        # Replanning events
        st.markdown("---")
        st.markdown("### ⚠ Dynamic Replanning Events")
        any_replan = False
        for i, vr in enumerate(R["victim_results"]):
            re = vr.get("replanning_event")
            if re:
                any_replan = True
                st.error(
                    f"**Victim {i+1} {vr['loc']}** — Road blocked at `{re['blocked_cell']}`\n\n"
                    f"- Old path: `{re['old_path']}`\n"
                    f"- New path: `{re['new_path']}`\n"
                    f"- Reason: {re['reason']}"
                )
        if not any_replan:
            st.success("No dynamic replanning events occurred this run.")

# ═══════════════════════════════════════════════════════
# TAB 5 — DECISION LOG
# ═══════════════════════════════════════════════════════
with t5:
    if R is None:
        st.info("Run simulation first.")
    else:
        st.markdown("### 📋 Full Decision & Event Log")
        col_e, col_d = st.columns(2)
        with col_e:
            st.markdown("#### 📡 Events")
            for msg in R["event_log"]:
                st.markdown(f'<div class="evt-box">{msg}</div>', unsafe_allow_html=True)
        with col_d:
            st.markdown("#### 🧠 Decisions")
            for msg in R["decision_log"]:
                st.markdown(f'<div class="dec-box">{msg}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# TAB 6 — KPI DASHBOARD
# ═══════════════════════════════════════════════════════
with t6:
    if R is None:
        st.info("Run simulation first.")
    else:
        m = R["final_metrics"]
        st.markdown("### 📊 Final KPI Dashboard")

        k1, k2, k3, k4, k5 = st.columns(5)
        kpis = [
            (k1, "Victims Saved",        f"{m['victims_saved']}/5",          "🧍"),
            (k2, "Avg Rescue Time",      f"{m['avg_time']:.1f} steps",       "⏱"),
            (k3, "Optimality Ratio",     f"{m['optimality_ratio']:.3f}",     "📐"),
            (k4, "Risk Exposure",        f"{m['risk_exposure']} cells",      "⚠️"),
            (k5, "Resource Utilization", f"{m['resource_utilization']:.0%}", "🚑"),
        ]
        for col, label, value, icon in kpis:
            col.markdown(
                f'<div class="metric-card"><div class="val">{icon} {value}</div>'
                f'<div class="lbl">{label}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        # Show saved PNG graphs if they exist
        import os
        graphs = [
            ("algorithm_comparison.png", "Algorithm Comparison (Victim 1)"),
            ("time_risk_tradeoff.png",   "Time vs Risk Trade-off"),
            ("optimality_ratio.png",     "Path Optimality Ratio"),
        ]
        cols = st.columns(3)
        for (fname, title), col in zip(graphs, cols):
            if os.path.exists(fname):
                with col:
                    st.markdown(f"**{title}**")
                    st.image(fname)
            else:
                col.info(f"{fname} not generated yet.")

        # Per-victim summary table
        st.markdown("---")
        st.markdown("### Per-Victim Summary")
        rows = []
        for i, vr in enumerate(R["victim_results"]):
            rows.append({
                "Victim": f"V{i+1} {vr['loc']}",
                "Severity": vr["severity"],
                "Survival%": f"{vr['survival_prob']:.0%}",
                "Fuzzy": f"{vr['fuzzy_priority']:.2f}",
                "Algorithm": vr["chosen_algorithm"],
                "Decision Reason": vr["reason"][:45] + "…" if len(vr["reason"]) > 45 else vr["reason"],
                "Replanned": "✅" if vr.get("replanning_event") else "—",
            })
        st.table(rows)