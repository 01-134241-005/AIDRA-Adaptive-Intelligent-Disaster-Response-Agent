
# pipeline.py — shared AI pipeline used by both main.py and streamlit_app.py
from config import AMBULANCES, MAX_PER_AMBULANCE
from environment import create_grid, get_victims
from search import astar, bfs, dfs, greedy, hill_climbing
from csp import allocate
from uncertainty import FuzzyRiskSystem
from dynamic import block_road
from metric import compute_metrics
from utils import log_decision, log_event, print_comparison
from ml import train_models, predict_severity
from collections import Counter
import random


def _analyze_path(path, grid):
    if not path:
        return {"length": None, "risk": None, "time": None}
    length = len(path)
    risk = sum(1 for x, y in path if grid[x][y] == 'R')
    time = sum(3 if grid[x][y] == 'R' else 1 for x, y in path)
    return {"length": length, "risk": risk, "time": time}


def run_simulation(seed=None):
    """
    Run the full AIDRA AI pipeline.
    Returns a rich dict with all results for display by main.py or streamlit_app.py.
    """
    if seed is not None:
        random.seed(seed)

    event_log = []      # list of (tag, message) tuples
    decision_log = []

    def _evt(msg):
        event_log.append(msg)
        log_event(msg)

    def _dec(msg):
        decision_log.append(msg)
        log_decision(msg)

    # ── 1. Environment ──────────────────────────────────────────────
    grid = create_grid()
    victims_raw = get_victims()
    start = (0, 0)

    _evt("=== AIDRA: Fully Integrated Intelligent Disaster Response Agent ===")
    _evt(f"Grid {len(grid)}x{len(grid[0])} | Ambulances: {AMBULANCES} | Capacity: {MAX_PER_AMBULANCE} each")

    # ── 2. ML Training ──────────────────────────────────────────────
    _evt("Training ML models (custom KNN & Naive Bayes)...")
    ml_results = train_models()
    best_model = ml_results["best_model"]

    # ── 3. Predict severities ───────────────────────────────────────
    updated_victims = []
    for loc, _ in victims_raw:
        sev = predict_severity(best_model, loc, grid)
        updated_victims.append((loc, sev))
        _evt(f"  Victim at {loc}: ML predicted severity = {sev}")

    # ── 4. Enforce 2 Critical / 2 Moderate / 1 Minor ───────────────
    required = {"Critical": 2, "Moderate": 2, "Minor": 1}
    current_counts = Counter(s for _, s in updated_victims)
    if current_counts != required:
        urgency_scores = []
        for loc, sev in updated_victims:
            dist = abs(loc[0]) + abs(loc[1])
            haz  = 1 if grid[loc[0]][loc[1]] == 'R' else 0
            urgency_scores.append((loc, sev, dist + haz * 5))
        urgency_scores.sort(key=lambda x: x[2], reverse=True)
        new_sevs = ["Critical", "Critical", "Moderate", "Moderate", "Minor"]
        for i, (loc, old_sev, _) in enumerate(urgency_scores):
            new_sev = new_sevs[i]
            if old_sev != new_sev:
                idx = next(j for j, (l, _) in enumerate(updated_victims) if l == loc)
                updated_victims[idx] = (loc, new_sev)
                _evt(f"  Adjusted {loc}: {old_sev} -> {new_sev} (enforce 2C/2M/1Mi spec)")

    # ── 5. Prioritised rescue order ─────────────────────────────────
    SEV_WEIGHT = {"Critical": 10, "Moderate": 5, "Minor": 1}
    urgency = []
    for loc, sev in updated_victims:
        dist = abs(loc[0]) + abs(loc[1])
        haz  = 1 if grid[loc[0]][loc[1]] == 'R' else 0
        score = dist + haz * 5 + SEV_WEIGHT[sev]
        urgency.append((loc, sev, score))
    urgency.sort(key=lambda x: x[2], reverse=True)
    _evt("Prioritised rescue order:")
    for i, (loc, sev, score) in enumerate(urgency, 1):
        _evt(f"  {i}. {loc} ({sev}) - urgency score {score}")

    # ── 6. Fuzzy system ─────────────────────────────────────────────
    fuzzy = FuzzyRiskSystem()

    # ── 7. CSP allocation ───────────────────────────────────────────
    _evt("Running CSP allocation (MRV + forward checking)...")
    allocation, backtrack_count = allocate(updated_victims, fuzzy, astar, start, grid)
    _dec(f"CSP allocation result: {allocation}")
    _dec(f"CSP backtrack count:   {backtrack_count}")

    # ── 8. Per-victim processing ────────────────────────────────────
    victim_results = []
    all_chosen_paths = []
    bfs_paths_all    = []
    algo_times_v1    = None
    algo_risks_v1    = None

    for v_idx, (victim_loc, severity) in enumerate(updated_victims):
        _evt(f"\n[Ambulance] Victim {v_idx+1} at {victim_loc} ({severity})")

        # Run all 5 search algorithms with trace
        p_bfs,  tr_bfs  = bfs(grid,           start, victim_loc, return_trace=True)
        p_ast,  tr_ast  = astar(grid,          start, victim_loc, return_trace=True)
        p_grd,  tr_grd  = greedy(grid,         start, victim_loc, return_trace=True)
        p_hil,  tr_hil  = hill_climbing(grid,  start, victim_loc, return_trace=True)
        p_dfs,  tr_dfs  = dfs(grid,            start, victim_loc, return_trace=True)

        bfs_paths_all.append(p_bfs)

        d_bfs = _analyze_path(p_bfs,  grid)
        d_ast = _analyze_path(p_ast,  grid)
        d_grd = _analyze_path(p_grd,  grid)
        d_hil = _analyze_path(p_hil,  grid)
        d_dfs = _analyze_path(p_dfs,  grid)

        # Print comparison table
        print_comparison(victim_loc, p_bfs, p_ast, p_grd, p_hil, p_dfs, grid)

        # Log search traces
        for algo, tr in [("BFS", tr_bfs), ("A*", tr_ast),
                         ("Greedy", tr_grd), ("Hill", tr_hil), ("DFS", tr_dfs)]:
            _evt(f"  [{algo}] nodes_expanded={tr['nodes_expanded']}  max_frontier={tr['max_frontier']}")

        if v_idx == 0:
            algo_names_v1 = ['BFS', 'A*', 'Greedy', 'Hill', 'DFS']
            algo_times_v1 = [d_bfs['time'], d_ast['time'], d_grd['time'], d_hil['time'], d_dfs['time']]
            algo_risks_v1 = [d_bfs['risk'], d_ast['risk'], d_grd['risk'], d_hil['risk'], d_dfs['risk']]

        # Survival probability → decision objective
        survival_prob = {"Critical": 0.3, "Moderate": 0.7, "Minor": 0.95}[severity]
        _dec(f"Survival probability: {survival_prob:.2f}")

        risk_score = d_ast['risk'] if p_ast else 0
        time_cost  = d_ast['time'] if p_ast else 99
        fuzzy_priority = fuzzy.compute(risk_score, time_cost)
        _dec(f"Fuzzy priority (risk={risk_score}, time={time_cost}): {fuzzy_priority:.2f}")

        options = {
            "BFS":    (p_bfs, d_bfs),
            "A*":     (p_ast, d_ast),
            "Greedy": (p_grd, d_grd),
            "Hill":   (p_hil, d_hil),
            "DFS":    (p_dfs, d_dfs),
        }
        options = {k: v for k, v in options.items() if v[1]["time"] is not None}

        if not options:
            _dec("No path found - victim unreachable")
            all_chosen_paths.append(None)
            victim_results.append({
                "loc": victim_loc, "severity": severity,
                "survival_prob": survival_prob, "fuzzy_priority": fuzzy_priority,
                "algo_comparison": {}, "chosen_algorithm": "—",
                "chosen_path": None, "reason": "UNREACHABLE", "replanning_event": None,
            })
            continue

        # Decision logic (explicit trade-off justification)
        if survival_prob < 0.5:
            best   = min(options.items(), key=lambda x: x[1][1]["time"])
            reason = "LOW SURVIVAL (<50%) -> FASTEST PATH (save before deterioration)"
        elif fuzzy_priority >= 7:
            best   = min(options.items(), key=lambda x: x[1][1]["risk"])
            reason = "HIGH FUZZY PRIORITY (>=7) -> SAFEST PATH (reduce hazard exposure)"
        elif fuzzy_priority >= 4:
            best   = min(options.items(), key=lambda x: x[1][1]["time"] + x[1][1]["risk"])
            reason = "MEDIUM PRIORITY (4-7) -> BALANCED PATH (time + risk combined)"
        else:
            best   = min(options.items(), key=lambda x: x[1][1]["time"])
            reason = "LOW PRIORITY (<4) -> FASTEST PATH (throughput focus)"

        chosen_algo = best[0]
        chosen_path = best[1][0]
        _dec(f"DECISION: {reason}")
        _dec(f"  Selected: {chosen_algo}  |  Time={best[1][1]['time']}  Risk={best[1][1]['risk']}")
        _dec(f"  Path: {chosen_path}")
        all_chosen_paths.append(chosen_path)

        # Build algo_comparison dict with trace info
        algo_comparison = {}
        for name, path_x, data_x, trace_x in [
            ("BFS",    p_bfs, d_bfs, tr_bfs),
            ("A*",     p_ast, d_ast, tr_ast),
            ("Greedy", p_grd, d_grd, tr_grd),
            ("Hill",   p_hil, d_hil, tr_hil),
            ("DFS",    p_dfs, d_dfs, tr_dfs),
        ]:
            algo_comparison[name] = {
                **data_x,
                "nodes_expanded": trace_x["nodes_expanded"],
                "max_frontier":   trace_x["max_frontier"],
            }

        # Dynamic event — 50 % chance of road block → replan
        replanning_event = None
        if random.random() < 0.5:
            old_path = chosen_path[:] if chosen_path else []
            blocked  = block_road(grid, [loc for loc, _ in updated_victims])
            if blocked:
                _evt(f"[Warning] Dynamic event: Road blocked at {blocked}")
                new_path = astar(grid, start, victim_loc)
                if new_path and new_path != chosen_path:
                    _dec("Replanning triggered!")
                    _dec(f"  Old path:     {old_path}")
                    _dec(f"  New path:     {new_path}")
                    _dec(f"  Reason:       Road at {blocked} blocked mid-rescue")
                    all_chosen_paths[-1] = new_path
                    chosen_path = new_path
                    replanning_event = {
                        "blocked_cell": blocked,
                        "old_path":     old_path,
                        "new_path":     new_path,
                        "reason":       f"Road at {blocked} blocked mid-rescue",
                    }
                else:
                    _dec("Replanning: Path unchanged or no alternative found")
            else:
                _evt("No free cell to block - skipping dynamic event")
        else:
            _evt("No dynamic event for this victim")

        victim_results.append({
            "loc":               victim_loc,
            "severity":          severity,
            "survival_prob":     survival_prob,
            "fuzzy_priority":    fuzzy_priority,
            "algo_comparison":   algo_comparison,
            "chosen_algorithm":  chosen_algo,
            "chosen_path":       chosen_path,
            "reason":            reason,
            "replanning_event":  replanning_event,
        })

    # ── 9. Final KPIs ───────────────────────────────────────────────
    _evt("\nComputing final KPIs and saving graphs...")
    metrics = compute_metrics(
        all_chosen_paths, grid, bfs_paths_all,
        algorithm_names=algo_names_v1 if algo_times_v1 else None,
        algorithm_times=algo_times_v1,
        algorithm_risks=algo_risks_v1,
    )

    return {
        "grid":             grid,
        "victims":          updated_victims,
        "prioritized_order": urgency,
        "csp_allocation":   allocation,
        "csp_backtracks":   backtrack_count,
        "ml_results":       ml_results,
        "victim_results":   victim_results,
        "final_metrics":    metrics,
        "event_log":        event_log,
        "decision_log":     decision_log,
    }
