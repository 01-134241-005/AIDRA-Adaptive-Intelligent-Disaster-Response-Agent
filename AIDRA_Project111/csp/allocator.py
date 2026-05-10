from config import AMBULANCES, MAX_PER_AMBULANCE
from utils.logger import log_decision

def allocate(victims, fuzzy_system, search_agent, start, grid):
    """
    CSP allocation with MRV heuristic and forward checking.
    Returns: (ambulances, backtrack_count)
    """
    # Compute fuzzy priority for each victim
    scored = []
    for loc, sev in victims:
        risk = 3 if grid[loc[0]][loc[1]] == 'R' else 0
        path = search_agent(grid, start, loc)
        time_cost = len(path) if path else 30
        priority = fuzzy_system.compute(risk, time_cost)
        scored.append((loc, priority, sev))
    
    # Initial ordering by fuzzy priority (fallback)
    scored.sort(key=lambda x: x[1], reverse=True)
    variables = [ (idx, loc, sev) for idx, (loc, _, sev) in enumerate(scored) ]
    
    # Domain for each variable: ambulance IDs (0..AMBULANCES-1)
    domains = {idx: list(range(AMBULANCES)) for idx, _, _ in variables}
    
    # Track remaining capacities
    capacities = {amb: MAX_PER_AMBULANCE for amb in range(AMBULANCES)}
    assignments = {}
    backtrack_count = 0
    
    def is_consistent(var_idx, amb, assignments):
        # Forward checking: capacity constraint
        if capacities[amb] <= 0:
            return False
        return True
    
    def select_unassigned_variable(assignments):
        """MRV: choose variable with smallest domain size"""
        unassigned = [(idx, loc, sev) for idx, loc, sev in variables if idx not in assignments]
        if not unassigned:
            return None
        # Compute current domain sizes (remaining ambulances with capacity > 0)
        best_var = None
        best_size = float('inf')
        for idx, loc, sev in unassigned:
            # Domain = ambulances with remaining capacity
            domain_size = sum(1 for amb in range(AMBULANCES) if capacities[amb] > 0)
            if domain_size < best_size:
                best_size = domain_size
                best_var = (idx, loc, sev)
        return best_var
    
    def order_domain_values(var_idx, loc, sev, assignments):
        """Value ordering: prefer ambulances with lower current load (not required)"""
        # Return ambulances sorted by remaining capacity (descending) – simple heuristic
        return sorted([amb for amb in range(AMBULANCES) if capacities[amb] > 0],
                      key=lambda a: capacities[a], reverse=True)
    
    def forward_check(var_idx, amb, assignments):
        """ Reduce capacities temporarily """
        capacities[amb] -= 1
        return True
    
    def backtrack(assignments):
        nonlocal backtrack_count
        if len(assignments) == len(variables):
            return True
        var = select_unassigned_variable(assignments)
        if var is None:
            return False
        idx, loc, sev = var
        for amb in order_domain_values(idx, loc, sev, assignments):
            if is_consistent(idx, amb, assignments):
                assignments[idx] = amb
                forward_check(idx, amb, assignments)
                if backtrack(assignments):
                    return True
                # Undo assignment and restore capacity
                del assignments[idx]
                capacities[amb] += 1
                backtrack_count += 1
        return False
    
    success = backtrack(assignments)
    if not success:
        log_decision(f"CSP failed after {backtrack_count} backtracks – insufficient capacity")
    else:
        log_decision(f"CSP succeeded with {backtrack_count} backtracks")
    
    # Build ambulance → victim list
    ambulances_dict = {amb: [] for amb in range(AMBULANCES)}
    for idx, amb in assignments.items():
        loc = variables[idx][1]
        ambulances_dict[amb].append(loc)
    
    return ambulances_dict, backtrack_count