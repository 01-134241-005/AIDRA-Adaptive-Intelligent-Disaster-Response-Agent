
def hill_climbing(grid, start, goal, return_trace=False):
    current = start
    path = [current]
    visited = set()
    rows, cols = len(grid), len(grid[0])
    nodes_expanded = 0
    max_frontier = 0

    while current != goal:
        visited.add(current)
        x, y = current
        neighbors = []
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if grid[nx][ny] != 'X' and (nx, ny) not in visited:
                    neighbors.append((nx, ny))
        max_frontier = max(max_frontier, len(neighbors))
        if not neighbors:
            if return_trace:
                return None, {"nodes_expanded": nodes_expanded, "max_frontier": max_frontier}
            return None
        next_node = min(neighbors, key=lambda n: abs(n[0]-goal[0]) + abs(n[1]-goal[1]))
        current = next_node
        path.append(current)
        nodes_expanded += 1

    if return_trace:
        return path, {"nodes_expanded": nodes_expanded, "max_frontier": max_frontier}
    return path