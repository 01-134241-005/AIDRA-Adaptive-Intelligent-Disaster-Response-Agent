
def dfs(grid, start, goal, return_trace=False):
    stack = [(start, [])]
    visited = set()
    rows, cols = len(grid), len(grid[0])
    nodes_expanded = 0
    max_frontier = 0

    while stack:
        max_frontier = max(max_frontier, len(stack))
        (x, y), path = stack.pop()
        if (x, y) in visited:
            continue
        visited.add((x, y))
        nodes_expanded += 1
        if (x, y) == goal:
            result_path = path + [(x, y)]
            if return_trace:
                return result_path, {"nodes_expanded": nodes_expanded, "max_frontier": max_frontier}
            return result_path
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] != 'X':
                stack.append(((nx, ny), path + [(x, y)]))

    if return_trace:
        return None, {"nodes_expanded": nodes_expanded, "max_frontier": max_frontier}
    return None