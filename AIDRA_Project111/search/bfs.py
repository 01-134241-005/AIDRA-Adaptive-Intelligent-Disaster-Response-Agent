
from collections import deque

def bfs(grid, start, goal, return_trace=False):
    rows, cols = len(grid), len(grid[0])
    q = deque([(start, [])])
    visited = {start}
    nodes_expanded = 0
    max_frontier = 0

    while q:
        max_frontier = max(max_frontier, len(q))
        (x, y), path = q.popleft()
        nodes_expanded += 1
        if (x, y) == goal:
            result_path = path + [(x, y)]
            if return_trace:
                return result_path, {"nodes_expanded": nodes_expanded, "max_frontier": max_frontier}
            return result_path
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if grid[nx][ny] != 'X' and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append(((nx, ny), path + [(x, y)]))

    if return_trace:
        return None, {"nodes_expanded": nodes_expanded, "max_frontier": max_frontier}
    return None