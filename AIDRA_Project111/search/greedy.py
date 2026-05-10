
import heapq

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def greedy(grid, start, goal, return_trace=False):
    pq = [(heuristic(start, goal), start, [])]
    visited = set()
    rows, cols = len(grid), len(grid[0])
    nodes_expanded = 0
    max_frontier = 0

    while pq:
        max_frontier = max(max_frontier, len(pq))
        _, (x, y), path = heapq.heappop(pq)
        if (x, y) == goal:
            result_path = path + [(x, y)]
            if return_trace:
                return result_path, {"nodes_expanded": nodes_expanded, "max_frontier": max_frontier}
            return result_path
        if (x, y) in visited:
            continue
        visited.add((x, y))
        nodes_expanded += 1
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] != 'X':
                heapq.heappush(pq, (heuristic((nx, ny), goal), (nx, ny), path + [(x, y)]))

    if return_trace:
        return None, {"nodes_expanded": nodes_expanded, "max_frontier": max_frontier}
    return None