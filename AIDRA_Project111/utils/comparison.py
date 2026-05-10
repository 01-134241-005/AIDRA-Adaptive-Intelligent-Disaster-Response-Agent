# -*- coding: utf-8 -*-
"""
Created on Fri May  8 13:19:16 2026

@author: Administrator
"""

def analyze_path(path, grid):
    if not path:
        return {"length": None, "risk": None, "time": None}
    length = len(path)
    risk = sum(1 for x,y in path if grid[x][y]=='R')
    time = 0
    for x,y in path:
        time += 3 if grid[x][y]=='R' else 1
    return {"length": length, "risk": risk, "time": time}

def safe(v):
    return v if v is not None else "N/A"

def print_comparison(victim, bfs, astar, greedy, hill, dfs, grid):
    bfs_d = analyze_path(bfs, grid)
    astar_d = analyze_path(astar, grid)
    greedy_d = analyze_path(greedy, grid)
    hill_d = analyze_path(hill, grid)
    dfs_d = analyze_path(dfs, grid)
    print("\n================= DECISION TABLE =================")
    print(f"Victim: {victim}")
    print("--------------------------------------------------")
    print(f"{'Algo':<10}{'Length':<10}{'Time':<10}{'Risk':<10}")
    print("--------------------------------------------------")
    print(f"{'BFS':<10}{safe(bfs_d['length']):<10}{safe(bfs_d['time']):<10}{safe(bfs_d['risk']):<10}")
    print(f"{'A*':<10}{safe(astar_d['length']):<10}{safe(astar_d['time']):<10}{safe(astar_d['risk']):<10}")
    print(f"{'Greedy':<10}{safe(greedy_d['length']):<10}{safe(greedy_d['time']):<10}{safe(greedy_d['risk']):<10}")
    print(f"{'Hill':<10}{safe(hill_d['length']):<10}{safe(hill_d['time']):<10}{safe(hill_d['risk']):<10}")
    print(f"{'DFS':<10}{safe(dfs_d['length']):<10}{safe(dfs_d['time']):<10}{safe(dfs_d['risk']):<10}")
    print("--------------------------------------------------")
    return bfs_d, astar_d, greedy_d, hill_d, dfs_d
