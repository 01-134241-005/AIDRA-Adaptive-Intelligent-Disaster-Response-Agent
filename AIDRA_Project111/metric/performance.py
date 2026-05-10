import matplotlib.pyplot as plt
import numpy as np

def compute_metrics(paths, grid, bfs_paths, algorithm_names=None, algorithm_times=None, algorithm_risks=None):
    total_time = sum(len(p) for p in paths if p)
    victims_saved = len([p for p in paths if p])
    avg_time = total_time / victims_saved if victims_saved else 0
    risk_exposure = 0
    for path in paths:
        if path:
            for x, y in path:
                if grid[x][y] == 'R':
                    risk_exposure += 1
    ratios = []
    for p, bfs_p in zip(paths, bfs_paths):
        if p and bfs_p:
            ratios.append(len(p) / len(bfs_p))
    optimality_ratio = sum(ratios)/len(ratios) if ratios else 0
    resource_utilization = victims_saved / 5
    
    # Generate graphs
    if algorithm_names and algorithm_times and algorithm_risks:
        # Bar chart: path lengths per algorithm (first victim as example)
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.bar(algorithm_names, algorithm_times, color='skyblue')
        plt.title('Rescue Time per Algorithm (Victim 1)')
        plt.ylabel('Time steps')
        plt.xticks(rotation=45)
        
        plt.subplot(1,2,2)
        plt.bar(algorithm_names, algorithm_risks, color='salmon')
        plt.title('Risk Exposure per Algorithm (Victim 1)')
        plt.ylabel('Risk count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('algorithm_comparison.png')
        plt.close()
        
        # Scatter plot: time vs risk
        plt.figure(figsize=(6,6))
        plt.scatter(algorithm_times, algorithm_risks, c='green', s=100)
        for i, name in enumerate(algorithm_names):
            plt.annotate(name, (algorithm_times[i], algorithm_risks[i]))
        plt.xlabel('Rescue Time')
        plt.ylabel('Risk Exposure')
        plt.title('Time-Risk Trade-off')
        plt.grid(True)
        plt.savefig('time_risk_tradeoff.png')
        plt.close()
    
    # Optimality ratio plot
    if ratios:
        plt.figure(figsize=(8,4))
        plt.plot(range(1, len(ratios)+1), ratios, marker='o', linestyle='-', color='purple')
        plt.axhline(y=1, color='r', linestyle='--', label='Optimal (BFS baseline)')
        plt.xlabel('Victim index')
        plt.ylabel('Optimality Ratio (path length / BFS length)')
        plt.title('Path Optimality Ratio per Victim')
        plt.legend()
        plt.grid(True)
        plt.savefig('optimality_ratio.png')
        plt.close()
    
    return {
        "victims_saved": victims_saved,
        "avg_time": avg_time,
        "risk_exposure": risk_exposure,
        "optimality_ratio": optimality_ratio,
        "resource_utilization": resource_utilization
    }