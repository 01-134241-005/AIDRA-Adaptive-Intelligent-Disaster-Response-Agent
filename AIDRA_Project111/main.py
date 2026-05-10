
# main.py — console entry point
from pipeline import run_simulation

def main():
    results = run_simulation()
    m = results["final_metrics"]

    print("\n=========== FINAL PERFORMANCE METRICS ===========")
    print(f"  Victims saved:        {m['victims_saved']} / 5")
    print(f"  Avg rescue time:      {m['avg_time']:.2f} steps")
    print(f"  Path optimality ratio:{m['optimality_ratio']:.3f}  (1.0 = BFS-optimal)")
    print(f"  Risk exposure:        {m['risk_exposure']} hazard cells traversed")
    print(f"  Resource utilization: {m['resource_utilization']:.1%}")

    ml = results["ml_results"]
    print("\n=========== ML MODEL COMPARISON ===========")
    for model_name, key in [("KNN", "knn"), ("Naive Bayes", "nb")]:
        r = ml[key]
        print(f"\n  {model_name}:")
        print(f"    Accuracy:        {r['accuracy']:.3f}")
        print(f"    Macro Precision: {r['macro_precision']:.3f}")
        print(f"    Macro Recall:    {r['macro_recall']:.3f}")
        print(f"    Macro F1:        {r['macro_f1']:.3f}")
        print(f"    Confusion matrix:")
        classes = ['Critical', 'Moderate', 'Minor']
        print(f"    {'':12s} " + "  ".join(f"{c[:3]:>5}" for c in classes))
        for actual in classes:
            row_vals = "  ".join(f"{r['confusion_matrix'][actual][pred]:>5}" for pred in classes)
            print(f"    {actual[:12]:12s} {row_vals}")
    print(f"\n  Best model: {ml['best_model_name']}")

    print("\nGraphs saved as:")
    print("  algorithm_comparison.png")
    print("  time_risk_tradeoff.png")
    print("  optimality_ratio.png")
    print("\n=== Simulation Complete ===")

if __name__ == "__main__":
    main()