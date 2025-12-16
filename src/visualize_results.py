import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_comparison_plots():
    output_dir = 'output'
    metrics_path = os.path.join(output_dir, 'model_comparison_metrics.csv')
    
    if not os.path.exists(metrics_path):
        print(f"Error: {metrics_path} not found.")
        return

    df = pd.read_csv(metrics_path)
    print("Loaded Metrics:")
    print(df.head())

    # Algorithms to compare
    algorithms = ['LSA', 'LDA', 'NMF', 'BERTopic']
    
    sns.set_style("whitegrid")

    for algo in algorithms:
        print(f"\nProcessing {algo}...")
        
        # 1. Get Baseline
        # Note: In main.py, names were "Baseline-LSA", "Baseline-BERTopic", etc.
        baseline_row = df[df['Model'] == f'Baseline-{algo}']
        
        if baseline_row.empty:
            print(f"  Warning: No baseline found for {algo}")
            continue
            
        baseline_score = baseline_row.iloc[0]['Coherence']
        baseline_label = f"Baseline (n=5)"
        
        # 2. Get Best Tuned
        # Names were "Tuned-LSA-3", "Tuned-BERTopic-15", etc.
        tuned_rows = df[df['Model'].str.contains(f'Tuned-{algo}')]
        
        if tuned_rows.empty:
            print(f"  Warning: No tuned models found for {algo}")
            continue
            
        # Find max coherence
        best_tuned_idx = tuned_rows['Coherence'].idxmax()
        best_tuned_row = tuned_rows.loc[best_tuned_idx]
        best_tuned_score = best_tuned_row['Coherence']
        # Extract N from identifier if possible, or just use full name
        best_tuned_name = best_tuned_row['Model'] # e.g. Tuned-LSA-3
        
        # 3. Plot 1: Baseline
        plt.figure(figsize=(4, 5))
        plt.bar(['Baseline'], [baseline_score], color='#bdc3c7', width=0.5)
        plt.title(f'{algo} Baseline\n(n=5)', fontsize=12)
        plt.ylim(0, 1.0)
        plt.ylabel('Coherence Score')
        plt.text(0, baseline_score, f'{baseline_score:.4f}', ha='center', va='bottom', fontweight='bold')
        save_path_base = os.path.join(output_dir, f'score_baseline_{algo}.png')
        plt.savefig(save_path_base)
        plt.close()
        print(f"  Saved {save_path_base}")

        # 4. Plot 2: Best Tuned
        plt.figure(figsize=(4, 5))
        plt.bar(['Best Tuned'], [best_tuned_score], color='#2ecc71', width=0.5)
        plt.title(f'{algo} Best Config\n({best_tuned_name})', fontsize=12)
        plt.ylim(0, 1.0)
        plt.ylabel('Coherence Score')
        plt.text(0, best_tuned_score, f'{best_tuned_score:.4f}', ha='center', va='bottom', fontweight='bold')
        save_path_tuned = os.path.join(output_dir, f'score_tuned_{algo}.png')
        plt.savefig(save_path_tuned)
        plt.close()
        print(f"  Saved {save_path_tuned}")

if __name__ == "__main__":
    generate_comparison_plots()
