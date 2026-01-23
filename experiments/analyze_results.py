import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from scipy import stats

class ResultsAnalyzer:
    """Analyze experiment results from CSV/JSONL."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.df = self._load_results()
    
    def _load_results(self) -> pd.DataFrame:
        """Load results from CSV."""
        csv_path = self.results_dir / "experiments.csv"
        
        if not csv_path.exists():
            print(f"No results found at {csv_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(csv_path)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        return df
    
    def summary_statistics(self) -> pd.DataFrame:
        """Compute summary statistics."""
        if len(self.df) == 0:
            return pd.DataFrame()
        
        summary = self.df.groupby(['policy_type', 'embedding_type', 'topology_type']).agg({
            'accuracy_improvement': ['mean', 'std', 'count'],
            'f1_improvement': ['mean', 'std'],
            'num_inferred_triples': ['mean', 'std'],
            'runtime_seconds': ['mean', 'std']
        }).round(4)
        
        return summary
    
    # ========================================================================
    # ABLATION A: Policy Analysis
    # ========================================================================
    
    def analyze_policy_ablation(self) -> Dict:
        """Analyze policy ablation results."""
        print("\n" + "="*80)
        print("ABLATION A: POLICY ANALYSIS")
        print("="*80)
        
        # Filter for policy ablation (fixed embedding and topology)
        policy_df = self.df[
            (self.df['embedding_type'] == 'transe') &
            (self.df['topology_type'] == 'raw')
        ].copy()
        
        if len(policy_df) == 0:
            print("No policy ablation data found")
            return {}
        
        # Group by policy
        policy_summary = policy_df.groupby('policy_type').agg({
            'accuracy_improvement': ['mean', 'std', 'count'],
            'f1_improvement': ['mean', 'std'],
            'num_inferred_triples': ['mean', 'std'],
            'runtime_seconds': ['mean', 'std']
        }).round(4)
        
        print("\nPolicy Performance Summary:")
        print(policy_summary)
        
        # Statistical significance tests
        print("\n" + "-"*80)
        print("Statistical Significance (vs full_adaptive):")
        print("-"*80)
        
        baseline = policy_df[policy_df['policy_type'] == 'full_adaptive']['accuracy_improvement'].values
        
        for policy in policy_df['policy_type'].unique():
            if policy == 'full_adaptive':
                continue
            
            variant = policy_df[policy_df['policy_type'] == policy]['accuracy_improvement'].values
            
            if len(baseline) > 1 and len(variant) > 1:
                t_stat, p_value = stats.ttest_ind(baseline, variant)
                sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                
                print(f"{policy:20s}: t={t_stat:6.3f}, p={p_value:.4f} {sig}")
        
        return {'summary': policy_summary}
    
    # ========================================================================
    # ABLATION B: Embedding Analysis
    # ========================================================================
    
    def analyze_embedding_ablation(self) -> Dict:
        """Analyze embedding ablation results."""
        print("\n" + "="*80)
        print("ABLATION B: EMBEDDING ANALYSIS")
        print("="*80)
        
        # Filter for embedding ablation (fixed policy and topology)
        embedding_df = self.df[
            (self.df['policy_type'] == 'full_adaptive') &
            (self.df['topology_type'] == 'raw')
        ].copy()
        
        if len(embedding_df) == 0:
            print("No embedding ablation data found")
            return {}
        
        # Group by embedding
        embedding_summary = embedding_df.groupby('embedding_type').agg({
            'accuracy_improvement': ['mean', 'std', 'count'],
            'f1_improvement': ['mean', 'std'],
            'runtime_seconds': ['mean', 'std']
        }).round(4)
        
        print("\nEmbedding Performance Summary:")
        print(embedding_summary)
        
        # Compare to baseline (none)
        if 'none' in embedding_df['embedding_type'].values:
            print("\n" + "-"*80)
            print("Improvement over FLUID Baseline (no embeddings):")
            print("-"*80)
            
            baseline_none = embedding_df[embedding_df['embedding_type'] == 'none']['accuracy_improvement'].mean()
            
            for emb_type in ['transe', 'distmult', 'complex']:
                if emb_type in embedding_df['embedding_type'].values:
                    emb_mean = embedding_df[embedding_df['embedding_type'] == emb_type]['accuracy_improvement'].mean()
                    improvement = emb_mean - baseline_none
                    print(f"{emb_type:15s}: +{improvement:6.4f}")
        
        return {'summary': embedding_summary}
    
    # ========================================================================
    # ABLATION C: Topology Analysis
    # ========================================================================
    
    def analyze_topology_ablation(self) -> Dict:
        """Analyze topology ablation results."""
        print("\n" + "="*80)
        print("ABLATION C: TOPOLOGY ANALYSIS (Publication-Ready)")
        print("="*80)
        
        # Filter for topology ablation (fixed policy and embedding)
        topology_df = self.df[
            (self.df['policy_type'] == 'full_adaptive') &
            (self.df['embedding_type'] == 'transe')
        ].copy()
        
        if len(topology_df) == 0:
            print("No topology ablation data found")
            return {}
        
        # Group by topology
        topology_summary = topology_df.groupby('topology_type').agg({
            'accuracy_improvement': ['mean', 'std', 'count'],
            'f1_improvement': ['mean', 'std'],
            'runtime_seconds': ['mean', 'std']
        }).round(4)
        
        print("\nTopology-Aware Performance Summary:")
        print(topology_summary)
        
        # Compare to raw baseline
        print("\n" + "-"*80)
        print("Improvement over Raw Graph:")
        print("-"*80)
        
        baseline_raw = topology_df[topology_df['topology_type'] == 'raw']['accuracy_improvement'].mean()
        
        for topo_type in ['community_aware', 'hub_weighted', 'hierarchy_weighted']:
            if topo_type in topology_df['topology_type'].values:
                topo_mean = topology_df[topology_df['topology_type'] == topo_type]['accuracy_improvement'].mean()
                improvement = topo_mean - baseline_raw
                print(f"{topo_type:20s}: +{improvement:6.4f}")
        
        return {'summary': topology_summary}
    
    # ========================================================================
    # Visualizations
    # ========================================================================
    
    def create_visualizations(self, output_dir: str = "visualizations"):
        """Create all visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if len(self.df) == 0:
            print("No data to visualize")
            return
        
        sns.set_style("whitegrid")
        
        # 1. Policy comparison
        self._plot_policy_comparison(output_path / "policy_comparison.png")
        
        # 2. Embedding comparison
        self._plot_embedding_comparison(output_path / "embedding_comparison.png")
        
        # 3. Topology comparison
        self._plot_topology_comparison(output_path / "topology_comparison.png")
        
        # 4. Comprehensive comparison
        self._plot_comprehensive(output_path / "comprehensive_comparison.png")
        
        print(f"\nVisualizations saved to {output_path}")
    
    def _plot_policy_comparison(self, output_path: Path):
        """Plot policy ablation comparison."""
        policy_df = self.df[
            (self.df['embedding_type'] == 'transe') &
            (self.df['topology_type'] == 'raw')
        ]
        
        if len(policy_df) == 0:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy improvement
        policy_order = policy_df.groupby('policy_type')['accuracy_improvement'].mean().sort_values(ascending=False).index
        
        ax = axes[0]
        policy_df_plot = policy_df.groupby('policy_type')['accuracy_improvement'].agg(['mean', 'std']).loc[policy_order]
        ax.bar(range(len(policy_df_plot)), policy_df_plot['mean'], yerr=policy_df_plot['std'], capsize=5)
        ax.set_xticks(range(len(policy_df_plot)))
        ax.set_xticklabels(policy_df_plot.index, rotation=45, ha='right')
        ax.set_ylabel('Accuracy Improvement')
        ax.set_title('Policy Ablation: Accuracy')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax.grid(axis='y', alpha=0.3)
        
        # Inferred triples
        ax = axes[1]
        policy_triples = policy_df.groupby('policy_type')['num_inferred_triples'].mean().loc[policy_order]
        ax.bar(range(len(policy_triples)), policy_triples.values)
        ax.set_xticks(range(len(policy_triples)))
        ax.set_xticklabels(policy_triples.index, rotation=45, ha='right')
        ax.set_ylabel('Number of Inferred Triples')
        ax.set_title('Policy Ablation: Inference Volume')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_embedding_comparison(self, output_path: Path):
        """Plot embedding ablation comparison."""
        embedding_df = self.df[
            (self.df['policy_type'] == 'full_adaptive') &
            (self.df['topology_type'] == 'raw')
        ]
        
        if len(embedding_df) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        embedding_order = ['none', 'random', 'transe', 'distmult', 'complex']
        embedding_order = [e for e in embedding_order if e in embedding_df['embedding_type'].values]
        
        emb_summary = embedding_df.groupby('embedding_type')['accuracy_improvement'].agg(['mean', 'std'])
        emb_summary = emb_summary.loc[embedding_order]
        
        ax.bar(range(len(emb_summary)), emb_summary['mean'], yerr=emb_summary['std'], capsize=5)
        ax.set_xticks(range(len(emb_summary)))
        ax.set_xticklabels(emb_summary.index, rotation=45, ha='right')
        ax.set_ylabel('Accuracy Improvement')
        ax.set_title('Embedding Ablation: Model Comparison')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_topology_comparison(self, output_path: Path):
        """Plot topology ablation comparison."""
        topology_df = self.df[
            (self.df['policy_type'] == 'full_adaptive') &
            (self.df['embedding_type'] == 'transe')
        ]
        
        if len(topology_df) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        topo_order = ['raw', 'community_aware', 'hub_weighted', 'hierarchy_weighted']
        topo_order = [t for t in topo_order if t in topology_df['topology_type'].values]
        
        topo_summary = topology_df.groupby('topology_type')['accuracy_improvement'].agg(['mean', 'std'])
        topo_summary = topo_summary.loc[topo_order]
        
        colors = ['gray', 'skyblue', 'coral', 'lightgreen'][:len(topo_summary)]
        
        ax.bar(range(len(topo_summary)), topo_summary['mean'], yerr=topo_summary['std'], 
               capsize=5, color=colors)
        ax.set_xticks(range(len(topo_summary)))
        ax.set_xticklabels(topo_summary.index, rotation=45, ha='right')
        ax.set_ylabel('Accuracy Improvement')
        ax.set_title('Topology Ablation: Graph Structure Awareness')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_comprehensive(self, output_path: Path):
        """Create comprehensive comparison plot."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Top-left: Policy
        ax = axes[0, 0]
        policy_df = self.df[(self.df['embedding_type'] == 'transe') & (self.df['topology_type'] == 'raw')]
        if len(policy_df) > 0:
            sns.boxplot(data=policy_df, x='policy_type', y='accuracy_improvement', ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_title('A. Policy Ablation')
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        
        # Top-right: Embedding
        ax = axes[0, 1]
        emb_df = self.df[(self.df['policy_type'] == 'full_adaptive') & (self.df['topology_type'] == 'raw')]
        if len(emb_df) > 0:
            sns.boxplot(data=emb_df, x='embedding_type', y='accuracy_improvement', ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_title('B. Embedding Ablation')
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        
        # Bottom-left: Topology
        ax = axes[1, 0]
        topo_df = self.df[(self.df['policy_type'] == 'full_adaptive') & (self.df['embedding_type'] == 'transe')]
        if len(topo_df) > 0:
            sns.boxplot(data=topo_df, x='topology_type', y='accuracy_improvement', ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_title('C. Topology Ablation')
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        
        # Bottom-right: Runtime comparison
        ax = axes[1, 1]
        if len(self.df) > 0:
            runtime_summary = self.df.groupby(['policy_type', 'embedding_type', 'topology_type'])['runtime_seconds'].mean().reset_index()
            top_10 = runtime_summary.nlargest(10, 'runtime_seconds')
            ax.barh(range(len(top_10)), top_10['runtime_seconds'])
            ax.set_yticks(range(len(top_10)))
            labels = [f"{row.policy_type[:8]}/{row.embedding_type[:6]}/{row.topology_type[:8]}" 
                     for _, row in top_10.iterrows()]
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel('Runtime (seconds)')
            ax.set_title('Runtime Comparison (Top 10)')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_latex_table(self, output_path: str = "results/ablation_table.tex"):
        """Export publication-ready LaTeX table."""
        if len(self.df) == 0:
            return
        
        # Create three separate tables
        latex = "% Ablation Study Tables\n\n"
        
        # Table A: Policy
        latex += "\\begin{table}[ht]\n\\centering\n"
        latex += "\\caption{Policy Ablation Results}\n"
        latex += "\\begin{tabular}{lcccc}\n\\hline\n"
        latex += "Policy & Acc. $\\Delta$ & F1 $\\Delta$ & Inferred Triples & Runtime (s) \\\\\n\\hline\n"
        
        policy_df = self.df[(self.df['embedding_type'] == 'transe') & (self.df['topology_type'] == 'raw')]
        for policy in policy_df['policy_type'].unique():
            subset = policy_df[policy_df['policy_type'] == policy]
            latex += f"{policy} & "
            latex += f"{subset['accuracy_improvement'].mean():.4f} $\\pm$ {subset['accuracy_improvement'].std():.4f} & "
            latex += f"{subset['f1_improvement'].mean():.4f} & "
            latex += f"{subset['num_inferred_triples'].mean():.0f} & "
            latex += f"{subset['runtime_seconds'].mean():.1f} \\\\\n"
        
        latex += "\\hline\n\\end{tabular}\n\\end{table}\n\n"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex)
        
        print(f"\nLaTeX table exported to {output_path}")


def main():
    """Main analysis entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze ablation study results")
    parser.add_argument('--results-dir', default='results', help='Results directory')
    parser.add_argument('--output-dir', default='visualizations', help='Output directory for plots')
    parser.add_argument('--latex', action='store_true', help='Export LaTeX tables')
    
    args = parser.parse_args()
    
    # Load and analyze
    analyzer = ResultsAnalyzer(args.results_dir)
    
    print(f"\nLoaded {len(analyzer.df)} experiments")
    
    # Run analyses
    analyzer.analyze_policy_ablation()
    analyzer.analyze_embedding_ablation()
    analyzer.analyze_topology_ablation()
    
    # Create visualizations
    analyzer.create_visualizations(args.output_dir)
    
    # Export LaTeX if requested
    if args.latex:
        analyzer.export_latex_table()
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
