"""
Robustness and consensus analysis utilities for interpretability results.

Computes cross-architecture and cross-method comparisons to identify
robust SNP importance rankings for publication.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau
from matplotlib_venn import venn2, venn3
import itertools

logger = logging.getLogger(__name__)


class RobustnessAnalyzer:
    """Analyze robustness of SNP importance across architectures and methods."""
    
    def __init__(self, output_base_dir: str):
        """Initialize analyzer.
        
        Args:
            output_base_dir: Base directory containing analysis results
        """
        self.output_base_dir = Path(output_base_dir)
        self.results = {
            'consensus_snps': {},
            'rank_correlations': {},
            'top_k_overlaps': {},
            'cross_method_agreement': {},
        }
    
    def load_rankings(self, checkpoint_dir: str, dataset_name: str, method: str) -> Optional[pd.DataFrame]:
        """Load SNP rankings for a specific checkpoint, dataset, and method.
        
        Args:
            checkpoint_dir: Checkpoint directory name
            dataset_name: Dataset name
            method: Method name ('shap', 'ig', 'lime')
            
        Returns:
            DataFrame with SNP rankings or None if not found
        """
        ranking_file = self.output_base_dir / checkpoint_dir / dataset_name / method / f'top_{method}_snps.csv'
        
        if not ranking_file.exists():
            logger.warning(f"Ranking file not found: {ranking_file}")
            return None
        
        return pd.read_csv(ranking_file)
    
    def compute_consensus_snps(
        self,
        checkpoint_dirs: List[str],
        dataset_name: str,
        methods: List[str] = None,
        top_k: int = 50,
        min_agreement_ratio: float = 0.5,
    ) -> pd.DataFrame:
        """Compute consensus SNPs across architectures and methods.
        
        Args:
            checkpoint_dirs: List of checkpoint directories
            dataset_name: Dataset name
            methods: Methods to include (default: all available)
            top_k: Number of top SNPs per model/method to consider
            min_agreement_ratio: Minimum fraction of models where SNP appears in top-k
            
        Returns:
            DataFrame with consensus SNP scores and agreement rates
        """
        logger.info(f"Computing consensus SNPs for {dataset_name}...")
        
        if methods is None:
            methods = ['shap', 'ig', 'lime']
        
        # Collect all top SNPs across models and methods
        snp_appearance_count = {}
        snp_rank_sums = {}
        snp_max_ranks = {}
        
        total_combinations = 0
        
        for checkpoint_dir, method in itertools.product(checkpoint_dirs, methods):
            ranking_df = self.load_rankings(checkpoint_dir, dataset_name, method)
            
            if ranking_df is None:
                continue
            
            total_combinations += 1
            
            # Get top-k SNPs
            top_snps = ranking_df.head(top_k)
            
            for _, row in top_snps.iterrows():
                snp_id = row['SNP_ID']
                rank = row['Rank']
                
                if snp_id not in snp_appearance_count:
                    snp_appearance_count[snp_id] = 0
                    snp_rank_sums[snp_id] = 0
                    snp_max_ranks[snp_id] = 0
                
                snp_appearance_count[snp_id] += 1
                snp_rank_sums[snp_id] += rank
                snp_max_ranks[snp_id] = max(snp_max_ranks[snp_id], rank)
        
        if total_combinations == 0:
            logger.warning(f"No ranking data found for {dataset_name}")
            return pd.DataFrame()
        
        # Create consensus DataFrame
        consensus_data = []
        for snp_id, count in snp_appearance_count.items():
            agreement_ratio = count / total_combinations
            
            if agreement_ratio >= min_agreement_ratio:
                consensus_data.append({
                    'SNP_ID': snp_id,
                    'Appearances': count,
                    'Total_Combinations': total_combinations,
                    'Agreement_Ratio': agreement_ratio,
                    'Mean_Rank': snp_rank_sums[snp_id] / count,
                    'Worst_Rank': snp_max_ranks[snp_id],
                })
        
        consensus_df = pd.DataFrame(consensus_data)
        consensus_df = consensus_df.sort_values('Agreement_Ratio', ascending=False)
        consensus_df['Consensus_Rank'] = range(1, len(consensus_df) + 1)
        
        logger.info(f"✓ Found {len(consensus_df)} consensus SNPs (agreement ≥ {min_agreement_ratio})")
        
        return consensus_df
    
    def compute_rank_correlations(
        self,
        checkpoint_dirs: List[str],
        dataset_name: str,
        methods: List[str] = None,
    ) -> pd.DataFrame:
        """Compute rank correlations between methods within same architecture.
        
        Args:
            checkpoint_dirs: List of checkpoint directories
            dataset_name: Dataset name
            methods: Methods to compare (default: ['shap', 'ig', 'lime'])
            
        Returns:
            DataFrame with correlation results
        """
        logger.info(f"Computing rank correlations for {dataset_name}...")
        
        if methods is None:
            methods = ['shap', 'ig', 'lime']
        
        correlations = []
        
        # For each checkpoint, compute correlations between methods
        for checkpoint_dir in checkpoint_dirs:
            # Load rankings for each method
            rankings = {}
            for method in methods:
                ranking_df = self.load_rankings(checkpoint_dir, dataset_name, method)
                if ranking_df is not None:
                    rankings[method] = ranking_df
            
            # Compute pairwise correlations
            method_pairs = list(itertools.combinations(sorted(rankings.keys()), 2))
            
            for method1, method2 in method_pairs:
                df1 = rankings[method1].set_index('SNP_ID')['Rank']
                df2 = rankings[method2].set_index('SNP_ID')['Rank']
                
                # Find common SNPs
                common_snps = df1.index.intersection(df2.index)
                
                if len(common_snps) < 3:  # Need at least 3 points for correlation
                    continue
                
                ranks1 = df1[common_snps].values
                ranks2 = df2[common_snps].values
                
                # Compute correlations
                spearman_corr, spearman_p = spearmanr(ranks1, ranks2)
                kendall_corr, kendall_p = kendalltau(ranks1, ranks2)
                
                correlations.append({
                    'Checkpoint': checkpoint_dir,
                    'Method1': method1,
                    'Method2': method2,
                    'Spearman_Correlation': spearman_corr,
                    'Spearman_Pvalue': spearman_p,
                    'Kendall_Correlation': kendall_corr,
                    'Kendall_Pvalue': kendall_p,
                    'Common_SNPs': len(common_snps),
                })
        
        corr_df = pd.DataFrame(correlations)
        logger.info(f"✓ Computed {len(corr_df)} correlation pairs")
        
        return corr_df
    
    def compute_top_k_overlap(
        self,
        checkpoint_dirs: List[str],
        dataset_name: str,
        methods: List[str] = None,
        top_k_values: List[int] = None,
    ) -> Dict[int, Dict[str, float]]:
        """Compute top-K overlap (Jaccard index) between methods.
        
        Args:
            checkpoint_dirs: List of checkpoint directories
            dataset_name: Dataset name
            methods: Methods to compare (default: ['shap', 'ig', 'lime'])
            top_k_values: K values to compute (default: [10, 20, 50])
            
        Returns:
            Dictionary mapping K to overlap results
        """
        if methods is None:
            methods = ['shap', 'ig', 'lime']
        
        if top_k_values is None:
            top_k_values = [10, 20, 50]
        
        logger.info(f"Computing top-K overlaps for {dataset_name}...")
        
        results = {}
        
        # Aggregate SNP sets across architectures
        method_snp_sets = {method: {} for method in methods}
        
        for checkpoint_dir in checkpoint_dirs:
            for method in methods:
                ranking_df = self.load_rankings(checkpoint_dir, dataset_name, method)
                if ranking_df is not None:
                    if method not in method_snp_sets:
                        method_snp_sets[method] = {}
                    
                    # Store top-K sets for this checkpoint
                    method_snp_sets[method][checkpoint_dir] = ranking_df
        
        # Compute overlaps for each K
        for k in top_k_values:
            overlaps = {}
            
            # Get top-K SNPs per method (aggregated across checkpoints)
            top_k_sets = {}
            for method in methods:
                all_snps = set()
                
                for checkpoint_dir, ranking_df in method_snp_sets[method].items():
                    all_snps.update(ranking_df.head(k)['SNP_ID'].values)
                
                top_k_sets[method] = all_snps
            
            # Compute pairwise overlaps
            method_pairs = list(itertools.combinations(sorted(top_k_sets.keys()), 2))
            
            for method1, method2 in method_pairs:
                set1 = top_k_sets[method1]
                set2 = top_k_sets[method2]
                
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                jaccard = intersection / union if union > 0 else 0.0
                
                key = f"{method1}-{method2}"
                overlaps[key] = {
                    'intersection': intersection,
                    'union': union,
                    'jaccard_index': jaccard,
                }
            
            results[k] = overlaps
        
        logger.info(f"✓ Computed top-K overlaps for K={top_k_values}")
        
        return results
    
    def generate_consensus_report(
        self,
        checkpoint_dirs: List[str],
        dataset_name: str,
        output_dir: str,
        top_k: int = 50,
        min_agreement_ratio: float = 0.5,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate comprehensive robustness report.
        
        Args:
            checkpoint_dirs: List of checkpoint directories
            dataset_name: Dataset name
            output_dir: Output directory for results
            top_k: Number of top SNPs to consider
            min_agreement_ratio: Minimum agreement ratio for consensus
            
        Returns:
            Tuple of (consensus_df, correlations_df)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Compute consensus SNPs
        consensus_df = self.compute_consensus_snps(
            checkpoint_dirs,
            dataset_name,
            top_k=top_k,
            min_agreement_ratio=min_agreement_ratio,
        )
        
        # Save consensus SNPs
        consensus_csv = output_path / f'consensus_snps_{dataset_name}.csv'
        consensus_df.to_csv(consensus_csv, index=False)
        logger.info(f"Consensus SNPs saved to: {consensus_csv}")
        
        # Compute rank correlations
        corr_df = self.compute_rank_correlations(checkpoint_dirs, dataset_name)
        
        # Save correlations
        corr_csv = output_path / f'rank_correlations_{dataset_name}.csv'
        corr_df.to_csv(corr_csv, index=False)
        logger.info(f"Rank correlations saved to: {corr_csv}")
        
        # Compute top-K overlaps
        overlaps = self.compute_top_k_overlap(checkpoint_dirs, dataset_name)
        
        # Save overlaps as JSON
        overlaps_json = output_path / f'top_k_overlaps_{dataset_name}.json'
        with open(overlaps_json, 'w') as f:
            json.dump(overlaps, f, indent=2)
        logger.info(f"Top-K overlaps saved to: {overlaps_json}")
        
        return consensus_df, corr_df


def create_consensus_visualization(
    consensus_df: pd.DataFrame,
    output_path: str,
    top_k: int = 30,
):
    """Create visualization of consensus SNPs.
    
    Args:
        consensus_df: Consensus SNP DataFrame
        output_path: Path to save figure
        top_k: Number of top SNPs to display
    """
    logger.info(f"Creating consensus SNP visualization...")
    
    top_consensus = consensus_df.head(top_k)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bar plot
    colors = plt.cm.viridis(top_consensus['Agreement_Ratio'] / top_consensus['Agreement_Ratio'].max())
    bars = ax.barh(range(len(top_consensus)), top_consensus['Agreement_Ratio'], color=colors)
    
    # Customize
    ax.set_yticks(range(len(top_consensus)))
    ax.set_yticklabels(top_consensus['SNP_ID'], fontsize=10)
    ax.set_xlabel('Agreement Ratio (fraction of models)', fontsize=12, fontweight='bold')
    ax.set_ylabel('SNP Identifier', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_k} Consensus SNPs\n(Robustness Across Architectures)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, top_consensus['Agreement_Ratio'])):
        ax.text(value + 0.01, i, f'{value:.2%}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Consensus visualization saved to: {output_path}")
    plt.close()


def create_correlation_heatmap(
    corr_df: pd.DataFrame,
    output_path: str,
):
    """Create heatmap of rank correlations.
    
    Args:
        corr_df: Correlations DataFrame
        output_path: Path to save figure
    """
    logger.info(f"Creating correlation heatmap...")
    
    # Pivot table for heatmap
    if len(corr_df) == 0:
        logger.warning("No correlations to visualize")
        return
    
    # Create method pair labels
    corr_df['Method_Pair'] = corr_df['Method1'] + ' vs ' + corr_df['Method2']
    
    # Group by checkpoint
    checkpoints = corr_df['Checkpoint'].unique()
    
    if len(checkpoints) <= 1:
        logger.info(f"Only {len(checkpoints)} checkpoint(s), skipping heatmap")
        return
    
    # Create heatmap of Spearman correlations
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pivot_data = corr_df.pivot_table(
        values='Spearman_Correlation',
        index='Method_Pair',
        columns='Checkpoint',
        aggfunc='mean'
    )
    
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={'label': 'Spearman Correlation'}
    )
    
    ax.set_title('Rank Correlations Between Methods\n(Spearman ρ)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Method Pair', fontsize=12, fontweight='bold')
    ax.set_xlabel('Checkpoint', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Correlation heatmap saved to: {output_path}")
    plt.close()
