import argparse
from pathlib import Path
from experiments.configs.ablation_configs import AblationConfigGenerator
from experiments.runners.experiment_runner import ExperimentRunner


def run_policy_ablation(dataset_path: str, dataset_name: str, output_dir: str):
    """Run Policy Ablation (A)."""
    print("\n" + "="*80)
    print("STARTING POLICY ABLATION (A)")
    print("="*80)
    
    # Generate configs
    configs = AblationConfigGenerator.generate_policy_ablation(
        dataset_name=dataset_name,
        dataset_path=dataset_path
    )
    
    print(f"Generated {len(configs)} configurations")
    print("Policy variants: full_adaptive, no_sameas, no_domain_range, taxonomy_only, fixed_policy, random_policy")
    print(f"Runs per variant: 3")
    
    # Run experiments
    runner = ExperimentRunner(output_dir=output_dir)
    results = runner.run_batch(configs)
    
    print(f"\nPolicy ablation complete: {len(results)} experiments")
    
    return results


def run_embedding_ablation(dataset_path: str, dataset_name: str, output_dir: str):
    """Run Embedding Ablation (B)."""
    print("\n" + "="*80)
    print("STARTING EMBEDDING ABLATION (B)")
    print("="*80)
    
    # Generate configs
    configs = AblationConfigGenerator.generate_embedding_ablation(
        dataset_name=dataset_name,
        dataset_path=dataset_path
    )
    
    print(f"Generated {len(configs)} configurations")
    print("Embedding variants: transe, distmult, complex, random, none")
    print(f"Runs per variant: 3")
    
    # Run experiments
    runner = ExperimentRunner(output_dir=output_dir)
    results = runner.run_batch(configs)
    
    print(f"\nEmbedding ablation complete: {len(results)} experiments")
    
    return results


def run_topology_ablation(dataset_path: str, dataset_name: str, output_dir: str):
    """Run Topology Ablation (C)."""
    print("\n" + "="*80)
    print("STARTING TOPOLOGY ABLATION (C) - Publication-Ready")
    print("="*80)
    
    # Generate configs
    configs = AblationConfigGenerator.generate_topology_ablation(
        dataset_name=dataset_name,
        dataset_path=dataset_path
    )
    
    print(f"Generated {len(configs)} configurations")
    print("Topology variants: raw, community_aware, hub_weighted, hierarchy_weighted")
    print(f"Runs per variant: 3")
    
    # Run experiments
    runner = ExperimentRunner(output_dir=output_dir)
    results = runner.run_batch(configs)
    
    print(f"\nTopology ablation complete: {len(results)} experiments")
    
    return results


def run_all_ablations(dataset_path: str, dataset_name: str, output_dir: str):
    """Run all three ablation studies."""
    print("\n" + "="*80)
    print("RUNNING COMPLETE ABLATION STUDY")
    print("="*80)
    print(f"Dataset: {dataset_name}")
    print(f"Path: {dataset_path}")
    print(f"Output: {output_dir}")
    
    all_results = []
    
    # A. Policy Ablation
    results_a = run_policy_ablation(dataset_path, dataset_name, output_dir)
    all_results.extend(results_a)
    
    # B. Embedding Ablation
    results_b = run_embedding_ablation(dataset_path, dataset_name, output_dir)
    all_results.extend(results_b)
    
    # C. Topology Ablation
    results_c = run_topology_ablation(dataset_path, dataset_name, output_dir)
    all_results.extend(results_c)
    
    print("\n" + "="*80)
    print(f"COMPLETE ABLATION STUDY FINISHED")
    print(f"Total experiments: {len(all_results)}")
    print(f"  - Policy ablation: {len(results_a)}")
    print(f"  - Embedding ablation: {len(results_b)}")
    print(f"  - Topology ablation: {len(results_c)}")
    print("="*80)
    
    return all_results


def run_custom_config(config_path: str, output_dir: str):
    """Run experiments from custom config file."""
    configs = AblationConfigGenerator.load_configs(config_path)
    
    print(f"\nLoaded {len(configs)} configurations from {config_path}")
    
    runner = ExperimentRunner(output_dir=output_dir)
    results = runner.run_batch(configs)
    
    return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="IFLUID Ablation Study Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all three ablations on a dataset
  python run_ablations.py --mode all --dataset data/lubm.rdf --name LUBM
  
  # Run only policy ablation
  python run_ablations.py --mode policy --dataset data/lubm.rdf --name LUBM
  
  # Run only embedding ablation
  python run_ablations.py --mode embedding --dataset data/lubm.rdf --name LUBM
  
  # Run only topology ablation (publication-ready)
  python run_ablations.py --mode topology --dataset data/lubm.rdf --name LUBM
  
  # Run from custom config file
  python run_ablations.py --mode custom --config experiments/configs/my_configs.json
  
  # Analyze results
  python experiments/analyze_results.py --results-dir results --output-dir visualizations

Ablation Studies:
  A. Policy Ablation (6 variants × 3 runs = 18 experiments)
     - full_adaptive: Adaptive policy based on graph statistics
     - no_sameas: Disabling entity consolidation
     - no_domain_range: Only taxonomic inference
     - taxonomy_only: SubClassOf and SubPropertyOf only
     - fixed_policy: All rules enabled without adaptivity
     - random_policy: Random rule assignment (sanity check)
  
  B. Embedding Ablation (5 variants × 3 runs = 15 experiments)
     - transe: Translating embeddings (current baseline)
     - distmult: Bilinear model (symmetric relations)
     - complex: Complex embeddings (asymmetric relations)
     - random: Random embeddings (sanity check)
     - none: No embeddings (FLUID baseline)
  
  C. Topology Ablation (4 variants × 3 runs = 12 experiments)
     - raw: No topology-aware weighting
     - community_aware: Emphasize modular structure
     - hub_weighted: Emphasize scale-free property
     - hierarchy_weighted: Emphasize deep hierarchical structure

Total: 45 experiments for complete ablation study
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['all', 'policy', 'embedding', 'topology', 'custom'],
        required=True,
        help='Which ablation study to run'
    )
    
    parser.add_argument(
        '--dataset',
        help='Path to RDF dataset (required for all/policy/embedding/topology)'
    )
    
    parser.add_argument(
        '--name',
        help='Dataset name (required for all/policy/embedding/topology)'
    )
    
    parser.add_argument(
        '--config',
        help='Path to custom config JSON file (required for custom mode)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='results',
        help='Output directory for results (default: results/)'
    )
    
    parser.add_argument(
        '--save-configs',
        help='Save generated configs to file (optional)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode in ['all', 'policy', 'embedding', 'topology']:
        if not args.dataset or not args.name:
            parser.error(f"--dataset and --name required for {args.mode} mode")
        
        # Check if dataset exists
        if not Path(args.dataset).exists():
            parser.error(f"Dataset not found: {args.dataset}")
    
    if args.mode == 'custom':
        if not args.config:
            parser.error("--config required for custom mode")
        
        if not Path(args.config).exists():
            parser.error(f"Config file not found: {args.config}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    if args.mode == 'all':
        results = run_all_ablations(args.dataset, args.name, args.output_dir)
    
    elif args.mode == 'policy':
        results = run_policy_ablation(args.dataset, args.name, args.output_dir)
    
    elif args.mode == 'embedding':
        results = run_embedding_ablation(args.dataset, args.name, args.output_dir)
    
    elif args.mode == 'topology':
        results = run_topology_ablation(args.dataset, args.name, args.output_dir)
    
    elif args.mode == 'custom':
        results = run_custom_config(args.config, args.output_dir)
    
    # Save configs if requested
    if args.save_configs and args.mode != 'custom':
        if args.mode == 'all':
            configs = (
                AblationConfigGenerator.generate_policy_ablation(args.name, args.dataset) +
                AblationConfigGenerator.generate_embedding_ablation(args.name, args.dataset) +
                AblationConfigGenerator.generate_topology_ablation(args.name, args.dataset)
            )
        elif args.mode == 'policy':
            configs = AblationConfigGenerator.generate_policy_ablation(args.name, args.dataset)
        elif args.mode == 'embedding':
            configs = AblationConfigGenerator.generate_embedding_ablation(args.name, args.dataset)
        elif args.mode == 'topology':
            configs = AblationConfigGenerator.generate_topology_ablation(args.name, args.dataset)
        
        AblationConfigGenerator.save_configs(configs, args.save_configs)
        print(f"\nSaved {len(configs)} configs to {args.save_configs}")
    
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE")
    print(f"Results saved to: {args.output_dir}/")
    print(f"  - experiments.jsonl (detailed results)")
    print(f"  - experiments.csv (tabular format)")
    print("\nTo analyze results, run:")
    print(f"  python experiments/analyze_results.py --results-dir {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
