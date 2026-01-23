import json
import uuid
import time
import random
from pathlib import Path
from typing import Dict, Optional
import torch
import numpy as np
import rdflib

from experiments.configs.ablation_configs import (
    ExperimentConfig, 
    EmbeddingType,
    PolicyType,
    TopologyType
)
from experiments.models.embeddings_ablation import (
    EmbeddingTrainerAblation,
    PayloadBuilderAblation
)
from experiments.runners.topology_ablation import (
    TopologyAnalyzer,
    TopologyWeighter,
    TopologyAwarePyGBuilder
)
from core.statistics import KGStatistics
from core.extraction import AdaptivePolicy, SymbolicExtractor
from core.summarization import FLUIDSummarizer
from models.gnn import SummaryGCN, LabelBuilder
from evaluation.benchmark import GNNEvaluator


class ExperimentLogger:
    """JSON Lines logger for experiments."""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.jsonl_path = self.output_dir / "experiments.jsonl"
        self.csv_path = self.output_dir / "experiments.csv"
        
        # Initialize CSV with headers if doesn't exist
        if not self.csv_path.exists():
            self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV file with headers."""
        import csv
        
        headers = [
            'experiment_id', 'timestamp', 'dataset_name',
            'policy_type', 'embedding_type', 'topology_type',
            'num_original_triples', 'num_enriched_triples', 'num_inferred_triples',
            'num_summary_nodes', 'num_valid_labels',
            'baseline_accuracy', 'baseline_f1',
            'enriched_accuracy', 'enriched_f1',
            'accuracy_improvement', 'f1_improvement',
            'runtime_seconds', 'random_seed'
        ]
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def log_experiment(self, config: ExperimentConfig, metrics: Dict):
        """Log experiment to both JSONL and CSV."""
        record = {
            "experiment_id": config.experiment_id,
            "timestamp": time.time(),
            "config": config.to_dict(),
            "metrics": metrics
        }
        
        # Write to JSONL
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        
        # Write to CSV
        self._append_to_csv(record)
    
    def _append_to_csv(self, record: Dict):
        """Append record to CSV."""
        import csv
        
        config = record['config']
        metrics = record['metrics']
        
        row = [
            record['experiment_id'],
            record['timestamp'],
            config['dataset_name'],
            config['policy']['type'],
            config['embedding']['type'],
            config['topology']['type'],
            metrics.get('num_original_triples', ''),
            metrics.get('num_enriched_triples', ''),
            metrics.get('num_inferred_triples', ''),
            metrics.get('num_summary_nodes', ''),
            metrics.get('num_valid_labels', ''),
            metrics.get('baseline_accuracy', ''),
            metrics.get('baseline_f1', ''),
            metrics.get('enriched_accuracy', ''),
            metrics.get('enriched_f1', ''),
            metrics.get('accuracy_improvement', ''),
            metrics.get('f1_improvement', ''),
            metrics.get('runtime_seconds', ''),
            config['random_seed']
        ]
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)


class ExperimentRunner:
    """Main experiment runner."""
    
    def __init__(self, output_dir: str = "results"):
        self.logger = ExperimentLogger(output_dir)
    
    def run(self, config: ExperimentConfig) -> Dict:
        """Run single experiment."""
        print(f"\n{'='*80}")
        print(f"Running: {config.experiment_id}")
        print(f"  Policy: {config.policy_config.policy_type.value}")
        print(f"  Embedding: {config.embedding_config.embedding_type.value}")
        print(f"  Topology: {config.topology_config.topology_type.value}")
        print(f"{'='*80}")
        
        # Set random seed
        self._set_seed(config.random_seed)
        
        start_time = time.time()
        
        try:
            # Load graph
            graph = self._load_graph(config.dataset_path)
            num_original_triples = len(graph)
            
            # Phase 1: Symbolic extraction with policy
            policy, enriched_graph = self._extract_symbolic(graph, config)
            num_inferred_triples = len(enriched_graph) - num_original_triples
            
            # Phase 2: FLUID summarization
            summary_graph, summary_nodes, node_map = self._summarize(enriched_graph)
            
            # Phase 3: Embedding training
            model, entity_to_id, payload = self._train_embeddings(
                enriched_graph, 
                graph,
                summary_nodes,
                config
            )
            
            # Phase 4: Topology analysis (if needed)
            topology_weights = self._analyze_topology(
                graph,
                summary_nodes,
                config
            )
            
            # Phase 5: Evaluation
            metrics = self._evaluate(
                summary_graph,
                summary_nodes,
                graph,
                payload,
                topology_weights,
                config
            )
            
            # Add basic metrics
            metrics.update({
                'num_original_triples': num_original_triples,
                'num_enriched_triples': len(enriched_graph),
                'num_inferred_triples': num_inferred_triples,
                'num_summary_nodes': len(summary_nodes),
                'runtime_seconds': time.time() - start_time,
                'policy_applied': policy
            })
            
            # Log results
            self.logger.log_experiment(config, metrics)
            
            print(f"\n✓ Completed in {metrics['runtime_seconds']:.2f}s")
            if metrics.get('accuracy_improvement') is not None:
                print(f"  Accuracy Δ: {metrics['accuracy_improvement']:.4f}")
                print(f"  F1 Δ: {metrics['f1_improvement']:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f"\n✗ Failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Log failure
            metrics = {
                'error': str(e),
                'runtime_seconds': time.time() - start_time
            }
            self.logger.log_experiment(config, metrics)
            
            return metrics
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def _load_graph(self, path: str) -> rdflib.Graph:
        """Load RDF graph."""
        print(f"Loading graph from {path}...")
        graph = rdflib.Graph()
        graph.parse(path)
        print(f"  Loaded {len(graph)} triples")
        return graph
    
    def _extract_symbolic(
        self,
        graph: rdflib.Graph,
        config: ExperimentConfig
    ) -> tuple:
        """Phase 1: Symbolic extraction."""
        print("Phase 1: Symbolic extraction...")
        
        # Compute statistics
        stats_computer = KGStatistics(graph)
        profile = stats_computer.normalize()
        
        # Determine policy
        if config.policy_config.adaptive:
            # Adaptive policy
            policy_maker = AdaptivePolicy()
            policy, reasoning = policy_maker.determine(profile, stats_computer.compute_all())
        else:
            # Fixed policy from config
            policy = {
                'subClassOf': config.policy_config.subClassOf,
                'subPropertyOf': config.policy_config.subPropertyOf,
                'sameAs': config.policy_config.sameAs,
                'domain_range': config.policy_config.domain_range,
                'rule_mining': config.policy_config.rule_mining,
                'max_depth': 10
            }
        
        # Apply overrides from config
        if config.policy_config.sameAs is not None:
            policy['sameAs'] = config.policy_config.sameAs
        if config.policy_config.domain_range is not None:
            policy['domain_range'] = config.policy_config.domain_range
        
        print(f"  Policy: {policy}")
        
        # Extract
        extractor = SymbolicExtractor(policy)
        enriched_graph = extractor.extract(graph)
        
        print(f"  Extracted {len(enriched_graph) - len(graph)} inferred triples")
        
        return policy, enriched_graph
    
    def _summarize(self, graph: rdflib.Graph) -> tuple:
        """Phase 2: FLUID summarization."""
        print("Phase 2: FLUID summarization...")
        
        summarizer = FLUIDSummarizer()
        summary_graph, summary_nodes, node_map = summarizer.summarize(graph)
        
        print(f"  Created {len(summary_nodes)} super-nodes")
        
        return summary_graph, summary_nodes, node_map
    
    def _train_embeddings(
        self,
        enriched_graph: rdflib.Graph,
        original_graph: rdflib.Graph,
        summary_nodes: Dict,
        config: ExperimentConfig
    ) -> tuple:
        """Phase 3: Embedding training."""
        print(f"Phase 3: Embedding training ({config.embedding_config.embedding_type.value})...")
        
        # Get inferred triples
        inferred_triples = list(set(enriched_graph) - set(original_graph))
        
        if not inferred_triples:
            print("  No inferred triples - skipping embedding training")
            return None, {}, None
        
        # Train embeddings
        trainer = EmbeddingTrainerAblation(
            config.embedding_config.embedding_type,
            {
                'dim': config.embedding_config.dim,
                'epochs': config.embedding_config.epochs,
                'batch_size': config.embedding_config.batch_size,
                'learning_rate': config.embedding_config.learning_rate,
                'margin': config.embedding_config.margin,
                'neg_samples': config.embedding_config.neg_samples
            }
        )
        
        model, entity_to_id = trainer.train(inferred_triples)
        
        # Build payload
        payload = None
        if model is not None:
            builder = PayloadBuilderAblation(config.embedding_config.dim)
            payload = builder.build(
                summary_nodes,
                model,
                entity_to_id,
                config.embedding_config.embedding_type
            )
            print(f"  Trained embeddings on {len(inferred_triples)} triples")
        
        return model, entity_to_id, payload
    
    def _analyze_topology(
        self,
        graph: rdflib.Graph,
        summary_nodes: Dict,
        config: ExperimentConfig
    ) -> Optional[Dict[int, float]]:
        """Phase 4: Topology analysis."""
        if config.topology_config.topology_type == TopologyType.RAW:
            return None
        
        print(f"Phase 4: Topology analysis ({config.topology_config.topology_type.value})...")
        
        # Analyze topology
        analyzer = TopologyAnalyzer(graph)
        topology_metrics = analyzer.analyze()
        
        # Compute weights
        weighter = TopologyWeighter(config.topology_config.topology_type)
        weights = weighter.compute_weights(
            summary_nodes,
            topology_metrics,
            {
                'community_weight': config.topology_config.community_weight,
                'hub_weight': config.topology_config.hub_weight,
                'hierarchy_weight': config.topology_config.hierarchy_weight
            }
        )
        
        print(f"  Computed topology weights (range: {min(weights.values()):.2f} - {max(weights.values()):.2f})")
        
        return weights
    
    def _evaluate(
        self,
        summary_graph: rdflib.Graph,
        summary_nodes: Dict,
        original_graph: rdflib.Graph,
        payload: Optional[Dict],
        topology_weights: Optional[Dict],
        config: ExperimentConfig
    ) -> Dict:
        """Phase 5: Evaluation."""
        print("Phase 5: Evaluation...")
        
        # Build labels
        labels = LabelBuilder.build(summary_nodes, original_graph)
        num_valid_labels = sum(1 for v in labels.values() if v is not None)
        
        print(f"  Labels: {num_valid_labels}/{len(labels)} valid")
        
        metrics = {'num_valid_labels': num_valid_labels}
        
        if num_valid_labels < 3:
            print("  Insufficient labels - skipping GNN evaluation")
            return metrics
        
        try:
            # Build PyG graphs
            builder = TopologyAwarePyGBuilder(config.embedding_config.dim)
            
            # Baseline (no payload, no topology)
            data_baseline = builder.build(summary_graph, None, labels, None)
            
            # Enriched (with payload and topology)
            data_enriched = builder.build(summary_graph, payload, labels, topology_weights)
            
            # Train and evaluate
            evaluator = GNNEvaluator(config)
            
            baseline_acc, baseline_f1 = evaluator.train_and_evaluate(data_baseline)
            enriched_acc, enriched_f1 = evaluator.train_and_evaluate(data_enriched)
            
            metrics.update({
                'baseline_accuracy': baseline_acc,
                'baseline_f1': baseline_f1,
                'enriched_accuracy': enriched_acc,
                'enriched_f1': enriched_f1,
                'accuracy_improvement': enriched_acc - baseline_acc,
                'f1_improvement': enriched_f1 - baseline_f1
            })
            
        except Exception as e:
            print(f"  Evaluation failed: {e}")
            metrics['evaluation_error'] = str(e)
        
        return metrics
    
    def run_batch(self, configs: list) -> list:
        """Run batch of experiments."""
        results = []
        
        print(f"\n{'='*80}")
        print(f"Starting batch of {len(configs)} experiments")
        print(f"{'='*80}\n")
        
        for i, config in enumerate(configs):
            print(f"\n[{i+1}/{len(configs)}]")
            metrics = self.run(config)
            results.append(metrics)
        
        print(f"\n{'='*80}")
        print(f"Batch complete: {len(results)} experiments")
        print(f"{'='*80}\n")
        
        return results
