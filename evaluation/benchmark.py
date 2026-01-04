from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
import rdflib
from torch_geometric.data import Data
from typing import Tuple, Dict
from models.gnn import LabelBuilder
from config.config import GNNConfig, IFLUIDConfig
from utils.logger import IFLUIDLogger
from utils.validators import Validators
from main import IFLUIDPipeline
from models.gnn import PyGGraphBuilder
from models.gnn import SummaryGCN


class GNNEvaluator:
    """Train and evaluate GNN models."""
    
    def __init__(self, config: GNNConfig):
        self.config = config
        self.logger = IFLUIDLogger()
    
    def train_and_evaluate(self, data: Data) -> Tuple[float, float]:
        """Train GNN and return test metrics."""
        num_classes = int(data.y.max()) + 1
        model = SummaryGCN(
            data.x.size(1), 
            self.config.hidden_dim, 
            num_classes
        )
        
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.config.learning_rate
        )
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Split data
        indices = list(range(data.num_nodes))
        train_idx, test_idx = train_test_split(
            indices, 
            test_size=self.config.test_split,
            random_state=self.config.random_seed
        )
        
        # Training loop
        for epoch in range(self.config.epochs):
            model.train()
            optimizer.zero_grad()
            
            out = model(data)
            loss = loss_fn(out[train_idx], data.y[train_idx])
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 50 == 0:
                self.logger.debug(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            logits = model(data)[test_idx]
            predictions = logits.argmax(dim=1)
            
            accuracy = accuracy_score(data.y[test_idx], predictions)
            f1 = f1_score(data.y[test_idx], predictions, average='macro')
        
        return accuracy, f1


class LUBMBenchmark:
    """LUBM benchmark evaluation."""
    
    def __init__(self, config: IFLUIDConfig):
        self.config = config
        self.logger = IFLUIDLogger()
    
    def run(self, rdf_path: str) -> Dict[str, any]:
        """Run full LUBM experiment."""
        self.logger.info(f"Loading LUBM dataset from {rdf_path}")
        
        # Validate input
        Validators.validate_file_exists(rdf_path)
        
        # Load graph
        graph = self._load_graph(rdf_path)
        Validators.validate_graph(graph)
        
        # Run pipeline
        pipeline = IFLUIDPipeline(self.config)
        results = pipeline.run(graph)
        
        # Build labels
        labels = LabelBuilder.build(results['summary_nodes'], graph)
        valid_labels = sum(1 for v in labels.values() if v is not None)
        
        self.logger.info(f"Labels: {valid_labels}/{len(labels)} valid")
        
        if valid_labels < 10:
            self.logger.warning("Very few valid labels - results may be unreliable")
        
        # Build PyG graphs
        graph_builder = PyGGraphBuilder(self.config.embedding.dim)
        
        data_baseline = graph_builder.build(
            results['summary_graph'], 
            payload=None, 
            labels=labels
        )
        
        data_enriched = graph_builder.build(
            results['summary_graph'],
            payload=results['payload'],
            labels=labels
        )
        
        self.logger.info(
            f"Graph: {data_baseline.num_nodes} nodes, "
            f"{data_baseline.num_edges} edges, "
            f"{int(data_baseline.y.max()) + 1} classes"
        )
        
        # Evaluate both models
        evaluator = GNNEvaluator(self.config.gnn)
        
        self.logger.info("Training FLUID baseline...")
        acc_baseline, f1_baseline = evaluator.train_and_evaluate(data_baseline)
        
        self.logger.info("Training inference-aware model...")
        acc_enriched, f1_enriched = evaluator.train_and_evaluate(data_enriched)
        
        return {
            'baseline_accuracy': acc_baseline,
            'baseline_f1': f1_baseline,
            'enriched_accuracy': acc_enriched,
            'enriched_f1': f1_enriched,
            'policy': results['policy'],
            'num_summary_nodes': len(results['summary_nodes']),
            'num_valid_labels': valid_labels,
            'num_inferred_triples': len(results['enriched_graph']) - len(graph)
        }
    
    def _load_graph(self, path: str) -> rdflib.Graph:
        """Load RDF graph from file."""
        graph = rdflib.Graph()
        graph.parse(path)
        return graph
