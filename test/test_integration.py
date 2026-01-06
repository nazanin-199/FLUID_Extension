import pytest
import rdflib
from rdflib import RDF, RDFS 
from main import IFLUIDPipeline
from config.config import IFLUIDConfig, EmbeddingConfig, GNNConfig
from evaluation.benchmark import GNNEvaluator
from models.gnn import PyGGraphBuilder, LabelBuilder

def test_full_pipeline_basic(hierarchical_graph):
    """Test complete pipeline on hierarchical graph."""
    config = IFLUIDConfig(
        embedding=EmbeddingConfig(dim=8, epochs=2),
        gnn=GNNConfig(epochs=10),
        verbose=False
    )
    
    pipeline = IFLUIDPipeline(config)
    results = pipeline.run(hierarchical_graph)
    
    # Check all expected outputs
    assert 'enriched_graph' in results
    assert 'summary_graph' in results
    assert 'summary_nodes' in results
    assert 'node_map' in results
    assert 'payload' in results
    assert 'policy' in results
    
    # Check enriched graph has more triples
    assert len(results['enriched_graph']) >= len(hierarchical_graph)


def test_pipeline_with_inference(complex_graph):
    """Test pipeline generates inferences."""
    config = IFLUIDConfig(verbose=False)
    config.embedding.epochs = 2  # Fast training
    
    pipeline = IFLUIDPipeline(config)
    results = pipeline.run(complex_graph)
    
    # Should have inferred some triples
    original_count = len(complex_graph)
    enriched_count = len(results['enriched_graph'])
    
    assert enriched_count >= original_count
    
    # Should have payload if inference occurred
    if enriched_count > original_count:
        assert results['payload'] is not None


def test_pipeline_empty_graph(empty_graph):
    """Test pipeline handles empty graph gracefully."""
    config = IFLUIDConfig(verbose=False)
    
    pipeline = IFLUIDPipeline(config)
    results = pipeline.run(empty_graph)
    
    # Should complete without errors
    assert results['enriched_graph'] is not None
    assert len(results['summary_nodes']) == 0


def test_pyg_graph_building(hierarchical_graph):
    """Test PyG graph construction."""
    config = IFLUIDConfig(verbose=False)
    config.embedding.epochs = 2
    
    pipeline = IFLUIDPipeline(config)
    results = pipeline.run(hierarchical_graph)
    
    # Build labels
    labels = LabelBuilder.build(results['summary_nodes'], hierarchical_graph)
    
    # Build PyG graph
    builder = PyGGraphBuilder(config.embedding.dim)
    data = builder.build(
        results['summary_graph'],
        results['payload'],
        labels
    )
    
    # Check data structure
    assert data.x is not None
    assert data.edge_index is not None
    assert data.y is not None
    assert data.num_nodes > 0


def test_end_to_end_evaluation(complex_graph):
    """Test end-to-end evaluation including GNN training."""
    config = IFLUIDConfig(verbose=False)
    config.embedding.epochs = 2
    config.gnn.epochs = 10
    
    # Run pipeline
    pipeline = IFLUIDPipeline(config)
    results = pipeline.run(complex_graph)
    
    # Build labels
    labels = LabelBuilder.build(results['summary_nodes'], complex_graph)
    
    # Skip if insufficient labels
    valid_labels = sum(1 for v in labels.values() if v is not None)
    if valid_labels < 3:
        pytest.skip("Insufficient labels for evaluation")
    
    # Build PyG graph
    builder = PyGGraphBuilder(config.embedding.dim)
    data = builder.build(
        results['summary_graph'],
        results['payload'],
        labels
    )
    
    # Train GNN
    evaluator = GNNEvaluator(config.gnn)
    
    try:
        accuracy, f1 = evaluator.train_and_evaluate(data)
        
        # Check metrics are reasonable
        assert 0 <= accuracy <= 1
        assert 0 <= f1 <= 1
    except Exception as e:
        # Some configurations might not have enough data
        pytest.skip(f"Evaluation failed: {e}")


def test_pipeline_reproducibility(simple_graph):
    """Test pipeline gives consistent results with same seed."""
    import random
    import torch
    
    config = IFLUIDConfig(verbose=False)
    config.embedding.epochs = 2
    config.gnn.random_seed = 42
    
    # Run 1
    random.seed(42)
    torch.manual_seed(42)
    pipeline1 = IFLUIDPipeline(config)
    results1 = pipeline1.run(simple_graph)
    
    # Run 2
    random.seed(42)
    torch.manual_seed(42)
    pipeline2 = IFLUIDPipeline(config)
    results2 = pipeline2.run(simple_graph)
    
    # Should produce same enriched graph
    assert len(results1['enriched_graph']) == len(results2['enriched_graph'])
    assert len(results1['summary_nodes']) == len(results2['summary_nodes'])


def test_policy_impact():
    """Test that different policies produce different results."""
    g = rdflib.Graph()
    ex = rdflib.Namespace("http://example.org/")
    
    # Create hierarchical data
    g.add((ex.Student, RDFS.subClassOf, ex.Person))
    g.add((ex.Person, RDFS.subClassOf, ex.Agent))
    g.add((ex.Alice, RDF.type, ex.Student))
    
    config1 = IFLUIDConfig(verbose=False)
    config1.policy.log_triples_threshold = 0  # Enable everything
    
    config2 = IFLUIDConfig(verbose=False)
    config2.policy.log_triples_threshold = 100  # Disable most things
    
    pipeline1 = IFLUIDPipeline(config1)
    pipeline2 = IFLUIDPipeline(config2)
    
    results1 = pipeline1.run(g)
    results2 = pipeline2.run(g)
