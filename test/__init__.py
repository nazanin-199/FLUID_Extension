import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Test utilities
def assert_graph_valid(graph):
    """Helper to validate RDF graph."""
    import rdflib
    assert isinstance(graph, rdflib.Graph)
    assert len(graph) >= 0


def assert_policy_valid(policy):
    """Helper to validate policy structure."""
    required_keys = ['subClassOf', 'subPropertyOf', 'sameAs', 'domain_range', 'max_depth']
    for key in required_keys:
        assert key in policy, f"Missing policy key: {key}"


def assert_statistics_valid(stats):
    """Helper to validate statistics dictionary."""
    required_keys = [
        'num_triples',
        'num_classes', 
        'num_instances',
        'num_properties',
    ]
    for key in required_keys:
        assert key in stats, f"Missing statistic: {key}"
        assert stats[key] >= 0, f"Negative value for {key}"


def create_test_graph(num_triples=10):
    """Create a simple test graph for quick tests."""
    import rdflib
    from rdflib import RDF, RDFS, Namespace, Literal
    
    g = rdflib.Graph()
    ex = Namespace("http://test.example.org/")
    
    for i in range(num_triples):
        g.add((ex[f"entity_{i}"], RDF.type, ex.TestClass))
        g.add((ex[f"entity_{i}"], ex.testProperty, Literal(f"value_{i}")))
    
    return g


# Test configuration helpers
def get_fast_config():
    """Get configuration for fast tests."""
    from ifluid.config.config import IFLUIDConfig, EmbeddingConfig, GNNConfig
    
    return IFLUIDConfig(
        embedding=EmbeddingConfig(
            dim=8,
            epochs=2,
            batch_size=16
        ),
        gnn=GNNConfig(
            hidden_dim=8,
            epochs=10
        ),
        verbose=False
    )


def get_full_config():
    """Get configuration for comprehensive tests."""
    from ifluid.config.config import IFLUIDConfig, EmbeddingConfig, GNNConfig
    
    return IFLUIDConfig(
        embedding=EmbeddingConfig(
            dim=32,
            epochs=20,
            batch_size=64
        ),
        gnn=GNNConfig(
            hidden_dim=32,
            epochs=100
        ),
        verbose=True
    )


# Export test utilities
__all__ = [
    'assert_graph_valid',
    'assert_policy_valid',
    'assert_statistics_valid',
    'create_test_graph',
    'get_fast_config',
    'get_full_config',
]
