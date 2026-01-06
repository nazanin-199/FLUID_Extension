import pytest
import rdflib
from rdflib import RDF, RDFS, OWL, Namespace, Literal, URIRef
import torch
import random
import numpy as np


# ============================================================================
# Test Configuration Fixtures
# ============================================================================
@pytest.fixture(scope="session")
def random_seed():
    """Set random seed for reproducibility."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed

# ============================================================================
# Basic Graph Fixtures
# ============================================================================
@pytest.fixture
def empty_graph():
    """
    Create an empty graph.
    
    Use for:
    - Edge case testing
    - Error handling validation
    - Ensuring functions handle empty inputs gracefully
    """
    return rdflib.Graph()

@pytest.fixture
def simple_graph():
    """
    Create a simple test graph with basic structure.
    
    Structure:
    - 2 person instances
    - 1 class hierarchy (Person -> Agent)
    - Simple properties (name, knows)
    
    Triples: ~5
    Use for: Quick unit tests
    """
    g = rdflib.Graph()
    ex = Namespace("http://example.org/")
    
    # Instances
    g.add((ex.Person1, RDF.type, ex.Person))
    g.add((ex.Person1, ex.name, Literal("Alice")))
    g.add((ex.Person1, ex.knows, ex.Person2))
    g.add((ex.Person2, RDF.type, ex.Person))
    
    # Simple hierarchy
    g.add((ex.Person, RDFS.subClassOf, ex.Agent))
    
    return g

# ============================================================================
# Specialized Graph Fixtures
# ============================================================================

@pytest.fixture
def hierarchical_graph():
    """
    Create a graph with deep class hierarchy.
    
    Structure:
    - 4-level hierarchy: Student -> Person -> Agent -> Thing
    - 3 instances at different levels
    - Properties connecting instances
    
    Triples: ~10
    Use for: Testing hierarchy analysis, subClassOf transitivity
    """
    g = rdflib.Graph()
    ex = Namespace("http://example.org/")
    
    # Deep class hierarchy (4 levels)
    g.add((ex.Student, RDFS.subClassOf, ex.Person))
    g.add((ex.Person, RDFS.subClassOf, ex.Agent))
    g.add((ex.Agent, RDFS.subClassOf, ex.Thing))
    
    # Instances at different levels
    g.add((ex.Alice, RDF.type, ex.Student))
    g.add((ex.Bob, RDF.type, ex.Person))
    g.add((ex.Charlie, RDF.type, ex.Agent))
    
    # Properties
    g.add((ex.Alice, ex.enrolled, ex.Course1))
    g.add((ex.Bob, ex.works, ex.Company1))
    g.add((ex.Charlie, ex.lives, ex.City1))
    
    return g

@pytest.fixture
def sameas_graph():
    """
    Create a graph with sameAs relations and clusters.
    
    Structure:
    - Chain: Entity1 -> Entity2 -> Entity3 (cluster of 3)
    - Pair: Entity4 -> Entity5 (cluster of 2)
    - Properties distributed across equivalent entities
    
    Triples: ~8
    Use for: Testing sameAs consolidation, cluster detection
    """
    g = rdflib.Graph()
    ex = Namespace("http://example.org/")
    
    # sameAs chain (cluster size: 3)
    g.add((ex.Entity1, OWL.sameAs, ex.Entity2))
    g.add((ex.Entity2, OWL.sameAs, ex.Entity3))
    
    # sameAs pair (cluster size: 2)
    g.add((ex.Entity4, OWL.sameAs, ex.Entity5))
    
    # Properties on equivalent entities
    g.add((ex.Entity1, ex.property1, Literal("value1")))
    g.add((ex.Entity3, ex.property2, Literal("value2")))  # Same cluster
    g.add((ex.Entity4, RDF.type, ex.Type1))
    g.add((ex.Entity5, ex.property3, Literal("value3")))  # Same cluster
    
    return g

@pytest.fixture
def domain_range_graph():
    """
    Create a graph with domain/range schema declarations.
    
    Structure:
    - 3 properties with domain/range constraints
    - Multiple instances using these properties
    - Mix of types and hierarchy
    
    Triples: ~10
    Use for: Testing domain/range propagation
    """
    g = rdflib.Graph()
    ex = Namespace("http://example.org/")
    
    # Schema declarations
    g.add((ex.worksFor, RDFS.domain, ex.Person))
    g.add((ex.worksFor, RDFS.range, ex.Organization))
    g.add((ex.teaches, RDFS.domain, ex.Professor))
    g.add((ex.teaches, RDFS.range, ex.Course))
    
    # Instances using properties
    g.add((ex.Alice, ex.worksFor, ex.Acme))
    g.add((ex.Bob, ex.worksFor, ex.TechCorp))
    g.add((ex.Carol, ex.teaches, ex.Course1))
    
    # Some explicit types
    g.add((ex.Alice, RDF.type, ex.Person))
    
    # Hierarchy
    g.add((ex.Professor, RDFS.subClassOf, ex.Person))
    
    return g

@pytest.fixture
def complex_graph():
    """
    Create a comprehensive graph combining all features.
    
    Structure:
    - Class hierarchy (3 levels)
    - sameAs relations
    - Domain/range declarations
    - Multiple instances and properties
    
    Triples: ~12
    Use for: Integration testing, comprehensive policy evaluation
    """
    g = rdflib.Graph()
    ex = Namespace("http://example.org/")
    
    # Class hierarchy
    g.add((ex.GradStudent, RDFS.subClassOf, ex.Student))
    g.add((ex.Student, RDFS.subClassOf, ex.Person))
    g.add((ex.Professor, RDFS.subClassOf, ex.Person))
    g.add((ex.Person, RDFS.subClassOf, ex.Agent))
    
    # sameAs relations
    g.add((ex.Alice, OWL.sameAs, ex.AliceSmith))
    
    # Domain/range schema
    g.add((ex.enrolled, RDFS.domain, ex.Student))
    g.add((ex.enrolled, RDFS.range, ex.Course))
    g.add((ex.advises, RDFS.domain, ex.Professor))
    
    # Instances with types
    g.add((ex.Alice, RDF.type, ex.GradStudent))
    g.add((ex.Bob, RDF.type, ex.Professor))
    g.add((ex.Carol, RDF.type, ex.Student))
    
    # Properties
    g.add((ex.Alice, ex.enrolled, ex.Course1))
    g.add((ex.AliceSmith, ex.advisor, ex.Bob))  # Uses sameAs
    g.add((ex.Bob, ex.advises, ex.Carol))
    
    return g

# ============================================================================
# Specialized Fixtures for Edge Cases
# ============================================================================

@pytest.fixture
def cyclic_hierarchy_graph():
    """
    Create a graph with cycles in class hierarchy.
    
    Use for: Testing cycle detection in hierarchy analysis
    """
    g = rdflib.Graph()
    ex = Namespace("http://example.org/")
    
    # Create a cycle: A -> B -> C -> A
    g.add((ex.ClassA, RDFS.subClassOf, ex.ClassB))
    g.add((ex.ClassB, RDFS.subClassOf, ex.ClassC))
    g.add((ex.ClassC, RDFS.subClassOf, ex.ClassA))  # Cycle!
    
    # Add an instance
    g.add((ex.Entity1, RDF.type, ex.ClassA))
    
    return g

@pytest.fixture
def multiple_inheritance_graph():
    """
    Create a graph with multiple inheritance (diamond pattern).
    
    Structure:
         Thing
        /     \
    Person   Agent
        \     /
        Student
    
    Use for: Testing multiple inheritance handling
    """
    g = rdflib.Graph()
    ex = Namespace("http://example.org/")
    
    # Diamond pattern
    g.add((ex.Person, RDFS.subClassOf, ex.Thing))
    g.add((ex.Agent, RDFS.subClassOf, ex.Thing))
    g.add((ex.Student, RDFS.subClassOf, ex.Person))
    g.add((ex.Student, RDFS.subClassOf, ex.Agent))  # Multiple inheritance
    
    # Instance
    g.add((ex.Alice, RDF.type, ex.Student))
    
    return g

@pytest.fixture
def large_hierarchy_graph():
    """
    Create a graph with broad hierarchy (many classes at same level).
    
    Structure: 1 root, 10 children, 20 grandchildren
    
    Use for: Testing performance and breadth handling
    """
    g = rdflib.Graph()
    ex = Namespace("http://example.org/")
    
    # Root
    root = ex.Root
    
    # 10 children
    for i in range(10):
        child = ex[f"Child{i}"]
        g.add((child, RDFS.subClassOf, root))
        
        # 2 grandchildren per child
        for j in range(2):
            grandchild = ex[f"GrandChild{i}_{j}"]
            g.add((grandchild, RDFS.subClassOf, child))
    
    return g


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def fast_config():
    """Fast configuration for quick tests."""
    from ifluid.config.config import IFLUIDConfig, EmbeddingConfig, GNNConfig
    
    return IFLUIDConfig(
        embedding=EmbeddingConfig(
            dim=8,
            epochs=2,
            batch_size=16,
            learning_rate=0.01
        ),
        gnn=GNNConfig(
            hidden_dim=8,
            epochs=10,
            learning_rate=0.01
        ),
        verbose=False
    )

@pytest.fixture
def full_config():
    """Full configuration for comprehensive tests."""
    from ifluid.config.config import IFLUIDConfig, EmbeddingConfig, GNNConfig
    
    return IFLUIDConfig(
        embedding=EmbeddingConfig(
            dim=32,
            epochs=40,
            batch_size=128,
            learning_rate=0.01
        ),
        gnn=GNNConfig(
            hidden_dim=32,
            epochs=200,
            learning_rate=0.01
        ),
        verbose=True
    )

# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary directory for test outputs."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_triples():
    """Sample triples for embedding training tests."""
    ex = rdflib.Namespace("http://example.org/")
    return [
        (ex.Entity1, ex.relation1, ex.Entity2),
        (ex.Entity2, ex.relation1, ex.Entity3),
        (ex.Entity1, ex.relation2, ex.Entity3),
        (ex.Entity3, ex.relation2, ex.Entity4),
    ]

def assert_graph_valid(graph):
    """Helper to validate RDF graph."""
    assert isinstance(graph, rdflib.Graph)
    assert len(graph) >= 0


def count_triples_by_predicate(graph, predicate):
    """Count triples with specific predicate."""
    return len(list(graph.triples((None, predicate, None))))
