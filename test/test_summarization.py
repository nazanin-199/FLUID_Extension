import pytest
from core.summarization import FLUIDSummarizer

def test_basic_summarization(simple_graph):
    """Test basic FLUID summarization."""
    summarizer = FLUIDSummarizer()
    summary_graph, summary_nodes, node_map = summarizer.summarize(simple_graph)
    
    # Check outputs exist
    assert summary_graph is not None
    assert isinstance(summary_nodes, dict)
    assert isinstance(node_map, dict)
    
    # Summary should have fewer or equal nodes
    assert len(summary_nodes) <= len(set(simple_graph.subjects()) | set(simple_graph.objects()))


def test_empty_graph_summarization(empty_graph):
    """Test summarization of empty graph."""
    summarizer = FLUIDSummarizer()
    summary_graph, summary_nodes, node_map = summarizer.summarize(empty_graph)
    
    assert len(summary_graph) == 0
    assert len(summary_nodes) == 0
    assert len(node_map) == 0


def test_summary_preserves_structure(hierarchical_graph):
    """Test that summary preserves graph structure."""
    summarizer = FLUIDSummarizer()
    summary_graph, summary_nodes, node_map = summarizer.summarize(hierarchical_graph)
    
    # Summary should have edges
    assert len(summary_graph) > 0
    
    # All nodes should be in summary
    for entity in set(hierarchical_graph.subjects()) | set(hierarchical_graph.objects()):
        if isinstance(entity, rdflib.URIRef):
            assert entity in node_map


def test_identical_descriptors_merge(simple_graph):
    """Test that entities with identical descriptors are merged."""
    # Add two entities with same type and properties
    ex = rdflib.Namespace("http://example.org/")
    g = rdflib.Graph()
    
    g.add((ex.Entity1, RDF.type, ex.Type1))
    g.add((ex.Entity1, ex.prop1, ex.Target))
    
    g.add((ex.Entity2, RDF.type, ex.Type1))
    g.add((ex.Entity2, ex.prop1, ex.Target))
    
    summarizer = FLUIDSummarizer()
    summary_graph, summary_nodes, node_map = summarizer.summarize(g)
    
    # Should merge into same super-node
    assert node_map[ex.Entity1] == node_map[ex.Entity2]
    
    # Should have only 1-2 super-nodes (entities + target)
    assert len(summary_nodes) <= 2


def test_different_descriptors_separate():
    """Test that entities with different descriptors are separate."""
    ex = rdflib.Namespace("http://example.org/")
    g = rdflib.Graph()
    
    g.add((ex.Entity1, RDF.type, ex.Type1))
    g.add((ex.Entity2, RDF.type, ex.Type2))  # Different type
    
    summarizer = FLUIDSummarizer()
    summary_graph, summary_nodes, node_map = summarizer.summarize(g)
    
    # Should have different super-nodes
    assert node_map[ex.Entity1] != node_map[ex.Entity2]


def test_summary_node_mapping_consistency(complex_graph):
    """Test that node mapping is consistent."""
    summarizer = FLUIDSummarizer()
    summary_graph, summary_nodes, node_map = summarizer.summarize(complex_graph)
    
    # Every entity should map to exactly one super-node
    for entity, super_node_id in node_map.items():
        assert entity in summary_nodes[super_node_id]
    
    # Every entity in summary_nodes should be in node_map
    for super_node_id, entities in summary_nodes.items():
        for entity in entities:
            assert entity in node_map
            assert node_map[entity] == super_node_id
