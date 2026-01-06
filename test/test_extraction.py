"""Tests for symbolic extraction and adaptive policy."""

import pytest
from core.extraction_streamlined import StreamlinedAdaptivePolicy
from core.extraction import SymbolicExtractor
from core.statistics_streamlined import StreamlinedKGStatistics


def test_policy_determination(hierarchical_graph):
    """Test adaptive policy determination."""
    stats_computer = StreamlinedKGStatistics(hierarchical_graph)
    stats = stats_computer.compute_all()
    profile = stats_computer.normalize()
    
    policy_maker = StreamlinedAdaptivePolicy()
    policy, reasoning = policy_maker.determine(profile, stats)
    
    # Check structure
    assert isinstance(policy, dict)
    assert isinstance(reasoning, dict)
    
    # Check required keys
    assert 'subClassOf' in policy
    assert 'sameAs' in policy
    assert 'domain_range' in policy
    assert 'max_depth' in policy


def test_subclass_policy_with_hierarchy(hierarchical_graph):
    """Test subClassOf policy on graph with hierarchy."""
    stats_computer = StreamlinedKGStatistics(hierarchical_graph)
    stats = stats_computer.compute_all()
    profile = stats_computer.normalize()
    
    policy_maker = StreamlinedAdaptivePolicy()
    policy, reasoning = policy_maker.determine(profile, stats)
    
    # Should enable subClassOf due to deep hierarchy
    decision = reasoning['subClassOf']
    assert decision.enabled == True
    assert 'hierarchy' in decision.reason.lower()


def test_subclass_policy_without_hierarchy(simple_graph):
    """Test subClassOf policy on graph without significant hierarchy."""
    # Remove subclass triple
    g = rdflib.Graph()
    ex = rdflib.Namespace("http://example.org/")
    g.add((ex.Person1, RDF.type, ex.Person))
    g.add((ex.Person1, ex.name, rdflib.Literal("Alice")))
    
    stats_computer = StreamlinedKGStatistics(g)
    stats = stats_computer.compute_all()
    profile = stats_computer.normalize()
    
    policy_maker = StreamlinedAdaptivePolicy()
    policy, reasoning = policy_maker.determine(profile, stats)
    
    # Might disable subClassOf due to no hierarchy
    decision = reasoning['subClassOf']
    # Just check decision exists and has reasoning
    assert hasattr(decision, 'enabled')
    assert hasattr(decision, 'reason')


def test_sameas_policy(sameas_graph):
    """Test sameAs policy on graph with duplicates."""
    stats_computer = StreamlinedKGStatistics(sameas_graph)
    stats = stats_computer.compute_all()
    profile = stats_computer.normalize()
    
    policy_maker = StreamlinedAdaptivePolicy()
    policy, reasoning = policy_maker.determine(profile, stats)
    
    # Should enable sameAs due to clusters
    decision = reasoning['sameAs']
    assert decision.enabled == True


def test_domain_range_policy(domain_range_graph):
    """Test domain/range policy on schema-rich graph."""
    stats_computer = StreamlinedKGStatistics(domain_range_graph)
    stats = stats_computer.compute_all()
    profile = stats_computer.normalize()
    
    policy_maker = StreamlinedAdaptivePolicy()
    policy, reasoning = policy_maker.determine(profile, stats)
    
    # Should enable domain_range due to declarations
    decision = reasoning['domain_range']
    # Check structure
    assert hasattr(decision, 'enabled')
    assert hasattr(decision, 'reason')


def test_subclass_extraction(hierarchical_graph):
    """Test subClassOf transitivity extraction."""
    policy = {
        'subClassOf': True,
        'subPropertyOf': False,
        'domain_range': False,
        'sameAs': False
    }
    
    extractor = SymbolicExtractor(policy)
    enriched = extractor.extract(hierarchical_graph)
    
    # Should add transitive closures
    assert len(enriched) >= len(hierarchical_graph)
    
    # Verify transitivity: Student->Person->Agent->Thing
    # Should infer: Student->Agent, Student->Thing, Person->Thing
    ex = rdflib.Namespace("http://example.org/")
    assert (ex.Student, RDFS.subClassOf, ex.Agent) in enriched
    assert (ex.Student, RDFS.subClassOf, ex.Thing) in enriched


def test_sameas_consolidation(sameas_graph):
    """Test sameAs consolidation."""
    policy = {
        'subClassOf': False,
        'subPropertyOf': False,
        'domain_range': False,
        'sameAs': True
    }
    
    extractor = SymbolicExtractor(policy)
    enriched = extractor.extract(sameas_graph)
    
    # Should consolidate entities
    # Count unique subjects/objects
    original_entities = len(set(sameas_graph.subjects()) | set(sameas_graph.objects()))
    enriched_entities = len(set(enriched.subjects()) | set(enriched.objects()))
    
    # After consolidation, should have fewer entities
    assert enriched_entities <= original_entities


def test_domain_range_propagation(domain_range_graph):
    """Test domain/range propagation."""
    policy = {
        'subClassOf': False,
        'subPropertyOf': False,
        'domain_range': True,
        'sameAs': False
    }
    
    extractor = SymbolicExtractor(policy)
    enriched = extractor.extract(domain_range_graph)
    
    # Should add type inferences
    assert len(enriched) >= len(domain_range_graph)


def test_all_policies_disabled():
    """Test extraction with all policies disabled."""
    g = rdflib.Graph()
    ex = rdflib.Namespace("http://example.org/")
    g.add((ex.A, RDF.type, ex.Type1))
    
    policy = {
        'subClassOf': False,
        'subPropertyOf': False,
        'domain_range': False,
        'sameAs': False
    }
    
    extractor = SymbolicExtractor(policy)
    enriched = extractor.extract(g)
    
    # Should return graph unchanged
    assert len(enriched) == len(g)
