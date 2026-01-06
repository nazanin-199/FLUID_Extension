"""Tests for symbolic extraction."""

import pytest
from ifluid.core.extraction import AdaptivePolicy, SymbolicExtractor
from ifluid.core.statistics import KGStatistics
from ifluid.config.config import PolicyConfig


def test_adaptive_policy(simple_graph):
    """Test adaptive policy determination."""
    stats = KGStatistics(simple_graph)
    profile = stats.normalize()
    
    policy_config = PolicyConfig()
    policy_maker = AdaptivePolicy(policy_config)
    policy = policy_maker.determine(profile)
    
    assert isinstance(policy, dict)
    assert 'subClassOf' in policy
    assert 'sameAs' in policy


def test_subclass_extraction(hierarchical_graph):
    """Test subClassOf transitivity extraction."""
    policy = {'subClassOf': True, 'subPropertyOf': False, 
              'domain_range': False, 'sameAs': False}
    extractor = SymbolicExtractor(policy)
    enriched = extractor.extract(hierarchical_graph)
    
    assert len(enriched) >= len(hierarchical_graph)


def test_sameas_consolidation(sameas_graph):
    """Test sameAs consolidation."""
    policy = {'subClassOf': False, 'subPropertyOf': False,
              'domain_range': False, 'sameAs': True}
    extractor = SymbolicExtractor(policy)
    enriched = extractor.extract(sameas_graph)
    
    # Should consolidate entities
    assert len(enriched) <= len(sameas_graph)
