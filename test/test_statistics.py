import pytest
from ifluid.core.statistics_streamlined import StreamlinedKGStatistics


def test_basic_counts(simple_graph):
    """Test basic statistics computation."""
    stats = StreamlinedKGStatistics(simple_graph)
    result = stats.compute_all()
    
    assert result['num_triples'] > 0
    assert result['num_classes'] >= 0
    assert result['num_instances'] >= 0
    assert result['num_properties'] >= 0


def test_empty_graph(empty_graph):
    """Test with empty graph."""
    stats = StreamlinedKGStatistics(empty_graph)
    result = stats.compute_all()
    
    assert result['num_triples'] == 0
    assert result['num_classes'] == 0
    assert result['num_instances'] == 0


def test_hierarchy_metrics(hierarchical_graph):
    """Test hierarchy analysis."""
    stats = StreamlinedKGStatistics(hierarchical_graph)
    result = stats.compute_all()
    
    assert result['num_subclass_triples'] > 0
    assert result['max_hierarchy_depth'] >= 3  # Student->Person->Agent->Thing
    assert result['avg_hierarchy_depth'] > 0


def test_no_hierarchy(simple_graph):
    """Test graph with minimal hierarchy."""
    stats = StreamlinedKGStatistics(simple_graph)
    result = stats.compute_all()
    
    # Should still work, just with low values
    assert 'max_hierarchy_depth' in result


def test_sameas_metrics(sameas_graph):
    """Test sameAs analysis."""
    stats = StreamlinedKGStatistics(sameas_graph)
    result = stats.compute_all()
    
    assert result['sameas_count'] > 0
    assert result['sameas_density'] > 0
    assert result['avg_sameas_cluster_size'] >= 2  # Has chains


def test_no_sameas(simple_graph):
    """Test graph without sameAs."""
    stats = StreamlinedKGStatistics(simple_graph)
    result = stats.compute_all()
    
    assert result['sameas_count'] == 0
    assert result['sameas_density'] == 0


def test_schema_metrics(domain_range_graph):
    """Test schema analysis."""
    stats = StreamlinedKGStatistics(domain_range_graph)
    result = stats.compute_all()
    
    assert result['num_domain_declarations'] > 0
    assert result['has_schema_declarations'] == True
    assert result['class_to_property_ratio'] >= 0


def test_inference_cost_estimation(hierarchical_graph):
    """Test inference cost estimation."""
    stats = StreamlinedKGStatistics(hierarchical_graph)
    result = stats.compute_all()
    
    # Should estimate some transitive closures
    assert 'subclass_inference_cost' in result
    assert 'domain_range_inference_cost' in result
    assert result['subclass_inference_cost'] >= 0


def test_normalize_profile(complex_graph):
    """Test profile normalization."""
    stats = StreamlinedKGStatistics(complex_graph)
    profile = stats.normalize()
    
    # Check all expected keys
    assert 'log_triples' in profile
    assert 'has_hierarchy' in profile
    assert 'sameas_density' in profile
    assert 'instance_class_ratio' in profile
    assert 'subclass_cost' in profile
    
    # Check value ranges
    assert profile['log_triples'] >= 0
    assert isinstance(profile['has_hierarchy'], bool)


def test_caching(simple_graph):
    """Test that statistics are cached."""
    stats = StreamlinedKGStatistics(simple_graph)
    
    # First computation
    result1 = stats.compute_all()
    
    # Second computation should use cache
    result2 = stats.compute_all()
    
    assert result1 == result2
    assert stats._cache is not None
