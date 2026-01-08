import math
import numpy as np
from typing import Dict, Set, Any, List, Tuple
import rdflib
from rdflib import RDF, RDFS, OWL
from collections import defaultdict, Counter

class KGStatistics:
    """
    1. SubClassOf: Is there hierarchy? How deep? How big will inference be?
    2. sameAs: Are there duplicates? How clustered?
    3. Domain/Range: Rich schema? Many instances?
    4. Rule mining: Large enough? Connected enough?
    """
    
    def __init__(self, graph: rdflib.Graph):
        self.graph = graph
        self._cache = {}
        
    def compute_all(self) -> Dict[str, float]:
        """Compute minimal, focused statistics."""
        if self._cache:
            return self._cache
            
        stats = {}
        stats.update(self._compute_basic_counts())
        stats.update(self._compute_hierarchy_metrics())
        stats.update(self._compute_sameas_metrics())
        stats.update(self._compute_schema_metrics())
        stats.update(self._estimate_inference_costs())
        
        self._cache = stats
        return stats
    
    #Basic Counts    
    def _compute_basic_counts(self) -> Dict[str, int]:
        """
        Basic counts needed for normalization and ratios.
        
        Policy impact:
        - num_triples: Determines if graph is large enough for rule mining
        - num_instances/classes: Determines instance density for domain/range
        """
        classes = set(o for _, _, o in self.graph.triples((None, RDF.type, None)))
        instances = set(s for s, _, _ in self.graph.triples((None, RDF.type, None)))
        properties = set(p for _, p, _ in self.graph 
                        if p not in [RDF.type, RDFS.subClassOf, RDFS.subPropertyOf, OWL.sameAs])
        
        return {
            'num_triples': len(self.graph),
            'num_classes': len(classes),
            'num_instances': len(instances),
            'num_properties': len(properties),
            # Derived: instance density (high → domain/range beneficial)
            'instance_to_class_ratio': len(instances) / max(1, len(classes)),
        }
    
    # Hierarchy Metrics (for subClassOf policy)
    def _compute_hierarchy_metrics(self) -> Dict[str, float]:
        """
        Hierarchy analysis for subClassOf transitivity decision.
        
        Key questions:
        1. Does hierarchy exist? → max_depth > 1
        2. Is it deep enough to benefit? → avg_depth > 2
        3. How complex? → num_subclass_triples
        
        Policy impact: If no hierarchy, skip subClassOf entirely.
        """
        subclass_triples = list(self.graph.triples((None, RDFS.subClassOf, None)))
        
        if not subclass_triples:
            return {
                'num_subclass_triples': 0,
                'max_hierarchy_depth': 0,
                'avg_hierarchy_depth': 0,
                'has_multiple_inheritance': False,
            }
        
        # Build parent mapping
        child_to_parents = defaultdict(set)
        for child, _, parent in subclass_triples:
            child_to_parents[child].add(parent)
        
        # Compute depths (BFS to handle cycles)
        all_classes = set(child_to_parents.keys()) | \
                     set(p for parents in child_to_parents.values() for p in parents)
        
        depths = []
        max_depth = 0
        has_multiple_inheritance = False
        
        for cls in all_classes:
            depth = self._compute_depth_bfs(cls, child_to_parents)
            depths.append(depth)
            max_depth = max(max_depth, depth)
            
            # Check for multiple inheritance
            if len(child_to_parents.get(cls, [])) > 1:
                has_multiple_inheritance = True
        
        return {
            'num_subclass_triples': len(subclass_triples),
            'max_hierarchy_depth': max_depth,
            'avg_hierarchy_depth': np.mean(depths) if depths else 0,
            'has_multiple_inheritance': has_multiple_inheritance,
        }
    
    def _compute_depth_bfs(self, cls, child_to_parents: Dict, max_depth: int = 50) -> int:
        """BFS depth computation (handles cycles and multiple paths)."""
        if cls not in child_to_parents:
            return 0
        
        visited = {cls}
        queue = [(cls, 0)]
        max_depth_found = 0
        
        while queue:
            current, depth = queue.pop(0)
            max_depth_found = max(max_depth_found, depth)
            
            if depth >= max_depth:
                continue
            
            for parent in child_to_parents.get(current, []):
                if parent not in visited:
                    visited.add(parent)
                    queue.append((parent, depth + 1))
        
        return max_depth_found
    
    # sameAs Metrics (for sameAs policy)    
    def _compute_sameas_metrics(self) -> Dict[str, float]:
        """
        sameAs analysis for consolidation decision.
        
        Key questions:
        1. Are there sameAs relations? → count > 0
        2. Are they clustered? → avg_cluster_size > 2
        3. Significant enough? → density, count
        
        Policy impact: Only consolidate if clusters are non-trivial.
        """
        sameas_triples = list(self.graph.triples((None, OWL.sameAs, None)))
        
        if not sameas_triples:
            return {
                'sameas_count': 0,
                'sameas_density': 0.0,
                'avg_sameas_cluster_size': 0.0,
            }
        
        # Union-find to compute cluster sizes
        parent = {}
        
        def find(x):
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            parent[find(x)] = find(y)
        
        for x, _, y in sameas_triples:
            union(x, y)
        
        # Count cluster sizes
        clusters = defaultdict(set)
        for entity in parent:
            clusters[find(entity)].add(entity)
        
        cluster_sizes = [len(cluster) for cluster in clusters.values()]
        
        return {
            'sameas_count': len(sameas_triples),
            'sameas_density': len(sameas_triples) / max(1, len(self.graph)),
            'avg_sameas_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
        }
    
    
    # Schema Richness (for domain/range policy)    
    def _compute_schema_metrics(self) -> Dict[str, float]:
        """
        Schema analysis for domain/range propagation decision.
        
        Key questions:
        1. Is schema rich (many properties vs classes)? → ratio < 2
        2. Do properties have domain/range? → count > 0
        
        Policy impact: Only propagate if schema is rich enough to benefit.
        """
        num_classes = self._cache.get('num_classes', len(set(o for _, _, o in self.graph.triples((None, RDF.type, None)))))
        num_properties = self._cache.get('num_properties', len(set(p for _, p, _ in self.graph if p not in [RDF.type, RDFS.subClassOf])))
        
        # Count domain/range declarations
        domain_count = len(list(self.graph.triples((None, RDFS.domain, None))))
        range_count = len(list(self.graph.triples((None, RDFS.range, None))))
        
        return {
            'class_to_property_ratio': num_classes / max(1, num_properties),
            'num_domain_declarations': domain_count,
            'num_range_declarations': range_count,
            'has_schema_declarations': (domain_count + range_count) > 0,
        }
    
    # Inference Cost Estimation (CRITICAL)
    def _estimate_inference_costs(self) -> Dict[str, int]:
        """
        Estimate how many triples each rule will add.
        This is CRITICAL - prevents policy from causing combinatorial explosion.
        Policy impact: Disable rules if estimated cost is too high.
        """
        
        # === SubClassOf transitivity cost ===
        child_to_parents = defaultdict(set)
        for child, _, parent in self.graph.triples((None, RDFS.subClassOf, None)):
            child_to_parents[child].add(parent)
        
        subclass_cost = 0
        for child in child_to_parents:
            # Find all ancestors via BFS
            ancestors = set()
            queue = list(child_to_parents[child])
            visited = set(child_to_parents[child])
            
            while queue and len(ancestors) < 1000:  # Safety limit
                parent = queue.pop(0)
                ancestors.add(parent)
                for grandparent in child_to_parents.get(parent, []):
                    if grandparent not in visited:
                        visited.add(grandparent)
                        queue.append(grandparent)
            
            # New triples = indirect ancestors
            indirect = ancestors - child_to_parents[child]
            subclass_cost += len(indirect)
        
        # === Domain/Range cost ===
        domain_range_cost = 0
        
        # Domain cost: count instances that use properties with domains
        for prop, _, domain_class in self.graph.triples((None, RDFS.domain, None)):
            usage = len(list(self.graph.triples((None, prop, None))))
            domain_range_cost += usage
        
        # Range cost: count instances that are objects of properties with ranges
        for prop, _, range_class in self.graph.triples((None, RDFS.range, None)):
            usage = len(list(self.graph.triples((None, prop, None))))
            domain_range_cost += usage
        
        return {
            'subclass_inference_cost': subclass_cost,
            'domain_range_inference_cost': domain_range_cost,
        }
    
    def normalize(self) -> Dict[str, float]:
        """
        Normalize statistics into profile for policy determination.
        Only includes metrics used in policy decisions.
        """
        stats = self.compute_all()
        
        return {
            # Size indicators
            'log_triples': math.log1p(stats['num_triples']),
            
            # Hierarchy indicators (subClassOf)
            'has_hierarchy': stats['max_hierarchy_depth'] > 1,
            'avg_hierarchy_depth': stats['avg_hierarchy_depth'],
            'max_hierarchy_depth': stats['max_hierarchy_depth'],
            
            # Duplication indicators (sameAs)
            'sameas_density': stats['sameas_density'],
            'sameas_cluster_size': stats['avg_sameas_cluster_size'],
            
            # Schema indicators (domain/range)
            'instance_class_ratio': stats['instance_to_class_ratio'],
            'class_property_ratio': stats['class_to_property_ratio'],
            'has_schema': stats['has_schema_declarations'],
            
            # Cost indicators (ALL rules)
            'subclass_cost': stats['subclass_inference_cost'],
            'domain_range_cost': stats['domain_range_inference_cost'],
        }
