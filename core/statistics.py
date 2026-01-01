from typing import Dict, Set
import rdflib
from rdflib import RDF, RDFS, OWL
import math


class KGStatistics:
    """Compute dataset-agnostic knowledge graph statistics."""
    
    def __init__(self, graph: rdflib.Graph):
        self.graph = graph
        self._stats_cache: Dict[str, Any] = {}
    
    def compute_all(self) -> Dict[str, float]:
        """Compute all statistics."""
        if self._stats_cache:
            return self._stats_cache
        
        self._stats_cache = {
            'num_triples': self._count_triples(),
            'num_classes': self._count_classes(),
            'num_properties': self._count_properties(),
            'sameas_density': self._compute_sameas_density(),
            'avg_class_depth': self._compute_avg_class_depth()
        }
        return self._stats_cache
    
    def _count_triples(self) -> int:
        return len(self.graph)
    
    def _count_classes(self) -> int:
        classes = set(o for _, _, o in self.graph.triples((None, RDF.type, None)))
        return len(classes)
    
    def _count_properties(self) -> int:
        properties = set(p for _, p, _ in self.graph)
        return len(properties)
    
    def _compute_sameas_density(self) -> float:
        sameas = list(self.graph.triples((None, OWL.sameAs, None)))
        return len(sameas) / max(1, len(self.graph))
    
    def _compute_avg_class_depth(self) -> float:
        """Compute average depth of class hierarchy."""
        classes = set(o for _, _, o in self.graph.triples((None, RDF.type, None)))
        depths = []
        
        for c in classes:
            depth = self._get_class_depth(c)
            depths.append(depth)
        
        return sum(depths) / max(1, len(depths))
    
    def _get_class_depth(self, cls: rdflib.term.Node, max_depth: int = 100) -> int:
        """Get depth of a class in hierarchy."""
        depth = 0
        current = cls
        visited = set()
        
        while depth < max_depth:
            if current in visited:
                break
            visited.add(current)
            
            parents = list(self.graph.objects(current, RDFS.subClassOf))
            if not parents:
                break
            
            current = parents[0]
            depth += 1
        
        return depth
    
    def normalize(self) -> Dict[str, float]:
        """Normalize statistics to profile."""
        stats = self.compute_all()
        return {
            'log_triples': math.log1p(stats['num_triples']),
            'class_property_ratio': stats['num_classes'] / max(1, stats['num_properties']),
            'sameas_density': stats['sameas_density'],
            'avg_class_depth': stats['avg_class_depth']
        }

