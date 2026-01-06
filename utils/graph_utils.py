import rdflib
from typing import Dict, Set, List, Tuple
from collections import defaultdict


class GraphUtils:
    """Utility functions for RDF graph operations."""
    
    @staticmethod
    def get_entity_types(graph: rdflib.Graph, entity: rdflib.term.Node) -> Set:
        """Get all types of an entity."""
        from rdflib import RDF
        return set(graph.objects(entity, RDF.type))
    
    @staticmethod
    def get_entity_properties(
        graph: rdflib.Graph, 
        entity: rdflib.term.Node,
        direction: str = 'out'
    ) -> Set:
        """
        Get properties of an entity.
        
        Args:
            graph: RDF graph
            entity: Entity node
            direction: 'out' for outgoing, 'in' for incoming, 'both' for both
        """
        properties = set()
        
        if direction in ('out', 'both'):
            properties.update(graph.predicates(entity, None))
        
        if direction in ('in', 'both'):
            properties.update(graph.predicates(None, entity))
        
        return properties
    
    @staticmethod
    def get_neighbors(
        graph: rdflib.Graph,
        entity: rdflib.term.Node,
        direction: str = 'out'
    ) -> Set:
        """Get neighboring entities."""
        neighbors = set()
        
        if direction in ('out', 'both'):
            neighbors.update(graph.objects(entity, None))
        
        if direction in ('in', 'both'):
            neighbors.update(graph.subjects(None, entity))
        
        return neighbors
    
    @staticmethod
    def compute_degree(
        graph: rdflib.Graph,
        entity: rdflib.term.Node,
        direction: str = 'both'
    ) -> int:
        """Compute node degree."""
        degree = 0
        
        if direction in ('out', 'both'):
            degree += len(list(graph.triples((entity, None, None))))
        
        if direction in ('in', 'both'):
            degree += len(list(graph.triples((None, None, entity))))
        
        return degree
    
    @staticmethod
    def get_graph_stats(graph: rdflib.Graph) -> Dict[str, int]:
        """Get basic graph statistics."""
        entities = set(graph.subjects()) | set(graph.objects())
        predicates = set(graph.predicates())
        
        return {
            'num_triples': len(graph),
            'num_entities': len(entities),
            'num_predicates': len(predicates),
            'num_subjects': len(set(graph.subjects())),
            'num_objects': len(set(graph.objects())),
        }
    
    @staticmethod
    def find_connected_components(graph: rdflib.Graph) -> List[Set]:
        """Find connected components using union-find."""
        entities = set(graph.subjects()) | set(graph.objects())
        parent = {e: e for e in entities}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            parent[find(x)] = find(y)
        
        # Union all connected entities
        for s, p, o in graph:
            if isinstance(s, rdflib.URIRef) and isinstance(o, rdflib.URIRef):
                union(s, o)
        
        # Group by root
        components = defaultdict(set)
        for entity in entities:
            components[find(entity)].add(entity)
        
        return list(components.values())
