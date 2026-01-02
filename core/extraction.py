from config.config import PolicyConfig
from typing import Dict, Any
from collections import defaultdict
import rdflib
from rdflib import RDF, RDFS, OWL


class AdaptivePolicy:
    """Determine extraction policy based on KG profile."""
    
    def __init__(self, config: PolicyConfig):
        self.config = config
    
    def determine(self, profile: Dict[str, float]) -> Dict[str, Any]:
        """Determine policy based on normalized profile."""
        return {
            'subClassOf': True,
            'subPropertyOf': True,
            'sameAs': profile['sameas_density'] > self.config.sameas_threshold,
            'domain_range': profile['class_property_ratio'] < self.config.class_property_ratio_threshold,
            'rule_mining': profile['log_triples'] > self.config.log_triples_threshold,
            'max_depth': (
                self.config.high_depth_max 
                if profile['avg_class_depth'] > self.config.depth_threshold 
                else self.config.low_depth_max
            )
        }


class SymbolicExtractor:
    """Extract symbolic inferences from knowledge graph."""
    
    def __init__(self, policy: Dict[str, Any]):
        self.policy = policy
        self.logger = IFLUIDLogger()
    
    def extract(self, graph: rdflib.Graph) -> rdflib.Graph:
        """Apply selective symbolic extraction based on policy."""
        enriched_graph = rdflib.Graph()
        
        # Copy original triples
        for triple in graph:
            enriched_graph.add(triple)
        
        # Apply extraction rules
        if self.policy.get('subClassOf', False):
            self._extract_subclass_transitivity(graph, enriched_graph)
        
        if self.policy.get('subPropertyOf', False):
            self._extract_subproperty_transitivity(graph, enriched_graph)
        
        if self.policy.get('domain_range', False):
            self._extract_domain_range(graph, enriched_graph)
        
        if self.policy.get('sameAs', False):
            self._consolidate_sameas(graph, enriched_graph)
        
        self.logger.info(f"Extracted {len(enriched_graph) - len(graph)} new triples")
        return enriched_graph
    
    def _extract_subclass_transitivity(
        self, 
        graph: rdflib.Graph, 
        enriched: rdflib.Graph
    ) -> None:
        """Extract transitive subClassOf relationships."""
        subclass_map = defaultdict(set)
        
        for c, _, p in graph.triples((None, RDFS.subClassOf, None)):
            subclass_map[c].add(p)
        
        for c in list(subclass_map.keys()):
            for p in list(subclass_map[c]):
                for pp in subclass_map.get(p, []):
                    enriched.add((c, RDFS.subClassOf, pp))
    
    def _extract_subproperty_transitivity(
        self, 
        graph: rdflib.Graph, 
        enriched: rdflib.Graph
    ) -> None:
        """Extract transitive subPropertyOf relationships."""
        subprop_map = defaultdict(set)
        
        for p, _, q in graph.triples((None, RDFS.subPropertyOf, None)):
            subprop_map[p].add(q)
        
        for p in list(subprop_map.keys()):
            for q in list(subprop_map[p]):
                for qq in subprop_map.get(q, []):
                    enriched.add((p, RDFS.subPropertyOf, qq))
    
    def _extract_domain_range(
        self, 
        graph: rdflib.Graph, 
        enriched: rdflib.Graph
    ) -> None:
        """Propagate domain and range constraints."""
        for p, _, C in graph.triples((None, RDFS.domain, None)):
            for s, _, _ in graph.triples((None, p, None)):
                enriched.add((s, RDF.type, C))
    
    def _consolidate_sameas(
        self, 
        graph: rdflib.Graph, 
        enriched: rdflib.Graph
    ) -> None:
        """Consolidate sameAs equivalence classes using union-find."""
        parent = {}
        
        def find(x):
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            parent[find(x)] = find(y)
        
        # Build equivalence classes
        for x, _, y in graph.triples((None, OWL.sameAs, None)):
            union(x, y)
        
        # Create representative mapping
        representatives = {x: find(x) for x in parent}
        
        # Replace entities with representatives
        for s, p, o in list(enriched):
            if s in representatives or o in representatives:
                enriched.remove((s, p, o))
                new_s = representatives.get(s, s)
                new_o = representatives.get(o, o)
                enriched.add((new_s, p, new_o))
