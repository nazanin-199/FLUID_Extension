from typing import Dict, List, Tuple, Set, FrozenSet
from collections import defaultdict
import rdflib
from rdflib import RDF


class FLUIDSummarizer:
    """FLUID graph summarization."""
    
    def __init__(self):
        self.logger = IFLUIDLogger()
    
    def summarize(
        self, 
        graph: rdflib.Graph
    ) -> Tuple[rdflib.Graph, Dict[int, List], Dict]:
        """
        Create FLUID summary.
        
        Returns:
            - Summary graph
            - Summary nodes (super-node ID -> list of entities)
            - Node mapping (entity -> super-node ID)
        """
        descriptors = self._compute_descriptors(graph)
        summary_nodes = {i: nodes for i, nodes in enumerate(descriptors.values())}
        node_map = {
            v: i for i, nodes in summary_nodes.items() for v in nodes
        }
        
        summary_graph = self._build_summary_graph(graph, node_map)
        
        self.logger.info(f"Created {len(summary_nodes)} super-nodes from {len(node_map)} entities")
        return summary_graph, summary_nodes, node_map
    
    def _compute_descriptors(
        self, 
        graph: rdflib.Graph
    ) -> Dict[Tuple[FrozenSet, FrozenSet, FrozenSet], List]:
        """Compute FLUID descriptors for each vertex."""
        descriptors = defaultdict(list)
        vertices = set(graph.subjects()) | set(graph.objects())
        
        for v in vertices:
            if not isinstance(v, rdflib.URIRef):
                continue
            
            # Explicit types
            types = frozenset(
                o for _, _, o in graph.triples((v, RDF.type, None))
                if isinstance(o, rdflib.URIRef)
            )
            
            # Outgoing predicates (excluding rdf:type)
            out_predicates = frozenset(
                p for _, p, _ in graph.triples((v, None, None))
                if p != RDF.type
            )
            
            # Incoming predicates
            in_predicates = frozenset(
                p for _, p, _ in graph.triples((None, None, v))
            )
            
            descriptor = (types, out_predicates, in_predicates)
            descriptors[descriptor].append(v)
        
        return descriptors
    
    def _build_summary_graph(
        self, 
        graph: rdflib.Graph, 
        node_map: Dict
    ) -> rdflib.Graph:
        """Build summary graph from node mapping."""
        summary = rdflib.Graph()
        
        for s, p, o in graph:
            if (isinstance(s, rdflib.URIRef) and 
                isinstance(o, rdflib.URIRef) and
                s in node_map and o in node_map):
                
                summary.add((
                    rdflib.URIRef(f"summary:{node_map[s]}"),
                    p,
                    rdflib.URIRef(f"summary:{node_map[o]}")
                ))
        
        return summary
