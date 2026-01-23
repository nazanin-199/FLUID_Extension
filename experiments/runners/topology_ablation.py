import torch
import rdflib
import networkx as nx
from typing import Dict, List, Set, Optional
from collections import defaultdict
import numpy as np
from experiments.configs.ablation_configs import TopologyType


class TopologyAnalyzer:
    """Analyze graph topology for feature weighting."""
    
    def __init__(self, graph: rdflib.Graph):
        self.graph = graph
        self._nx_graph = None
        self._communities = None
        self._centrality = None
        self._hierarchy_depth = None
    
    def analyze(self) -> Dict[str, any]:
        """Compute all topology metrics."""
        self._build_networkx_graph()
        
        return {
            'communities': self._compute_communities(),
            'centrality': self._compute_centrality(),
            'hierarchy_depth': self._compute_hierarchy_depth(),
        }
    
    def _build_networkx_graph(self):
        """Convert RDF graph to NetworkX for analysis."""
        self._nx_graph = nx.Graph()
        
        for s, p, o in self.graph:
            if isinstance(s, rdflib.URIRef) and isinstance(o, rdflib.URIRef):
                self._nx_graph.add_edge(str(s), str(o))
    
    def _compute_communities(self) -> Dict:
        """Detect communities using Louvain algorithm."""
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(self._nx_graph)
            
            # Group nodes by community
            communities = defaultdict(list)
            for node, comm_id in partition.items():
                communities[comm_id].append(node)
            
            return {
                'partition': partition,
                'num_communities': len(communities),
                'modularity': community_louvain.modularity(partition, self._nx_graph)
            }
        except ImportError:
            # Fallback if python-louvain not installed
            return {'partition': {}, 'num_communities': 0, 'modularity': 0.0}
    
    def _compute_centrality(self) -> Dict:
        """Compute node centrality metrics."""
        degree_centrality = nx.degree_centrality(self._nx_graph)
        
        # For large graphs, use approximate betweenness
        if len(self._nx_graph) > 1000:
            betweenness = nx.betweenness_centrality(self._nx_graph, k=100)
        else:
            betweenness = nx.betweenness_centrality(self._nx_graph)
        
        return {
            'degree': degree_centrality,
            'betweenness': betweenness
        }
    
    def _compute_hierarchy_depth(self) -> Dict[str, int]:
        """Compute hierarchy depth for each node."""
        from rdflib import RDFS
        
        depths = {}
        
        # Build hierarchy
        child_to_parents = defaultdict(set)
        for child, _, parent in self.graph.triples((None, RDFS.subClassOf, None)):
            child_to_parents[str(child)].add(str(parent))
        
        # Compute depths via BFS
        for node in self._nx_graph.nodes():
            depth = self._compute_node_depth(node, child_to_parents)
            depths[node] = depth
        
        return depths
    
    def _compute_node_depth(self, node: str, child_to_parents: Dict, max_depth: int = 20) -> int:
        """Compute depth of a node in hierarchy."""
        if node not in child_to_parents:
            return 0
        
        visited = {node}
        queue = [(node, 0)]
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


class TopologyWeighter:
    """Apply topology-aware weighting to graph features."""
    
    def __init__(self, topology_type: TopologyType):
        self.topology_type = topology_type
    
    def compute_weights(
        self,
        summary_nodes: Dict[int, List],
        topology_metrics: Dict,
        config: Dict
    ) -> Dict[int, float]:
        """Compute weight for each super-node based on topology."""
        if self.topology_type == TopologyType.RAW:
            # No weighting - all nodes equal
            return {nid: 1.0 for nid in summary_nodes.keys()}
        
        elif self.topology_type == TopologyType.COMMUNITY_AWARE:
            return self._community_weights(summary_nodes, topology_metrics, config)
        
        elif self.topology_type == TopologyType.HUB_WEIGHTED:
            return self._hub_weights(summary_nodes, topology_metrics, config)
        
        elif self.topology_type == TopologyType.HIERARCHY_WEIGHTED:
            return self._hierarchy_weights(summary_nodes, topology_metrics, config)
        
        else:
            raise ValueError(f"Unknown topology type: {self.topology_type}")
    
    def _community_weights(
        self,
        summary_nodes: Dict[int, List],
        topology_metrics: Dict,
        config: Dict
    ) -> Dict[int, float]:
        """Weight based on community structure."""
        partition = topology_metrics['communities']['partition']
        weights = {}
        
        for nid, entities in summary_nodes.items():
            # Count community diversity
            communities = set()
            for entity in entities:
                entity_str = str(entity)
                if entity_str in partition:
                    communities.add(partition[entity_str])
            
            # More diverse communities = higher weight
            diversity = len(communities)
            weights[nid] = 1.0 + config.get('community_weight', 1.0) * np.log1p(diversity)
        
        return self._normalize_weights(weights)
    
    def _hub_weights(
        self,
        summary_nodes: Dict[int, List],
        topology_metrics: Dict,
        config: Dict
    ) -> Dict[int, float]:
        """Weight based on hub centrality."""
        centrality = topology_metrics['centrality']['degree']
        weights = {}
        
        for nid, entities in summary_nodes.items():
            # Average centrality of entities
            centralities = []
            for entity in entities:
                entity_str = str(entity)
                if entity_str in centrality:
                    centralities.append(centrality[entity_str])
            
            if centralities:
                avg_centrality = np.mean(centralities)
                weights[nid] = 1.0 + config.get('hub_weight', 1.0) * avg_centrality
            else:
                weights[nid] = 1.0
        
        return self._normalize_weights(weights)
    
    def _hierarchy_weights(
        self,
        summary_nodes: Dict[int, List],
        topology_metrics: Dict,
        config: Dict
    ) -> Dict[int, float]:
        """Weight based on hierarchy depth."""
        hierarchy_depth = topology_metrics['hierarchy_depth']
        weights = {}
        
        for nid, entities in summary_nodes.items():
            # Average hierarchy depth
            depths = []
            for entity in entities:
                entity_str = str(entity)
                if entity_str in hierarchy_depth:
                    depths.append(hierarchy_depth[entity_str])
            
            if depths:
                avg_depth = np.mean(depths)
                weights[nid] = 1.0 + config.get('hierarchy_weight', 1.0) * np.log1p(avg_depth)
            else:
                weights[nid] = 1.0
        
        return self._normalize_weights(weights)
    
    def _normalize_weights(self, weights: Dict[int, float]) -> Dict[int, float]:
        """Normalize weights to [0.5, 2.0] range."""
        if not weights:
            return weights
        
        values = list(weights.values())
        min_val = min(values)
        max_val = max(values)
        
        if max_val - min_val < 1e-6:
            return {k: 1.0 for k in weights.keys()}
        
        # Normalize to [0, 1] then scale to [0.5, 2.0]
        normalized = {}
        for k, v in weights.items():
            norm = (v - min_val) / (max_val - min_val)
            normalized[k] = 0.5 + 1.5 * norm
        
        return normalized


class TopologyAwarePyGBuilder:
    """Build PyG graphs with topology-aware features."""
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
    
    def build(
        self,
        summary_graph: rdflib.Graph,
        payload: Optional[Dict[int, torch.Tensor]],
        labels: Optional[Dict[int, any]],
        topology_weights: Optional[Dict[int, float]] = None
    ):
        """Build PyG Data with optional topology weighting."""
        from torch_geometric.data import Data
        
        nodes = list(set(summary_graph.subjects()) | set(summary_graph.objects()))
        node_to_id = {n: i for i, n in enumerate(nodes)}
        
        # Build edge index
        edge_index = self._build_edges(summary_graph, node_to_id)
        
        # Build node features with topology weighting
        x = self._build_features(nodes, node_to_id, payload, topology_weights)
        
        # Build labels
        y = self._build_labels(nodes, node_to_id, labels) if labels else None
        
        return Data(x=x, edge_index=edge_index, y=y)
    
    def _build_edges(self, graph: rdflib.Graph, node_to_id: Dict) -> torch.Tensor:
        """Build edge index tensor."""
        edges = []
        for s, p, o in graph:
            if s in node_to_id and o in node_to_id:
                edges.append([node_to_id[s], node_to_id[o]])
        
        if edges:
            return torch.tensor(edges).t().contiguous()
        return torch.empty((2, 0), dtype=torch.long)
    
    def _build_features(
        self,
        nodes: List,
        node_to_id: Dict,
        payload: Optional[Dict[int, torch.Tensor]],
        topology_weights: Optional[Dict[int, float]]
    ) -> torch.Tensor:
        """Build node feature matrix with optional topology weighting."""
        num_nodes = len(nodes)
        
        # Structural identity (one-hot)
        x_struct = torch.eye(num_nodes)
        
        # Semantic embeddings
        x_payload = torch.zeros(num_nodes, self.embedding_dim)
        has_payload = torch.zeros(num_nodes, 1)
        
        # Topology weights
        x_topology = torch.ones(num_nodes, 1)
        
        if payload:
            for node, node_id in node_to_id.items():
                if str(node).startswith("summary:"):
                    summary_id = int(str(node).split(":")[-1])
                    
                    if summary_id in payload:
                        x_payload[node_id] = payload[summary_id]
                        has_payload[node_id] = 1.0
                    
                    if topology_weights and summary_id in topology_weights:
                        x_topology[node_id] = topology_weights[summary_id]
        
        # Concatenate: [structural | semantic | has_payload | topology_weight]
        return torch.cat([x_struct, x_payload, has_payload, x_topology], dim=1)
    
    def _build_labels(self, nodes: List, node_to_id: Dict, labels: Dict[int, any]) -> torch.Tensor:
        """Build label tensor."""
        label_to_idx = {lbl: idx for idx, lbl in enumerate(set(labels.values()))}
        
        y_list = []
        for node, node_id in node_to_id.items():
            if str(node).startswith("summary:"):
                summary_id = int(str(node).split(":")[-1])
                label = labels.get(summary_id)
                y_list.append(label_to_idx.get(label, 0))
            else:
                y_list.append(0)
        
        return torch.tensor(y_list, dtype=torch.long)
