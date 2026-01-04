import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from typing import List, Dict, Optional
from collections import Counter
import rdflib
from rdflib import RDF


class SummaryGCN(nn.Module):
    """GCN model for FLUID summary graph classification."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
    
    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class PyGGraphBuilder:
    """Build PyTorch Geometric graph from FLUID summary."""
    
    def __init__(self, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim
    
    def build(
        self, 
        summary_graph: rdflib.Graph,
        payload: Optional[Dict[int, torch.Tensor]] = None,
        labels: Optional[Dict[int, any]] = None
    ) -> Data:
        """
        Build PyG Data object.
        
        Features: [structural_identity || semantic_embedding || has_payload_flag]
        """
        nodes = list(set(summary_graph.subjects()) | set(summary_graph.objects()))
        node_to_id = {n: i for i, n in enumerate(nodes)}
        
        # Build edge index
        edge_index = self._build_edges(summary_graph, node_to_id)
        
        # Build node features
        x = self._build_features(nodes, node_to_id, payload)
        
        # Build labels
        y = self._build_labels(nodes, node_to_id, labels) if labels else None
        
        return Data(x=x, edge_index=edge_index, y=y)
    
    def _build_edges(
        self, 
        graph: rdflib.Graph, 
        node_to_id: Dict
    ) -> torch.Tensor:
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
        payload: Optional[Dict[int, torch.Tensor]]
    ) -> torch.Tensor:
        """Build node feature matrix."""
        num_nodes = len(nodes)
        
        # Structural identity (one-hot)
        x_struct = torch.eye(num_nodes)
        
        # Semantic embeddings
        x_payload = torch.zeros(num_nodes, self.embedding_dim)
        has_payload = torch.zeros(num_nodes, 1)
        
        if payload:
            for node, node_id in node_to_id.items():
                # Extract summary ID from URI
                if str(node).startswith("summary:"):
                    summary_id = int(str(node).split(":")[-1])
                    if summary_id in payload:
                        x_payload[node_id] = payload[summary_id]
                        has_payload[node_id] = 1.0
        
        # Concatenate all features
        return torch.cat([x_struct, x_payload, has_payload], dim=1)
    
    def _build_labels(
        self, 
        nodes: List, 
        node_to_id: Dict,
        labels: Dict[int, any]
    ) -> torch.Tensor:
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


class LabelBuilder:
    """Build labels for FLUID summary nodes."""
    
    @staticmethod
    def build(
        summary_nodes: Dict[int, List], 
        original_graph: rdflib.Graph
    ) -> Dict[int, any]:
        """Build labels using majority voting from original graph."""
        labels = {}
        
        for summary_id, entities in summary_nodes.items():
            counter = Counter()
            for entity in entities:
                for _, _, type_class in original_graph.triples((entity, RDF.type, None)):
                    counter[type_class] += 1
            
            if counter:
                labels[summary_id] = counter.most_common(1)[0][0]
            else:
                labels[summary_id] = None
        
        return labels

