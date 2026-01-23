import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
import random
from experiments.configs.ablation_configs import EmbeddingType


class BaseEmbeddingModel(nn.Module):
    """Base class for embedding models."""
    
    def __init__(self, n_entities: int, n_relations: int, dim: int):
        super().__init__()
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.dim = dim
    
    def score(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        """Score triples. Lower is better."""
        raise NotImplementedError
    
    def forward(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        return self.score(heads, relations, tails)


class TransEModel(BaseEmbeddingModel):
    """TransE: Translating Embeddings (current baseline)."""
    
    def __init__(self, n_entities: int, n_relations: int, dim: int = 32):
        super().__init__(n_entities, n_relations, dim)
        self.entity_embeddings = nn.Embedding(n_entities, dim)
        self.relation_embeddings = nn.Embedding(n_relations, dim)
        
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
    
    def score(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        """TransE scoring: ||h + r - t||"""
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)
        return torch.norm(h + r - t, p=2, dim=1)


class DistMultModel(BaseEmbeddingModel):
    """DistMult: Handles symmetric relations better."""
    
    def __init__(self, n_entities: int, n_relations: int, dim: int = 32):
        super().__init__(n_entities, n_relations, dim)
        self.entity_embeddings = nn.Embedding(n_entities, dim)
        self.relation_embeddings = nn.Embedding(n_relations, dim)
        
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
    
    def score(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        """DistMult scoring: -(<h, r, t>) (negative because lower is better)"""
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)
        
        # Element-wise product and sum
        scores = torch.sum(h * r * t, dim=1)
        return -scores  # Negate because we want lower scores to be better


class ComplExModel(BaseEmbeddingModel):
    """ComplEx: Handles asymmetric relations with complex numbers."""
    
    def __init__(self, n_entities: int, n_relations: int, dim: int = 32):
        super().__init__(n_entities, n_relations, dim)
        
        # Split dimension for real and imaginary parts
        self.entity_embeddings_real = nn.Embedding(n_entities, dim)
        self.entity_embeddings_imag = nn.Embedding(n_entities, dim)
        self.relation_embeddings_real = nn.Embedding(n_relations, dim)
        self.relation_embeddings_imag = nn.Embedding(n_relations, dim)
        
        nn.init.xavier_uniform_(self.entity_embeddings_real.weight.data)
        nn.init.xavier_uniform_(self.entity_embeddings_imag.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings_real.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings_imag.weight.data)
    
    def score(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        """ComplEx scoring using Hermitian dot product."""
        h_real = self.entity_embeddings_real(heads)
        h_imag = self.entity_embeddings_imag(heads)
        r_real = self.relation_embeddings_real(relations)
        r_imag = self.relation_embeddings_imag(relations)
        t_real = self.entity_embeddings_real(tails)
        t_imag = self.entity_embeddings_imag(tails)
        
        # ComplEx scoring function
        score = torch.sum(
            h_real * r_real * t_real +
            h_real * r_imag * t_imag +
            h_imag * r_real * t_imag -
            h_imag * r_imag * t_real,
            dim=1
        )
        
        return -score  # Negate for consistency


class RandomEmbeddingModel(BaseEmbeddingModel):
    """Random embeddings (sanity check)."""
    
    def __init__(self, n_entities: int, n_relations: int, dim: int = 32):
        super().__init__(n_entities, n_relations, dim)
        self.entity_embeddings = nn.Embedding(n_entities, dim)
        self.relation_embeddings = nn.Embedding(n_relations, dim)
        
        # Random initialization (no training)
        nn.init.normal_(self.entity_embeddings.weight.data, mean=0, std=0.1)
        nn.init.normal_(self.relation_embeddings.weight.data, mean=0, std=0.1)
    
    def score(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        """Random scoring (just for consistency)."""
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)
        return torch.norm(h + r - t, p=2, dim=1)


class EmbeddingTrainerAblation:
    """Unified trainer for different embedding models."""
    
    def __init__(self, embedding_type: EmbeddingType, config: Dict):
        self.embedding_type = embedding_type
        self.config = config
        self.model: Optional[BaseEmbeddingModel] = None
        self.entity_to_id: Dict = {}
        self.relation_to_id: Dict = {}
    
    def train(self, inferred_triples: List[Tuple]) -> Tuple[Optional[BaseEmbeddingModel], Dict]:
        """Train embedding model on inferred triples."""
        if self.embedding_type == EmbeddingType.NONE:
            # No embeddings - FLUID baseline
            return None, {}
        
        if not inferred_triples:
            return None, {}
        
        # Build vocabularies
        entities, relations = self._build_vocabularies(inferred_triples)
        indexed_triples = self._index_triples(inferred_triples)
        
        # Initialize model based on type
        self.model = self._create_model(len(entities), len(relations))
        
        if self.embedding_type == EmbeddingType.RANDOM:
            # Don't train random embeddings
            return self.model, self.entity_to_id
        
        # Train model
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self._train_loop(indexed_triples, optimizer, list(range(len(entities))))
        
        return self.model, self.entity_to_id
    
    def _create_model(self, n_entities: int, n_relations: int) -> BaseEmbeddingModel:
        """Create model based on embedding type."""
        dim = self.config['dim']
        
        if self.embedding_type == EmbeddingType.TRANSE:
            return TransEModel(n_entities, n_relations, dim)
        elif self.embedding_type == EmbeddingType.DISTMULT:
            return DistMultModel(n_entities, n_relations, dim)
        elif self.embedding_type == EmbeddingType.COMPLEX:
            return ComplExModel(n_entities, n_relations, dim)
        elif self.embedding_type == EmbeddingType.RANDOM:
            return RandomEmbeddingModel(n_entities, n_relations, dim)
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")
    
    def _build_vocabularies(self, triples: List[Tuple]) -> Tuple[List, List]:
        """Build entity and relation vocabularies."""
        entities = list({x for s, p, o in triples for x in (s, o)})
        relations = list({p for _, p, _ in triples})
        
        self.entity_to_id = {e: i for i, e in enumerate(entities)}
        self.relation_to_id = {r: i for i, r in enumerate(relations)}
        
        return entities, relations
    
    def _index_triples(self, triples: List[Tuple]) -> List[Tuple[int, int, int]]:
        """Convert triples to integer indices."""
        return [
            (self.entity_to_id[s], self.relation_to_id[p], self.entity_to_id[o])
            for s, p, o in triples
        ]
    
    def _train_loop(self, triples: List[Tuple[int, int, int]], optimizer, all_entities: List[int]):
        """Main training loop."""
        for epoch in range(self.config['epochs']):
            random.shuffle(triples)
            epoch_loss = self._train_epoch(triples, optimizer, all_entities)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{self.config['epochs']}, Loss: {epoch_loss:.4f}")
    
    def _train_epoch(self, triples: List[Tuple[int, int, int]], optimizer, all_entities: List[int]) -> float:
        """Train single epoch."""
        total_loss = 0.0
        num_batches = 0
        batch_size = self.config['batch_size']
        
        for batch_start in range(0, len(triples), batch_size):
            batch = triples[batch_start:batch_start + batch_size]
            loss = self._train_batch(batch, optimizer, all_entities)
            total_loss += loss
            num_batches += 1
        
        return total_loss / max(1, num_batches)
    
    def _train_batch(self, batch: List[Tuple[int, int, int]], optimizer, all_entities: List[int]) -> float:
        """Train single batch."""
        pos_samples, neg_samples = self._generate_samples(batch, all_entities)
        
        # Compute scores
        pos_scores = self.model(*pos_samples)
        neg_scores = self.model(*neg_samples)
        
        # Margin ranking loss
        loss = torch.relu(self.config['margin'] + pos_scores - neg_scores).mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Normalize embeddings (only for TransE and DistMult)
        if self.embedding_type in [EmbeddingType.TRANSE, EmbeddingType.DISTMULT]:
            with torch.no_grad():
                self.model.entity_embeddings.weight.data = torch.nn.functional.normalize(
                    self.model.entity_embeddings.weight.data, p=2, dim=1
                )
                self.model.relation_embeddings.weight.data = torch.nn.functional.normalize(
                    self.model.relation_embeddings.weight.data, p=2, dim=1
                )
        
        return loss.item()
    
    def _generate_samples(self, batch: List[Tuple[int, int, int]], all_entities: List[int]) -> Tuple:
        """Generate positive and negative samples."""
        h_pos, r_pos, t_pos = [], [], []
        h_neg, r_neg, t_neg = [], [], []
        
        for h, r, t in batch:
            for _ in range(self.config['neg_samples']):
                h_pos.append(h)
                r_pos.append(r)
                t_pos.append(t)
                
                # Corrupt head or tail
                if random.random() < 0.5:
                    h_corrupt = h
                    while h_corrupt == h:
                        h_corrupt = random.choice(all_entities)
                    h_neg.append(h_corrupt)
                    r_neg.append(r)
                    t_neg.append(t)
                else:
                    t_corrupt = t
                    while t_corrupt == t:
                        t_corrupt = random.choice(all_entities)
                    h_neg.append(h)
                    r_neg.append(r)
                    t_neg.append(t_corrupt)
        
        pos = (
            torch.tensor(h_pos, dtype=torch.long),
            torch.tensor(r_pos, dtype=torch.long),
            torch.tensor(t_pos, dtype=torch.long)
        )
        neg = (
            torch.tensor(h_neg, dtype=torch.long),
            torch.tensor(r_neg, dtype=torch.long),
            torch.tensor(t_neg, dtype=torch.long)
        )
        
        return pos, neg


class PayloadBuilderAblation:
    """Build payload with different embedding types."""
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
    
    def build(
        self, 
        summary_nodes: Dict[int, List],
        model: Optional[BaseEmbeddingModel],
        entity_to_id: Dict,
        embedding_type: EmbeddingType
    ) -> Optional[Dict[int, torch.Tensor]]:
        """Build payload based on embedding type."""
        if embedding_type == EmbeddingType.NONE or model is None:
            return None
        
        payload = {}
        
        for summary_id, entities in summary_nodes.items():
            vectors = []
            
            for entity in entities:
                if entity in entity_to_id:
                    entity_id = entity_to_id[entity]
                    
                    # Extract embeddings based on model type
                    if isinstance(model, (TransEModel, DistMultModel, RandomEmbeddingModel)):
                        emb = model.entity_embeddings(torch.tensor(entity_id)).detach()
                    elif isinstance(model, ComplExModel):
                        # Concatenate real and imaginary parts
                        emb_real = model.entity_embeddings_real(torch.tensor(entity_id)).detach()
                        emb_imag = model.entity_embeddings_imag(torch.tensor(entity_id)).detach()
                        emb = torch.cat([emb_real, emb_imag])
                    else:
                        raise ValueError(f"Unknown model type: {type(model)}")
                    
                    vectors.append(emb)
            
            if vectors:
                mean_vec = torch.stack(vectors).mean(dim=0)
                payload[summary_id] = torch.nn.functional.normalize(mean_vec, p=2, dim=0)
            else:
                payload[summary_id] = torch.zeros(self.embedding_dim)
        
        return payload
