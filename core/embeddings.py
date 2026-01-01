import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
import random

class InferenceEmbedder(nn.Module):
    """TransE-based embedding model for inferred triples."""
    
    def __init__(self, n_entities: int, n_relations: int, dim: int = 32):
        super().__init__()
        self.entity_embeddings = nn.Embedding(n_entities, dim)
        self.relation_embeddings = nn.Embedding(n_relations, dim)
        
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
    
    def score(
        self, 
        heads: torch.Tensor, 
        relations: torch.Tensor, 
        tails: torch.Tensor
    ) -> torch.Tensor:
        """TransE scoring: ||h + r - t||"""
        h_emb = self.entity_embeddings(heads)
        r_emb = self.relation_embeddings(relations)
        t_emb = self.entity_embeddings(tails)
        return torch.norm(h_emb + r_emb - t_emb, p=2, dim=1)
    
    def forward(
        self, 
        heads: torch.Tensor, 
        relations: torch.Tensor, 
        tails: torch.Tensor
    ) -> torch.Tensor:
        return self.score(heads, relations, tails)


class EmbeddingTrainer:
    """Train embeddings on inferred triples."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.logger = IFLUIDLogger()
        self.model: Optional[InferenceEmbedder] = None
        self.entity_to_id: Dict = {}
        self.relation_to_id: Dict = {}
    
    def train(
        self, 
        inferred_triples: List[Tuple]
    ) -> Tuple[Optional[InferenceEmbedder], Dict]:
        """Train embeddings on inferred triples."""
        if not inferred_triples:
            self.logger.warning("No inferred triples to train on!")
            return None, {}
        
        # Build vocabularies
        entities, relations = self._build_vocabularies(inferred_triples)
        indexed_triples = self._index_triples(inferred_triples)
        
        self.logger.info(
            f"Training on {len(indexed_triples)} triples, "
            f"{len(entities)} entities, {len(relations)} relations"
        )
        
        # Initialize model
        self.model = InferenceEmbedder(
            len(entities), 
            len(relations), 
            self.config.dim
        )
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.learning_rate
        )
        
        # Training loop
        self._train_loop(indexed_triples, optimizer, list(range(len(entities))))
        
        return self.model, self.entity_to_id
    
    def _build_vocabularies(
        self, 
        triples: List[Tuple]
    ) -> Tuple[List, List]:
        """Build entity and relation vocabularies."""
        entities = list({x for s, p, o in triples for x in (s, o)})
        relations = list({p for _, p, _ in triples})
        
        self.entity_to_id = {e: i for i, e in enumerate(entities)}
        self.relation_to_id = {r: i for i, r in enumerate(relations)}
        
        return entities, relations
    
    def _index_triples(
        self, 
        triples: List[Tuple]
    ) -> List[Tuple[int, int, int]]:
        """Convert triples to integer indices."""
        return [
            (self.entity_to_id[s], self.relation_to_id[p], self.entity_to_id[o])
            for s, p, o in triples
        ]
    
    def _train_loop(
        self, 
        triples: List[Tuple[int, int, int]], 
        optimizer: torch.optim.Optimizer,
        all_entities: List[int]
    ) -> None:
        """Main training loop."""
        for epoch in range(self.config.epochs):
            random.shuffle(triples)
            epoch_loss = self._train_epoch(triples, optimizer, all_entities)
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch + 1}/{self.config.epochs}, Loss: {epoch_loss:.4f}")
    
    def _train_epoch(
        self, 
        triples: List[Tuple[int, int, int]], 
        optimizer: torch.optim.Optimizer,
        all_entities: List[int]
    ) -> float:
        """Train single epoch."""
        total_loss = 0.0
        num_batches = 0
        
        for batch_start in range(0, len(triples), self.config.batch_size):
            batch = triples[batch_start:batch_start + self.config.batch_size]
            loss = self._train_batch(batch, optimizer, all_entities)
            total_loss += loss
            num_batches += 1
        
        return total_loss / max(1, num_batches)
    
    def _train_batch(
        self, 
        batch: List[Tuple[int, int, int]], 
        optimizer: torch.optim.Optimizer,
        all_entities: List[int]
    ) -> float:
        """Train single batch."""
        # Generate positive and negative samples
        pos_samples, neg_samples = self._generate_samples(batch, all_entities)
        
        # Compute scores
        pos_scores = self.model(*pos_samples)
        neg_scores = self.model(*neg_samples)
        
        # Margin ranking loss
        loss = torch.relu(self.config.margin + pos_scores - neg_scores).mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Normalize embeddings
        with torch.no_grad():
            self.model.entity_embeddings.weight.data = torch.nn.functional.normalize(
                self.model.entity_embeddings.weight.data, p=2, dim=1
            )
            self.model.relation_embeddings.weight.data = torch.nn.functional.normalize(
                self.model.relation_embeddings.weight.data, p=2, dim=1
            )
        
        return loss.item()
    
    def _generate_samples(
        self, 
        batch: List[Tuple[int, int, int]], 
        all_entities: List[int]
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """Generate positive and negative samples."""
        h_pos, r_pos, t_pos = [], [], []
        h_neg, r_neg, t_neg = [], [], []
        
        for h, r, t in batch:
            for _ in range(self.config.neg_samples):
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


class PayloadBuilder:
    """Build embedding payload for FLUID summary nodes."""
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
    
    def build(
        self, 
        summary_nodes: Dict[int, List], 
        model: InferenceEmbedder, 
        entity_to_id: Dict
    ) -> Dict[int, torch.Tensor]:
        """Build payload by averaging entity embeddings per super-node."""
        payload = {}
        
        for summary_id, entities in summary_nodes.items():
            vectors = []
            for entity in entities:
                if entity in entity_to_id:
                    entity_id = entity_to_id[entity]
                    emb = model.entity_embeddings(torch.tensor(entity_id)).detach()
                    vectors.append(emb)
            
            if vectors:
                mean_vec = torch.stack(vectors).mean(dim=0)
                payload[summary_id] = torch.nn.functional.normalize(mean_vec, p=2, dim=0)
            else:
                payload[summary_id] = torch.zeros(self.embedding_dim)
        
        return payload
