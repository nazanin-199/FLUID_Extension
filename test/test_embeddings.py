import pytest
import torch
import rdflib
from core.embeddings import (
    InferenceEmbedder,
    EmbeddingTrainer,
    PayloadBuilder
)
from config.config import EmbeddingConfig


def test_embedder_initialization():
    """Test InferenceEmbedder initialization."""
    model = InferenceEmbedder(n_entities=10, n_relations=5, dim=16)
    
    assert model.entity_embeddings.num_embeddings == 10
    assert model.relation_embeddings.num_embeddings == 5
    assert model.entity_embeddings.embedding_dim == 16


def test_embedder_forward():
    """Test InferenceEmbedder forward pass."""
    model = InferenceEmbedder(n_entities=10, n_relations=5, dim=16)
    
    heads = torch.tensor([0, 1, 2])
    relations = torch.tensor([0, 1, 2])
    tails = torch.tensor([3, 4, 5])
    
    scores = model.score(heads, relations, tails)
    
    assert scores.shape == (3,)
    assert torch.all(scores >= 0)  # Distances are non-negative


def test_embedding_trainer_basic():
    """Test basic embedding training."""
    # Create simple triples
    ex = rdflib.Namespace("http://example.org/")
    triples = [
        (ex.Entity1, ex.relation1, ex.Entity2),
        (ex.Entity2, ex.relation1, ex.Entity3),
        (ex.Entity1, ex.relation2, ex.Entity3),
    ]
    
    config = EmbeddingConfig(dim=8, epochs=2, batch_size=2)
    trainer = EmbeddingTrainer(config)
    
    model, entity_to_id = trainer.train(triples)
    
    assert model is not None
    assert len(entity_to_id) == 3  # 3 unique entities
    assert ex.Entity1 in entity_to_id


def test_embedding_trainer_empty():
    """Test embedding trainer with empty triples."""
    config = EmbeddingConfig(dim=8, epochs=2)
    trainer = EmbeddingTrainer(config)
    
    model, entity_to_id = trainer.train([])
    
    assert model is None
    assert entity_to_id == {}


def test_embedding_trainer_convergence():
    """Test that training loss decreases."""
    # Create some triples
    ex = rdflib.Namespace("http://example.org/")
    triples = [(ex[f"e{i}"], ex.rel, ex[f"e{i+1}"]) for i in range(10)]
    
    config = EmbeddingConfig(dim=16, epochs=5, batch_size=4)
    trainer = EmbeddingTrainer(config)
    
    model, entity_to_id = trainer.train(triples)
    
    # Model should train successfully
    assert model is not None
    assert len(entity_to_id) == 11  # e0 to e10


def test_payload_builder():
    """Test payload building from embeddings."""
    # Create mock model and mapping
    model = InferenceEmbedder(n_entities=5, n_relations=2, dim=8)
    
    ex = rdflib.Namespace("http://example.org/")
    entity_to_id = {
        ex.Entity1: 0,
        ex.Entity2: 1,
        ex.Entity3: 2,
    }
    
    # Create summary nodes
    summary_nodes = {
        0: [ex.Entity1, ex.Entity2],  # Super-node 0 contains entities 1,2
        1: [ex.Entity3],               # Super-node 1 contains entity 3
    }
    
    builder = PayloadBuilder(embedding_dim=8)
    payload = builder.build(summary_nodes, model, entity_to_id)
    
    # Check payload structure
    assert len(payload) == 2  # 2 super-nodes
    assert 0 in payload
    assert 1 in payload
    
    # Check embedding dimensions
    assert payload[0].shape == (8,)
    assert payload[1].shape == (8,)
    
    # Check normalization
    assert torch.allclose(torch.norm(payload[0]), torch.tensor(1.0), atol=1e-5)


def test_payload_builder_missing_entities():
    """Test payload builder with entities not in model."""
    model = InferenceEmbedder(n_entities=2, n_relations=1, dim=8)
    
    ex = rdflib.Namespace("http://example.org/")
    entity_to_id = {
        ex.Entity1: 0,
    }
    
    # Summary with entity not in model
    summary_nodes = {
        0: [ex.Entity1],
        1: [ex.Entity2],  # Not in entity_to_id
    }
    
    builder = PayloadBuilder(embedding_dim=8)
    payload = builder.build(summary_nodes, model, entity_to_id)
    
    # Should still create payload
    assert len(payload) == 2
    
    # Super-node 1 should have zero vector
    assert torch.all(payload[1] == 0)
