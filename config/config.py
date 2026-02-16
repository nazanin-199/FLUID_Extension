from dataclasses import dataclass, field
from typing import Dict, Any
import json


@dataclass
class EmbeddingConfig:
    """Configuration for embedding training."""
    dim: int = 32
    epochs: int = 40
    margin: float = 1.0
    neg_samples: int = 5
    batch_size: int = 128
    learning_rate: float = 0.01


@dataclass
class GNNConfig:
    """Configuration for GNN training."""
    hidden_dim: int = 32
    epochs: int = 200
    learning_rate: float = 0.01
    test_split: float = 0.3
    random_seed: int = 42


@dataclass
class PolicyConfig:
    """Configuration for adaptive extraction policy."""
    sameas_threshold: float = 0.001
    class_property_ratio_threshold: float = 5.0
    log_triples_threshold: float = 5.0
    low_depth_max: int = 10
    high_depth_max: int = 3
    depth_threshold: float = 5.0


@dataclass
class IFLUIDConfig:
    """Main configuration container."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    verbose: bool = True

    @classmethod
    def from_json(cls, path: str) -> 'IFLUIDConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            embedding=EmbeddingConfig(**data.get('embedding', {})),
            gnn=GNNConfig(**data.get('gnn', {})),
            policy=PolicyConfig(**data.get('policy', {})),
            verbose=data.get('verbose', True)
        )

    def to_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        data = {
            'embedding': self.embedding.__dict__,
            'gnn': self.gnn.__dict__,
            'policy': self.policy.__dict__,
            'verbose': self.verbose
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
