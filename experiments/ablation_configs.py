from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from enum import Enum
import json


class PolicyType(Enum):
    """Policy ablation variants."""
    FULL_ADAPTIVE = "full_adaptive"
    NO_SAMEAS = "no_sameas"
    NO_DOMAIN_RANGE = "no_domain_range"
    TAXONOMY_ONLY = "taxonomy_only"
    FIXED_POLICY = "fixed_policy"
    RANDOM_POLICY = "random_policy"


class EmbeddingType(Enum):
    """Embedding ablation variants."""
    TRANSE = "transe"           # Current baseline
    DISTMULT = "distmult"       # Symmetric relations
    COMPLEX = "complex"         # Asymmetric relations
    RANDOM = "random"           # Sanity check
    NONE = "none"               # FLUID baseline (no embeddings)


class TopologyType(Enum):
    """Graph topology ablation variants."""
    RAW = "raw"                           # Baseline - no modification
    COMMUNITY_AWARE = "community_aware"   # Modular structure emphasis
    HUB_WEIGHTED = "hub_weighted"         # Scale-free property emphasis
    HIERARCHY_WEIGHTED = "hierarchy_weighted"  # Deep hierarchical graphs


@dataclass
class PolicyConfig:
    policy_type: PolicyType
    subClassOf: Optional[bool] = None
    subPropertyOf: Optional[bool] = None
    sameAs: Optional[bool] = None
    domain_range: Optional[bool] = None
    rule_mining: Optional[bool] = None
    adaptive: bool = True
    
    @classmethod
    def from_type(cls, policy_type: PolicyType) -> 'PolicyConfig':
        """Create config from policy type."""
        if policy_type == PolicyType.FULL_ADAPTIVE:
            return cls(
                policy_type=policy_type,
                adaptive=True,
                subClassOf=None,
                subPropertyOf=None,
                sameAs=None,
                domain_range=None,
                rule_mining=None
            )
        
        elif policy_type == PolicyType.NO_SAMEAS:
            return cls(
                policy_type=policy_type,
                adaptive=True,
                sameAs=False,  
                subClassOf=None,
                subPropertyOf=None,
                domain_range=None,
                rule_mining=None
            )
        
        elif policy_type == PolicyType.NO_DOMAIN_RANGE:
            return cls(
                policy_type=policy_type,
                adaptive=True,
                domain_range=False,  
                subClassOf=None,
                subPropertyOf=None,
                sameAs=None,
                rule_mining=None
            )
        
        elif policy_type == PolicyType.TAXONOMY_ONLY:
            return cls(
                policy_type=policy_type,
                adaptive=False,
                subClassOf=True,
                subPropertyOf=True,
                sameAs=False,
                domain_range=False,
                rule_mining=False
            )
        
        elif policy_type == PolicyType.FIXED_POLICY:
            return cls(
                policy_type=policy_type,
                adaptive=False,
                subClassOf=True,
                subPropertyOf=True,
                sameAs=True,
                domain_range=True,
                rule_mining=False
            )
        
        elif policy_type == PolicyType.RANDOM_POLICY:
            import random
            return cls(
                policy_type=policy_type,
                adaptive=False,
                subClassOf=random.choice([True, False]),
                subPropertyOf=random.choice([True, False]),
                sameAs=random.choice([True, False]),
                domain_range=random.choice([True, False]),
                rule_mining=False
            )
        
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")


@dataclass
class EmbeddingConfig:
    embedding_type: EmbeddingType
    dim: int = 32
    epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 0.01
    margin: float = 1.0
    neg_samples: int = 5
    
    @classmethod
    def from_type(cls, embedding_type: EmbeddingType, dim: int = 32) -> 'EmbeddingConfig':
        """Create config from embedding type."""
        return cls(
            embedding_type=embedding_type,
            dim=dim,
            epochs=40 if embedding_type != EmbeddingType.NONE else 0,
            batch_size=128,
            learning_rate=0.01,
            margin=1.0,
            neg_samples=5
        )


@dataclass
class TopologyConfig:
    topology_type: TopologyType
    community_weight: float = 1.0
    hub_weight: float = 1.0
    hierarchy_weight: float = 1.0
    
    @classmethod
    def from_type(cls, topology_type: TopologyType) -> 'TopologyConfig':
        """Create config from topology type."""
        if topology_type == TopologyType.RAW:
            return cls(
                topology_type=topology_type,
                community_weight=1.0,
                hub_weight=1.0,
                hierarchy_weight=1.0
            )
        
        elif topology_type == TopologyType.COMMUNITY_AWARE:
            return cls(
                topology_type=topology_type,
                community_weight=2.0,  # Emphasize communities
                hub_weight=1.0,
                hierarchy_weight=1.0
            )
        
        elif topology_type == TopologyType.HUB_WEIGHTED:
            return cls(
                topology_type=topology_type,
                community_weight=1.0,
                hub_weight=2.0,  # Emphasize hubs
                hierarchy_weight=1.0
            )
        
        elif topology_type == TopologyType.HIERARCHY_WEIGHTED:
            return cls(
                topology_type=topology_type,
                community_weight=1.0,
                hub_weight=1.0,
                hierarchy_weight=2.0  # Emphasize hierarchy
            )
        
        else:
            raise ValueError(f"Unknown topology type: {topology_type}")


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    experiment_id: str
    dataset_name: str
    dataset_path: str
    
    # Ablation configurations
    policy_config: PolicyConfig
    embedding_config: EmbeddingConfig
    topology_config: TopologyConfig
    
    # GNN configuration
    gnn_hidden_dim: int = 32
    gnn_epochs: int = 200
    gnn_learning_rate: float = 0.01
    test_split: float = 0.3
    
    # Runtime
    random_seed: int = 42
    verbose: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'experiment_id': self.experiment_id,
            'dataset_name': self.dataset_name,
            'dataset_path': self.dataset_path,
            'policy': {
                'type': self.policy_config.policy_type.value,
                **{k: v for k, v in asdict(self.policy_config).items() 
                   if k != 'policy_type'}
            },
            'embedding': {
                'type': self.embedding_config.embedding_type.value,
                **{k: v for k, v in asdict(self.embedding_config).items() 
                   if k != 'embedding_type'}
            },
            'topology': {
                'type': self.topology_config.topology_type.value,
                **{k: v for k, v in asdict(self.topology_config).items() 
                   if k != 'topology_type'}
            },
            'gnn': {
                'hidden_dim': self.gnn_hidden_dim,
                'epochs': self.gnn_epochs,
                'learning_rate': self.gnn_learning_rate,
                'test_split': self.test_split
            },
            'random_seed': self.random_seed,
            'verbose': self.verbose
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary."""
        policy_data = data['policy']
        embedding_data = data['embedding']
        topology_data = data['topology']
        gnn_data = data['gnn']
        
        return cls(
            experiment_id=data['experiment_id'],
            dataset_name=data['dataset_name'],
            dataset_path=data['dataset_path'],
            policy_config=PolicyConfig(
                policy_type=PolicyType(policy_data['type']),
                **{k: v for k, v in policy_data.items() if k != 'type'}
            ),
            embedding_config=EmbeddingConfig(
                embedding_type=EmbeddingType(embedding_data['type']),
                **{k: v for k, v in embedding_data.items() if k != 'type'}
            ),
            topology_config=TopologyConfig(
                topology_type=TopologyType(topology_data['type']),
                **{k: v for k, v in topology_data.items() if k != 'type'}
            ),
            gnn_hidden_dim=gnn_data['hidden_dim'],
            gnn_epochs=gnn_data['epochs'],
            gnn_learning_rate=gnn_data['learning_rate'],
            test_split=gnn_data['test_split'],
            random_seed=data['random_seed'],
            verbose=data['verbose']
        )


class AblationConfigGenerator:    
    @staticmethod
    def generate_policy_ablation(
        dataset_name: str,
        dataset_path: str,
        base_seed: int = 42
    ) -> List[ExperimentConfig]:
        """Generate configs for policy ablation."""
        configs = []
        
        for policy_type in PolicyType:
            for run in range(3):  # 3 runs per variant
                configs.append(ExperimentConfig(
                    experiment_id=f"{dataset_name}_{policy_type.value}_run{run}",
                    dataset_name=dataset_name,
                    dataset_path=dataset_path,
                    policy_config=PolicyConfig.from_type(policy_type),
                    embedding_config=EmbeddingConfig.from_type(EmbeddingType.TRANSE),
                    topology_config=TopologyConfig.from_type(TopologyType.RAW),
                    random_seed=base_seed + run
                ))
        
        return configs
    
    @staticmethod
    def generate_embedding_ablation(
        dataset_name: str,
        dataset_path: str,
        base_seed: int = 42
    ) -> List[ExperimentConfig]:
        """Generate configs for embedding ablation."""
        configs = []
        
        for embedding_type in EmbeddingType:
            for run in range(3):
                configs.append(ExperimentConfig(
                    experiment_id=f"{dataset_name}_{embedding_type.value}_run{run}",
                    dataset_name=dataset_name,
                    dataset_path=dataset_path,
                    policy_config=PolicyConfig.from_type(PolicyType.FULL_ADAPTIVE),
                    embedding_config=EmbeddingConfig.from_type(embedding_type),
                    topology_config=TopologyConfig.from_type(TopologyType.RAW),
                    random_seed=base_seed + run
                ))
        
        return configs
    
    @staticmethod
    def generate_topology_ablation(
        dataset_name: str,
        dataset_path: str,
        base_seed: int = 42
    ) -> List[ExperimentConfig]:
        """Generate configs for topology ablation."""
        configs = []
        
        for topology_type in TopologyType:
            for run in range(3):
                configs.append(ExperimentConfig(
                    experiment_id=f"{dataset_name}_{topology_type.value}_run{run}",
                    dataset_name=dataset_name,
                    dataset_path=dataset_path,
                    policy_config=PolicyConfig.from_type(PolicyType.FULL_ADAPTIVE),
                    embedding_config=EmbeddingConfig.from_type(EmbeddingType.TRANSE),
                    topology_config=TopologyConfig.from_type(topology_type),
                    random_seed=base_seed + run
                ))
        
        return configs
    
    @staticmethod
    def generate_full_factorial(
        dataset_name: str,
        dataset_path: str,
        base_seed: int = 42,
        num_runs: int = 1
    ) -> List[ExperimentConfig]:
        """Generate full factorial design (all combinations)."""
        configs = []
        
        for policy_type in PolicyType:
            for embedding_type in EmbeddingType:
                for topology_type in TopologyType:
                    for run in range(num_runs):
                        configs.append(ExperimentConfig(
                            experiment_id=f"{dataset_name}_{policy_type.value}_{embedding_type.value}_{topology_type.value}_run{run}",
                            dataset_name=dataset_name,
                            dataset_path=dataset_path,
                            policy_config=PolicyConfig.from_type(policy_type),
                            embedding_config=EmbeddingConfig.from_type(embedding_type),
                            topology_config=TopologyConfig.from_type(topology_type),
                            random_seed=base_seed + run
                        ))
        
        return configs
    
    @staticmethod
    def save_configs(configs: List[ExperimentConfig], output_path: str):
        """Save configs to JSON file."""
        with open(output_path, 'w') as f:
            json.dump([c.to_dict() for c in configs], f, indent=2)
    
    @staticmethod
    def load_configs(input_path: str) -> List[ExperimentConfig]:
        """Load configs from JSON file."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        return [ExperimentConfig.from_dict(d) for d in data]
