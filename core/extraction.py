from config.config import PolicyConfig
from typing import Dict, Any, Tuple
from collections import defaultdict
import rdflib
from rdflib import RDF, RDFS, OWL
from utils.logger import IFLUIDLogger
from dataclasses import dataclass


@dataclass
class PolicyDecision:
    enabled: bool
    reason: str
    cost_estimate: int = 0


class AdaptivePolicy:
    """
    Design principles:
    1. Every threshold must be justified
    2. Every metric must directly inform a decision
    3. Prefer simple rules over complex scoring
    """
    
    def __init__(self):
        # Thresholds with justification
        self.thresholds = {
            # SubClassOf: Enable if hierarchy exists and cost is manageable
            'subclass_min_depth': 2,          # Need at least 2 levels to benefit
            'subclass_max_cost': 10000,       # Avoid combinatorial explosion
            
            # sameAs: Enable if non-trivial clusters exist
            'sameas_min_density': 0.001,      # At least 0.1% of triples
            'sameas_min_cluster': 2,          # Clusters of 2+ entities
            'sameas_min_count': 5,            # At least 5 relations
            
            # Domain/Range: Enable if schema-rich with many instances
            'domain_min_instances': 5,        # Need instances to benefit
            'domain_max_ratio': 3,            # Properties >> Classes
            'domain_max_cost': 5000,          # Manageable inference
            
            # Rule mining: Enable for large, well-structured graphs
            'rule_min_triples': 1000,         # Need data for patterns
        }
    
    def determine(self, profile: Dict, stats: Dict) -> Tuple[Dict, Dict]:
        """
        Determine policy with simple, clear rules.
        
        Returns:
            policy: Dict of enabled rules
            reasoning: Dict of PolicyDecision objects
        """
        policy = {}
        reasoning = {}
        
        # === SubClassOf Transitivity ===
        reasoning['subClassOf'] = self._decide_subclass(profile, stats)
        policy['subClassOf'] = reasoning['subClassOf'].enabled
        
        # === SubPropertyOf ===
        policy['subPropertyOf'] = True
        reasoning['subPropertyOf'] = PolicyDecision(
            enabled=True,
            reason="SubPropertyOf is low-cost and generally beneficial"
        )
        
        # === sameAs Consolidation ===
        reasoning['sameAs'] = self._decide_sameas(profile, stats)
        policy['sameAs'] = reasoning['sameAs'].enabled
        
        # === Domain/Range Propagation ===
        reasoning['domain_range'] = self._decide_domain_range(profile, stats)
        policy['domain_range'] = reasoning['domain_range'].enabled
        
        # === Rule Mining ===
        reasoning['rule_mining'] = self._decide_rule_mining(profile, stats)
        policy['rule_mining'] = reasoning['rule_mining'].enabled
        
        # === Max Depth ===
        policy['max_depth'] = self._decide_max_depth(profile, stats)
        reasoning['max_depth'] = PolicyDecision(
            enabled=True,
            reason=f"Set to {policy['max_depth']} based on hierarchy depth={profile['max_hierarchy_depth']}"
        )
        
        return policy, reasoning
    
    def _decide_subclass(self, profile: Dict, stats: Dict) -> PolicyDecision:
        """
        Decision: Enable subClassOf if hierarchy exists and cost is acceptable.
        
        Logic:
        1. Must have hierarchy (depth > min_depth)
        2. Cost must be manageable (< max_cost)
        """
        depth = profile['avg_hierarchy_depth']
        cost = profile['subclass_cost']
        
        # Check 1: Does hierarchy exist?
        if not profile['has_hierarchy']:
            return PolicyDecision(
                enabled=False,
                reason="No class hierarchy detected",
                cost_estimate=0
            )
        
        # Check 2: Is depth sufficient?
        if depth < self.thresholds['subclass_min_depth']:
            return PolicyDecision(
                enabled=False,
                reason=f"Hierarchy too shallow (depth={depth:.1f} < {self.thresholds['subclass_min_depth']})",
                cost_estimate=cost
            )
        
        # Check 3: Is cost manageable?
        if cost > self.thresholds['subclass_max_cost']:
            return PolicyDecision(
                enabled=False,
                reason=f"Inference cost too high ({cost} > {self.thresholds['subclass_max_cost']})",
                cost_estimate=cost
            )
        
        # All checks passed
        return PolicyDecision(
            enabled=True,
            reason=f"Hierarchy exists (depth={depth:.1f}), cost acceptable ({cost} triples)",
            cost_estimate=cost
        )
    
    def _decide_sameas(self, profile: Dict, stats: Dict) -> PolicyDecision:
        """
        Decision: Enable sameAs if non-trivial clusters exist.
        
        Logic:
        1. Must have minimum density
        2. Must have meaningful clusters (size > 2)
        3. Must have minimum count
        """
        density = profile['sameas_density']
        cluster_size = profile['sameas_cluster_size']
        count = stats['sameas_count']
        
        # Check 1: Sufficient density?
        if density < self.thresholds['sameas_min_density']:
            return PolicyDecision(
                enabled=False,
                reason=f"Low sameAs density ({density:.4f} < {self.thresholds['sameas_min_density']})"
            )
        
        # Check 2: Meaningful clusters?
        if cluster_size < self.thresholds['sameas_min_cluster']:
            return PolicyDecision(
                enabled=False,
                reason=f"Trivial clusters (avg={cluster_size:.1f} < {self.thresholds['sameas_min_cluster']})"
            )
        
        # Check 3: Sufficient count?
        if count < self.thresholds['sameas_min_count']:
            return PolicyDecision(
                enabled=False,
                reason=f"Too few sameAs relations ({count} < {self.thresholds['sameas_min_count']})"
            )
        
        return PolicyDecision(
            enabled=True,
            reason=f"Non-trivial clusters detected (avg={cluster_size:.1f}, count={count})"
        )
    
    def _decide_domain_range(self, profile: Dict, stats: Dict) -> PolicyDecision:
        """
        Decision: Enable domain/range if schema-rich with instances.
        
        Logic:
        1. Must have schema declarations
        2. Must have instances to benefit
        3. Must have rich schema (properties >> classes)
        4. Cost must be acceptable
        """
        has_schema = profile['has_schema']
        instance_ratio = profile['instance_class_ratio']
        class_prop_ratio = profile['class_property_ratio']
        cost = profile['domain_range_cost']
        
        # Check 1: Schema declarations exist?
        if not has_schema:
            return PolicyDecision(
                enabled=False,
                reason="No domain/range declarations found"
            )
        
        # Check 2: Enough instances?
        if instance_ratio < self.thresholds['domain_min_instances']:
            return PolicyDecision(
                enabled=False,
                reason=f"Too few instances per class ({instance_ratio:.1f} < {self.thresholds['domain_min_instances']})"
            )
        
        # Check 3: Rich schema?
        if class_prop_ratio > self.thresholds['domain_max_ratio']:
            return PolicyDecision(
                enabled=False,
                reason=f"Schema not rich enough (C/P ratio={class_prop_ratio:.1f} > {self.thresholds['domain_max_ratio']})"
            )
        
        # Check 4: Cost acceptable?
        if cost > self.thresholds['domain_max_cost']:
            return PolicyDecision(
                enabled=False,
                reason=f"Inference cost too high ({cost} > {self.thresholds['domain_max_cost']})",
                cost_estimate=cost
            )
        
        return PolicyDecision(
            enabled=True,
            reason=f"Rich schema with instances (ratio={instance_ratio:.1f}, cost={cost})",
            cost_estimate=cost
        )
    
    def _decide_rule_mining(self, profile: Dict, stats: Dict) -> PolicyDecision:
        """
        Decision: Enable rule mining for large graphs.
        
        Logic: Graph must be large enough for patterns to emerge.
        """
        num_triples = stats['num_triples']
        
        if num_triples < self.thresholds['rule_min_triples']:
            return PolicyDecision(
                enabled=False,
                reason=f"Graph too small for rule mining ({num_triples} < {self.thresholds['rule_min_triples']})"
            )
        
        return PolicyDecision(
            enabled=True,
            reason=f"Graph large enough for patterns ({num_triples} triples)"
        )
    
    def _decide_max_depth(self, profile: Dict, stats: Dict) -> int:
        """
        Decision: Set max depth based on hierarchy depth.
        
        Simple heuristic:
        - Shallow (≤3): 10 (allow full expansion)
        - Medium (4-6): 5
        - Deep (>6): 3 (limit to prevent explosion)
        """
        max_depth = profile['max_hierarchy_depth']
        
        if max_depth <= 3:
            return 10
        elif max_depth <= 6:
            return 5
        else:
            return 3

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
