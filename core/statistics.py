import math
import numpy as np
from typing import Dict, Set, Any, List, Tuple
import rdflib
from rdflib import RDF, RDFS, OWL
from collections import defaultdict, Counter


class KGStatistics:
    """
    1. SubClassOf: Is there hierarchy? How deep? How big will inference be?
    2. sameAs: Are there duplicates? How clustered?
    3. Domain/Range: Rich schema? Many instances?
    4. Rule mining: Large enough? Connected enough?

    topology analysis:
    - Sparsity/density for graph structure
    - Degree distribution for connectivity patterns
    - Clustering coefficient for community detection
    - Branching factor for hierarchy shape
    """

    def __init__(self, graph: rdflib.Graph):
        self.graph = graph
        self._cache = {}

    def compute_all(self) -> Dict[str, float]:
        """Compute comprehensive statistics including topology."""
        if self._cache:
            return self._cache

        stats = {}

        stats.update(self._compute_basic_counts())
        stats.update(self._compute_topology_metrics())
        stats.update(self._compute_hierarchy_metrics())
        stats.update(self._compute_sameas_metrics())
        stats.update(self._compute_schema_metrics())
        stats.update(self._estimate_inference_costs())

        self._cache = stats
        return stats

    def _compute_basic_counts(self) -> Dict[str, int]:
        """Basic counts."""
        classes = set(o for _, _, o in self.graph.triples((None, RDF.type, None)))
        instances = set(s for s, _, _ in self.graph.triples((None, RDF.type, None)))
        properties = set(p for _, p, _ in self.graph
                         if p not in [RDF.type, RDFS.subClassOf, RDFS.subPropertyOf, OWL.sameAs])

        return {
            'num_triples': len(self.graph),
            'num_classes': len(classes),
            'num_instances': len(instances),
            'num_properties': len(properties),
            'instance_to_class_ratio': len(instances) / max(1, len(classes)),
        }

    def _compute_topology_metrics(self) -> Dict[str, float]:
        """
        Compute graph topology metrics.

        Metrics:
        1. Graph density: |E| / |V|²
        2. Average degree: mean connectivity
        3. Max degree: highest connectivity
        4. Degree skew: max_degree / median_degree (distribution shape)
        5. Clustering coefficient: local community signal
        """
        # Get all entities (vertices)
        entities = set(self.graph.subjects()) | set(self.graph.objects())
        entities = {e for e in entities if isinstance(e, rdflib.URIRef)}

        if not entities:
            return {
                'num_entities': 0,
                'graph_density': 0.0,
                'avg_degree': 0.0,
                'max_degree': 0,
                'median_degree': 0.0,
                'degree_skew': 0.0,
                'clustering_coefficient': 0.0,
            }

        num_vertices = len(entities)
        num_edges = len(self.graph)

        # Graph density: |E| / |V|²
        # Measures how sparse/dense the graph is
        max_possible_edges = num_vertices * num_vertices
        graph_density = num_edges / max(1, max_possible_edges)

        # Degree distribution
        degrees = []
        degree_map = {}  # For clustering coefficient calculation

        for entity in entities:
            out_degree = len(list(self.graph.triples((entity, None, None))))
            in_degree = len(list(self.graph.triples((None, None, entity))))
            total_degree = out_degree + in_degree
            degrees.append(total_degree)
            degree_map[entity] = total_degree

        avg_degree = np.mean(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0
        median_degree = np.median(degrees) if degrees else 0

        # Degree skew: indicates if graph has hubs (high skew) or uniform connectivity (low skew)
        # High skew (>10) = hub structure, Low skew (<3) = uniform
        degree_skew = max_degree / max(1, median_degree)

        # Clustering coefficient (approximate, for community signal)
        # Measures how much neighbors of a node are connected to each other
        clustering_coefficient = self._compute_clustering_coefficient(entities, degree_map)

        return {
            'num_entities': num_vertices,
            'graph_density': graph_density,
            'avg_degree': avg_degree,
            'max_degree': max_degree,
            'median_degree': median_degree,
            'degree_skew': degree_skew,
            'clustering_coefficient': clustering_coefficient,
        }

    def _compute_clustering_coefficient(
        self,
        entities: Set,
        degree_map: Dict,
        sample_size: int = 100
    ) -> float:
        """
        Compute approximate local clustering coefficient.
        For computational efficiency, sample up to 100 nodes.
        Clustering coefficient = (# of closed triplets) / (# of all triplets)

        High clustering (>0.3) indicates community structure.
        """
        # Sample nodes to make computation tractable
        sampled_entities = list(entities)
        if len(sampled_entities) > sample_size:
            sampled_entities = np.random.choice(sampled_entities, sample_size, replace=False)

        clustering_scores = []

        for entity in sampled_entities:
            # Get neighbors (both in and out)
            neighbors = set()

            # Outgoing neighbors
            for _, _, o in self.graph.triples((entity, None, None)):
                if isinstance(o, rdflib.URIRef):
                    neighbors.add(o)

            # Incoming neighbors
            for s, _, _ in self.graph.triples((None, None, entity)):
                if isinstance(s, rdflib.URIRef):
                    neighbors.add(s)

            neighbors.discard(entity)  # Remove self

            if len(neighbors) < 2:
                continue  # Need at least 2 neighbors for clustering

            # Count connections between neighbors
            neighbor_list = list(neighbors)
            possible_connections = len(neighbor_list) * (len(neighbor_list) - 1) / 2
            actual_connections = 0

            for i, n1 in enumerate(neighbor_list):
                for n2 in neighbor_list[i+1:]:
                    # Check if n1 and n2 are connected
                    if (n1, None, n2) in self.graph or (n2, None, n1) in self.graph:
                        actual_connections += 1

            if possible_connections > 0:
                local_clustering = actual_connections / possible_connections
                clustering_scores.append(local_clustering)

        return np.mean(clustering_scores) if clustering_scores else 0.0

    def _compute_hierarchy_metrics(self) -> Dict[str, float]:
        """
        Enhanced hierarchy analysis with branching factor.
        Branching factor = |subClassOf edges| / |classes with children|
        - High branching (>3): Wide, shallow hierarchy
        - Low branching (<2): Deep, narrow hierarchy
        """
        subclass_triples = list(self.graph.triples((None, RDFS.subClassOf, None)))

        if not subclass_triples:
            return {
                'num_subclass_triples': 0,
                'max_hierarchy_depth': 0,
                'avg_hierarchy_depth': 0,
                'branching_factor': 0.0,
                'has_multiple_inheritance': False,
            }

        # Build parent-child mappings
        child_to_parents = defaultdict(set)
        parent_to_children = defaultdict(set)

        for child, _, parent in subclass_triples:
            child_to_parents[child].add(parent)
            parent_to_children[parent].add(child)

        # Compute depths
        all_classes = set(child_to_parents.keys()) | set(parent_to_children.keys())

        depths = []
        max_depth = 0
        has_multiple_inheritance = False

        for cls in all_classes:
            depth = self._compute_depth_bfs(cls, child_to_parents)
            depths.append(depth)
            max_depth = max(max_depth, depth)

            if len(child_to_parents.get(cls, [])) > 1:
                has_multiple_inheritance = True

        # Branching factor: average number of children per parent
        # Indicates hierarchy shape: wide vs narrow
        num_parents_with_children = len(parent_to_children)
        total_children = sum(len(children) for children in parent_to_children.values())
        branching_factor = total_children / max(1, num_parents_with_children)

        return {
            'num_subclass_triples': len(subclass_triples),
            'max_hierarchy_depth': max_depth,
            'avg_hierarchy_depth': np.mean(depths) if depths else 0,
            'branching_factor': branching_factor,
            'has_multiple_inheritance': has_multiple_inheritance,
        }

    def _compute_depth_bfs(self, cls, child_to_parents: Dict, max_depth: int = 50) -> int:
        """BFS depth computation (handles cycles and multiple paths)."""
        if cls not in child_to_parents:
            return 0

        visited = {cls}
        queue = [(cls, 0)]
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

    def _compute_sameas_metrics(self) -> Dict[str, float]:
        """
        Analyze owl:sameAs usage for duplication and clustering.
        Computes cluster sizes using Union-Find (Disjoint Set Union).
        """
        sameas_triples = list(self.graph.triples((None, OWL.sameAs, None)))

        if not sameas_triples:
            return {
                'sameas_count': 0,
                'sameas_density': 0.0,
                'avg_sameas_cluster_size': 0.0,
            }

        parent = {}

        def find(x):
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            parent[find(x)] = find(y)

        for x, _, y in sameas_triples:
            union(x, y)

        clusters = defaultdict(set)
        for entity in parent:
            clusters[find(entity)].add(entity)

        cluster_sizes = [len(cluster) for cluster in clusters.values()]

        return {
            'sameas_count': len(sameas_triples),
            'sameas_density': len(sameas_triples) / max(1, len(self.graph)),
            'avg_sameas_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
        }

    def _compute_schema_metrics(self) -> Dict[str, float]:
        num_classes = self._cache.get('num_classes',
                                      len(set(o for _, _, o in self.graph.triples((None, RDF.type, None)))))
        num_properties = self._cache.get('num_properties',
                                         len(set(p for _, p, _ in self.graph if p not in [RDF.type, RDFS.subClassOf])))

        domain_count = len(list(self.graph.triples((None, RDFS.domain, None))))
        range_count = len(list(self.graph.triples((None, RDFS.range, None))))

        return {
            'class_to_property_ratio': num_classes / max(1, num_properties),
            'num_domain_declarations': domain_count,
            'num_range_declarations': range_count,
            'has_schema_declarations': (domain_count + range_count) > 0,
        }

    def _estimate_inference_costs(self) -> Dict[str, int]:
        child_to_parents = defaultdict(set)
        for child, _, parent in self.graph.triples((None, RDFS.subClassOf, None)):
            child_to_parents[child].add(parent)

        subclass_cost = 0
        for child in child_to_parents:
            ancestors = set()
            queue = list(child_to_parents[child])
            visited = set(child_to_parents[child])

            while queue and len(ancestors) < 1000:
                parent = queue.pop(0)
                ancestors.add(parent)
                for grandparent in child_to_parents.get(parent, []):
                    if grandparent not in visited:
                        visited.add(grandparent)
                        queue.append(grandparent)

            indirect = ancestors - child_to_parents[child]
            subclass_cost += len(indirect)

        domain_range_cost = 0
        for prop, _, domain_class in self.graph.triples((None, RDFS.domain, None)):
            usage = len(list(self.graph.triples((None, prop, None))))
            domain_range_cost += usage

        for prop, _, range_class in self.graph.triples((None, RDFS.range, None)):
            usage = len(list(self.graph.triples((None, prop, None))))
            domain_range_cost += usage

        return {
            'subclass_inference_cost': subclass_cost,
            'domain_range_inference_cost': domain_range_cost,
        }

    def normalize(self) -> Dict[str, float]:
        """
        Normalize statistics for policy determination.
        Enhanced with topology metrics.
        """
        stats = self.compute_all()

        return {
            # Size indicators
            'log_triples': math.log1p(stats['num_triples']),

            # Topology indicators
            'graph_density': stats.get('graph_density', 0),
            'avg_degree': stats.get('avg_degree', 0),
            'degree_skew': stats.get('degree_skew', 0),
            'clustering_coefficient': stats.get('clustering_coefficient', 0),

            # Hierarchy indicators
            'has_hierarchy': stats['max_hierarchy_depth'] > 1,
            'avg_hierarchy_depth': stats['avg_hierarchy_depth'],
            'max_hierarchy_depth': stats['max_hierarchy_depth'],
            'branching_factor': stats.get('branching_factor', 0),

            # Duplication indicators
            'sameas_density': stats['sameas_density'],
            'sameas_cluster_size': stats['avg_sameas_cluster_size'],

            # Schema indicators
            'instance_class_ratio': stats['instance_to_class_ratio'],
            'class_property_ratio': stats['class_to_property_ratio'],
            'has_schema': stats['has_schema_declarations'],

            # Cost indicators
            'subclass_cost': stats['subclass_inference_cost'],
            'domain_range_cost': stats['domain_range_inference_cost'],
        }
