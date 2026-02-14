# Inference-Aware FLUID Summarization (IFLUID)

A research framework for scalable knowledge graph summarization that preserves semantic completeness through adaptive inference extraction and embedding-based payload enrichment.

## Overview

IFLUID extends the FLUID structural summarization framework by selectively incorporating logical inferences (subClassOf, subPropertyOf, sameAs, domain/range constraints) into compact semantic representations—without explicitly materializing all inferred triples in the graph structure.

Traditional approaches face a scalability dilemma:
- Full logical materialization preserves semantics but explodes graph size
- Pure structural summarization loses critical semantic relationships

IFLUID resolves this trade-off by:
1. Profiling the input knowledge graph to determine an adaptive extraction policy
2. Selectively extracting only high-value inferences based on graph characteristics
3. Training TransE-style embeddings on extracted inferences
4. Encoding semantic information into payload vectors attached to FLUID super-nodes
5. Preserving both structural guarantees (exact equivalence classes) and semantic richness where it matters most

This enables efficient downstream tasks (e.g., GNN-based node classification) on large knowledge graphs while maintaining interpretability and scalability.

## Architecture

The IFLUID pipeline consists of five stages:

1. **Graph Profiling**  
   Computes normalized statistics (log triples, class-property ratio, sameAs density, average class depth) to characterize the input KG.

2. **Adaptive Policy Determination**  
   Uses configurable thresholds to decide which inference types to extract:
   - `sameAs` consolidation when density > threshold
   - Domain/range propagation when class-property ratio is low
   - Rule mining when graph size exceeds threshold
   - Depth-limited transitive closure based on hierarchy depth

3. **Selective Symbolic Extraction**  
   Applies only the enabled inference rules to enrich the graph with high-value triples while avoiding combinatorial explosion.

4. **FLUID Structural Summarization**  
   Groups vertices into super-nodes using exact FLUID descriptors (types, outgoing predicates, incoming predicates).

5. **Semantic Payload Construction**  
   Trains TransE embeddings on extracted inferences and builds payload vectors by averaging entity embeddings within each super-node.

## Installation

```bash
git clone https://github.com/your-username/ifluid.git
cd ifluid
pip install -r requirements.txt
