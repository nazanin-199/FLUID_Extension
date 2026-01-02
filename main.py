import rdflib
from typing import Dict
from core.statistics import KGStatistics
from core.extraction import AdaptivePolicy, SymbolicExtractor
from core.summarization import FLUIDSummarizer
from core.embeddings import EmbeddingTrainer, PayloadBuilder
from utils.logger import IFLUIDLogger
from config.config import IFLUIDConfig

class IFLUIDPipeline:
    """Main IFLUID pipeline orchestrator."""
    
    def __init__(self, config: IFLUIDConfig):
        self.config = config
        self.logger = IFLUIDLogger()
    
    def run(self, graph: rdflib.Graph) -> Dict[str, any]:
        """
        Run complete IFLUID pipeline.
        
        Returns dictionary with:
            - enriched_graph: Graph with inferred triples
            - summary_graph: FLUID summary
            - summary_nodes: Super-node to entity mapping
            - node_map: Entity to super-node mapping
            - payload: Embedding payload
            - policy: Extraction policy used
        """
        self.logger.info(f"Input graph: {len(graph)} triples")
        
        # 1. Compute statistics and determine policy
        stats_computer = KGStatistics(graph)
        profile = stats_computer.normalize()
        
        policy_maker = AdaptivePolicy(self.config.policy)
        policy = policy_maker.determine(profile)
        
        if self.config.verbose:
            self.logger.info(f"Extraction policy: {policy}")
        
        # 2. Symbolic extraction
        extractor = SymbolicExtractor(policy)
        enriched_graph = extractor.extract(graph)
        
        inferred_triples = list(set(enriched_graph) - set(graph))
        self.logger.info(f"Extracted {len(inferred_triples)} inferred triples")
        
        # 3. FLUID summarization
        summarizer = FLUIDSummarizer()
        summary_graph, summary_nodes, node_map = summarizer.summarize(enriched_graph)
        
        # 4. Train embeddings on inferred triples
        payload = None
        if inferred_triples:
            trainer = EmbeddingTrainer(self.config.embedding)
            model, entity_to_id = trainer.train(inferred_triples)
            
            if model:
                payload_builder = PayloadBuilder(self.config.embedding.dim)
                payload = payload_builder.build(summary_nodes, model, entity_to_id)
        else:
            self.logger.warning("No inferred triples - skipping embedding training")
        
        return {
            'enriched_graph': enriched_graph,
            'summary_graph': summary_graph,
            'summary_nodes': summary_nodes,
            'node_map': node_map,
            'payload': payload,
            'policy': policy
        }
