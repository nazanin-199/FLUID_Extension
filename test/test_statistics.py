import unittest
import rdflib
from rdflib import RDF, RDFS
from core.statistics import KGStatistics

class TestKGStatistics(unittest.TestCase):
    def setUp(self):
        self.graph = rdflib.Graph()
        # Add test triples
        ex = rdflib.Namespace("http://example.org/")
        self.graph.add((ex.entity1, RDF.type, ex.Class1))
        self.graph.add((ex.entity1, ex.property1, ex.entity2))
    
    def test_compute_statistics(self):
        stats = KGStatistics(self.graph)
        result = stats.compute_all()
        
        self.assertGreater(result['num_triples'], 0)
        self.assertGreater(result['num_classes'], 0)
    
    def test_normalize(self):
        stats = KGStatistics(self.graph)
        profile = stats.normalize()
        
        self.assertIn('log_triples', profile)
        self.assertGreater(profile['log_triples'], 0)

if __name__ == '__main__':
    unittest.main()
