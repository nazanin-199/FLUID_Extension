"""Pytest fixtures for IFLUID tests."""

import pytest
import rdflib
from rdflib import RDF, RDFS, OWL, Namespace


@pytest.fixture
def simple_graph():
    """Create a simple test graph."""
    g = rdflib.Graph()
    ex = Namespace("http://example.org/")
    
    # Add some triples
    g.add((ex.Person1, RDF.type, ex.Person))
    g.add((ex.Person1, ex.name, rdflib.Literal("Alice")))
    g.add((ex.Person1, ex.knows, ex.Person2))
    g.add((ex.Person2, RDF.type, ex.Person))
    g.add((ex.Person, RDFS.subClassOf, ex.Agent))
    
    return g


@pytest.fixture
def hierarchical_graph():
    """Create a graph with class hierarchy."""
    g = rdflib.Graph()
    ex = Namespace("http://example.org/")
    
    # Class hierarchy
    g.add((ex.Student, RDFS.subClassOf, ex.Person))
    g.add((ex.Person, RDFS.subClassOf, ex.Agent))
    g.add((ex.Agent, RDFS.subClassOf, ex.Thing))
    
    # Instances
    g.add((ex.Alice, RDF.type, ex.Student))
    g.add((ex.Bob, RDF.type, ex.Person))
    
    return g


@pytest.fixture
def sameas_graph():
    """Create a graph with sameAs relations."""
    g = rdflib.Graph()
    ex = Namespace("http://example.org/")
    
    g.add((ex.Entity1, OWL.sameAs, ex.Entity2))
    g.add((ex.Entity2, OWL.sameAs, ex.Entity3))
    g.add((ex.Entity1, ex.property1, rdflib.Literal("value")))
    
    return g
