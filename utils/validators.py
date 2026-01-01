from typing import Any
import rdflib


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class Validators:
    """Input validation utilities."""
    
    @staticmethod
    def validate_graph(G: rdflib.Graph) -> None:
        """Validate RDF graph is non-empty."""
        if not isinstance(G, rdflib.Graph):
            raise ValidationError(f"Expected rdflib.Graph, got {type(G)}")
        if len(G) == 0:
            raise ValidationError("Graph is empty")
    
    @staticmethod
    def validate_positive_int(value: int, name: str) -> None:
        """Validate positive integer."""
        if not isinstance(value, int) or value <= 0:
            raise ValidationError(f"{name} must be positive integer, got {value}")
    
    @staticmethod
    def validate_probability(value: float, name: str) -> None:
        """Validate probability in [0, 1]."""
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValidationError(f"{name} must be in [0, 1], got {value}")
    
    @staticmethod
    def validate_file_exists(path: str) -> None:
        """Validate file exists."""
        from pathlib import Path
        if not Path(path).exists():
            raise ValidationError(f"File not found: {path}")
