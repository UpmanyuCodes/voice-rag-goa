"""Chunking Strategy Factory and Registry."""

from typing import Dict, Type, List
from src.chunking.base import BaseChunker
from src.chunking.fixed_window import FixedWindowChunker
from src.chunking.semantic_boundary import SemanticBoundaryChunker
from src.chunking.metadata_aware import MetadataAwareChunker
from src.chunking.hierarchical import HierarchicalChunker


_STRATEGY_MAP: Dict[str, Type[BaseChunker]] = {
    "fixed_window": FixedWindowChunker,
    "semantic_boundary": SemanticBoundaryChunker,
    "metadata_aware": MetadataAwareChunker,
    "hierarchical": HierarchicalChunker,
}


def get_chunker(strategy_name: str = "metadata_aware", **kwargs) -> BaseChunker:
    """Instantiates a chunker by strategy name.
    
    Args:
        strategy_name: One of 'fixed_window', 'semantic_boundary', 'metadata_aware', 'hierarchical'
        **kwargs: Additional parameters passed to chunker constructor
        
    Returns:
        Instance of BaseChunker
    """
    chunker_cls = _STRATEGY_MAP.get(strategy_name.lower())
    if not chunker_cls:
        raise ValueError(
            f"Unknown chunking strategy '{strategy_name}'. "
            f"Available options: {list(_STRATEGY_MAP.keys())}"
        )
    return chunker_cls(**kwargs)


def list_chunking_strategies() -> List[Dict[str, str]]:
    """Lists all available chunking strategies with their descriptions."""
    strategies = []
    for name, cls in _STRATEGY_MAP.items():
        instance = cls()
        strategies.append({
            "id": name,
            "name": instance.name,
            "description": instance.description
        })
    return strategies
