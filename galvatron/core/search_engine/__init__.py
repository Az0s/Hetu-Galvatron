"""
Galvatron Search Engine Module.

This module provides both the original and refactored search engine implementations
for backward compatibility and future development.
"""

# Original search engine for backward compatibility
from .search_engine import GalvatronSearchEngine as GalvatronSearchEngineOriginal

# New refactored components
from .search_engine_factory import SearchEngineFactory
from .config_types import (
    ModelLayerConfig, 
    HardwareConfig, 
    ModelProfile, 
    SearchConfig, 
    CostModelArgs, 
    SearchEngineConfig
)
from .config_path_manager import ConfigPathManager
from .profile_data_loader import ProfileDataLoader

# For backward compatibility, export the factory as the main interface
def create_search_engine(args, path, model_layer_configs, model_name):
    """Create a search engine using the new factory pattern."""
    return SearchEngineFactory.create_search_engine(args, path, model_layer_configs, model_name)

# Keep the original interface for backward compatibility
GalvatronSearchEngine = GalvatronSearchEngineOriginal

# Default export is the new factory-created search engine
__all__ = [
    'create_search_engine',
    'SearchEngineFactory',
    'GalvatronSearchEngine',
    'GalvatronSearchEngineOriginal',
    'ModelLayerConfig',
    'HardwareConfig', 
    'ModelProfile',
    'SearchConfig',
    'CostModelArgs',
    'SearchEngineConfig',
    'ConfigPathManager',
    'ProfileDataLoader'
]