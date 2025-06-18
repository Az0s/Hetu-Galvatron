"""
Configuration data classes for Galvatron Search Engine.

This module contains dataclasses that encapsulate different types of configuration
data, replacing the scattered attributes in the original GalvatronSearchEngine class.
"""
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable


@dataclass
class ModelLayerConfig:
    """Model layer configuration data."""
    hiddensize_list: List[int]
    layernum_list: List[int]
    seqlen_list: List[int]
    num_layertype: int
    
    @classmethod
    def from_configs(cls, model_layer_configs: Optional[List[Dict[str, Any]]]) -> 'ModelLayerConfig':
        """Create ModelLayerConfig from raw config list."""
        if model_layer_configs is None:
            return cls([], [], [], 0)
        
        hiddensize_list = [config['hidden_size'] for config in model_layer_configs]
        layernum_list = [config['layer_num'] for config in model_layer_configs]
        seqlen_list = [config['seq_len'] for config in model_layer_configs]
        num_layertype = len(layernum_list)
        
        return cls(hiddensize_list, layernum_list, seqlen_list, num_layertype)


@dataclass
class HardwareConfig:
    """Hardware configuration data."""
    allreduce_bandwidth: Dict
    allreduce_comm_coe: Dict
    p2p_bandwidth: Dict
    p2p_comm_coe: Dict
    overlap_coe: float
    sp_allreduce: Dict
    sp_all2all: Dict


@dataclass
class ModelProfile:
    """Model profiling data."""
    param_sizes: List[int]
    act_sizes: List[Dict]
    other_memory_pp_off: Dict
    other_memory_pp_on: Dict
    time_profiled_list: List
    other_time_profiled_list: List


@dataclass
class SearchConfig:
    """Search configuration data."""
    min_bsz: Optional[int] = None
    max_bsz: Optional[int] = None
    bsz_scale: Optional[int] = None
    BSZs: Optional[List[int]] = None
    strategies: Optional[List] = None
    search_history: Optional[Dict] = None


@dataclass
class CostModelArgs:
    """Cost model arguments collection."""
    model_args_list: List
    train_args_list: List
    parallel_args_list: List
    profile_model_args_list: List
    profile_hardware_args_list: List


@dataclass
class SearchEngineConfig:
    """Complete search engine configuration."""
    model_layer_config: ModelLayerConfig
    hardware_config: HardwareConfig
    model_profile: ModelProfile
    search_config: SearchConfig
    cost_model_args: CostModelArgs
    
    # Core engine settings
    model_type: str = 'gpt'
    use_pipeline_costmodel: bool = False
    memory_constraint: int = 0
    optimal_chunk_func: Optional[Callable] = None