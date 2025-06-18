"""
Configuration path management for Galvatron Search Engine.

This module handles all file path related operations, separating path management
from the core search engine logic.
"""
import os
from typing import Dict, Optional


class ConfigPathManager:
    """Manages configuration file paths for Galvatron Search Engine."""
    
    def __init__(self, args, base_path: str, model_name: str):
        """
        Initialize path manager.
        
        Args:
            args: Command line arguments with path overrides
            base_path: Base directory path
            model_name: Model name for config file naming
        """
        self.args = args
        self.base_path = base_path
        self.model_name = model_name
        
    def get_memory_config_path(self) -> str:
        """Get memory profiling configuration file path."""
        memory_config_name = f'memory_profiling_{self.args.mixed_precision}_{self.model_name}.json'
        
        if self.args.memory_profiling_path is None:
            memory_config_dir = os.path.join(self.base_path, 'configs')
        else:
            memory_config_dir = self.args.memory_profiling_path
            
        return os.path.join(memory_config_dir, memory_config_name)
    
    def get_time_config_path(self) -> str:
        """Get time profiling configuration file path."""
        time_config_name = f"computation_profiling_{self.args.mixed_precision}_{self.model_name}.json"
        
        if self.args.time_profiling_path is None:
            time_config_dir = os.path.join(self.base_path, "configs")
        else:
            time_config_dir = self.args.time_profiling_path
            
        return os.path.join(time_config_dir, time_config_name)
    
    def get_hardware_config_paths(self) -> Dict[str, str]:
        """Get all hardware configuration file paths."""
        hardware_configs_dir = '../../profile_hardware/hardware_configs/'
        
        # Allreduce bandwidth config
        if self.args.allreduce_bandwidth_config_path is None:
            allreduce_dir = os.path.join(self.base_path, hardware_configs_dir)
        else:
            allreduce_dir = self.args.allreduce_bandwidth_config_path
        allreduce_name = f'allreduce_bandwidth_{self.args.num_nodes}nodes_{self.args.num_gpus_per_node}gpus_per_node.json'
        allreduce_path = os.path.join(allreduce_dir, allreduce_name)
        
        # P2P bandwidth config
        if self.args.p2p_bandwidth_config_path is None:
            p2p_dir = os.path.join(self.base_path, hardware_configs_dir)
        else:
            p2p_dir = self.args.p2p_bandwidth_config_path
        p2p_name = f'p2p_bandwidth_{self.args.num_nodes}nodes_{self.args.num_gpus_per_node}gpus_per_node.json'
        p2p_path = os.path.join(p2p_dir, p2p_name)
        
        # Overlap coefficient config
        if self.args.overlap_coe_path is None:
            overlap_dir = os.path.join(self.base_path, hardware_configs_dir)
        else:
            overlap_dir = self.args.overlap_coe_path
        overlap_name = 'overlap_coefficient.json'
        overlap_path = os.path.join(overlap_dir, overlap_name)
        
        # SP time config
        if self.args.sp_time_path is None:
            sp_time_dir = os.path.join(self.base_path, hardware_configs_dir)
        else:
            sp_time_dir = self.args.sp_time_path
        sp_time_name = f'sp_time_{self.args.num_nodes}nodes_{self.args.num_gpus_per_node}gpus_per_node.json'
        sp_time_path = os.path.join(sp_time_dir, sp_time_name)
        
        return {
            'allreduce': allreduce_path,
            'p2p': p2p_path,
            'overlap': overlap_path,
            'sp_time': sp_time_path
        }
    
    def get_output_config_path(self, memory_constraint: int) -> str:
        """Get output configuration file path."""
        if self.args.output_config_path is None:
            config_dir = os.path.join(self.base_path, 'configs/')
        else:
            config_dir = self.args.output_config_path
            
        # Build filename with various options
        mixed_precision = f'_{self.args.mixed_precision}'
        settle_bsz = f'_bsz{self.args.settle_bsz}' if self.args.settle_bsz > 0 else ''
        
        off_options = []
        if self.args.disable_dp:
            off_options.append('dp')
        if self.args.disable_tp:
            off_options.append('tp')
        if self.args.disable_pp:
            off_options.append('pp')
        if self.args.disable_sdp:
            off_options.append('sdp')
        if self.args.disable_ckpt:
            off_options.append('ckpt')
        if self.args.disable_tp_consec:
            off_options.append('tpconsec')
        off_options_str = f'_[{"_".join(off_options)}_off]' if len(off_options) else ''
        
        output_config_name = (f'galvatron_config_{self.model_name}_{self.args.num_nodes}nodes_'
                             f'{self.args.num_gpus_per_node}gpus_per_node_{memory_constraint//1024}GB'
                             f'{mixed_precision}{settle_bsz}{off_options_str}.json')
        
        return os.path.join(config_dir, output_config_name)