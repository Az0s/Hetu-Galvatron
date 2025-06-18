"""
Factory for creating Galvatron Search Engine instances.

This module provides a factory class that orchestrates the creation of all
components needed for the search engine, implementing the factory pattern.
"""
from typing import List, Dict, Any, Optional
from .config_types import ModelLayerConfig, SearchEngineConfig, SearchConfig, CostModelArgs
from .config_path_manager import ConfigPathManager
from .profile_data_loader import ProfileDataLoader
from .cost_model_args import ModelArgs, ParallelArgs, TrainArgs, ProfileModelArgs, ProfileHardwareArgs


def optimal_chunk_func_default(local_bsz, strategy, microbatch_size, min_tp):
    """Default optimal chunk function - moved here to avoid circular imports."""
    import numpy as np
    assert(strategy[1] % min_tp == 0)
    local_bsz = local_bsz // (strategy[1] // min_tp)
    chunk = np.ceil(local_bsz / microbatch_size)
    chunk = 1 if chunk == 0 else chunk
    return chunk


class SearchEngineFactory:
    """Factory for creating GalvatronSearchEngine instances."""
    
    @staticmethod
    def create_search_engine(args, path: str, model_layer_configs: Optional[List[Dict[str, Any]]], model_name: str):
        """
        Create a complete search engine instance.
        
        Args:
            args: Command line arguments
            path: Base path for configuration files
            model_layer_configs: Model layer configuration data
            model_name: Name of the model
            
        Returns:
            Configured GalvatronSearchEngine instance
        """
        # Create model layer configuration
        model_layer_config = ModelLayerConfig.from_configs(model_layer_configs)
        
        # Create path manager
        path_manager = ConfigPathManager(args, path, model_name)
        
        # Create data loader and load profiles
        data_loader = ProfileDataLoader(args, model_layer_config, path_manager)
        model_profile = data_loader.load_model_profile()
        hardware_config = data_loader.load_hardware_config()
        
        # Create cost model arguments
        cost_model_args = SearchEngineFactory._create_cost_model_args(
            args, model_layer_config, model_profile, hardware_config
        )
        
        # Create search configuration
        search_config = SearchConfig()
        
        # Create complete configuration
        config = SearchEngineConfig(
            model_layer_config=model_layer_config,
            hardware_config=hardware_config,
            model_profile=model_profile,
            search_config=search_config,
            cost_model_args=cost_model_args,
            model_type='gpt',
            use_pipeline_costmodel=args.use_pipeline_costmodel,
            memory_constraint=args.memory_constraint * 1024,
            optimal_chunk_func=optimal_chunk_func_default
        )
        
        # Import here to avoid circular imports
        from .search_engine_refactored import GalvatronSearchEngine
        
        return GalvatronSearchEngine(args, config, path_manager)
    
    @staticmethod
    def _create_cost_model_args(args, model_layer_config: ModelLayerConfig, 
                               model_profile, hardware_config) -> CostModelArgs:
        """Create cost model arguments for each layer type."""
        model_args_list, train_args_list, parallel_args_list = [], [], []
        profile_model_args_list, profile_hardware_args_list = [], []
        
        for i in range(model_layer_config.num_layertype):
            model_args = ModelArgs(
                parameter_size=model_profile.param_sizes[i],
                seq_length=model_layer_config.seqlen_list[i],
                hidden_size=model_layer_config.hiddensize_list[i],
                layer_num=model_layer_config.layernum_list[i],
            )
            
            train_args = TrainArgs(
                mixed_precision=False if args.mixed_precision == 'fp32' else True,
                async_grad_reduce=args.async_grad_reduce,
            )
            
            # Set optimal chunk function
            optimal_chunk_func = lambda local_bsz, strategy, mbsz, min_tp: optimal_chunk_func_default(local_bsz, strategy, args.microbatch_size if hasattr(args, 'microbatch_size') else 1, min_tp)
            
            parallel_args = ParallelArgs(
                use_zero2_for_dp=True if args.default_dp_type == 'zero2' else False,
                disable_vtp=args.disable_vtp,
                sequence_parallel=args.sequence_parallel,
                sp_space=args.sp_space,
                pipeline_type=args.pipeline_type,
                optimal_chunk_func=optimal_chunk_func,
            )
            
            profile_model_args = ProfileModelArgs(
                tp_activation_per_bsz_dict=model_profile.act_sizes[i],
                other_memory_pp_off=model_profile.other_memory_pp_off,
                other_memory_pp_on=model_profile.other_memory_pp_on,
                forward_computation_time=model_profile.time_profiled_list[i],
                other_time_profiled=model_profile.other_time_profiled_list[0],
            )
            
            profile_hardware_args = ProfileHardwareArgs(
                bct_fct_coe=2,
                extra_overhead=0,
                comm_coe_dict=hardware_config.allreduce_comm_coe,
                dp_overlap_coe=hardware_config.overlap_coe,
                bct_overlap_coe=hardware_config.overlap_coe,
                p2p_comm_coe_dict=hardware_config.p2p_comm_coe,
                costmodel_coe=args.costmodel_coe,
                allreduce_dict=hardware_config.sp_allreduce,
                all2all_dict=hardware_config.sp_all2all,
            )
            
            model_args_list.append(model_args)
            train_args_list.append(train_args)
            parallel_args_list.append(parallel_args)
            profile_model_args_list.append(profile_model_args)
            profile_hardware_args_list.append(profile_hardware_args)
        
        return CostModelArgs(
            model_args_list=model_args_list,
            train_args_list=train_args_list,
            parallel_args_list=parallel_args_list,
            profile_model_args_list=profile_model_args_list,
            profile_hardware_args_list=profile_hardware_args_list
        )