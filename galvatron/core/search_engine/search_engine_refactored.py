"""
Refactored Galvatron Search Engine.

This module contains the refactored GalvatronSearchEngine class that focuses on
core search logic while delegating data loading and configuration management
to specialized classes.
"""
import os
import copy
import logging
import numpy as np
from galvatron.utils import (
    form_strategy, 
    print_strategies,
    strategy2config,
    array2str,
    write_json_config,
)
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from .cost_model import pipeline_costmodel
from .cost_model import MemoryCostModel, TimeCostModel
from .dynamic_programming import DpOnModel
from .utils import ensure_log_dir, get_thread_logger
from .config_types import SearchEngineConfig, SearchConfig
from .config_path_manager import ConfigPathManager


class GalvatronSearchEngine:
    """
    Refactored Galvatron Search Engine focused on core search functionality.
    
    This class has been refactored to follow the single responsibility principle,
    focusing only on the core search and optimization logic while delegating
    data loading and configuration management to specialized classes.
    """
    
    def __init__(self, args, config: SearchEngineConfig, path_manager: ConfigPathManager):
        """
        Initialize search engine with dependency injection.
        
        Args:
            args: Command line arguments
            config: Complete search engine configuration
            path_manager: Configuration path manager
        """
        self.args = args
        self.config = config
        self.path_manager = path_manager
        
        # Extract commonly used configurations for backward compatibility
        self.model_layer_config = config.model_layer_config
        self.hardware_config = config.hardware_config
        self.model_profile = config.model_profile
        self.search_config = config.search_config
        self.cost_model_args = config.cost_model_args
        
        # Core engine settings
        self.model_type = config.model_type
        self.use_pipeline_costmodel = config.use_pipeline_costmodel
        self.memory_constraint = config.memory_constraint
        self.optimal_chunk_func = config.optimal_chunk_func
        
        # Initialize search state
        self.search_config.search_history = {}
        self.strategies = None
        
        # Set up args.gpu_num for compatibility
        args.gpu_num = args.num_nodes * args.num_gpus_per_node
    
    # =============== Legacy compatibility methods ===============
    @property
    def hiddensize_list(self):
        """Legacy property for backward compatibility."""
        return self.model_layer_config.hiddensize_list
    
    @property
    def layernum_list(self):
        """Legacy property for backward compatibility."""
        return self.model_layer_config.layernum_list
    
    @property
    def seqlen_list(self):
        """Legacy property for backward compatibility."""
        return self.model_layer_config.seqlen_list
    
    @property
    def num_layertype(self):
        """Legacy property for backward compatibility."""
        return self.model_layer_config.num_layertype
    
    @property
    def param_sizes(self):
        """Legacy property for backward compatibility."""
        return self.model_profile.param_sizes
    
    @property
    def act_sizes(self):
        """Legacy property for backward compatibility."""
        return self.model_profile.act_sizes
    
    @property
    def other_memory_pp_off(self):
        """Legacy property for backward compatibility."""
        return self.model_profile.other_memory_pp_off
    
    @property
    def other_memory_pp_on(self):
        """Legacy property for backward compatibility."""
        return self.model_profile.other_memory_pp_on
    
    @property
    def time_profiled_list(self):
        """Legacy property for backward compatibility."""
        return self.model_profile.time_profiled_list
    
    @property
    def other_time_profiled_list(self):
        """Legacy property for backward compatibility."""
        return self.model_profile.other_time_profiled_list
    
    @property
    def allreduce_bandwidth(self):
        """Legacy property for backward compatibility."""
        return self.hardware_config.allreduce_bandwidth
    
    @property
    def allreduce_comm_coe(self):
        """Legacy property for backward compatibility."""
        return self.hardware_config.allreduce_comm_coe
    
    @property
    def p2p_bandwidth(self):
        """Legacy property for backward compatibility."""
        return self.hardware_config.p2p_bandwidth
    
    @property
    def p2p_comm_coe(self):
        """Legacy property for backward compatibility."""
        return self.hardware_config.p2p_comm_coe
    
    @property
    def overlap_coe(self):
        """Legacy property for backward compatibility."""
        return self.hardware_config.overlap_coe
    
    @property
    def sp_allreduce(self):
        """Legacy property for backward compatibility."""
        return self.hardware_config.sp_allreduce
    
    @property
    def sp_all2all(self):
        """Legacy property for backward compatibility."""
        return self.hardware_config.sp_all2all
    
    @property
    def model_args_list(self):
        """Legacy property for backward compatibility."""
        return self.cost_model_args.model_args_list
    
    @property
    def train_args_list(self):
        """Legacy property for backward compatibility."""
        return self.cost_model_args.train_args_list
    
    @property
    def parallel_args_list(self):
        """Legacy property for backward compatibility."""
        return self.cost_model_args.parallel_args_list
    
    @property
    def profile_model_args_list(self):
        """Legacy property for backward compatibility."""
        return self.cost_model_args.profile_model_args_list
    
    @property
    def profile_hardware_args_list(self):
        """Legacy property for backward compatibility."""
        return self.cost_model_args.profile_hardware_args_list
    
    @property
    def search_history(self):
        """Legacy property for backward compatibility."""
        return self.search_config.search_history
    
    @search_history.setter
    def search_history(self, value):
        """Legacy property setter for backward compatibility."""
        self.search_config.search_history = value
    
    @property
    def min_bsz(self):
        """Legacy property for backward compatibility."""
        return self.search_config.min_bsz
    
    @min_bsz.setter
    def min_bsz(self, value):
        """Legacy property setter for backward compatibility."""
        self.search_config.min_bsz = value
    
    @property
    def max_bsz(self):
        """Legacy property for backward compatibility."""
        return self.search_config.max_bsz
    
    @max_bsz.setter
    def max_bsz(self, value):
        """Legacy property setter for backward compatibility."""
        self.search_config.max_bsz = value
    
    @property
    def bsz_scale(self):
        """Legacy property for backward compatibility."""
        return self.search_config.bsz_scale
    
    @bsz_scale.setter
    def bsz_scale(self, value):
        """Legacy property setter for backward compatibility."""
        self.search_config.bsz_scale = value
    
    @property
    def BSZs(self):
        """Legacy property for backward compatibility."""
        return self.search_config.BSZs
    
    @BSZs.setter
    def BSZs(self, value):
        """Legacy property setter for backward compatibility."""
        self.search_config.BSZs = value
    
    # =============== Core Search Engine Methods ===============
    def initialize_search_engine(self):
        """Initialize the search engine - now only handles strategy generation."""
        self.generate_strategies()
        self.show_search_info()
    
    def set_microbatch_func(self, microbatch_size, max_chunk):
        """Set microbatch function for the search engine."""
        self.optimal_chunk_func = lambda local_bsz, strategy, mbsz, min_tp: optimal_chunk_func_default(local_bsz, strategy, microbatch_size, min_tp)
    
    # =============== Core Search Logic (unchanged) ===============
    def parallelism_optimization(self):
        """Core parallelism optimization algorithm."""
        print('='*25, 'Galvatron Search Engine Start Searching','='*25)
        self.set_searching_bsz()
        
        print('-----', '[Searching Memory Info]', 'Memory constraint:', self.memory_constraint, 'MB', '-----')
        results = dict()
        self.search_history = dict()
        temp_strategies = copy.deepcopy(self.strategies)
        max_throughput, optimal_bsz, max_bsz = -1, -1, -1
        
        total_min_tp = []
        i = 1
        while i<=self.args.gpu_num and i <= self.args.max_tp_deg:
            total_min_tp.append(i)
            i *= 2
        if self.args.disable_vtp:
            total_min_tp = [1]
        if not self.args.global_memory_buffer:
            total_max_tp = [self.args.max_tp_deg]
            sp_search_speace = [1, 3]
        else:
            total_max_tp = total_min_tp
            sp_search_speace = [1, 2, 3] # 1 tp, 2 sp, 3 tp+sp
        
        if self.args.sp_space == 'tp+sp':
            total_vsp = [0, 1]
        elif self.args.sp_space == 'tp':
            total_vsp = [0]
            sp_search_speace = [1]
        elif self.args.sp_space == 'sp':
            assert False,"Only sp mode unsupport now."
            total_vsp = [1]
            sp_search_speace = [2]

        if self.args.disable_sdp:
            total_embed_sdp = [0]
        else:
            total_embed_sdp = [0, 1]

        def search_for_chunk(bsz, chunk, min_tp, max_tp, vsp, embed_sdp):
            log_dir = self.args.log_dir + '/%s_%dnodes_%dgpus_%dGB'%(self.args.model_name if hasattr(self.args, 'model_name') else 'model', self.args.num_nodes, self.args.num_gpus_per_node, self.memory_constraint//1024)
            log_dir = ensure_log_dir(log_dir)
            logger = get_thread_logger(bsz, chunk, min_tp, max_tp, vsp, embed_sdp, log_dir)
            logger.info(f"Starting search for bsz={bsz}, chunk={chunk}, min_tp={min_tp}, max_tp={max_tp}, vsp={vsp}, embed_sdp={embed_sdp}")

            results = dict()
            
            for sp_search in sp_search_speace:
                if sp_search == 1 and vsp == 1:
                    continue
                if sp_search == 2 and vsp == 0:
                    continue
        
                strategies = [s for s in temp_strategies if min_tp <= s[1] and max_tp >= s[1]]
                strategies = [s for s in strategies if chunk <= bsz // (self.args.gpu_num // s[0] // min_tp) ]
                if sp_search == 1:
                    strategies = [s for s in strategies if 'sp' not in s[-1] or ('sp' in s[-1] and s[-1]['sp'] == 0)]
                if sp_search == 2:
                    strategies = [s for s in strategies if 'sp' not in s[-1] or ('sp' in s[-1] and s[-1]['sp'] == 1)]
                if len(strategies) == 0:
                    continue
                
                pp_deg_list = sorted(list(set([s[0] for s in strategies])))
                
                pp_deg_list = [pp for pp in pp_deg_list if pp * min_tp <= self.args.gpu_num and bsz % (self.args.gpu_num // pp // min_tp) == 0]
                
                if len(pp_deg_list) == 0:
                    continue
                
                strategies = [s for s in strategies if s[0] in pp_deg_list]
                
                mbsz_dict = dict() # calc micro batch size in different pp size when tp = min_tp
                for pp in pp_deg_list:
                    mbsz_dict[pp] = (bsz // (self.args.gpu_num // pp // min_tp) + chunk - 1) // chunk
                
                # strict mode: search chunk must be equal to real chunk 
                strategies = [s for s in strategies if chunk == (bsz // (self.args.gpu_num // s[0] // min_tp) + mbsz_dict[s[0]] - 1) // mbsz_dict[s[0]]]
                
                if len(strategies) == 0:
                    continue

                pp_stage_dict = get_pp_stage_for_bsz(strategies, self.model_args_list, self.train_args_list, self.parallel_args_list, self.profile_model_args_list, self.layernum_list, bsz, mbsz_dict)
                
                results[sp_search] = self.dynamic_programming(strategies, bsz, chunk, mbsz_dict, pp_stage_dict, min_tp, max_tp, vsp, embed_sdp, sp_search, logger)
                results[sp_search]['pp_stage_dict'] = copy.deepcopy(pp_stage_dict)
            return results
        
        # Continue with the rest of the parallelism_optimization method...
        # (The rest of the method remains the same as in the original)
        if self.args.parallel_search:
            import concurrent.futures
            import threading
            import time
    
            all_tasks = []
            for bsz in self.BSZs:
                results[bsz] = dict()
                chunk_list = range(1, bsz+1)
                if self.args.settle_chunk != -1:
                    chunk_list = [self.args.settle_chunk]
                
                for chunk in chunk_list:
                    if bsz % chunk != 0:
                        continue
                    results[bsz][chunk] = dict()
                    for min_tp in total_min_tp:
                        results[bsz][chunk][min_tp] = dict()
                        for max_tp in total_max_tp:
                            if min_tp > max_tp:
                                continue
                            results[bsz][chunk][min_tp][max_tp] = dict()
                            for vsp in total_vsp:
                                results[bsz][chunk][min_tp][max_tp][vsp] = dict()
                                for embed_sdp in total_embed_sdp:
                                    results[bsz][chunk][min_tp][max_tp][vsp][embed_sdp] = dict()
                                    all_tasks.append((bsz, chunk, min_tp, max_tp, vsp, embed_sdp))
            
            results_lock = threading.Lock()

            import multiprocessing
            if hasattr(self.args, 'worker') and self.args.worker > 0:
                num_threads = min(self.args.worker, len(all_tasks))
            else:
                num_threads = min(multiprocessing.cpu_count() * 2, len(all_tasks))
            print(f"Starting parallel search with {num_threads} threads for {len(all_tasks)} tasks...")
            
            def process_task(bsz, chunk, min_tp, max_tp, vsp, embed_sdp):
                thread_id = threading.get_ident() % 1000
                print(f"[Thread {thread_id:03d}] Start processing: bsz={bsz}, chunk={chunk}, min_tp={min_tp}, max_tp={max_tp}, vsp={vsp}, embed_sdp={embed_sdp}", flush=True)

                chunk_results = search_for_chunk(bsz, chunk, min_tp, max_tp, vsp, embed_sdp)
                with results_lock:
                    results[bsz][chunk][min_tp][max_tp][vsp][embed_sdp] = copy.deepcopy(chunk_results)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(process_task, bsz, chunk, min_tp, max_tp, vsp, embed_sdp) for bsz, chunk, min_tp, max_tp, vsp, embed_sdp in all_tasks]
                concurrent.futures.wait(futures)
        else:
            for bsz in self.BSZs:
                results[bsz] = dict()
                chunk_list = range(1,bsz+1)
                if self.args.settle_chunk != -1:
                    chunk_list = [self.args.settle_chunk]
                for chunk in chunk_list:
                    results[bsz][chunk] = dict()
                    if bsz % chunk != 0:
                        continue
                    for min_tp in total_min_tp:
                        results[bsz][chunk][min_tp] = dict()
                        for max_tp in total_max_tp:
                            if min_tp > max_tp:
                                continue
                            results[bsz][chunk][min_tp][max_tp] = dict()
                            for vsp in total_vsp:
                                results[bsz][chunk][min_tp][max_tp][vsp] = dict()
                                for embed_sdp in total_embed_sdp:
                                    print(f"Start processing: bsz={bsz}, chunk={chunk}, min_tp={min_tp}, max_tp={max_tp}, vsp={vsp}, embed_sdp={embed_sdp}", flush=True)

                                    results[bsz][chunk][min_tp][max_tp][vsp][embed_sdp] = search_for_chunk(bsz, chunk, min_tp, max_tp, vsp, embed_sdp)

        for bsz in results:
            for chunk in results[bsz]:
                for min_tp in results[bsz][chunk]:
                    for max_tp in results[bsz][chunk][min_tp]:
                        for vsp in results[bsz][chunk][min_tp][max_tp]:
                            for embed_sdp in results[bsz][chunk][min_tp][max_tp][vsp]:
                                for sp_search in results[bsz][chunk][min_tp][max_tp][vsp][embed_sdp]:
                                    throughput = results[bsz][chunk][min_tp][max_tp][vsp][embed_sdp][sp_search]['throughput']
                                    pp_stage_dict = results[bsz][chunk][min_tp][max_tp][vsp][embed_sdp][sp_search]['pp_stage_dict']
                                    if throughput > max_throughput:
                                        max_throughput = throughput
                                        optimal_bsz = bsz
                                        optimal_chunk = chunk
                                        optimal_min_tp = min_tp
                                        optimal_max_tp = max_tp
                                        optimal_vsp = vsp
                                        optimal_embed_sdp = embed_sdp
                                        optimal_sp_search = sp_search
                                        optimal_pp_stage_dict = pp_stage_dict

        if max_throughput > 0:
            print('\nFinal results of max memory %d MB:'%self.memory_constraint)
            re = results[optimal_bsz][optimal_chunk][optimal_min_tp][optimal_max_tp][optimal_vsp][optimal_embed_sdp][optimal_sp_search]
            re['vsp'] = optimal_vsp
            re['embed_sdp'] = optimal_embed_sdp
            print(f"Optimal bsz = {optimal_bsz} Optimal chunk = {optimal_chunk} Optimal vocab tp = {re['vtp']} Optimal vocab sp = {optimal_vsp} Optimal embed sdp = {optimal_embed_sdp} Max throughput={re['throughput']} samples/s")
            print(f"pp_deg={re['min_pp_deg']} Minimized timecost={re['min_cost']} Memory remaining={re['mem_remain']} Memory cost={re['mem_cost']}")
            print(f"Min_tp={optimal_min_tp} Max_tp={optimal_max_tp} ")
            print_strategies(re['min_res_list'])
            
            self.save_results(re, optimal_bsz, optimal_chunk, optimal_pp_stage_dict)
        else:
            print("No valid configuration found.")
        
        print("-----------------------------------------")
        print('='*25, 'Galvatron Search Engine End Searching','='*25)

        return max_throughput
    
    # Include other necessary methods from the original class...
    # (Copy the remaining methods that are needed for the core functionality)


# Import utility functions that were originally part of the main file
from .search_engine import (
    pp_division_memory_balanced, get_pp_stage_for_bsz, get_cost_all_stages,
    get_layer_costs, pp_division_even, optimal_chunk_func_default, check_optimal_chunks
)