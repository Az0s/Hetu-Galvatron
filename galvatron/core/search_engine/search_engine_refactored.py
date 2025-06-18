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
    
    def set_searching_bsz(self):
        """Set searching batch sizes based on arguments."""
        args = self.args
        # Set Searching BSZs
        if args.settle_bsz is not None and args.settle_bsz > 0:
            self.min_bsz = self.max_bsz = args.settle_bsz
            self.bsz_scale = 0
            self.BSZs = [args.settle_bsz]
            print('-----', '[Searching Batch Sizes Info]', 'Settle bsz:', args.settle_bsz, '-----')
            return
        self.bsz_scale = args.bsz_scale

        if args.recommend_min_bsz:
            recommend_bsz = self.recommend_min_bsz(self.bsz_scale)
            args.min_bsz = recommend_bsz if recommend_bsz > 0 else args.min_bsz
        
        self.min_bsz = max(args.min_bsz, self.bsz_scale)
        self.min_bsz = self.min_bsz // self.bsz_scale * self.bsz_scale
        self.max_bsz = int(np.ceil(args.max_bsz / self.bsz_scale) * self.bsz_scale) if args.max_bsz % self.bsz_scale else (args.max_bsz+self.bsz_scale)
        self.BSZs = list(range(self.min_bsz, self.max_bsz, self.bsz_scale))
        self.max_bsz = self.BSZs[-1]
        print('-----', '[Searching Batch Sizes Info]', 'Min bsz:', self.min_bsz, 'Max bsz:', self.max_bsz, 'bsz_scale:', self.bsz_scale, '-----')

    def recommend_min_bsz(self, scale):
        """Recommend minimum batch size based on baseline strategies."""
        prune_percent = 0.65
        args = self.args
        gpu_num = args.gpu_num
        if not args.search_space in ['full', 'dp+pp', 'dp+tp']:
            return -1
        baselines = []
        if not args.disable_dp:
            baselines.append([1,1,gpu_num,{'fsdp':0}])
        if not args.disable_sdp:
            baselines.append([1,1,gpu_num,{'fsdp':1}])
        if not args.disable_tp:
            baselines.append([1,gpu_num,1,{'fsdp':0}])
        max_bsz_baselines = [self.estimate_strategy_max_bsz([s], scale) for s in baselines]
        max_bsz, min_bsz = np.max(max_bsz_baselines), np.min(max_bsz_baselines)
        bsz_start = int((min_bsz*(1-prune_percent)+max_bsz*prune_percent)//scale*scale)
        bsz_start = bsz_start if bsz_start > scale else scale
        return bsz_start

    def estimate_strategy_max_bsz(self, strategies, scale):
        """Estimate maximum batch size for given strategies."""
        max_bsz = 0
        bsz = scale
        while True:
            from .search_engine import get_pp_stage_for_bsz
            pp_stage_dict = get_pp_stage_for_bsz(strategies, self.model_args_list, self.train_args_list, self.parallel_args_list, self.profile_model_args_list, self.layernum_list, bsz)
            dp_on_model = DpOnModel(strategies, MemoryCostModel, TimeCostModel, 
                                    model_args_list=self.model_args_list, train_args_list=self.train_args_list,
                                    parallel_args_list=self.parallel_args_list, profile_model_args_list=self.profile_model_args_list,
                                    profile_hardware_args_list=self.profile_hardware_args_list,
                                    max_mem=self.memory_constraint, layer_num=self.layernum_list, sequence_len = self.seqlen_list, 
                                    multi_layer_type = True, pp_stage_dict = pp_stage_dict,
                                    comm_coe_dict=self.allreduce_comm_coe, gpu_num=self.args.gpu_num,
                                    config = self.args)
            min_cost, min_res_list, min_pp_deg, mem_remain, mem_cost, min_vtp = dp_on_model.fit(bsz, 1, 1, 0, 1, print_=False, mbsz_dict = {1:bsz})
            if min_pp_deg == -1:
                max_bsz = bsz - scale
                break
            bsz += scale
        return max_bsz

    def dynamic_programming(self, strategies, bsz, chunk, mbsz_dict, pp_stage_dict, min_tp, max_tp, vsp, embed_sdp, sp_search, logger):
        """Execute dynamic programming optimization."""
        args = self.args
        logger.info(f'bsz={bsz} {pp_stage_dict}')
        dp_on_model = DpOnModel(strategies, 
                                MemoryCostModel, 
                                TimeCostModel, 
                                model_args_list=self.model_args_list,
                                train_args_list=self.train_args_list,
                                parallel_args_list=self.parallel_args_list,
                                profile_model_args_list=self.profile_model_args_list,
                                profile_hardware_args_list=self.profile_hardware_args_list,
                                max_mem=self.memory_constraint,
                                layer_num=self.layernum_list,
                                sequence_len = self.seqlen_list,
                                multi_layer_type = True,
                                pp_stage_dict = pp_stage_dict,
                                search_history=self.search_history,
                                comm_coe_dict=self.allreduce_comm_coe,
                                gpu_num=args.gpu_num,
                                model_microbatch_after_dp=args.use_pipeline_costmodel,
                                pipeline_type=args.pipeline_type,
                                config = self.args,
                                logger=logger)
        
        logger.info(f"****Searching with bsz={bsz} chunk={chunk} min_tp={min_tp} max_tp={max_tp} vsp={vsp} embed_sdp={embed_sdp} sp_search={sp_search}****")
        from .search_engine import check_optimal_chunks
        chunk_dict = check_optimal_chunks(args.gpu_num, strategies, self.optimal_chunk_func, bsz, mbsz_dict, min_tp)
        logger.info(f'Chunk_dict for bsz {bsz}: {chunk_dict}')
        logger.info(f'Mbsz_dict for bsz {bsz}: {mbsz_dict}')
        
        min_cost, min_res_list, min_pp_deg, mem_remain, mem_cost, min_vtp = dp_on_model.fit(bsz, min_tp, max_tp, vsp, embed_sdp, sp_search, mbsz_dict = mbsz_dict)
        throughput = bsz / min_cost
        logger.info(f"[Optimal pp_deg={min_pp_deg}] Minimized timecost={min_cost} Memory remaining={mem_remain} Memory cost={mem_cost} Vocab tp={min_vtp}")
        logger.info(f"Max throughput={throughput} samples/s")
        print_strategies(min_res_list, logger)
        result = {'min_cost': min_cost, 'min_res_list': min_res_list, 'min_pp_deg': min_pp_deg, 
                        'mem_remain': mem_remain, 'mem_cost': mem_cost, 'throughput': throughput, "vtp": min_vtp}
        return result

    def save_results(self, results, bsz, chunk, pp_stage_dict):
        """Save optimization results to configuration file."""
        re, optimal_bsz = results, bsz
        args = self.args
        if re['min_pp_deg'] > 0 and re['min_res_list'] is not None:
            result_strategy = []
            if isinstance(re['min_res_list'],list) and isinstance(re['min_res_list'][0],list) and isinstance(re['min_res_list'][0][0],list):
                for l in re['min_res_list']:
                    result_strategy += l
            else:
                result_strategy = re['min_res_list']
            
            config = strategy2config(result_strategy)
            config['checkpoint'] = array2str([1 if 'cpt' in s[-1] and s[-1]['cpt'] else 0 for s in result_strategy])
            config['global_bsz'] = optimal_bsz
            config['chunks'] = chunk
            config['pp_division'] = array2str(pp_stage_dict[config['pp_deg']])
            config['pipeline_type'] = args.pipeline_type
            config['default_dp_type'] = args.default_dp_type
            config['vtp'] = re['vtp']
            config['vsp'] = re['vsp']
            config['embed_sdp'] = re['embed_sdp']
            
            # Use path manager to get output path
            config_path = self.path_manager.get_output_config_path(self.memory_constraint)
            print(config_path)
            write_json_config(config, config_path)
            print('Already written optimized parallelism config into galvatron config file %s!'%(config_path))

    def generate_strategies(self):
        """Generate parallelization strategies."""
        args = self.args
        gpu_num = args.gpu_num
        strategies = self.generate_dp_tp_pp_sdp()
        if args.search_space == 'dp+tp':
            args.disable_sdp = 1
            args.disable_pp = 1
        elif args.search_space == 'dp+pp':
            args.disable_sdp = 1
            args.disable_tp = 1
        elif args.search_space == '3d':
            args.disable_sdp = 1
        if args.search_space in ['3d', 'dp', 'tp', 'pp', 'sdp']:
            self.strategies = strategies
            args.disable_ckpt = 1
            return strategies
        strategies_new = []
        assert(not(args.disable_sdp and args.disable_dp))
        for s in strategies:
            if args.disable_dp and s[2] > 1 and 'fsdp' in s[-1] and s[-1]['fsdp'] == 0:
                continue
            if args.disable_sdp and s[2] > 1 and 'fsdp' in s[-1] and s[-1]['fsdp'] == 1:
                continue
            if args.disable_tp and s[1] > 1:
                continue
            if args.disable_pp and s[0] > 1:
                continue
            if args.disable_tp_consec and 'tp' in s[-1] and s[-1]['tp'] == 0:
                continue
            if s[1] > args.max_tp_deg:
                continue
            if s[0] > args.max_pp_deg:
                continue
            strategies_new.append(s)
        strategies = strategies_new

        if not args.disable_ckpt:
            strategies_cpt = []
            for s in strategies:
                s_cpt = copy.deepcopy(s)
                s_cpt[-1]['cpt']=1
                strategies_cpt.append(s_cpt)
            strategies += strategies_cpt
        self.strategies = strategies
        return strategies
    
    def generate_dp_tp_pp_sdp(self, gpu_num=None, search_space=None):
        """Generate DP, TP, PP, SDP strategies."""
        args = self.args
        gpu_num = args.gpu_num if gpu_num is None else gpu_num
        search_space = args.search_space if search_space is None else search_space
        i, total = 1, []
        while i<=gpu_num:
            total.append(i)
            i *= 2
        if args.search_space == 'full':
            strategies = []
            for pp in total:
                for tp in total:
                    if pp*tp<=gpu_num:
                        dp = gpu_num // (pp * tp) 
                        if tp==1 or tp == gpu_num/pp:
                            if dp == 1:
                                strategies.append([pp,tp,dp,{}])
                            else:
                                strategies.append([pp,tp,dp,{'fsdp':0}])
                                strategies.append([pp,tp,dp,{'fsdp':1}])
                        else:
                            strategies.append([pp,tp,dp,{'tp':0,'fsdp':0}])
                            strategies.append([pp,tp,dp,{'tp':0,'fsdp':1}])
                            strategies.append([pp,tp,dp,{'tp':1,'fsdp':0}])
                            strategies.append([pp,tp,dp,{'tp':1,'fsdp':1}])
        elif args.search_space == 'dp+tp':
            strategies = []
            pp = 1
            for tp in total:
                if pp*tp<=gpu_num:
                    dp = gpu_num // (pp * tp) 
                    if tp==1 or tp == gpu_num/pp:
                        if dp == 1:
                            strategies.append([pp,tp,dp,{}])
                        else:
                            strategies.append([pp,tp,dp,{'fsdp':0}])
                    else:
                        strategies.append([pp,tp,dp,{'tp':0,'fsdp':0}])
                        strategies.append([pp,tp,dp,{'tp':1,'fsdp':0}])
        elif args.search_space == 'dp+pp':
            strategies = []
            tp = 1
            for pp in total:
                if pp*tp<=gpu_num:
                    dp = gpu_num // (pp * tp) 
                    if tp==1 or tp == gpu_num/pp:
                        if dp == 1:
                            strategies.append([pp,tp,dp,{}])
                        else:
                            strategies.append([pp,tp,dp,{'fsdp':0}])
                    else:
                        strategies.append([pp,tp,dp,{'tp':0,'fsdp':0}])
                        strategies.append([pp,tp,dp,{'tp':1,'fsdp':0}])
        elif args.search_space == '3d':
            strategies = [[2,2,gpu_num//4,{'tp':1,'fsdp':0}]]
        elif args.search_space == 'dp':
            strategies = [[1,1,gpu_num,{'fsdp':0}]]
        elif args.search_space == 'sdp':
            strategies = [[1,1,gpu_num,{'fsdp':1}]]
        elif args.search_space == 'tp':
            strategies = [[1,args.max_tp_deg,gpu_num//args.max_tp_deg,{'fsdp':0}]]
            if strategies[0][2] > 1:
                strategies[0][-1]['tp'] = 1
        elif args.search_space == 'pp':
            strategies = [[args.max_pp_deg,1,gpu_num//args.max_pp_deg,{'fsdp':0}]]
        
        if args.sp_space == 'tp':
            for strategie in strategies:
                if strategie[1] > 1:
                    strategie[-1]['sp'] = 0
        elif args.sp_space == 'sp':
            for strategie in strategies:
                if strategie[1] > 1:
                    strategie[-1]['sp'] = 1
        elif args.sp_space == 'tp+sp':
            new_strategies = []
            for strategie in strategies:
                if strategie[1] > 1:
                    strategie[-1]['sp'] = 0
                    new_strategies.append(copy.deepcopy(strategie))
                    strategie[-1]['sp'] = 1
                    new_strategies.append(copy.deepcopy(strategie))
                else:
                    new_strategies.append(copy.deepcopy(strategie))
            return new_strategies
        return strategies

    def show_search_info(self):
        """Display search engine configuration information."""
        console = Console()
        
        # Create title
        title = Text("🚀 Galvatron Search Engine Configuration", style="bold cyan")
        console.print(Panel(title, border_style="cyan"))
        
        # Optimization Configs Section
        opt_table = Table(title="⚙️ Optimization Configs", title_style="bold blue", border_style="blue")
        opt_table.add_column("Configuration", style="cyan", width=20)
        opt_table.add_column("Value", style="white")
        
        opt_table.add_row("Memory Constraint", f"{self.args.memory_constraint} GB")
        opt_table.add_row("Pipeline Type", self.args.pipeline_type)
        opt_table.add_row("Default DP Type", self.args.default_dp_type)
        opt_table.add_row("Mixed Precision", self.args.mixed_precision)
        
        console.print(opt_table)
        console.print()
        
        # Search Space Section
        search_panel = Panel.fit(
            f"[bold yellow]Search Space Strategies:[/bold yellow]\n[dim]Total strategies: {len(self.strategies)}[/dim]",
            border_style="yellow"
        )
        console.print(search_panel)
        
        # Environment Configs Section  
        env_table = Table(title="🌐 Environment Configs", title_style="bold green", border_style="green")
        env_table.add_column("Hardware Metric", style="cyan", width=25)
        env_table.add_column("Value", style="white")
        
        env_table.add_row("Allreduce Bandwidth", f"{self.allreduce_bandwidth} GB/s")
        env_table.add_row("Allreduce Comm Coefficient", f"{self.allreduce_comm_coe} ms/MB")
        env_table.add_row("P2P Bandwidth", f"{self.p2p_bandwidth} GB/s") 
        env_table.add_row("P2P Comm Coefficient", f"{self.p2p_comm_coe} ms/MB")
        env_table.add_row("Overlap Coefficient", str(self.overlap_coe))
        
        console.print(env_table)
        console.print()
        
        # Model Configs Section
        model_table = Table(title="🤖 Model Configs", title_style="bold magenta", border_style="magenta")
        model_table.add_column("Model Property", style="cyan", width=20)
        model_table.add_column("Value", style="white")
        
        model_table.add_row("Model Name", str(getattr(self.args, 'model_name', 'Unknown')))
        model_table.add_row("Num Layertype", str(self.num_layertype))
        model_table.add_row("Layer Numbers", str(self.layernum_list))
        model_table.add_row("Hidden Sizes", str(self.hiddensize_list))
        model_table.add_row("Sequence Lengths", str(self.seqlen_list))
        
        console.print(model_table)
        console.print()
        
        # Model Computation Configs
        comp_panel = Panel.fit(
            f"[bold red]⚡ Model Computation Configs[/bold red]\n"
            f"Forward computation time: {self.time_profiled_list}",
            border_style="red"
        )
        console.print(comp_panel)
        console.print()
        
        # Model Memory Configs
        mem_table = Table(title="💾 Model Memory Configs", title_style="bold yellow", border_style="yellow")
        mem_table.add_column("Memory Component", style="cyan", width=25)
        mem_table.add_column("Details", style="white")
        
        mem_table.add_row("Parameter Memory Cost", str(self.param_sizes))
        mem_table.add_row("Activation Memory (per bsz)", str(self.act_sizes))
        mem_table.add_row("Other Memory (pp=1)", str(self.other_memory_pp_off))
        mem_table.add_row("Other Memory (pp>1)", str(self.other_memory_pp_on))
        
        console.print(mem_table)
        console.print()
        
        # Footer
        footer = Text("✅ Configuration Summary Complete", style="bold green")
        console.print(Panel(footer, border_style="green"))


# Import utility functions that were originally part of the main file
def optimal_chunk_func_default(local_bsz, strategy, microbatch_size, min_tp):
    """Default optimal chunk function."""
    import numpy as np
    assert(strategy[1] % min_tp == 0)
    local_bsz = local_bsz // (strategy[1] // min_tp)
    chunk = np.ceil(local_bsz / microbatch_size)
    chunk = 1 if chunk == 0 else chunk
    return chunk


# Import needed utility functions
def get_pp_stage_for_bsz(strategies, model_args_list, train_args_list, parallel_args_list, profile_model_args_list, layer_num_list, bsz, mbsz_dict=None, single_layer_even=True):
    """Import this function from the original search_engine module."""
    from .search_engine import get_pp_stage_for_bsz as original_get_pp_stage_for_bsz
    return original_get_pp_stage_for_bsz(strategies, model_args_list, train_args_list, parallel_args_list, profile_model_args_list, layer_num_list, bsz, mbsz_dict, single_layer_even)