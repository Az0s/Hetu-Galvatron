"""
Profile data loading and preprocessing for Galvatron Search Engine.

This module handles all data loading and preprocessing operations, including
different profiling modes (static, batch, sequence) and data fitting algorithms.
"""
import copy
from typing import Dict, List, Tuple, Any
from scipy.optimize import curve_fit
from galvatron.utils import (
    read_allreduce_bandwidth_config, 
    read_json_config, 
    read_p2p_bandwidth_config,
    remap_config,
    num2str
)
from .config_types import ModelLayerConfig, HardwareConfig, ModelProfile
from .config_path_manager import ConfigPathManager


class ProfileDataLoader:
    """Handles profile data loading and preprocessing."""
    
    def __init__(self, args, model_layer_config: ModelLayerConfig, path_manager: ConfigPathManager):
        """
        Initialize profile data loader.
        
        Args:
            args: Command line arguments
            model_layer_config: Model layer configuration
            path_manager: Configuration path manager
        """
        self.args = args
        self.model_layer_config = model_layer_config
        self.path_manager = path_manager
        
    def load_model_profile(self) -> ModelProfile:
        """Load and process model profiling data."""
        # Load raw configs
        time_config = read_json_config(self.path_manager.get_time_config_path())
        memory_config = read_json_config(self.path_manager.get_memory_config_path())
        memory_config = self._convert_keys_to_int(memory_config)
        
        # Process time data based on mode
        time_profiled_list, other_time_profiled_list = self._process_time_data(time_config)
        
        # Process memory data based on mode  
        param_sizes, act_sizes, other_memory_pp_off, other_memory_pp_on = self._process_memory_data(memory_config)
        
        return ModelProfile(
            param_sizes=param_sizes,
            act_sizes=act_sizes,
            other_memory_pp_off=other_memory_pp_off,
            other_memory_pp_on=other_memory_pp_on,
            time_profiled_list=time_profiled_list,
            other_time_profiled_list=other_time_profiled_list
        )
    
    def load_hardware_config(self) -> HardwareConfig:
        """Load hardware configuration data."""
        hardware_paths = self.path_manager.get_hardware_config_paths()
        
        # Load allreduce bandwidth config
        allreduce_bandwidth, allreduce_comm_coe = read_allreduce_bandwidth_config(
            hardware_paths['allreduce'], gpu_num=self.args.gpu_num
        )
        
        # Load P2P bandwidth config
        p2p_bandwidth, p2p_comm_coe = read_p2p_bandwidth_config(hardware_paths['p2p'])
        
        # Load overlap coefficient
        overlap_coe = read_json_config(hardware_paths['overlap'])['overlap_coe']
        
        # Load SP time config
        sp_config = read_json_config(hardware_paths['sp_time'])
        sp_allreduce = remap_config(sp_config, "allreduce")
        sp_all2all = remap_config(sp_config, "all2all")
        
        return HardwareConfig(
            allreduce_bandwidth=allreduce_bandwidth,
            allreduce_comm_coe=allreduce_comm_coe,
            p2p_bandwidth=p2p_bandwidth,
            p2p_comm_coe=p2p_comm_coe,
            overlap_coe=overlap_coe,
            sp_allreduce=sp_allreduce,
            sp_all2all=sp_all2all
        )
    
    def _convert_keys_to_int(self, d: Any) -> Any:
        """Convert string keys to integers where applicable."""
        if isinstance(d, dict):
            new_dict = {}
            for k, v in d.items():
                if isinstance(k, str) and k.isdigit():
                    new_dict[int(k)] = self._convert_keys_to_int(v)
                else:
                    new_dict[k] = self._convert_keys_to_int(v)
            return new_dict
        return d
    
    def _process_time_data(self, time_config: Dict) -> Tuple[List, List]:
        """Process time profiling data based on the configured mode."""
        if self.args.time_profile_mode == 'static':
            return self._process_time_data_static(time_config)
        elif self.args.time_profile_mode == 'batch':
            return self._process_time_data_batch(time_config)
        elif self.args.time_profile_mode == 'sequence':
            return self._process_time_data_sequence(time_config)
        else:
            raise ValueError(f"Unknown time profile mode: {self.args.time_profile_mode}")
    
    def _process_time_data_static(self, time_config: Dict) -> Tuple[List, List]:
        """Process time data in static mode."""
        time_profiled_list = []
        other_time_profiled_list = []
        
        for i in range(self.model_layer_config.num_layertype):
            for s, t in time_config.items():
                if s.startswith(f'layertype_{i}_'):
                    time_profiled_list.append(t)
                if s.startswith('layertype_other_'):
                    other_time_profiled_list.append(t)
        
        return time_profiled_list, other_time_profiled_list
    
    def _process_time_data_batch(self, time_config: Dict) -> Tuple[List, List]:
        """Process time data in batch mode with curve fitting."""
        time_profiled_list = []
        
        for i in range(self.model_layer_config.num_layertype):
            x_data = []
            y_data = []
            for s, t in time_config.items():
                if (s.startswith(f'layertype_{i}_') and 
                    f'_seq{self.model_layer_config.seqlen_list[i]}' in s):
                    x_data.append(int(s.split('_')[-2][3:]))
                    y_data.append(t * x_data[-1])
            
            assert len(x_data) >= 8, f"Different bsz in computation profile of layertype_{i} should not be lower than 8."
            
            def linear_func(x, m, c):
                return m * x + c
            
            popt, pcov = curve_fit(linear_func, x_data, y_data)
            print("Fitted parameters:", popt)
            time_profiled_list.append(popt)
        
        # Process other time data
        other_time_profiled_list = []
        for i in range(self.model_layer_config.num_layertype):
            x_data = []
            y_data = []
            for s, t in time_config.items():
                if (s.startswith('layertype_other_') and 
                    f'_seq{self.model_layer_config.seqlen_list[i]}' in s):
                    x_data.append(int(s.split('_')[-2][3:]))
                    y_data.append(t * x_data[-1])
            
            assert len(x_data) >= 8, f"Different bsz in computation profile of layertype_other_{i} should not be lower than 8."
            
            def linear_func(x, m, c):
                return m * x + c
            
            popt, pcov = curve_fit(linear_func, x_data, y_data)
            print("Fitted parameters other:", popt)
            other_time_profiled_list.append(popt)
        
        return time_profiled_list, other_time_profiled_list
    
    def _process_time_data_sequence(self, time_config: Dict) -> Tuple[List, List]:
        """Process time data in sequence mode with quadratic fitting."""
        time_profiled_list = []
        
        for i in range(self.model_layer_config.num_layertype):
            x_data = []
            y_data = []
            for s, t in time_config.items():
                if s.startswith(f'layertype_{i}_') and "_bsz1_" in s:
                    x_data.append(int(s.split('seq')[-1]))
                    y_data.append(t)
            
            def quadratic_func(x, a, b, c):
                return a * x * x + b * x + c
            
            popt, pcov = curve_fit(quadratic_func, x_data, y_data)
            print("Fitted parameters:", popt)
            time_profiled_list.append(quadratic_func(self.model_layer_config.seqlen_list[i], *popt))
        
        # Process other time data
        other_time_profiled_list = []
        for i in range(self.model_layer_config.num_layertype):
            x_data = []
            y_data = []
            for s, t in time_config.items():
                if s.startswith('layertype_other_') and "_bsz1_" in s:
                    x_data.append(int(s.split('seq')[-1]))
                    y_data.append(t)
            
            def linear_func(x, m, c):
                return m * x + c
            
            popt, pcov = curve_fit(linear_func, x_data, y_data)
            print("Fitted parameters other:", popt)
            other_time_profiled_list.append(linear_func(self.model_layer_config.seqlen_list[i], *popt))
        
        return time_profiled_list, other_time_profiled_list
    
    def _process_memory_data(self, memory_config: Dict) -> Tuple[List[int], List[Dict], Dict, Dict]:
        """Process memory profiling data based on the configured mode."""
        if self.args.memory_profile_mode == 'static':
            return self._process_memory_data_static(memory_config)
        elif self.args.memory_profile_mode == 'sequence':
            return self._process_memory_data_sequence(memory_config)
        else:
            raise ValueError(f"Unknown memory profile mode: {self.args.memory_profile_mode}")
    
    def _process_memory_data_static(self, memory_config: Dict) -> Tuple[List[int], List[Dict], Dict, Dict]:
        """Process memory data in static mode."""
        param_sizes = [0] * self.model_layer_config.num_layertype
        act_sizes = [{} for _ in range(self.model_layer_config.num_layertype)]
        
        if self.args.sequence_parallel:
            for i in range(self.model_layer_config.num_layertype):
                layer_mem_config = memory_config[f'layertype_{i}_sp']
                parameter_size = layer_mem_config[self.model_layer_config.seqlen_list[i]]['parameter_size']
                tp_activation_per_bsz_dict = layer_mem_config[self.model_layer_config.seqlen_list[i]]['tp_activation_per_bsz_dict'].copy()
                param_sizes[i] = parameter_size
                act_sizes[i] = tp_activation_per_bsz_dict
            
            seq_info = num2str(self.model_layer_config.seqlen_list, 'seq')[3:]
            if seq_info.isdigit():
                seq_info = int(seq_info)
            other_memory_pp_off = memory_config['other_memory_pp_off_sp'][int(seq_info)]
            other_memory_pp_on = {
                'first_stage': memory_config['other_memory_pp_on_first_sp'][seq_info], 
                'last_stage': memory_config['other_memory_pp_on_last_sp'][seq_info]
            }
        else:
            for i in range(self.model_layer_config.num_layertype):
                layer_mem_config = memory_config[f'layertype_{i}']
                parameter_size = layer_mem_config[self.model_layer_config.seqlen_list[i]]['parameter_size']
                tp_activation_per_bsz_dict = layer_mem_config[self.model_layer_config.seqlen_list[i]]['tp_activation_per_bsz_dict'].copy()
                param_sizes[i] = parameter_size
                act_sizes[i] = tp_activation_per_bsz_dict
            
            seq_info = num2str(self.model_layer_config.seqlen_list, 'seq')[3:]
            if seq_info.isdigit():
                seq_info = int(seq_info)
            other_memory_pp_off = memory_config['other_memory_pp_off'][seq_info]
            other_memory_pp_on = {
                'first_stage': memory_config['other_memory_pp_on_first'][seq_info], 
                'last_stage': memory_config['other_memory_pp_on_last'][seq_info]
            }
        
        return param_sizes, act_sizes, other_memory_pp_off, other_memory_pp_on
    
    def _process_memory_data_sequence(self, memory_config: Dict) -> Tuple[List[int], List[Dict], Dict, Dict]:
        """Process memory data in sequence mode."""
        assert self.args.sequence_parallel, "Sequence parallel is required for sequence memory profiling."
        assert self.model_layer_config.num_layertype == 1, "Only support num(layertype) == 1 for sequence memory profiling."
        
        param_sizes = [0] * self.model_layer_config.num_layertype
        act_sizes = [{} for _ in range(self.model_layer_config.num_layertype)]
        
        maxseq_list = []
        for i in range(self.model_layer_config.num_layertype):
            layer_mem_config = memory_config[f'layertype_{i}_sp']
            seqs = layer_mem_config.keys()
            maxseq = max([int(seq) for seq in seqs])
            minseq = min([int(seq) for seq in seqs])
            maxseq_list.append(maxseq)
            parameter_size = layer_mem_config[minseq]['parameter_size']
            tp_activation_per_bsz_dict = layer_mem_config[maxseq]['tp_activation_per_bsz_dict'].copy()
            param_sizes[i] = parameter_size
            act_sizes[i] = tp_activation_per_bsz_dict
            for tp in act_sizes[i]:
                act_sizes[i][tp] = act_sizes[i][tp] / maxseq * self.model_layer_config.seqlen_list[i]
        
        other_memory_pp_off = memory_config['other_memory_pp_off_sp'][maxseq_list[0]]
        other_memory_pp_on = {
            'first_stage': memory_config['other_memory_pp_on_first_sp'][maxseq_list[0]], 
            'last_stage': memory_config['other_memory_pp_on_last_sp'][maxseq_list[-1]]
        }
        
        for tp in other_memory_pp_off['activation']:
            # TODO: reasonable scaling when len(seqlen_list) > 1
            other_memory_pp_off['activation'][tp] = (
                2/3 * other_memory_pp_off['activation'][tp] + 
                1/3 * other_memory_pp_off['activation'][tp] / maxseq_list[0] * self.model_layer_config.seqlen_list[0]
            )
            # first stage is not scaled
            other_memory_pp_on['first_stage']['activation'][tp] = other_memory_pp_on['first_stage']['activation'][tp]
            # last stage is scaled
            other_memory_pp_on['last_stage']['activation'][tp] = (
                other_memory_pp_on['last_stage']['activation'][tp] / maxseq_list[-1] * self.model_layer_config.seqlen_list[-1]
            )
        
        return param_sizes, act_sizes, other_memory_pp_off, other_memory_pp_on