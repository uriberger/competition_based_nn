#############################################
#### Competition and Synchrony Model V2 #####
######### By Uri Berger, January 20 #########
#############################################

import numpy as np
import math
import matplotlib.pyplot as plt
import enum

############################
# Configuration parameters #
############################

# Connection of inhibitory neurons to inhibitory neurons from other layers
IIN_PROJECTIONS = False

# Fast connection between inhibitory neurons- NEED TO TEST
IIN_FAST_CONNECTIONS_FACTOR = 1

# Direct connection between the sensory and response layers
SENSORY_RESPONSE_CONNECTION = True

# High-level action response neurons innervate high-level object prediction neurons
HL_AC_RESPONSE_HL_OB_PREDICTION_CONNECTION = False

# Top level actions innervation by the cortex
TL_AC_INNERVATION = True

# Connection of inhibitory neurons only to specific excitatory neurons- NOT IMPLEMENTED YET
SPATIAL_IINS = False

###################
# Model functions #
###################

# General parameters
ll_ob_num = 10
ll_ac_num = 8
hl_ob_num = 7
ml_ac_num = 3
tl_ac_num = 4
hl_ac_num = ml_ac_num+tl_ac_num
excitatory_precentage = 0.8
iin_num = math.ceil((ll_ob_num+ll_ac_num+hl_ob_num+hl_ac_num) * \
                        (1-excitatory_precentage)/excitatory_precentage)
unit_num = 40
#layer_num = 5
layer_num = 1
response_layer = layer_num-1
#response_layer = 1

active_tl_ac_node = 0

# Competition parameters
#default_competition_len = 20000
default_competition_len = 10000

class ModelClass:
    
    default_configuration = {
        # Neuron parameters
        'excitatory_threshold' : 0.4,
        'inhibitory_threshold' : 0.2,
        'sensory_input_strength' : 0.13333,
        
        # Competition parameters
        'ex_sync_window' : 80,
        'ex_sync_threshold' : 10,
        'ex_unsync_threshold' : 1,
        'iin_sync_window' : 20,
        'iin_sync_threshold' : 15,
        'after_iin_sync_waiting' : 2000,
        
        # Normalization parameter
        'Z_ex_ex_th_ratio' : 2.4,
        'Z_in_ex_th_ratio' : 19,
        
        'Z_ll_ob_to_ll_ob_Z_ex_ratio' : 0.1,
        'Z_hl_ob_to_ll_ob_Z_ex_ratio' : 0.6,
        'Z_ll_ac_to_ll_ob_Z_ex_ratio' : 0.1,
        'Z_ml_ac_to_ll_ob_Z_ex_ratio' : 0.1,
        'Z_tl_ac_to_ll_ob_Z_ex_ratio' : 0.1,
        
        'Z_ll_ob_to_hl_ob_Z_ex_ratio' : 0.6,
        'Z_hl_ob_to_hl_ob_Z_ex_ratio' : 0.1,
        'Z_ll_ac_to_hl_ob_Z_ex_ratio' : 0.1,
        'Z_ml_ac_to_hl_ob_Z_ex_ratio' : 0.1,
        'Z_tl_ac_to_hl_ob_Z_ex_ratio' : 0.1,
        
        'Z_ll_ob_to_ll_ac_Z_ex_ratio' : 0.3,
        'Z_hl_ob_to_ll_ac_Z_ex_ratio' : 0.1,
        'Z_ll_ac_to_ll_ac_Z_ex_ratio' : 0.1,
        'Z_ml_ac_to_ll_ac_Z_ex_ratio' : 0.4,
        'Z_tl_ac_to_ll_ac_Z_ex_ratio' : 0.1,
        
        'Z_ll_ob_to_ml_ac_Z_ex_ratio' : 0.1,
        'Z_hl_ob_to_ml_ac_Z_ex_ratio' : 0.1,
        'Z_ll_ac_to_ml_ac_Z_ex_ratio' : 0.1,
        'Z_ml_ac_to_ml_ac_Z_ex_ratio' : 0.1,
        'Z_tl_ac_to_ml_ac_Z_ex_ratio' : 0.6,
        
        'Z_ll_ob_to_tl_ac_Z_ex_ratio' : 0.1,
        'Z_hl_ob_to_tl_ac_Z_ex_ratio' : 0.6,
        'Z_ll_ac_to_tl_ac_Z_ex_ratio' : 0.1,
        'Z_ml_ac_to_tl_ac_Z_ex_ratio' : 0.1,
        'Z_tl_ac_to_tl_ac_Z_ex_ratio' : 0.1,
    
        'Z_ll_ob_to_in_Z_ex_ratio' : 0.2,
        'Z_hl_ob_to_in_Z_ex_ratio' : 0.2,
        'Z_ll_ac_to_in_Z_ex_ratio' : 0.2,
        'Z_ml_ac_to_in_Z_ex_ratio' : 0.2,
        'Z_tl_ac_to_in_Z_ex_ratio' : 0.2,
        
        'Z_forward_ex_th_ratio' : 0.4,
        'Z_backward_ex_th_ratio' : 0.2,
        'Z_sensory_response_ex_th_ratio' : 0.4,
        'Z_response_prediction_ex_th_ratio' : 0.2,
    
        # Learning parameters
        'eta' : 0.0001,
        'comp_len_zeta_ratio' : 1500,
        'gamma_ex' : 0.05,
        'gamma_in' : 0.14,
        }
    
    def __init__(self, configuration, load_from_file, quiet):
        self.conf = {key : configuration[key] if key in configuration else ModelClass.default_configuration[key] for key in ModelClass.default_configuration.keys()}        
        self.quiet = quiet
        self.init_normalization_parameters()
        self.init_data_structures(load_from_file)
        
    def my_print(self, my_str):
        if not self.quiet:
            print(my_str)

    def save_synapse_strength(self):
        # Save the synapse strength matrix to a file.
        trained_strength = [self.synapse_strength]
        file_name = "synapse_strength"
        file_name += "_" + str(unit_num) + "_neurons_" + str(layer_num) + "_layers"
        np.save(file_name, trained_strength)
    
    def init_data_structures(self, load_from_file):
        ''' Initialize the data structures.
        load_from_file- if different from 'None', holds the suffix of the file to be
        loaded. '''
        self.synapse_strength = []
        
        if not load_from_file:
            # Initialize random synapse strength
            for post_layer in range(layer_num):
                post_layer_synapses = []
                for pre_layer in range(layer_num):
                    cur_synapse_strength = np.random.rand(unit_num, unit_num)
                    if IIN_PROJECTIONS or pre_layer == post_layer:
                        ''' In a non-projection configuration, inhibitory neurons only effect
                        neurons in the same layer'''
                        cur_synapse_strength[:, -iin_num:] = (-1) * cur_synapse_strength[:, -iin_num:]
                    post_layer_synapses.append(cur_synapse_strength)
                self.synapse_strength.append(post_layer_synapses)
            
            # Hard coded good weights
            for layer in range(layer_num):
                ll_act_begin = ll_ob_num+hl_ob_num
                '''self.synapse_strength[layer][layer][ll_act_begin,0] = 0
                self.synapse_strength[layer][layer][ll_act_begin,1] = 1
                self.synapse_strength[layer][layer][ll_act_begin,2] = 1
                self.synapse_strength[layer][layer][ll_act_begin,3] = 0
                self.synapse_strength[layer][layer][ll_act_begin,4] = 1
                self.synapse_strength[layer][layer][ll_act_begin,5] = 0
                self.synapse_strength[layer][layer][ll_act_begin,6] = 0
                self.synapse_strength[layer][layer][ll_act_begin,7] = 0
                self.synapse_strength[layer][layer][ll_act_begin,8] = 0
                self.synapse_strength[layer][layer][ll_act_begin,9] = 0
                
                self.synapse_strength[layer][layer][ll_act_begin+1,0] = 0
                self.synapse_strength[layer][layer][ll_act_begin+1,1] = 0
                self.synapse_strength[layer][layer][ll_act_begin+1,2] = 1
                self.synapse_strength[layer][layer][ll_act_begin+1,3] = 1
                self.synapse_strength[layer][layer][ll_act_begin+1,4] = 1
                self.synapse_strength[layer][layer][ll_act_begin+1,5] = 0
                self.synapse_strength[layer][layer][ll_act_begin+1,6] = 0
                self.synapse_strength[layer][layer][ll_act_begin+1,7] = 0
                self.synapse_strength[layer][layer][ll_act_begin+1,8] = 0
                self.synapse_strength[layer][layer][ll_act_begin+1,9] = 0
                
                self.synapse_strength[layer][layer][ll_act_begin+2,0] = 0
                self.synapse_strength[layer][layer][ll_act_begin+2,1] = 0
                self.synapse_strength[layer][layer][ll_act_begin+2,2] = 0
                self.synapse_strength[layer][layer][ll_act_begin+2,3] = 1
                self.synapse_strength[layer][layer][ll_act_begin+2,4] = 1
                self.synapse_strength[layer][layer][ll_act_begin+2,5] = 0
                self.synapse_strength[layer][layer][ll_act_begin+2,6] = 0
                self.synapse_strength[layer][layer][ll_act_begin+2,7] = 0
                self.synapse_strength[layer][layer][ll_act_begin+2,8] = 0
                self.synapse_strength[layer][layer][ll_act_begin+2,9] = 0
                
                self.synapse_strength[layer][layer][ll_act_begin+3,0] = 0
                self.synapse_strength[layer][layer][ll_act_begin+3,1] = 0
                self.synapse_strength[layer][layer][ll_act_begin+3,2] = 0
                self.synapse_strength[layer][layer][ll_act_begin+3,3] = 1
                self.synapse_strength[layer][layer][ll_act_begin+3,4] = 1
                self.synapse_strength[layer][layer][ll_act_begin+3,5] = 0
                self.synapse_strength[layer][layer][ll_act_begin+3,6] = 0
                self.synapse_strength[layer][layer][ll_act_begin+3,7] = 0
                self.synapse_strength[layer][layer][ll_act_begin+3,8] = 0
                self.synapse_strength[layer][layer][ll_act_begin+3,9] = 0
                
                self.synapse_strength[layer][layer][ll_act_begin+4,0] = 0
                self.synapse_strength[layer][layer][ll_act_begin+4,1] = 0
                self.synapse_strength[layer][layer][ll_act_begin+4,2] = 0
                self.synapse_strength[layer][layer][ll_act_begin+4,3] = 0
                self.synapse_strength[layer][layer][ll_act_begin+4,4] = 0
                self.synapse_strength[layer][layer][ll_act_begin+4,5] = 0
                self.synapse_strength[layer][layer][ll_act_begin+4,6] = 1
                self.synapse_strength[layer][layer][ll_act_begin+4,7] = 1
                self.synapse_strength[layer][layer][ll_act_begin+4,8] = 0
                self.synapse_strength[layer][layer][ll_act_begin+4,9] = 0
                
                self.synapse_strength[layer][layer][ll_act_begin+5,0] = 0
                self.synapse_strength[layer][layer][ll_act_begin+5,1] = 0
                self.synapse_strength[layer][layer][ll_act_begin+5,2] = 0
                self.synapse_strength[layer][layer][ll_act_begin+5,3] = 0
                self.synapse_strength[layer][layer][ll_act_begin+5,4] = 0
                self.synapse_strength[layer][layer][ll_act_begin+5,5] = 1
                self.synapse_strength[layer][layer][ll_act_begin+5,6] = 1
                self.synapse_strength[layer][layer][ll_act_begin+5,7] = 0
                self.synapse_strength[layer][layer][ll_act_begin+5,8] = 0
                self.synapse_strength[layer][layer][ll_act_begin+5,9] = 0
                
                self.synapse_strength[layer][layer][ll_act_begin+6,0] = 0
                self.synapse_strength[layer][layer][ll_act_begin+6,1] = 0
                self.synapse_strength[layer][layer][ll_act_begin+6,2] = 0
                self.synapse_strength[layer][layer][ll_act_begin+6,3] = 0
                self.synapse_strength[layer][layer][ll_act_begin+6,4] = 0
                self.synapse_strength[layer][layer][ll_act_begin+6,5] = 1
                self.synapse_strength[layer][layer][ll_act_begin+6,6] = 0
                self.synapse_strength[layer][layer][ll_act_begin+6,7] = 0
                self.synapse_strength[layer][layer][ll_act_begin+6,8] = 0
                self.synapse_strength[layer][layer][ll_act_begin+6,9] = 0
                
                self.synapse_strength[layer][layer][ll_act_begin+7,0] = 0
                self.synapse_strength[layer][layer][ll_act_begin+7,1] = 0
                self.synapse_strength[layer][layer][ll_act_begin+7,2] = 0
                self.synapse_strength[layer][layer][ll_act_begin+7,3] = 0
                self.synapse_strength[layer][layer][ll_act_begin+7,4] = 0
                self.synapse_strength[layer][layer][ll_act_begin+7,5] = 1
                self.synapse_strength[layer][layer][ll_act_begin+7,6] = 0
                self.synapse_strength[layer][layer][ll_act_begin+7,7] = 0
                self.synapse_strength[layer][layer][ll_act_begin+7,8] = 0
                self.synapse_strength[layer][layer][ll_act_begin+7,9] = 0'''
                
        else:
            file_name = "synapse_strength"
            file_name += "_" + str(unit_num) + "_neurons_" + str(layer_num) + "_layers"
            file_name += ".npy"
            
            trained_strength = np.load(file_name)
        
            self.synapse_strength = trained_strength[0]
            
        self.prev_act = []
        for _ in range(layer_num):
            self.prev_act.append(np.zeros((unit_num, 1)))
        
        self.before_prev_act = []
        for _ in range(layer_num):
            self.before_prev_act.append(np.zeros((unit_num, 1)))
        
        self.prev_input_to_neurons = []
        for _ in range(layer_num):
            self.prev_input_to_neurons.append(np.zeros((unit_num, 1)))
        
        self.zero_matrices = []
        for post_layer in range(layer_num):
            self.zero_matrices.append([])
            for pre_layer in range(layer_num):
                zero_matrix = np.zeros((unit_num,unit_num))
                if pre_layer == post_layer:
                    zero_matrix[:,:] = 1
                    # No self loops
                    np.fill_diagonal(zero_matrix, 0)
                if pre_layer == post_layer-1 or pre_layer == post_layer+1 or \
                (SENSORY_RESPONSE_CONNECTION and pre_layer == 0 and post_layer == layer_num-1): # Response neurons are directly innervated by sensory neurons
                    for unit_ind in range(unit_num-iin_num):
                        zero_matrix[unit_ind,unit_ind] = 1
                if HL_AC_RESPONSE_HL_OB_PREDICTION_CONNECTION and pre_layer == layer_num-1 and post_layer == layer_num-2:
                    # High-level action response neurons innervate high-level object prediction neurons
                    hl_ob_begin = ll_ob_num
                    hl_ac_begin = ll_ob_num+hl_ob_num+ll_ac_num
                    for row in range(hl_ob_begin,hl_ob_begin+hl_ob_num):
                        for col in range(hl_ac_begin,hl_ac_begin+hl_ac_num):
                            zero_matrix[row,col] = 1
                if IIN_PROJECTIONS:
                    # Inhibitory neurons are connected through different layers
                    excitatory_unit_num = unit_num - iin_num
                    zero_matrix[excitatory_unit_num:,excitatory_unit_num:] = 1
                            
                self.zero_matrices[post_layer].append(zero_matrix)
        
        self.fix_synapse_strength()
    
    def init_normalization_parameters(self):
        # Initialize normalization parameters
        self.conf['Z_ex'] = self.conf['Z_ex_ex_th_ratio'] * self.conf['excitatory_threshold']
        self.conf['Z_in'] = (1-excitatory_precentage)*self.conf['excitatory_threshold']*self.conf['Z_in_ex_th_ratio']
        
        self.conf['Z_ll_ob_to_ll_ob'] = self.conf['Z_ll_ob_to_ll_ob_Z_ex_ratio'] * self.conf['Z_ex']
        self.conf['Z_hl_ob_to_ll_ob'] = self.conf['Z_hl_ob_to_ll_ob_Z_ex_ratio'] * self.conf['Z_ex']
        self.conf['Z_ll_ac_to_ll_ob'] = self.conf['Z_ll_ac_to_ll_ob_Z_ex_ratio'] * self.conf['Z_ex']
        self.conf['Z_ml_ac_to_ll_ob'] = self.conf['Z_ml_ac_to_ll_ob_Z_ex_ratio'] * self.conf['Z_ex']
        self.conf['Z_tl_ac_to_ll_ob'] = self.conf['Z_tl_ac_to_ll_ob_Z_ex_ratio'] * self.conf['Z_ex']
        
        self.conf['Z_ll_ob_to_hl_ob'] = self.conf['Z_ll_ob_to_hl_ob_Z_ex_ratio'] * self.conf['Z_ex']
        self.conf['Z_hl_ob_to_hl_ob'] = self.conf['Z_hl_ob_to_hl_ob_Z_ex_ratio'] * self.conf['Z_ex']
        self.conf['Z_ll_ac_to_hl_ob'] = self.conf['Z_ll_ac_to_hl_ob_Z_ex_ratio'] * self.conf['Z_ex']
        self.conf['Z_ml_ac_to_hl_ob'] = self.conf['Z_ml_ac_to_hl_ob_Z_ex_ratio'] * self.conf['Z_ex']
        self.conf['Z_tl_ac_to_hl_ob'] = self.conf['Z_tl_ac_to_hl_ob_Z_ex_ratio'] * self.conf['Z_ex']
    
        self.conf['Z_ll_ob_to_ll_ac'] = self.conf['Z_ll_ob_to_ll_ac_Z_ex_ratio'] * self.conf['Z_ex']
        self.conf['Z_hl_ob_to_ll_ac'] = self.conf['Z_hl_ob_to_ll_ac_Z_ex_ratio'] * self.conf['Z_ex']
        self.conf['Z_ll_ac_to_ll_ac'] = self.conf['Z_ll_ac_to_ll_ac_Z_ex_ratio'] * self.conf['Z_ex']
        self.conf['Z_ml_ac_to_ll_ac'] = self.conf['Z_ml_ac_to_ll_ac_Z_ex_ratio'] * self.conf['Z_ex']
        self.conf['Z_tl_ac_to_ll_ac'] = self.conf['Z_tl_ac_to_ll_ac_Z_ex_ratio'] * self.conf['Z_ex']
    
        self.conf['Z_ll_ob_to_ml_ac'] = self.conf['Z_ll_ob_to_ml_ac_Z_ex_ratio'] * self.conf['Z_ex']
        self.conf['Z_hl_ob_to_ml_ac'] = self.conf['Z_hl_ob_to_ml_ac_Z_ex_ratio'] * self.conf['Z_ex']
        self.conf['Z_ll_ac_to_ml_ac'] = self.conf['Z_ll_ac_to_ml_ac_Z_ex_ratio'] * self.conf['Z_ex']
        self.conf['Z_ml_ac_to_ml_ac'] = self.conf['Z_ml_ac_to_ml_ac_Z_ex_ratio'] * self.conf['Z_ex']
        self.conf['Z_tl_ac_to_ml_ac'] = self.conf['Z_tl_ac_to_ml_ac_Z_ex_ratio'] * self.conf['Z_ex']
        
        self.conf['Z_ll_ob_to_tl_ac'] = self.conf['Z_ll_ob_to_tl_ac_Z_ex_ratio'] * self.conf['Z_ex']
        self.conf['Z_hl_ob_to_tl_ac'] = self.conf['Z_hl_ob_to_tl_ac_Z_ex_ratio'] * self.conf['Z_ex']
        self.conf['Z_ll_ac_to_tl_ac'] = self.conf['Z_ll_ac_to_tl_ac_Z_ex_ratio'] * self.conf['Z_ex']
        self.conf['Z_ml_ac_to_tl_ac'] = self.conf['Z_ml_ac_to_tl_ac_Z_ex_ratio'] * self.conf['Z_ex']
        self.conf['Z_tl_ac_to_tl_ac'] = self.conf['Z_tl_ac_to_tl_ac_Z_ex_ratio'] * self.conf['Z_ex']
    
        self.conf['Z_ll_ob_to_in'] = self.conf['Z_ll_ob_to_in_Z_ex_ratio'] * self.conf['Z_ex']
        self.conf['Z_hl_ob_to_in'] = self.conf['Z_hl_ob_to_in_Z_ex_ratio'] * self.conf['Z_ex']
        self.conf['Z_ll_ac_to_in'] = self.conf['Z_ll_ac_to_in_Z_ex_ratio'] * self.conf['Z_ex']
        self.conf['Z_ml_ac_to_in'] = self.conf['Z_ml_ac_to_in_Z_ex_ratio'] * self.conf['Z_ex']
        self.conf['Z_tl_ac_to_in'] = self.conf['Z_tl_ac_to_in_Z_ex_ratio'] * self.conf['Z_ex']
        
        self.conf['Z_forward'] = self.conf['Z_forward_ex_th_ratio'] * self.conf['excitatory_threshold']
        self.conf['Z_backward'] = self.conf['Z_backward_ex_th_ratio'] * self.conf['excitatory_threshold']
        self.conf['Z_sensory_response'] = self.conf['Z_sensory_response_ex_th_ratio'] * self.conf['excitatory_threshold']
        self.conf['Z_response_prediction'] = self.conf['Z_response_prediction_ex_th_ratio'] * self.conf['excitatory_threshold']
    
    def fix_synapse_strength(self):
        # Normalize the synapses strength, and enforce the invariants.
        excitatory_unit_num = unit_num-iin_num
        
        for post_layer in range(layer_num):
            for pre_layer in range(layer_num):
                self.synapse_strength[post_layer][pre_layer] = np.multiply(self.synapse_strength[post_layer][pre_layer], self.zero_matrices[post_layer][pre_layer])
                if post_layer == pre_layer+1:
                    normalized_weight = self.conf['Z_forward']
                    for unit_ind in range(unit_num-iin_num):
                        self.synapse_strength[post_layer][pre_layer][unit_ind,unit_ind] = normalized_weight
                if post_layer == pre_layer-1:
                    normalized_weight = self.conf['Z_backward']
                    for unit_ind in range(unit_num-iin_num):
                        self.synapse_strength[post_layer][pre_layer][unit_ind,unit_ind] = normalized_weight
                if post_layer == pre_layer:
                    # Make sure excitatory weights in inter node are all excitatory
                    self.synapse_strength[post_layer][pre_layer][:,:excitatory_unit_num][self.synapse_strength[post_layer][pre_layer][:,:excitatory_unit_num] < 0] = 0
                    
                    # Normalize incoming excitatory weights to each unit
                    '''normalizing_factor = Z_ex
                    excitatory_row_sums = (self.synapse_strength[post_layer][pre_layer][:,:excitatory_unit_num].sum(axis=1).reshape(unit_num,1).repeat(excitatory_unit_num, axis=1))/normalizing_factor'''
                    ll_ob_begin = 0
                    hl_ob_begin = ll_ob_begin + ll_ob_num
                    ll_ac_begin = hl_ob_begin + hl_ob_num
                    ml_ac_begin = ll_ac_begin + ll_ac_num
                    tl_ac_begin = ml_ac_begin + ml_ac_num
                    
                    ll_ob_to_ll_ob_row_sum = (self.synapse_strength[post_layer][pre_layer][ll_ob_begin:ll_ob_begin+ll_ob_num,ll_ob_begin:ll_ob_begin+ll_ob_num].sum(axis=1).reshape(ll_ob_num,1).repeat(ll_ob_num, axis=1))/self.conf['Z_ll_ob_to_ll_ob']
                    hl_ob_to_ll_ob_row_sum = (self.synapse_strength[post_layer][pre_layer][ll_ob_begin:ll_ob_begin+ll_ob_num,hl_ob_begin:hl_ob_begin+hl_ob_num].sum(axis=1).reshape(ll_ob_num,1).repeat(hl_ob_num, axis=1))/self.conf['Z_hl_ob_to_ll_ob']
                    ll_ac_to_ll_ob_row_sum = (self.synapse_strength[post_layer][pre_layer][ll_ob_begin:ll_ob_begin+ll_ob_num,ll_ac_begin:ll_ac_begin+ll_ac_num].sum(axis=1).reshape(ll_ob_num,1).repeat(ll_ac_num, axis=1))/self.conf['Z_ll_ac_to_ll_ob']
                    ml_ac_to_ll_ob_row_sum = (self.synapse_strength[post_layer][pre_layer][ll_ob_begin:ll_ob_begin+ll_ob_num,ml_ac_begin:ml_ac_begin+ml_ac_num].sum(axis=1).reshape(ll_ob_num,1).repeat(ml_ac_num, axis=1))/self.conf['Z_ml_ac_to_ll_ob']
                    tl_ac_to_ll_ob_row_sum = (self.synapse_strength[post_layer][pre_layer][ll_ob_begin:ll_ob_begin+ll_ob_num,tl_ac_begin:tl_ac_begin+tl_ac_num].sum(axis=1).reshape(ll_ob_num,1).repeat(tl_ac_num, axis=1))/self.conf['Z_tl_ac_to_ll_ob']
                    ll_ob_row_sum = np.concatenate((ll_ob_to_ll_ob_row_sum,hl_ob_to_ll_ob_row_sum,ll_ac_to_ll_ob_row_sum,ml_ac_to_ll_ob_row_sum,tl_ac_to_ll_ob_row_sum),axis=1)
                    
                    ll_ob_to_hl_ob_row_sum = (self.synapse_strength[post_layer][pre_layer][hl_ob_begin:hl_ob_begin+hl_ob_num,ll_ob_begin:ll_ob_begin+ll_ob_num].sum(axis=1).reshape(hl_ob_num,1).repeat(ll_ob_num, axis=1))/self.conf['Z_ll_ob_to_hl_ob']
                    hl_ob_to_hl_ob_row_sum = (self.synapse_strength[post_layer][pre_layer][hl_ob_begin:hl_ob_begin+hl_ob_num,hl_ob_begin:hl_ob_begin+hl_ob_num].sum(axis=1).reshape(hl_ob_num,1).repeat(hl_ob_num, axis=1))/self.conf['Z_hl_ob_to_hl_ob']
                    ll_ac_to_hl_ob_row_sum = (self.synapse_strength[post_layer][pre_layer][hl_ob_begin:hl_ob_begin+hl_ob_num,ll_ac_begin:ll_ac_begin+ll_ac_num].sum(axis=1).reshape(hl_ob_num,1).repeat(ll_ac_num, axis=1))/self.conf['Z_ll_ac_to_hl_ob']
                    ml_ac_to_hl_ob_row_sum = (self.synapse_strength[post_layer][pre_layer][hl_ob_begin:hl_ob_begin+hl_ob_num,ml_ac_begin:ml_ac_begin+ml_ac_num].sum(axis=1).reshape(hl_ob_num,1).repeat(ml_ac_num, axis=1))/self.conf['Z_ml_ac_to_hl_ob']
                    tl_ac_to_hl_ob_row_sum = (self.synapse_strength[post_layer][pre_layer][hl_ob_begin:hl_ob_begin+hl_ob_num,tl_ac_begin:tl_ac_begin+tl_ac_num].sum(axis=1).reshape(hl_ob_num,1).repeat(tl_ac_num, axis=1))/self.conf['Z_tl_ac_to_hl_ob']
                    hl_ob_row_sum = np.concatenate((ll_ob_to_hl_ob_row_sum,hl_ob_to_hl_ob_row_sum,ll_ac_to_hl_ob_row_sum,ml_ac_to_hl_ob_row_sum,tl_ac_to_hl_ob_row_sum),axis=1)
                    
                    ll_ob_to_ll_ac_row_sum = (self.synapse_strength[post_layer][pre_layer][ll_ac_begin:ll_ac_begin+ll_ac_num,ll_ob_begin:ll_ob_begin+ll_ob_num].sum(axis=1).reshape(ll_ac_num,1).repeat(ll_ob_num, axis=1))/self.conf['Z_ll_ob_to_ll_ac']
                    hl_ob_to_ll_ac_row_sum = (self.synapse_strength[post_layer][pre_layer][ll_ac_begin:ll_ac_begin+ll_ac_num,hl_ob_begin:hl_ob_begin+hl_ob_num].sum(axis=1).reshape(ll_ac_num,1).repeat(hl_ob_num, axis=1))/self.conf['Z_hl_ob_to_ll_ac']
                    ll_ac_to_ll_ac_row_sum = (self.synapse_strength[post_layer][pre_layer][ll_ac_begin:ll_ac_begin+ll_ac_num,ll_ac_begin:ll_ac_begin+ll_ac_num].sum(axis=1).reshape(ll_ac_num,1).repeat(ll_ac_num, axis=1))/self.conf['Z_ll_ac_to_ll_ac']
                    ml_ac_to_ll_ac_row_sum = (self.synapse_strength[post_layer][pre_layer][ll_ac_begin:ll_ac_begin+ll_ac_num,ml_ac_begin:ml_ac_begin+ml_ac_num].sum(axis=1).reshape(ll_ac_num,1).repeat(ml_ac_num, axis=1))/self.conf['Z_ml_ac_to_ll_ac']
                    tl_ac_to_ll_ac_row_sum = (self.synapse_strength[post_layer][pre_layer][ll_ac_begin:ll_ac_begin+ll_ac_num,tl_ac_begin:tl_ac_begin+tl_ac_num].sum(axis=1).reshape(ll_ac_num,1).repeat(tl_ac_num, axis=1))/self.conf['Z_tl_ac_to_ll_ac']
                    ll_ac_row_sum = np.concatenate((ll_ob_to_ll_ac_row_sum,hl_ob_to_ll_ac_row_sum,ll_ac_to_ll_ac_row_sum,ml_ac_to_ll_ac_row_sum,tl_ac_to_ll_ac_row_sum),axis=1)
                    
                    ll_ob_to_ml_ac_row_sum = (self.synapse_strength[post_layer][pre_layer][ml_ac_begin:ml_ac_begin+ml_ac_num,ll_ob_begin:ll_ob_begin+ll_ob_num].sum(axis=1).reshape(ml_ac_num,1).repeat(ll_ob_num, axis=1))/self.conf['Z_ll_ob_to_ml_ac']
                    hl_ob_to_ml_ac_row_sum = (self.synapse_strength[post_layer][pre_layer][ml_ac_begin:ml_ac_begin+ml_ac_num,hl_ob_begin:hl_ob_begin+hl_ob_num].sum(axis=1).reshape(ml_ac_num,1).repeat(hl_ob_num, axis=1))/self.conf['Z_hl_ob_to_ml_ac']
                    ll_ac_to_ml_ac_row_sum = (self.synapse_strength[post_layer][pre_layer][ml_ac_begin:ml_ac_begin+ml_ac_num,ll_ac_begin:ll_ac_begin+ll_ac_num].sum(axis=1).reshape(ml_ac_num,1).repeat(ll_ac_num, axis=1))/self.conf['Z_ll_ac_to_ml_ac']
                    ml_ac_to_ml_ac_row_sum = (self.synapse_strength[post_layer][pre_layer][ml_ac_begin:ml_ac_begin+ml_ac_num,ml_ac_begin:ml_ac_begin+ml_ac_num].sum(axis=1).reshape(ml_ac_num,1).repeat(ml_ac_num, axis=1))/self.conf['Z_ml_ac_to_ml_ac']
                    tl_ac_to_ml_ac_row_sum = (self.synapse_strength[post_layer][pre_layer][ml_ac_begin:ml_ac_begin+ml_ac_num,tl_ac_begin:tl_ac_begin+tl_ac_num].sum(axis=1).reshape(ml_ac_num,1).repeat(tl_ac_num, axis=1))/self.conf['Z_tl_ac_to_ml_ac']
                    ml_ac_row_sum = np.concatenate((ll_ob_to_ml_ac_row_sum,hl_ob_to_ml_ac_row_sum,ll_ac_to_ml_ac_row_sum,ml_ac_to_ml_ac_row_sum,tl_ac_to_ml_ac_row_sum),axis=1)
                    
                    ll_ob_to_tl_ac_row_sum = (self.synapse_strength[post_layer][pre_layer][tl_ac_begin:tl_ac_begin+tl_ac_num,ll_ob_begin:ll_ob_begin+ll_ob_num].sum(axis=1).reshape(tl_ac_num,1).repeat(ll_ob_num, axis=1))/self.conf['Z_ll_ob_to_tl_ac']
                    hl_ob_to_tl_ac_row_sum = (self.synapse_strength[post_layer][pre_layer][tl_ac_begin:tl_ac_begin+tl_ac_num,hl_ob_begin:hl_ob_begin+hl_ob_num].sum(axis=1).reshape(tl_ac_num,1).repeat(hl_ob_num, axis=1))/self.conf['Z_hl_ob_to_tl_ac']
                    ll_ac_to_tl_ac_row_sum = (self.synapse_strength[post_layer][pre_layer][tl_ac_begin:tl_ac_begin+tl_ac_num,ll_ac_begin:ll_ac_begin+ll_ac_num].sum(axis=1).reshape(tl_ac_num,1).repeat(ll_ac_num, axis=1))/self.conf['Z_ll_ac_to_tl_ac']
                    ml_ac_to_tl_ac_row_sum = (self.synapse_strength[post_layer][pre_layer][tl_ac_begin:tl_ac_begin+tl_ac_num,ml_ac_begin:ml_ac_begin+ml_ac_num].sum(axis=1).reshape(tl_ac_num,1).repeat(ml_ac_num, axis=1))/self.conf['Z_ml_ac_to_tl_ac']
                    tl_ac_to_tl_ac_row_sum = (self.synapse_strength[post_layer][pre_layer][tl_ac_begin:tl_ac_begin+tl_ac_num,tl_ac_begin:tl_ac_begin+tl_ac_num].sum(axis=1).reshape(tl_ac_num,1).repeat(tl_ac_num, axis=1))/self.conf['Z_tl_ac_to_tl_ac']
                    tl_ac_row_sum = np.concatenate((ll_ob_to_tl_ac_row_sum,hl_ob_to_tl_ac_row_sum,ll_ac_to_tl_ac_row_sum,ml_ac_to_tl_ac_row_sum,tl_ac_to_tl_ac_row_sum),axis=1)
                    
                    ll_ob_to_in_row_sum = (self.synapse_strength[post_layer][pre_layer][excitatory_unit_num:unit_num,ll_ob_begin:ll_ob_begin+ll_ob_num].sum(axis=1).reshape(iin_num,1).repeat(ll_ob_num, axis=1))/self.conf['Z_ll_ob_to_in']
                    hl_ob_to_in_row_sum = (self.synapse_strength[post_layer][pre_layer][excitatory_unit_num:unit_num,hl_ob_begin:hl_ob_begin+hl_ob_num].sum(axis=1).reshape(iin_num,1).repeat(hl_ob_num, axis=1))/self.conf['Z_hl_ob_to_in']
                    ll_ac_to_in_row_sum = (self.synapse_strength[post_layer][pre_layer][excitatory_unit_num:unit_num,ll_ac_begin:ll_ac_begin+ll_ac_num].sum(axis=1).reshape(iin_num,1).repeat(ll_ac_num, axis=1))/self.conf['Z_ll_ac_to_in']
                    ml_ac_to_in_row_sum = (self.synapse_strength[post_layer][pre_layer][excitatory_unit_num:unit_num,ml_ac_begin:ml_ac_begin+ml_ac_num].sum(axis=1).reshape(iin_num,1).repeat(ml_ac_num, axis=1))/self.conf['Z_ml_ac_to_in']
                    tl_ac_to_in_row_sum = (self.synapse_strength[post_layer][pre_layer][excitatory_unit_num:unit_num,tl_ac_begin:tl_ac_begin+tl_ac_num].sum(axis=1).reshape(iin_num,1).repeat(tl_ac_num, axis=1))/self.conf['Z_tl_ac_to_in']
                    in_from_ex_row_sum = np.concatenate((ll_ob_to_in_row_sum,hl_ob_to_in_row_sum,ll_ac_to_in_row_sum,ml_ac_to_in_row_sum,tl_ac_to_in_row_sum),axis=1)
                    
                    excitatory_row_sums = np.concatenate((ll_ob_row_sum,hl_ob_row_sum,ll_ac_row_sum,ml_ac_row_sum,tl_ac_row_sum,in_from_ex_row_sum),axis=0)
                    
                    # Make sure inhibitory weights in inter node are all inhibitory
                    self.synapse_strength[post_layer][pre_layer][:,excitatory_unit_num:][self.synapse_strength[post_layer][pre_layer][:,excitatory_unit_num:] > 0] = 0
                    # Normalize incoming inhibitory weights to each unit
                    if IIN_PROJECTIONS:
                        ''' Inhibitory neurons innervate excitatory neurons from the same layer,
                        and inhibitory neurons from all layers'''
                        normalizing_factor_for_ex_neurons = self.conf['Z_in']
                        normalizing_factor_for_in_neurons = self.conf['Z_in']/layer_num
                        in_to_ex_row_sums = (-1)*((self.synapse_strength[post_layer][pre_layer][:excitatory_unit_num,excitatory_unit_num:].sum(axis=1).reshape(excitatory_unit_num,1).repeat(iin_num, axis=1))/normalizing_factor_for_ex_neurons)
                        in_to_in_row_sums = (-1)*((self.synapse_strength[post_layer][pre_layer][excitatory_unit_num:,excitatory_unit_num:].sum(axis=1).reshape(iin_num,1).repeat(iin_num, axis=1))/normalizing_factor_for_in_neurons)
                        inhibitory_row_sums = np.concatenate((in_to_ex_row_sums, in_to_in_row_sums))
                    else:
                        normalizing_factor = self.conf['Z_in']
                        inhibitory_row_sums = (-1)*((self.synapse_strength[post_layer][pre_layer][:,excitatory_unit_num:].sum(axis=1).reshape(unit_num,1).repeat(iin_num, axis=1))/normalizing_factor)
                    
                    row_sums = np.concatenate((excitatory_row_sums,inhibitory_row_sums),axis=1)
                    row_sums[row_sums == 0] = 1
                    self.synapse_strength[post_layer][pre_layer] = self.synapse_strength[post_layer][pre_layer]/row_sums
                elif IIN_PROJECTIONS:
                    # Inhibitory neurons are connected through different layers
                    # Make sure weights are all inhibitory
                    self.synapse_strength[post_layer][pre_layer][excitatory_unit_num:,excitatory_unit_num:][self.synapse_strength[post_layer][pre_layer][excitatory_unit_num:,excitatory_unit_num:] > 0] = 0
                    # Normalize incoming inhibitory weights to each unit
                    normalizing_factor = self.conf['Z_in'] / layer_num
                    inhibitory_row_sums = (-1)*((self.synapse_strength[post_layer][pre_layer][excitatory_unit_num:,excitatory_unit_num:].sum(axis=1).reshape(iin_num,1).repeat(iin_num, axis=1))/normalizing_factor)
                    row_sums = np.pad(inhibitory_row_sums,((excitatory_unit_num,0),(excitatory_unit_num,0)), 'constant')
                    row_sums[row_sums == 0] = 1
                    self.synapse_strength[post_layer][pre_layer] = self.synapse_strength[post_layer][pre_layer]/row_sums
                if SENSORY_RESPONSE_CONNECTION and post_layer == layer_num-1 and pre_layer == 0:
                    normalized_weight = self.conf['Z_sensory_response']
                    # Sensory neurons directly innervate response neurons
                    for unit_ind in range(unit_num):
                        self.synapse_strength[post_layer][pre_layer][unit_ind,unit_ind] = normalized_weight
                if HL_AC_RESPONSE_HL_OB_PREDICTION_CONNECTION and post_layer == layer_num-2 and pre_layer == layer_num-1:
                    # High-level action response neurons innervate high-level object prediction neurons
                    hl_ob_begin = ll_ob_num
                    hl_ac_begin = ll_ob_num+hl_ob_num+ll_ac_num
                    self.synapse_strength[post_layer][pre_layer][hl_ob_begin:hl_ob_begin+hl_ob_num,hl_ac_begin:hl_ac_begin+hl_ac_num][self.synapse_strength[post_layer][pre_layer][hl_ob_begin:hl_ob_begin+hl_ob_num,hl_ac_begin:hl_ac_begin+hl_ac_num] < 0] = 0
                    
                    # Normalize incoming weights to each unit
                    normalizing_factor = self.conf['Z_response_prediction']
                    row_sums = (self.synapse_strength[post_layer][pre_layer][hl_ob_begin:hl_ob_begin+hl_ob_num,hl_ac_begin:hl_ac_begin+hl_ac_num].sum(axis=1).reshape(hl_ob_num,1).repeat(hl_ac_num, axis=1))/normalizing_factor
                    row_sums = np.pad(row_sums, ((hl_ob_begin,unit_num-hl_ob_begin-hl_ob_num),(hl_ac_begin,unit_num-hl_ac_begin-hl_ac_num)), 'constant')
                    row_sums[row_sums == 0] = 1
                    self.synapse_strength[post_layer][pre_layer] = self.synapse_strength[post_layer][pre_layer]/row_sums
                    
                # Top level actions doesn't get innervated by the cortex
                if not TL_AC_INNERVATION:
                    top_level_action_begin = ll_ob_num+hl_ob_num+ll_ac_num+ml_ac_num
                    self.synapse_strength[post_layer][pre_layer][top_level_action_begin:top_level_action_begin+tl_ac_num,:] = 0
        
    def calculate_winners(self, input_vec, begin_ind, node_num,stop_when_reach_def_comp_len,stop_when_resolved):
        # Given an input, simulate the dynamics of the system, for iter_num time steps
        fire_history = []
        
        # Stopping criterion
        need_to_wait_for_def_comp_len = stop_when_reach_def_comp_len
        need_to_wait_for_comp_resolution = stop_when_resolved
        
        for _ in range(layer_num):
            cur_history = []
            for _ in range(unit_num):
                cur_history.append([])
            fire_history.append(cur_history)
        
        synched_iins = []
        winner_loser_list = [-1] * node_num
            
        t = 0
        remaining_inhibitory_iterations = 0
        while True:
            if t % 1000 == 0:
                self.my_print('t='+str(t))
            
            # Propagate external input
            if remaining_inhibitory_iterations > 0:
                self.prop_external_input(input_vec,True)
                remaining_inhibitory_iterations -= 1
            else:
                self.prop_external_input(input_vec)
                remaining_inhibitory_iterations = IIN_FAST_CONNECTIONS_FACTOR-1
                
            # Document fire history
            for l in range(layer_num):
                for unit_ind in range(unit_num):
                    if self.prev_act[l][unit_ind, 0] == 1:
                        fire_history[l][unit_ind].append(t)
            if need_to_wait_for_comp_resolution:
                ''' The first step of the competition is- synchronization of the IINs.
                We need to wait until they are synchronized before we check if the competition
                is resolved. '''          
                if len(synched_iins) == 0:
                    ''' IINs were not synched until now. Go over all pairs of IINs and check if
                    they are now synched. '''       
                    for first_iin in range(unit_num-iin_num, unit_num):
                        for second_iin in range(first_iin+1, unit_num):
                            # Check if iins where synched in the last sync window
                            if len(fire_history[response_layer][first_iin]) < self.conf['iin_sync_window'] or len(fire_history[response_layer][second_iin]) < self.conf['iin_sync_window']:
                                # Not enough firing- still can't determine synchronization
                                continue
                            intersection_of_last_window = [x for x in fire_history[response_layer][first_iin][-self.conf['iin_sync_window']:] if x in fire_history[response_layer][second_iin][-self.conf['iin_sync_window']:]]
                            if len(intersection_of_last_window) >= self.conf['iin_sync_threshold']:
                                synched_iins = [first_iin, second_iin]
                                break
                        if len(synched_iins) > 0:
                            # IINs are synched
                            sync_time_step = t
                            self.my_print('sync time step: ' + str(t))
                            break
                elif t > sync_time_step+self.conf['after_iin_sync_waiting']:
                    ''' After the initial sync, we wait some more time ("after_iin_sync_waiting")
                    to make sure the synchronization is complete.
                    After the waiting, we check if the competition is resolved. '''
                    competition_resolved = True
                    
                    iin_firing = set().union(*([[x for x in fire_history[response_layer][iin_ind] if x > t-self.conf['ex_sync_window']*10] for iin_ind in range(unit_num-iin_num,unit_num)]))
                    for unit_ind in range(begin_ind, begin_ind+node_num):
                        winner_loser_ind = unit_ind - begin_ind
                        if winner_loser_list[winner_loser_ind] != -1:
                            continue # resolved
                        ex_intersection_of_last_window = [x for x in fire_history[response_layer][unit_ind][-self.conf['ex_sync_window']:] if x in iin_firing]
                        if len(ex_intersection_of_last_window) >= self.conf['ex_sync_threshold']:
                            winner_loser_list[winner_loser_ind] = 1 # winner
                        elif len(ex_intersection_of_last_window) <= self.conf['ex_unsync_threshold']:
                            winner_loser_list[winner_loser_ind] = 0 # loser
                        else:
                            competition_resolved = False # unresolved yet
                            
                    if competition_resolved:
                        need_to_wait_for_comp_resolution = False
                        if not need_to_wait_for_def_comp_len:
                            break
                
            t += 1
            if t == default_competition_len:
                need_to_wait_for_def_comp_len = False
                if not need_to_wait_for_comp_resolution:
                    break
        
        self.my_print('winner list firing: ' + str([len(a) for a in fire_history[response_layer][begin_ind:begin_ind+node_num]]))
        
        return winner_loser_list, t, fire_history
        
    def convert_winner_list_to_action(self, winner_list):
        side_size = int(ll_ac_num / 2)
        
        right_count = np.sum(winner_list[:side_size])
        left_count = np.sum(winner_list[side_size:])
        
        action = (right_count - left_count) % ll_ac_num
        return action
    
    def update_synapse_strength_long_term(self, winning_action_list, prev_sensory_input, before_prev_sensory_input, comp_len):
        
        zeta = comp_len / self.conf['comp_len_zeta_ratio']
        delta_input_strength = np.sum(prev_sensory_input-before_prev_sensory_input)
        max_possible_input_strength = self.conf['sensory_input_strength'] * ll_ob_num
        normalized_delta_input_strength = delta_input_strength / max_possible_input_strength
        total_strength_change = normalized_delta_input_strength * zeta
        before_prev_input_strength = np.sum(before_prev_sensory_input)
        strength_change_vec = before_prev_sensory_input * (total_strength_change / before_prev_input_strength)
        strength_change_vec = np.pad(strength_change_vec, ((0,unit_num-ll_ob_num),(0,0)), 'constant')
        
        ll_action_begin = ll_ob_num + hl_ob_num
        
        for layer_ind in range(layer_num):
            for node_ind in range(len(winning_action_list)):
                winner_loser_factor = (winning_action_list[node_ind]-0.5)*2
                real_node_ind = node_ind+ll_action_begin
                self.synapse_strength[layer_ind][layer_ind][[real_node_ind],:] += winner_loser_factor*strength_change_vec.transpose()
                
        self.fix_synapse_strength()
                
    def update_synapse_strength_short_term(self):
        # Update the synapses strength according the a Hebbian learning rule
        for post_layer in range(layer_num):
            for pre_layer in range(layer_num):
                post_layer_prev_act = self.prev_act[post_layer]
                normalizing_excitatory_vec = np.ones((unit_num-iin_num,1)) * self.conf['gamma_ex']
                normalizing_inhibitory_vec = np.ones((iin_num,1)) * self.conf['gamma_in']
                normalizing_vec = np.concatenate((normalizing_excitatory_vec, normalizing_inhibitory_vec))
                normalized_post_layer_prev_act = post_layer_prev_act - normalizing_vec
                
                pre_layer_before_prev_act = self.before_prev_act[pre_layer]
                
                update_mat = np.matmul(normalized_post_layer_prev_act, pre_layer_before_prev_act.transpose())
                # Strengthen inhibitory neurons weights by making them more negative (and not more positive)
                update_mat[:,-iin_num:] = (-1) * update_mat[:,-iin_num:]
                    
                self.synapse_strength[post_layer][pre_layer] = self.synapse_strength[post_layer][pre_layer] + self.conf['eta'] * update_mat
        
        self.fix_synapse_strength()
        
    def prop_external_input(self, sensory_input_vec, only_inhibitory=False):
        # Simulate the dynamics of the system for a single time step
        new_act = []
        new_input = []
        
        for post_layer in range(layer_num):
            cur_input = np.zeros((unit_num, 1))
            if post_layer == 0 and not only_inhibitory:
                input_from_prev_layer = np.pad(sensory_input_vec, ((0,unit_num-ll_ob_num),(0,0)), 'constant')
                cur_input = np.add(cur_input, input_from_prev_layer)
            for pre_layer in range(layer_num):
                input_from_pre_layer = np.matmul(self.synapse_strength[post_layer][pre_layer], self.prev_act[pre_layer])
                if only_inhibitory:
                    input_from_pre_layer[:-iin_num,0] = 0
                cur_input = np.add(cur_input, input_from_pre_layer)
            
            if (not TL_AC_INNERVATION) and (post_layer == layer_num-1):
                # The response neuron of the top level action is not innervated, so we should innervate it from outside
                top_level_action_begin = ll_ob_num+hl_ob_num+ll_ac_num+ml_ac_num
                cur_input[top_level_action_begin+active_tl_ac_node,0] = self.conf['sensory_input_strength']
                
            ''' Accumulating input and refractory period: If a neuron fired in the last time step,
            we subtract its previous input from its current input. Otherwise- we add its previous
            input to its current input. '''
            prev_input_factor = (1 - 2 * self.prev_act[post_layer])
            cur_input = np.add(cur_input, prev_input_factor * self.prev_input_to_neurons[post_layer])
            
            # Make sure the input is non-negative
            cur_input = np.where(cur_input>=0, cur_input, 0)
            
            if only_inhibitory:
                # Excitatory neurons doesn't change in an only-inhibitory stage
                cur_input[:-iin_num] = self.prev_input_to_neurons[post_layer][:-iin_num]
            
            cur_act = np.concatenate((self.excitatory_activation_function(cur_input[:cur_input.shape[0]-iin_num,[0]]),
                                  self.inhibitory_activation_function(cur_input[cur_input.shape[0]-iin_num:,[0]])),
                                  axis=0)
            new_act.append(cur_act)
            new_input.append(cur_input)
        
        self.before_prev_act = self.prev_act
        self.prev_act = new_act
        self.prev_input_to_neurons = new_input
        
        self.update_synapse_strength_short_term()
        
    def plot_activation(self, fire_history, from_time_step, to_time_step):
        ''' Plot the activations of all neurons in a range of time steps.
        Ideally not more than 500 time steps. '''
        plt.ioff()
        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        
        left_border = np.arange(from_time_step,from_time_step+2,1)
        bottom_border1 = np.ones(2)
        bottom_border2 = np.zeros(2)
        top_border1 = np.ones(2) * unit_num-1
        top_border2 = np.ones(2) * unit_num
        right_border = np.arange(to_time_step,to_time_step+2,1)
        ax.fill_between(left_border,bottom_border1,bottom_border2,color='white')
        ax.fill_between(left_border,top_border1,top_border2,color='white')
        ax.fill_between(right_border,bottom_border1,bottom_border2,color='white')
        
        for unit_ind in range(unit_num):
            for t_step in range(from_time_step, to_time_step):
                if t_step in fire_history[unit_ind]:
                    x1 = np.arange(t_step,t_step+2,1)
                    y1 = np.ones(2)*unit_ind
                    y2 = np.ones(2)*(unit_ind+1)
                    if unit_ind < unit_num-iin_num:
                        ax.fill_between(x1,y1,y2,color='black')
                    else:
                        ax.fill_between(x1,y1,y2,color='red')
        
        ax.grid()
        
        file_name = 'activation_' + str(from_time_step) + '_to_' + str(to_time_step) + '.png' 
        plt.savefig(file_name)
        plt.close()
    
    def excitatory_activation_function(self, x):
        # Linear activation function for excitatory neurons 
        return 0 + (x >= self.conf['excitatory_threshold'])
    
    def inhibitory_activation_function(self, x):
        # Linear activation function for inhibitory neurons
        return 0 + (x >= self.conf['inhibitory_threshold'])

#######################
# Simulator functions #
#######################

# Parameters
OBST = 0
nose_length = 2
world_height = 50
world_length = 50
goals = [(24,25)]

def init_world(vertical_size, horizontal_size, obstacle_list):
    world = np.zeros((vertical_size, horizontal_size))
    world[:,:] = 1 - OBST
    
    for obstacle in obstacle_list:
        upper_left = obstacle[0]
        obstacle_size = obstacle[1]
        
        lower_left = (upper_left[0] + obstacle_size - 1, upper_left[1])
        upper_right = (upper_left[0], upper_left[1] + obstacle_size - 1)
        
        world[upper_left[0]:lower_left[0]+1, upper_left[1]:upper_right[1]+1] = OBST
        
    return world

def advance(direction, cur_loc):
    row = cur_loc[0]
    col = cur_loc[1]
    
    if direction == 0:
        row = row - 1
    elif direction == 1:
        col = col + 1
        row = row - 1
    elif direction == 2:
        col = col + 1
    elif direction == 3:
        col = col + 1
        row = row + 1
    elif direction == 4:
        row = row + 1
    elif direction == 5:
        col = col - 1
        row = row + 1
    elif direction == 6:
        col = col - 1
    elif direction == 7:
        col = col - 1
        row = row - 1
    
    return row,col

def is_edge(loc, world_shape):
    return loc[0] == world_shape[0] or loc[1] == world_shape[1] or \
        loc[0] == -1 or loc[1] == -1

def print_world(world, player, goal):
    nose = []
    player_loc = (player[0], player[1])
    orientation = player[2]
    cur_nose_loc = player_loc
    for _ in range(nose_length):
        cur_nose_loc = advance(orientation, cur_nose_loc)
        if is_edge(cur_nose_loc, world.shape) or goal == cur_nose_loc:
            break
        nose.append(cur_nose_loc)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    hori_ticks = np.arange(0, world.shape[0]+1, 1)
    verti_ticks = np.arange(0, world.shape[1]+1, 1)
    ax.set_xticks(hori_ticks)
    ax.set_yticks(verti_ticks)
    plt.ylim(50,0)
    
    left_border = np.arange(0,2,1)
    bottom_border1 = np.ones(2)
    bottom_border2 = np.zeros(2)
    top_border1 = np.ones(2) * world.shape[0]-1
    top_border2 = np.ones(2) * world.shape[0]
    right_border = np.arange(world.shape[1]-1,world.shape[1]+1,1)
    ax.fill_between(left_border,bottom_border1,bottom_border2,color='white')
    ax.fill_between(left_border,top_border1,top_border2,color='white')
    ax.fill_between(right_border,bottom_border1,bottom_border2,color='white')
    
    for row in range(world.shape[0]):
        for col in range(world.shape[0]):
            if world[row][col] == OBST:
                x1 = np.arange(col,col+2,1)
                y1 = np.ones(2)*row
                y2 = np.ones(2)*(row+1)
                ax.fill_between(x1,y1,y2,color='black')
            if player[0] == row and player[1] == col:
                x1 = np.arange(col,col+2,1)
                y1 = np.ones(2)*row
                y2 = np.ones(2)*(row+1)
                ax.fill_between(x1,y1,y2,color='blue')
            if goal[0] == row and goal[1] == col:
                x1 = np.arange(col,col+2,1)
                y1 = np.ones(2)*row
                y2 = np.ones(2)*(row+1)
                ax.fill_between(x1,y1,y2,color='red')
            if (row,col) in nose:
                x1 = np.arange(col,col+2,1)
                y1 = np.ones(2)*row
                y2 = np.ones(2)*(row+1)
                ax.fill_between(x1,y1,y2,color='royalblue')
    
    ax.grid()
    plt.show()
    print(player_loc)

def generate_state_from_simulator(world, cur_player):
    state_vector = np.zeros((ll_ob_num, 1))
    
    for goal in goals:
        row_diff = goal[0]-cur_player[0]
        col_diff = goal[1]-cur_player[1]
        
        if row_diff == 0 and col_diff == 0:
            state_vector[-1] = 1
            state_vector[-2] = 1
        else:
            if col_diff > 0:
                offset = math.pi/2
            else:
                offset = 3*(math.pi/2)
            if col_diff == 0:
                row_col_ratio = math.inf
            else:
                row_col_ratio = row_diff/col_diff
            
            dist_from_goal = (row_diff**2 + col_diff**2)**0.5
            max_distance = (world.shape[0]**2 + world.shape[1]**2)**0.5
            normalized_distance = dist_from_goal / max_distance
            dist_val = (-1)*normalized_distance + 1    
            
            angle_to_goal = np.arctan(row_col_ratio) + offset
            normalized_angle = (angle_to_goal / (math.pi/4)) % 8
            relative_angle = (normalized_angle - cur_player[2]) % 8
            floor_int = int(relative_angle)
            state_vector[floor_int] = dist_val*(1 - (relative_angle - floor_int))
            state_vector[(floor_int + 1) % ll_ac_num] = dist_val*(relative_angle - floor_int)
    
    return state_vector

def get_shortest_path_to_target_len(player, goal):
    return max(abs(player[0] - goal[0]), abs(player[1] - goal[1]))

def evaluate(world, goals, cur_player, iter_num, shortest_path_len, max_rounds_num):
    if (cur_player[0],cur_player[1]) in goals:
        longest_interval_length = max_rounds_num - shortest_path_len
        my_interval_length = max_rounds_num - iter_num
        return 0.5 + 0.5 * (my_interval_length/longest_interval_length)
    else:
        max_possible_dist = (world.shape[0]**2 + world.shape[1]**2)**0.5
        my_dist = ((goals[0][0] - cur_player[0])**2 + (goals[0][1] - cur_player[1])**2)**0.5
        return 0.5 - 0.5 * (my_dist/max_possible_dist)

##############################
# Parameter calibration tool #
##############################

class WinnerAnalysisCode(enum.Enum):
    NORMAL = 0
    ONLY_LOSERS = 1 # Need to increase sync_window and Z_ob_to_ac, and decrease sync_threshold
    ONLY_WINNERS = 2 # Need to increase Z_in, and decrease Z_ob_to_ac
    MIX = 3 # Decrease threshold-window-ratio
    NO_JUMP = 4 # Need to increase Z_in
    LOSERS_AFTER_JUMP = 5 # Decrease threshold-window-ratio
    WINNERS_BEFORE_JUMP = 6 # Increase threshold-window-ratio
    MULTIPLE_JUMP = 7 # Need to increase sync_threshold

class FireHistoryAnalysisCode(enum.Enum):
    NORMAL = 0
    LOW_FIRING_RATE = 1
    HIGH_FIRING_RATE = 2

jump_factor = 2
jump_diff = 10
jump_compare_factor = 2
non_dichotomized_codes = [WinnerAnalysisCode.MIX,
                          WinnerAnalysisCode.NO_JUMP,
                          WinnerAnalysisCode.LOSERS_AFTER_JUMP,
                          WinnerAnalysisCode.WINNERS_BEFORE_JUMP,
                          WinnerAnalysisCode.MULTIPLE_JUMP]
high_firing_rate_threshold = 0.01

def analyze_competition(winner_list, competitiors_fire_history):
    competitiors_fire_count = [len(x) for x in competitiors_fire_history]
    sorted_indices = sorted(range(len(competitiors_fire_count)), key=lambda k: competitiors_fire_count[k])
    finished_losers = (winner_list[sorted_indices[0]] == 1)
    
    # First pass- make sure that there's no mixture between winners and losers
    for i in range(len(sorted_indices)):
        if finished_losers and winner_list[sorted_indices[i]] == 0:
            return WinnerAnalysisCode.MIX
        if (not finished_losers) and winner_list[sorted_indices[i]] == 1:
            finished_losers = True
    
    # Handle edge cases: only losers and only winners
    if winner_list[sorted_indices[-1]] == 0:
        return WinnerAnalysisCode.ONLY_LOSERS
    elif winner_list[sorted_indices[0]] == 1:
        return WinnerAnalysisCode.ONLY_WINNERS
    
    finished_losers = (winner_list[sorted_indices[0]] == 1)          
    jumps = []
    # Second pass- collect jumps
    for i in range(len(sorted_indices)-1):
        smaller = competitiors_fire_count[sorted_indices[i]]
        bigger = competitiors_fire_count[sorted_indices[i+1]]
        if bigger > (smaller * jump_factor) and bigger > (smaller + jump_diff):
            # Jump
            jumps.append((i,bigger-smaller))
    
    # Make sure one jump is much bigger than the rest
    jumps.sort(key=lambda x:x[1])
    if len(jumps) == 0:
        return WinnerAnalysisCode.NO_JUMP
    if len(jumps) > 1:
        if jumps[-1][1] < jump_compare_factor*jumps[-2][1]:
            return WinnerAnalysisCode.MULTIPLE_JUMP
    jump_index = jumps[-1][0]
    
    # Make sure the biggest jump is in the right location
    if winner_list[sorted_indices[jump_index]] == 1:
        return WinnerAnalysisCode.WINNERS_BEFORE_JUMP
    if winner_list[sorted_indices[jump_index+1]] == 0:
        return WinnerAnalysisCode.LOSERS_AFTER_JUMP
    
    return WinnerAnalysisCode.NORMAL

def analyze_fire_history(competitiors_fire_history, competition_len):
    competitiors_fire_count = [len(x) for x in competitiors_fire_history]
    print('\tFire count: ' + str(competitiors_fire_count))
    sorted_indices = sorted(range(len(competitiors_fire_count)), key=lambda k: competitiors_fire_count[k])
    
    jumps = []
    # Collect jumps and general diffs
    for i in range(len(sorted_indices)-1):
        smaller = competitiors_fire_count[sorted_indices[i]]
        bigger = competitiors_fire_count[sorted_indices[i+1]]
        if bigger > (smaller * jump_factor) and bigger > (smaller + jump_diff):
            # Jump
            jumps.append((i,bigger-smaller))
        
    # Make sure one jump is much bigger than the rest
    while True:
        jumps.sort(key=lambda x:x[1])
        if len(jumps) == 0:
            # No jumps
            break
        if len(jumps) > 1:
            if jumps[-1][1] < jump_compare_factor*jumps[-2][1]:
            # The biggest jump is not significantly bigger than the rest of the jumps
                break
        return FireHistoryAnalysisCode.NORMAL
    
    # If we got here the jumps are incorrect. Need to understand the rate of the firing
    firing_rate_average = np.mean(competitiors_fire_count)
    if firing_rate_average > high_firing_rate_threshold * competition_len:
        return FireHistoryAnalysisCode.HIGH_FIRING_RATE
    else:
        return FireHistoryAnalysisCode.LOW_FIRING_RATE

def change_natural_number_by_factor(num, factor):
    if factor > 1:
        return max(num+1,round(factor * num))
    elif factor == 1:
        return num
    else:
        return max(1,min(num-1,round(factor * num)))
    

def calibrate_competition_parameters(configuration):
    print('Starting competition parameters calibration...')
    
    ''' Welcome to the competition parameters calibration tool.
    Competition parameters are parameters that helps us deciding if the competition was
    resolved.
    Our main goal is to find parameters that predict the true winners and losers. The
    true winners are the neurons with the higher firing rate in the end of the
    simulation, while our prediction may occur before the end of the simulation.
    We assume that by the time the function is called we already calibrated the brain
    parameters. '''
    
    configuration['ex_sync_window'] = 95
    configuration['ex_sync_threshold'] = 6
    configuration['ex_unsync_threshold'] = 1
    
    only_losers_count = 0
    only_winners_count = 0
    normal_count = 0
    dichotomized_count = 0
    
    max_iter_num = 100
    for i in range(max_iter_num):
        print('Starting iteration #' + str(i))
        print('\t' + str(configuration))
        
        starting_loc = (0,0)
        while starting_loc == (0,0):
            starting_loc = (int(np.random.rand() * world_height), int(np.random.rand() * world_length))
        world = init_world(world_height, world_length, [((0,0),1)])
        initial_player = (starting_loc[0],starting_loc[1],0)
        action_begin_loc = ll_ob_num+hl_ob_num
        
        model = ModelClass(configuration,False,True)
        
        input_vec = generate_state_from_simulator(world, initial_player)
        input_vec *= model.conf['sensory_input_strength']
        winner_list, _, fire_history = model.calculate_winners(input_vec, action_begin_loc, ll_ac_num,True,True)
        output_code = analyze_competition(winner_list, fire_history[response_layer][action_begin_loc:action_begin_loc+ll_ac_num])
        
        print('\tOutput code: ' + str(output_code))
        
        if output_code in non_dichotomized_codes:
            only_losers_count = 0
            only_winners_count = 0
            normal_count = 0
            dichotomized_count = 0
            if output_code == WinnerAnalysisCode.MIX:
                # Increase the sync window, keep the threshold-window ratio
                configuration['ex_sync_threshold'] = change_natural_number_by_factor(configuration['ex_sync_threshold'],1.1)
                configuration['ex_sync_window'] = change_natural_number_by_factor(configuration['ex_sync_window'],1.1)
                # Do nothing
            elif output_code == WinnerAnalysisCode.NO_JUMP:
                # Do nothing
                '''configuration['Z_in'] = 1.1 * configuration['Z_in']'''
            elif output_code == WinnerAnalysisCode.LOSERS_AFTER_JUMP:
                # Decrease threshold-window-ratio
                configuration['ex_sync_threshold'] = change_natural_number_by_factor(configuration['ex_sync_threshold'],0.9)
                configuration['ex_sync_window'] = change_natural_number_by_factor(configuration['ex_sync_window'],1.1)
            elif output_code == WinnerAnalysisCode.WINNERS_BEFORE_JUMP:
                # Increase threshold-window-ratio
                configuration['ex_sync_threshold'] = change_natural_number_by_factor(configuration['ex_sync_threshold'],1.1)
                configuration['ex_sync_window'] = change_natural_number_by_factor(configuration['ex_sync_window'],0.9)
            elif output_code == WinnerAnalysisCode.MULTIPLE_JUMP:
                # Need to increase sync_threshold
                configuration['ex_sync_threshold'] = change_natural_number_by_factor(configuration['ex_sync_threshold'],1.1)
        else: # We got a dichotomized code
            dichotomized_count += 1
            if output_code == WinnerAnalysisCode.ONLY_LOSERS:
                only_losers_count += 1
            elif output_code == WinnerAnalysisCode.ONLY_WINNERS:
                only_winners_count += 1
            elif output_code == WinnerAnalysisCode.NORMAL:
                normal_count += 1
        
        ''' If we had a sequence of 5 dichotomized codes in a row("a window"), that means we're
        in a good region.
        We still need to make sure we don't get too much only losers/only winners.
        We define "too much" to be more than 2 in the previous window.
        We also demand that there will be at least 3 normal codes in the previous window. ''' 
        if dichotomized_count == 5:
            if only_losers_count > 2:
                # Need to decrease threshold-window ratio
                configuration['ex_sync_threshold'] = change_natural_number_by_factor(configuration['ex_sync_threshold'],0.9)
                configuration['ex_sync_window'] = change_natural_number_by_factor(configuration['ex_sync_window'],1.1)
            elif only_winners_count > 2:
                # Need to increase threshold-window ratio
                configuration['ex_sync_threshold'] = change_natural_number_by_factor(configuration['ex_sync_threshold'],1.1)
                configuration['ex_sync_window'] = change_natural_number_by_factor(configuration['ex_sync_window'],0.9)
            elif normal_count > 2:
                break
            only_losers_count = 0
            only_winners_count = 0
            normal_count = 0
            dichotomized_count = 0
            
    if i == max_iter_num-1:
        print('Unable to calibrate competition parameters')
        assert(False)
    
    print('Calibrated competition parameters successfully!')
    
def calibrate_brain_parameters(configuration):
    print('Starting brain parameters calibration tool...')
    
    ''' Welcome to the brain parameter calibration tool.
    Brain parameters are parameters that exist in reality, such as the normalization
    parameters.
    Our main goal is to find parameters that create good separation between winners and
    losers, but we want to do it while maximizing Z_in (which is statistically proven to
    promote good separation). So we start with a very high Z_in, and try it. If it's too
    high- we'll lower it down and stop when we first see good results. '''
    Z_ex = ModelClass.default_configuration['Z_ex_ex_th_ratio']*ModelClass.default_configuration['excitatory_threshold']
    
    configuration['Z_in'] = (iin_num/unit_num) * ModelClass.default_configuration['excitatory_threshold'] * 20
    highest_Z_ll_ob_to_ll_ac_possible = 0.6 * Z_ex
    configuration['Z_ll_ob_to_ll_ac'] = highest_Z_ll_ob_to_ll_ac_possible
    
    # Competition parameters- irrelevant for now
    configuration['ex_sync_window'] = 0
    configuration['ex_sync_threshold'] = 0
    configuration['ex_unsync_threshold'] = 0
    
    normal_count = 0
    low_firing_rate_count = 0
    high_firing_rate_count = 0
    window_len = 6
    good_parameters_threshold = 4
    
    max_iter_num = 100
    for i in range(max_iter_num):
        print('Starting iteration #' + str(i))
        print('\t' + str(configuration))
        
        # Create some random state
        starting_loc = (0,0)
        while starting_loc == (0,0):
            starting_loc = (int(np.random.rand() * world_height), int(np.random.rand() * world_length))
        world = init_world(world_height, world_length, [((0,0),1)])
        initial_player = (starting_loc[0],starting_loc[1],0)
        goals = [(24,25)]
        
        model = ModelClass(configuration,False,True)
        action_begin_loc = ll_ob_num+hl_ob_num
        
        input_vec = generate_state_from_simulator(world, initial_player, goals)
        input_vec *= model.conf['sensory_input_strength']
        _, competition_len, fire_history = model.calculate_winners(input_vec, action_begin_loc, ll_ac_num,True,False)
        output_code = analyze_fire_history(fire_history[response_layer][action_begin_loc:action_begin_loc+ll_ac_num], competition_len)
        
        print('\tOutput code: ' + str(output_code))
        
        if output_code == FireHistoryAnalysisCode.NORMAL:
            normal_count += 1
        elif output_code == FireHistoryAnalysisCode.LOW_FIRING_RATE:
            low_firing_rate_count +=1
        elif output_code == FireHistoryAnalysisCode.HIGH_FIRING_RATE:
            high_firing_rate_count +=1
        
        if normal_count == good_parameters_threshold:
            # calibrated good parameters
            break
        
        if low_firing_rate_count > (window_len - good_parameters_threshold):
            # Low firing rate- we need to fix the problem
            print('Firing rate is too low- changing parameters')
            if configuration['Z_ll_ob_to_ll_ac'] + 0.02 * Z_ex > highest_Z_ll_ob_to_ll_ac_possible:
                configuration['Z_in'] = configuration['Z_in'] - (iin_num/unit_num) * ModelClass.default_configuration['excitatory_threshold']
            else:
                configuration['Z_ll_ob_to_ll_ac'] = configuration['Z_ll_ob_to_ll_ac'] + 0.02 * Z_ex
            
            normal_count = 0
            low_firing_rate_count = 0
            high_firing_rate_count = 0
            
        if high_firing_rate_count > (window_len - good_parameters_threshold):
            # High firing rate- we need to fix the problem
            print('Firing rate is too high- changing parameters')
            configuration['Z_in'] = configuration['Z_in'] + (iin_num/unit_num) * ModelClass.default_configuration['excitatory_threshold']
            configuration['Z_ll_ob_to_ll_ac'] = configuration['Z_ll_ob_to_ll_ac'] - 0.02 * Z_ex
            
            normal_count = 0
            low_firing_rate_count = 0
            high_firing_rate_count = 0
        
        if low_firing_rate_count > 1 and high_firing_rate_count > 1:
            # No separation. We need more inhibition
            print('Firing rates are not separated- changing parameters')
            configuration['Z_in'] = configuration['Z_in'] + (iin_num/unit_num) * ModelClass.default_configuration['excitatory_threshold']
            
            normal_count = 0
            low_firing_rate_count = 0
            high_firing_rate_count = 0
            
    if i == max_iter_num-1:
        print('Unable to calibrate brain parameters')
        assert(False)
    
    print('Calibrated brain parameters successfully!')

def calibrate_parameters(configuration):
    print('Starting parameters calibration tool...')
    
    configuration['quiet'] = True
    
    calibrate_brain_parameters(configuration)
    calibrate_competition_parameters(configuration)

#############
# Main code #
#############
def main(load_from_file, configuration):
    quiet = False
    
    starting_loc = (0,0)
    while starting_loc == (0,0):
        starting_loc = (int(np.random.rand() * world_height), int(np.random.rand() * world_length))
    world = init_world(world_height, world_length, [((0,0),1)])
    initial_player = (starting_loc[0],starting_loc[1],0)
    
    model = ModelClass(configuration,load_from_file,quiet)
    
    cur_player = initial_player
    if not quiet:
        print(cur_player)
        
    shortest_path_to_target_len = get_shortest_path_to_target_len(cur_player, goals[0])
    max_rounds = 5 * shortest_path_to_target_len
    
    action_begin_loc = ll_ob_num+hl_ob_num
    
    prev_sensory_input_vec = None
    sensory_input_vec = None
    winner_list = None
    comp_len = 0
    
    for i in range(max_rounds):
        prev_sensory_input_vec = sensory_input_vec
        sensory_input_vec = generate_state_from_simulator(world, cur_player, goals)
        sensory_input_vec *= model.conf['sensory_input_strength']
        if not quiet:
            print('sensory input vec: ' + str((sensory_input_vec.transpose() > 0).astype(int)))
        if i > 0:
            model.update_synapse_strength_long_term(winner_list, sensory_input_vec, prev_sensory_input_vec, comp_len)
        
        winner_list, comp_len, _ = model.calculate_winners(sensory_input_vec, action_begin_loc, ll_ac_num,False,True)
        next_move = model.convert_winner_list_to_action(winner_list)
        if not quiet:
            print('winner list: ' + str(winner_list))
            print('move: ' + str(next_move))
    
        new_orientation = (cur_player[2] + next_move) % 8
        new_player_loc = advance(new_orientation, (cur_player[0], cur_player[1]))
        if not quiet:
            print('step number ' + str(i))
        if is_edge(new_player_loc, world.shape) or world[new_player_loc] == OBST:
            if not quiet:
                print('Encountered an obstacle. Not moving')
            cur_player = (cur_player[0], cur_player[1], new_orientation)
        else:
            cur_player = (new_player_loc[0], new_player_loc[1], new_orientation)
            if not quiet:
                print(cur_player)
         
        if new_player_loc in goals:
            prev_sensory_input_vec = sensory_input_vec
            sensory_input_vec = generate_state_from_simulator(world, cur_player, goals)
            sensory_input_vec *= model.conf['sensory_input_strength']
            if not quiet:
                print('sensory input vec: ' + str((sensory_input_vec.transpose() > 0).astype(int)))
            model.update_synapse_strength_long_term(winner_list, sensory_input_vec, prev_sensory_input_vec, comp_len)
            break
    
    model.save_synapse_strength()
    score = evaluate(world, goals, cur_player, i+1, shortest_path_to_target_len, max_rounds)
    return score
    
'''scores = []
epoch_num = 60
for epoch in range(epoch_num):
    print('***')
    print('epoch ' + str(epoch))
    print('***')
    configuration = {}
    
    if epoch == 0:
        load_from_file = False
    else:
        load_from_file = True
    score = main(load_from_file, configuration)
    scores.append(score)
    
plt.plot(range(epoch_num), scores)
plt.savefig('res')'''

'''configuration = {}
calibrate_parameters(configuration)
print(configuration)'''

mean_dist = 0
for i in range(world_height):
    for j in range(world_height):
        mean_dist += (1/(world_height*world_length))*((i-goals[0][0])**2+(j-goals[0][1])**2)**0.5
        
max_distance = (world_height**2 + world_length**2)**0.5
mean_dist_val = 1-mean_dist/max_distance
model = ModelClass({},False,True)
mu_s = (mean_dist_val * model.conf['sensory_input_strength'])/8

T = np.identity(6)
T = model.conf['excitatory_threshold']*T
T[5,5] = model.conf['inhibitory_threshold']

Z = np.zeros((6,6))

Z[0,0] = model.conf['Z_ll_ob_to_ll_ob']
Z[0,1] = model.conf['Z_hl_ob_to_ll_ob']
Z[0,2] = model.conf['Z_ll_ac_to_ll_ob']
Z[0,3] = model.conf['Z_ml_ac_to_ll_ob']
Z[0,4] = model.conf['Z_tl_ac_to_ll_ob']
Z[0,5] = (-1)*(model.conf['Z_in'])

Z[1,0] = model.conf['Z_ll_ob_to_hl_ob']
Z[1,1] = model.conf['Z_hl_ob_to_hl_ob']
Z[1,2] = model.conf['Z_ll_ac_to_hl_ob']
Z[1,3] = model.conf['Z_ml_ac_to_hl_ob']
Z[1,4] = model.conf['Z_tl_ac_to_hl_ob']
Z[1,5] = (-1)*(model.conf['Z_in'])

Z[2,0] = model.conf['Z_ll_ob_to_ll_ac']
Z[2,1] = model.conf['Z_hl_ob_to_ll_ac']
Z[2,2] = model.conf['Z_ll_ac_to_ll_ac']
Z[2,3] = model.conf['Z_ml_ac_to_ll_ac']
Z[2,4] = model.conf['Z_tl_ac_to_ll_ac']
Z[2,5] = (-1)*(model.conf['Z_in'])

Z[3,0] = model.conf['Z_ll_ob_to_ml_ac']
Z[3,1] = model.conf['Z_hl_ob_to_ml_ac']
Z[3,2] = model.conf['Z_ll_ac_to_ml_ac']
Z[3,3] = model.conf['Z_ml_ac_to_ml_ac']
Z[3,4] = model.conf['Z_tl_ac_to_ml_ac']
Z[3,5] = (-1)*(model.conf['Z_in'])

Z[4,0] = model.conf['Z_ll_ob_to_tl_ac']
Z[4,1] = model.conf['Z_hl_ob_to_tl_ac']
Z[4,2] = model.conf['Z_ll_ac_to_tl_ac']
Z[4,3] = model.conf['Z_ml_ac_to_tl_ac']
Z[4,4] = model.conf['Z_tl_ac_to_tl_ac']
Z[4,5] = (-1)*(model.conf['Z_in'])

Z[5,0] = model.conf['Z_ll_ob_to_in']
Z[5,1] = model.conf['Z_hl_ob_to_in']
Z[5,2] = model.conf['Z_ll_ac_to_in']
Z[5,3] = model.conf['Z_ml_ac_to_in']
Z[5,4] = model.conf['Z_tl_ac_to_in']
Z[5,5] = (-1)*(model.conf['Z_in'])

b = np.zeros((6,1))
b[0,0] = (-1)*mu_s

iter_num = 5000
fire_rate = np.zeros((unit_num,1))
for cur_iter in range(iter_num):
    model = ModelClass({},False,True)
    print('Starting iter ' + str(cur_iter))
    starting_loc = (int(np.random.rand() * world_height), int(np.random.rand() * world_length))
    world = init_world(world_height, world_length, [((0,0),1)])
    initial_player = (starting_loc[0],starting_loc[1],0)
    action_begin_loc = ll_ob_num+hl_ob_num
    
    input_vec = generate_state_from_simulator(world, initial_player)
    input_vec *= model.conf['sensory_input_strength']
    _, _, fire_history = model.calculate_winners(input_vec, action_begin_loc, ll_ac_num,True,False)
    fire_count = np.array([len(a) for a in fire_history[0]]).reshape((unit_num,1))
    fire_rate += fire_count / default_competition_len
fire_rate /= iter_num
zone_fire_rate = np.zeros((6,1))
ll_ob_begin = 0
hl_ob_begin = ll_ob_begin + ll_ob_num
ll_ac_begin = hl_ob_begin + hl_ob_num
ml_ac_begin = ll_ac_begin + ll_ac_num
tl_ac_begin = ml_ac_begin + ml_ac_num
in_begin = tl_ac_begin + tl_ac_num
zone_fire_rate[0,0] = np.sum(fire_rate[ll_ob_begin:ll_ob_begin+ll_ob_num,0])/ll_ob_num
zone_fire_rate[1,0] = np.sum(fire_rate[hl_ob_begin:hl_ob_begin+hl_ob_num,0])/hl_ob_num
zone_fire_rate[2,0] = np.sum(fire_rate[ll_ac_begin:ll_ac_begin+ll_ac_num,0])/ll_ac_num
zone_fire_rate[3,0] = np.sum(fire_rate[ml_ac_begin:ml_ac_begin+ml_ac_num,0])/ml_ac_num
zone_fire_rate[4,0] = np.sum(fire_rate[tl_ac_begin:tl_ac_begin+tl_ac_num,0])/tl_ac_num
zone_fire_rate[5,0] = np.sum(fire_rate[in_begin:in_begin+iin_num,0])/iin_num
    
print(np.linalg.solve((Z-T),b).transpose())
print(zone_fire_rate.transpose())