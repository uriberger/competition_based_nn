#############################################
#### Competition and Synchrony Model V2 #####
######### By Uri Berger, January 20 #########
#############################################

import numpy as np
import math
import matplotlib.pyplot as plt
import enum

########################## #
# Configuration parameters #
############################

# Connection of inhibitory neurons to inhibitory neurons from other layers
IIN_PROJECTIONS = False

# Fast connection between inhibitory neurons
IIN_FAST_CONNECTIONS = False

# Connection of inhibitory neurons only to specific excitatory neurons
SPATIAL_IINS = False

###################
# Model functions #
###################

# Parameters
excitatory_precentage = 0.8
ll_ob_num = 10
ll_ac_num = 8
hl_ob_num = 7
ml_ac_num = 3
tl_ac_num = 4
hl_ac_num = ml_ac_num+tl_ac_num
iin_num = math.ceil((ll_ob_num+ll_ac_num+hl_ob_num+hl_ac_num) * \
                        (1-excitatory_precentage)/excitatory_precentage)
unit_num = 40
excitatory_threshold = 0.4
inhibitory_threshold = 0.2
sensory_input_strength = excitatory_threshold / 3
eta = 0.0001
comp_len_zeta_ratio = 1500
gamma_ex = 0.05
gamma_in = 0.14
active_ll_ob_threshold = excitatory_threshold/20
layer_num = 5
winner_window_len = 10
#response_layer = layer_num-1
response_layer = 0

# Normalization constants
Z_ex = excitatory_threshold * 2

Z_ll_ob_to_ll_ob = 0.1 * Z_ex
Z_hl_ob_to_ll_ob = 0.6 * Z_ex 
Z_ll_ac_to_ll_ob = 0.1 * Z_ex
Z_ml_ac_to_ll_ob = 0.1 * Z_ex
Z_tl_ac_to_ll_ob = 0.1 * Z_ex

Z_ll_ob_to_hl_ob = 0.6 * Z_ex
Z_hl_ob_to_hl_ob = 0.1 * Z_ex
Z_ll_ac_to_hl_ob = 0.1 * Z_ex
Z_ml_ac_to_hl_ob = 0.1 * Z_ex
Z_tl_ac_to_hl_ob = 0.1 * Z_ex

Z_hl_ob_to_ll_ac = 0.1 * Z_ex
Z_ll_ac_to_ll_ac = 0.1 * Z_ex
Z_ml_ac_to_ll_ac = 0.4 * Z_ex
Z_tl_ac_to_ll_ac = 0.1 * Z_ex

Z_ll_ob_to_ml_ac = 0.1 * Z_ex
Z_hl_ob_to_ml_ac = 0.1 * Z_ex
Z_ll_ac_to_ml_ac = 0.1 * Z_ex
Z_ml_ac_to_ml_ac = 0.1 * Z_ex
Z_tl_ac_to_ml_ac = 0.6 * Z_ex

Z_ll_ob_to_tl_ac = 0.1 * Z_ex
Z_hl_ob_to_tl_ac = 0.6 * Z_ex
Z_ll_ac_to_tl_ac = 0.1 * Z_ex
Z_ml_ac_to_tl_ac = 0.1 * Z_ex
Z_tl_ac_to_tl_ac = 0.1 * Z_ex

Z_ll_ob_to_in = 0.2 * Z_ex
Z_hl_ob_to_in = 0.2 * Z_ex 
Z_ll_ac_to_in = 0.2 * Z_ex
Z_ml_ac_to_in = 0.2 * Z_ex
Z_tl_ac_to_in = 0.2 * Z_ex

#Z_in = (iin_num/unit_num) * excitatory_threshold * 11
Z_forward = 0.3 * excitatory_threshold
Z_backward = 0.1 * excitatory_threshold
Z_sensory_response = 0.2 * excitatory_threshold
Z_response_prediction = 0.2 * excitatory_threshold

# Data structures
synapse_strength = None
zero_matrices = None
before_prev_act = None
prev_act = None
prev_input = None

class ModelClass:
    
    def __init__(self, configuration):
        self.Z_in = configuration['Z_in']
        self.ex_sync_window = configuration['ex_sync_window']
        self.ex_sync_threshold = configuration['ex_sync_threshold']
        self.ex_unsync_threshold = configuration['ex_unsync_threshold']
        self.Z_ll_ob_to_ll_ac = configuration['Z_ll_ob_to_ll_ac']
        self.quiet = configuration['quiet']

    def save_synapse_strength(self):
        # Save the synapse strength matrix to a file.
        trained_strength = [synapse_strength]
        file_name = "synapse_strength"
        file_name += "_" + str(unit_num) + "_neurons_" + str(layer_num) + "_layers"
        np.save(file_name, trained_strength)
    
    def initialize(self, load_from_file = False):
        ''' Initialize the data structures.
        load_from_file- if different from 'None', holds the suffix of the file to be
        loaded. ''' 
        global synapse_strength
        global zero_matrices
        synapse_strength = []
        
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
                synapse_strength.append(post_layer_synapses)
            
            # Hard coded good weights
            '''for layer in range(layer_num):
                ll_act_begin = ll_ob_num+hl_ob_num
                synapse_strength[layer][layer][ll_act_begin,0] = 0
                synapse_strength[layer][layer][ll_act_begin,1] = 1
                synapse_strength[layer][layer][ll_act_begin,2] = 1
                synapse_strength[layer][layer][ll_act_begin,3] = 0
                synapse_strength[layer][layer][ll_act_begin,4] = 1
                synapse_strength[layer][layer][ll_act_begin,5] = 0
                synapse_strength[layer][layer][ll_act_begin,6] = 0
                synapse_strength[layer][layer][ll_act_begin,7] = 0
                synapse_strength[layer][layer][ll_act_begin,8] = 0
                synapse_strength[layer][layer][ll_act_begin,9] = 0
                
                synapse_strength[layer][layer][ll_act_begin+1,0] = 0
                synapse_strength[layer][layer][ll_act_begin+1,1] = 0
                synapse_strength[layer][layer][ll_act_begin+1,2] = 1
                synapse_strength[layer][layer][ll_act_begin+1,3] = 1
                synapse_strength[layer][layer][ll_act_begin+1,4] = 1
                synapse_strength[layer][layer][ll_act_begin+1,5] = 0
                synapse_strength[layer][layer][ll_act_begin+1,6] = 0
                synapse_strength[layer][layer][ll_act_begin+1,7] = 0
                synapse_strength[layer][layer][ll_act_begin+1,8] = 0
                synapse_strength[layer][layer][ll_act_begin+1,9] = 0
                
                synapse_strength[layer][layer][ll_act_begin+2,0] = 0
                synapse_strength[layer][layer][ll_act_begin+2,1] = 0
                synapse_strength[layer][layer][ll_act_begin+2,2] = 0
                synapse_strength[layer][layer][ll_act_begin+2,3] = 1
                synapse_strength[layer][layer][ll_act_begin+2,4] = 1
                synapse_strength[layer][layer][ll_act_begin+2,5] = 0
                synapse_strength[layer][layer][ll_act_begin+2,6] = 0
                synapse_strength[layer][layer][ll_act_begin+2,7] = 0
                synapse_strength[layer][layer][ll_act_begin+2,8] = 0
                synapse_strength[layer][layer][ll_act_begin+2,9] = 0
                
                synapse_strength[layer][layer][ll_act_begin+3,0] = 0
                synapse_strength[layer][layer][ll_act_begin+3,1] = 0
                synapse_strength[layer][layer][ll_act_begin+3,2] = 0
                synapse_strength[layer][layer][ll_act_begin+3,3] = 1
                synapse_strength[layer][layer][ll_act_begin+3,4] = 1
                synapse_strength[layer][layer][ll_act_begin+3,5] = 0
                synapse_strength[layer][layer][ll_act_begin+3,6] = 0
                synapse_strength[layer][layer][ll_act_begin+3,7] = 0
                synapse_strength[layer][layer][ll_act_begin+3,8] = 0
                synapse_strength[layer][layer][ll_act_begin+3,9] = 0
                
                synapse_strength[layer][layer][ll_act_begin+4,0] = 0
                synapse_strength[layer][layer][ll_act_begin+4,1] = 0
                synapse_strength[layer][layer][ll_act_begin+4,2] = 0
                synapse_strength[layer][layer][ll_act_begin+4,3] = 0
                synapse_strength[layer][layer][ll_act_begin+4,4] = 0
                synapse_strength[layer][layer][ll_act_begin+4,5] = 0
                synapse_strength[layer][layer][ll_act_begin+4,6] = 1
                synapse_strength[layer][layer][ll_act_begin+4,7] = 1
                synapse_strength[layer][layer][ll_act_begin+4,8] = 0
                synapse_strength[layer][layer][ll_act_begin+4,9] = 0
                
                synapse_strength[layer][layer][ll_act_begin+5,0] = 0
                synapse_strength[layer][layer][ll_act_begin+5,1] = 0
                synapse_strength[layer][layer][ll_act_begin+5,2] = 0
                synapse_strength[layer][layer][ll_act_begin+5,3] = 0
                synapse_strength[layer][layer][ll_act_begin+5,4] = 0
                synapse_strength[layer][layer][ll_act_begin+5,5] = 1
                synapse_strength[layer][layer][ll_act_begin+5,6] = 1
                synapse_strength[layer][layer][ll_act_begin+5,7] = 0
                synapse_strength[layer][layer][ll_act_begin+5,8] = 0
                synapse_strength[layer][layer][ll_act_begin+5,9] = 0
                
                synapse_strength[layer][layer][ll_act_begin+6,0] = 0
                synapse_strength[layer][layer][ll_act_begin+6,1] = 0
                synapse_strength[layer][layer][ll_act_begin+6,2] = 0
                synapse_strength[layer][layer][ll_act_begin+6,3] = 0
                synapse_strength[layer][layer][ll_act_begin+6,4] = 0
                synapse_strength[layer][layer][ll_act_begin+6,5] = 1
                synapse_strength[layer][layer][ll_act_begin+6,6] = 0
                synapse_strength[layer][layer][ll_act_begin+6,7] = 0
                synapse_strength[layer][layer][ll_act_begin+6,8] = 0
                synapse_strength[layer][layer][ll_act_begin+6,9] = 0
                
                synapse_strength[layer][layer][ll_act_begin+7,0] = 0
                synapse_strength[layer][layer][ll_act_begin+7,1] = 0
                synapse_strength[layer][layer][ll_act_begin+7,2] = 0
                synapse_strength[layer][layer][ll_act_begin+7,3] = 0
                synapse_strength[layer][layer][ll_act_begin+7,4] = 0
                synapse_strength[layer][layer][ll_act_begin+7,5] = 1
                synapse_strength[layer][layer][ll_act_begin+7,6] = 0
                synapse_strength[layer][layer][ll_act_begin+7,7] = 0
                synapse_strength[layer][layer][ll_act_begin+7,8] = 0
                synapse_strength[layer][layer][ll_act_begin+7,9] = 0'''
                
        else:
            file_name = "synapse_strength"
            file_name += "_" + str(unit_num) + "_neurons_" + str(layer_num) + "_layers"
            file_name += ".npy"
            
            trained_strength = np.load(file_name)
        
            synapse_strength = trained_strength[0]
            
        global prev_act
        prev_act = []
        for _ in range(layer_num):
            prev_act.append(np.zeros((unit_num, 1)))
        
        global before_prev_act
        before_prev_act = []
        for _ in range(layer_num):
            before_prev_act.append(np.zeros((unit_num, 1)))
        
        global prev_input
        prev_input = []
        for _ in range(layer_num):
            prev_input.append(np.zeros((unit_num, 1)))
        
        zero_matrices = []
        for post_layer in range(layer_num):
            zero_matrices.append([])
            for pre_layer in range(layer_num):
                zero_matrix = np.zeros((unit_num,unit_num))
                if pre_layer == post_layer:
                    zero_matrix[:,:] = 1
                    # No self loops
                    np.fill_diagonal(zero_matrix, 0)
                if pre_layer == post_layer-1 or pre_layer == post_layer+1 or \
                (pre_layer == 0 and post_layer == layer_num-1): # Response neurons are directly innervated by sensory neurons
                    for unit_ind in range(unit_num-iin_num):
                        zero_matrix[unit_ind,unit_ind] = 1
                if pre_layer == layer_num-1 and post_layer == layer_num-2:
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
                            
                zero_matrices[post_layer].append(zero_matrix)
        
        self.fix_synapse_strength()
    
    def fix_synapse_strength(self):
        # Normalize the synapses strength, and enforce the invariants.
        global synapse_strength
        
        excitatory_unit_num = unit_num-iin_num
        
        for post_layer in range(layer_num):
            for pre_layer in range(layer_num):
                synapse_strength[post_layer][pre_layer] = np.multiply(synapse_strength[post_layer][pre_layer], zero_matrices[post_layer][pre_layer])
                if post_layer == pre_layer+1:
                    normalized_weight = Z_forward
                    for unit_ind in range(unit_num-iin_num):
                        synapse_strength[post_layer][pre_layer][unit_ind,unit_ind] = normalized_weight
                if post_layer == pre_layer-1:
                    normalized_weight = Z_backward
                    for unit_ind in range(unit_num-iin_num):
                        synapse_strength[post_layer][pre_layer][unit_ind,unit_ind] = normalized_weight
                if post_layer == pre_layer:
                    # Make sure excitatory weights in inter node are all excitatory
                    synapse_strength[post_layer][pre_layer][:,:excitatory_unit_num][synapse_strength[post_layer][pre_layer][:,:excitatory_unit_num] < 0] = 0
                    
                    # Normalize incoming excitatory weights to each unit
                    '''normalizing_factor = Z_ex
                    excitatory_row_sums = (synapse_strength[post_layer][pre_layer][:,:excitatory_unit_num].sum(axis=1).reshape(unit_num,1).repeat(excitatory_unit_num, axis=1))/normalizing_factor'''
                    ll_ob_begin = 0
                    hl_ob_begin = ll_ob_begin + ll_ob_num
                    ll_ac_begin = hl_ob_begin + hl_ob_num
                    ml_ac_begin = ll_ac_begin + ll_ac_num
                    tl_ac_begin = ml_ac_begin + ml_ac_num
                    
                    ll_ob_to_ll_ob_row_sum = (synapse_strength[post_layer][pre_layer][ll_ob_begin:ll_ob_begin+ll_ob_num,ll_ob_begin:ll_ob_begin+ll_ob_num].sum(axis=1).reshape(ll_ob_num,1).repeat(ll_ob_num, axis=1))/Z_ll_ob_to_ll_ob
                    hl_ob_to_ll_ob_row_sum = (synapse_strength[post_layer][pre_layer][ll_ob_begin:ll_ob_begin+ll_ob_num,hl_ob_begin:hl_ob_begin+hl_ob_num].sum(axis=1).reshape(ll_ob_num,1).repeat(hl_ob_num, axis=1))/Z_hl_ob_to_ll_ob
                    ll_ac_to_ll_ob_row_sum = (synapse_strength[post_layer][pre_layer][ll_ob_begin:ll_ob_begin+ll_ob_num,ll_ac_begin:ll_ac_begin+ll_ac_num].sum(axis=1).reshape(ll_ob_num,1).repeat(ll_ac_num, axis=1))/Z_ll_ac_to_ll_ob
                    ml_ac_to_ll_ob_row_sum = (synapse_strength[post_layer][pre_layer][ll_ob_begin:ll_ob_begin+ll_ob_num,ml_ac_begin:ml_ac_begin+ml_ac_num].sum(axis=1).reshape(ll_ob_num,1).repeat(ml_ac_num, axis=1))/Z_ml_ac_to_ll_ob
                    tl_ac_to_ll_ob_row_sum = (synapse_strength[post_layer][pre_layer][ll_ob_begin:ll_ob_begin+ll_ob_num,tl_ac_begin:tl_ac_begin+tl_ac_num].sum(axis=1).reshape(ll_ob_num,1).repeat(tl_ac_num, axis=1))/Z_tl_ac_to_ll_ob
                    ll_ob_row_sum = np.concatenate((ll_ob_to_ll_ob_row_sum,hl_ob_to_ll_ob_row_sum,ll_ac_to_ll_ob_row_sum,ml_ac_to_ll_ob_row_sum,tl_ac_to_ll_ob_row_sum),axis=1)
                    
                    ll_ob_to_hl_ob_row_sum = (synapse_strength[post_layer][pre_layer][hl_ob_begin:hl_ob_begin+hl_ob_num,ll_ob_begin:ll_ob_begin+ll_ob_num].sum(axis=1).reshape(hl_ob_num,1).repeat(ll_ob_num, axis=1))/Z_ll_ob_to_hl_ob
                    hl_ob_to_hl_ob_row_sum = (synapse_strength[post_layer][pre_layer][hl_ob_begin:hl_ob_begin+hl_ob_num,hl_ob_begin:hl_ob_begin+hl_ob_num].sum(axis=1).reshape(hl_ob_num,1).repeat(hl_ob_num, axis=1))/Z_hl_ob_to_hl_ob
                    ll_ac_to_hl_ob_row_sum = (synapse_strength[post_layer][pre_layer][hl_ob_begin:hl_ob_begin+hl_ob_num,ll_ac_begin:ll_ac_begin+ll_ac_num].sum(axis=1).reshape(hl_ob_num,1).repeat(ll_ac_num, axis=1))/Z_ll_ac_to_hl_ob
                    ml_ac_to_hl_ob_row_sum = (synapse_strength[post_layer][pre_layer][hl_ob_begin:hl_ob_begin+hl_ob_num,ml_ac_begin:ml_ac_begin+ml_ac_num].sum(axis=1).reshape(hl_ob_num,1).repeat(ml_ac_num, axis=1))/Z_ml_ac_to_hl_ob
                    tl_ac_to_hl_ob_row_sum = (synapse_strength[post_layer][pre_layer][hl_ob_begin:hl_ob_begin+hl_ob_num,tl_ac_begin:tl_ac_begin+tl_ac_num].sum(axis=1).reshape(hl_ob_num,1).repeat(tl_ac_num, axis=1))/Z_tl_ac_to_hl_ob
                    hl_ob_row_sum = np.concatenate((ll_ob_to_hl_ob_row_sum,hl_ob_to_hl_ob_row_sum,ll_ac_to_hl_ob_row_sum,ml_ac_to_hl_ob_row_sum,tl_ac_to_hl_ob_row_sum),axis=1)
                    
                    ll_ob_to_ll_ac_row_sum = (synapse_strength[post_layer][pre_layer][ll_ac_begin:ll_ac_begin+ll_ac_num,ll_ob_begin:ll_ob_begin+ll_ob_num].sum(axis=1).reshape(ll_ac_num,1).repeat(ll_ob_num, axis=1))/self.Z_ll_ob_to_ll_ac
                    hl_ob_to_ll_ac_row_sum = (synapse_strength[post_layer][pre_layer][ll_ac_begin:ll_ac_begin+ll_ac_num,hl_ob_begin:hl_ob_begin+hl_ob_num].sum(axis=1).reshape(ll_ac_num,1).repeat(hl_ob_num, axis=1))/Z_hl_ob_to_ll_ac
                    ll_ac_to_ll_ac_row_sum = (synapse_strength[post_layer][pre_layer][ll_ac_begin:ll_ac_begin+ll_ac_num,ll_ac_begin:ll_ac_begin+ll_ac_num].sum(axis=1).reshape(ll_ac_num,1).repeat(ll_ac_num, axis=1))/Z_ll_ac_to_ll_ac
                    ml_ac_to_ll_ac_row_sum = (synapse_strength[post_layer][pre_layer][ll_ac_begin:ll_ac_begin+ll_ac_num,ml_ac_begin:ml_ac_begin+ml_ac_num].sum(axis=1).reshape(ll_ac_num,1).repeat(ml_ac_num, axis=1))/Z_ml_ac_to_ll_ac
                    tl_ac_to_ll_ac_row_sum = (synapse_strength[post_layer][pre_layer][ll_ac_begin:ll_ac_begin+ll_ac_num,tl_ac_begin:tl_ac_begin+tl_ac_num].sum(axis=1).reshape(ll_ac_num,1).repeat(tl_ac_num, axis=1))/Z_tl_ac_to_ll_ac
                    ll_ac_row_sum = np.concatenate((ll_ob_to_ll_ac_row_sum,hl_ob_to_ll_ac_row_sum,ll_ac_to_ll_ac_row_sum,ml_ac_to_ll_ac_row_sum,tl_ac_to_ll_ac_row_sum),axis=1)
                    
                    ll_ob_to_ml_ac_row_sum = (synapse_strength[post_layer][pre_layer][ml_ac_begin:ml_ac_begin+ml_ac_num,ll_ob_begin:ll_ob_begin+ll_ob_num].sum(axis=1).reshape(ml_ac_num,1).repeat(ll_ob_num, axis=1))/Z_ll_ob_to_ml_ac
                    hl_ob_to_ml_ac_row_sum = (synapse_strength[post_layer][pre_layer][ml_ac_begin:ml_ac_begin+ml_ac_num,hl_ob_begin:hl_ob_begin+hl_ob_num].sum(axis=1).reshape(ml_ac_num,1).repeat(hl_ob_num, axis=1))/Z_hl_ob_to_ml_ac
                    ll_ac_to_ml_ac_row_sum = (synapse_strength[post_layer][pre_layer][ml_ac_begin:ml_ac_begin+ml_ac_num,ll_ac_begin:ll_ac_begin+ll_ac_num].sum(axis=1).reshape(ml_ac_num,1).repeat(ll_ac_num, axis=1))/Z_ll_ac_to_ml_ac
                    ml_ac_to_ml_ac_row_sum = (synapse_strength[post_layer][pre_layer][ml_ac_begin:ml_ac_begin+ml_ac_num,ml_ac_begin:ml_ac_begin+ml_ac_num].sum(axis=1).reshape(ml_ac_num,1).repeat(ml_ac_num, axis=1))/Z_ml_ac_to_ml_ac
                    tl_ac_to_ml_ac_row_sum = (synapse_strength[post_layer][pre_layer][ml_ac_begin:ml_ac_begin+ml_ac_num,tl_ac_begin:tl_ac_begin+tl_ac_num].sum(axis=1).reshape(ml_ac_num,1).repeat(tl_ac_num, axis=1))/Z_tl_ac_to_ml_ac
                    ml_ac_row_sum = np.concatenate((ll_ob_to_ml_ac_row_sum,hl_ob_to_ml_ac_row_sum,ll_ac_to_ml_ac_row_sum,ml_ac_to_ml_ac_row_sum,tl_ac_to_ml_ac_row_sum),axis=1)
                    
                    ll_ob_to_tl_ac_row_sum = (synapse_strength[post_layer][pre_layer][tl_ac_begin:tl_ac_begin+tl_ac_num,ll_ob_begin:ll_ob_begin+ll_ob_num].sum(axis=1).reshape(tl_ac_num,1).repeat(ll_ob_num, axis=1))/Z_ll_ob_to_tl_ac
                    hl_ob_to_tl_ac_row_sum = (synapse_strength[post_layer][pre_layer][tl_ac_begin:tl_ac_begin+tl_ac_num,hl_ob_begin:hl_ob_begin+hl_ob_num].sum(axis=1).reshape(tl_ac_num,1).repeat(hl_ob_num, axis=1))/Z_hl_ob_to_tl_ac
                    ll_ac_to_tl_ac_row_sum = (synapse_strength[post_layer][pre_layer][tl_ac_begin:tl_ac_begin+tl_ac_num,ll_ac_begin:ll_ac_begin+ll_ac_num].sum(axis=1).reshape(tl_ac_num,1).repeat(ll_ac_num, axis=1))/Z_ll_ac_to_tl_ac
                    ml_ac_to_tl_ac_row_sum = (synapse_strength[post_layer][pre_layer][tl_ac_begin:tl_ac_begin+tl_ac_num,ml_ac_begin:ml_ac_begin+ml_ac_num].sum(axis=1).reshape(tl_ac_num,1).repeat(ml_ac_num, axis=1))/Z_ml_ac_to_tl_ac
                    tl_ac_to_tl_ac_row_sum = (synapse_strength[post_layer][pre_layer][tl_ac_begin:tl_ac_begin+tl_ac_num,tl_ac_begin:tl_ac_begin+tl_ac_num].sum(axis=1).reshape(tl_ac_num,1).repeat(tl_ac_num, axis=1))/Z_tl_ac_to_tl_ac
                    tl_ac_row_sum = np.concatenate((ll_ob_to_tl_ac_row_sum,hl_ob_to_tl_ac_row_sum,ll_ac_to_tl_ac_row_sum,ml_ac_to_tl_ac_row_sum,tl_ac_to_tl_ac_row_sum),axis=1)
                    
                    ll_ob_to_in_row_sum = (synapse_strength[post_layer][pre_layer][excitatory_unit_num:unit_num,ll_ob_begin:ll_ob_begin+ll_ob_num].sum(axis=1).reshape(iin_num,1).repeat(ll_ob_num, axis=1))/Z_ll_ob_to_in
                    hl_ob_to_in_row_sum = (synapse_strength[post_layer][pre_layer][excitatory_unit_num:unit_num,hl_ob_begin:hl_ob_begin+hl_ob_num].sum(axis=1).reshape(iin_num,1).repeat(hl_ob_num, axis=1))/Z_hl_ob_to_in
                    ll_ac_to_in_row_sum = (synapse_strength[post_layer][pre_layer][excitatory_unit_num:unit_num,ll_ac_begin:ll_ac_begin+ll_ac_num].sum(axis=1).reshape(iin_num,1).repeat(ll_ac_num, axis=1))/Z_ll_ac_to_in
                    ml_ac_to_in_row_sum = (synapse_strength[post_layer][pre_layer][excitatory_unit_num:unit_num,ml_ac_begin:ml_ac_begin+ml_ac_num].sum(axis=1).reshape(iin_num,1).repeat(ml_ac_num, axis=1))/Z_ml_ac_to_in
                    tl_ac_to_in_row_sum = (synapse_strength[post_layer][pre_layer][excitatory_unit_num:unit_num,tl_ac_begin:tl_ac_begin+tl_ac_num].sum(axis=1).reshape(iin_num,1).repeat(tl_ac_num, axis=1))/Z_tl_ac_to_in
                    in_from_ex_row_sum = np.concatenate((ll_ob_to_in_row_sum,hl_ob_to_in_row_sum,ll_ac_to_in_row_sum,ml_ac_to_in_row_sum,tl_ac_to_in_row_sum),axis=1)
                    
                    excitatory_row_sums = np.concatenate((ll_ob_row_sum,hl_ob_row_sum,ll_ac_row_sum,ml_ac_row_sum,tl_ac_row_sum,in_from_ex_row_sum),axis=0)
                    
                    # Make sure inhibitory weights in inter node are all inhibitory
                    synapse_strength[post_layer][pre_layer][:,excitatory_unit_num:][synapse_strength[post_layer][pre_layer][:,excitatory_unit_num:] > 0] = 0
                    # Normalize incoming inhibitory weights to each unit
                    if IIN_PROJECTIONS:
                        ''' Inhibitory neurons innervate excitatory neurons from the same layer,
                        and inhibitory neurons from all layers'''
                        normalizing_factor_for_ex_neurons = self.Z_in
                        normalizing_factor_for_in_neurons = self.Z_in/layer_num
                        in_to_ex_row_sums = (-1)*((synapse_strength[post_layer][pre_layer][:excitatory_unit_num,excitatory_unit_num:].sum(axis=1).reshape(excitatory_unit_num,1).repeat(iin_num, axis=1))/normalizing_factor_for_ex_neurons)
                        in_to_in_row_sums = (-1)*((synapse_strength[post_layer][pre_layer][excitatory_unit_num:,excitatory_unit_num:].sum(axis=1).reshape(iin_num,1).repeat(iin_num, axis=1))/normalizing_factor_for_in_neurons)
                        inhibitory_row_sums = np.concatenate((in_to_ex_row_sums, in_to_in_row_sums))
                    else:
                        normalizing_factor = self.Z_in
                        inhibitory_row_sums = (-1)*((synapse_strength[post_layer][pre_layer][:,excitatory_unit_num:].sum(axis=1).reshape(unit_num,1).repeat(iin_num, axis=1))/normalizing_factor)
                    
                    row_sums = np.concatenate((excitatory_row_sums,inhibitory_row_sums),axis=1)
                    row_sums[row_sums == 0] = 1
                    synapse_strength[post_layer][pre_layer] = synapse_strength[post_layer][pre_layer]/row_sums
                elif IIN_PROJECTIONS:
                    # Inhibitory neurons are connected through different layers
                    # Make sure weights are all inhibitory
                    synapse_strength[post_layer][pre_layer][excitatory_unit_num:,excitatory_unit_num:][synapse_strength[post_layer][pre_layer][excitatory_unit_num:,excitatory_unit_num:] > 0] = 0
                    # Normalize incoming inhibitory weights to each unit
                    normalizing_factor = self.Z_in / layer_num
                    inhibitory_row_sums = (-1)*((synapse_strength[post_layer][pre_layer][excitatory_unit_num:,excitatory_unit_num:].sum(axis=1).reshape(iin_num,1).repeat(iin_num, axis=1))/normalizing_factor)
                    row_sums = np.pad(inhibitory_row_sums,((excitatory_unit_num,0),(excitatory_unit_num,0)), 'constant')
                    row_sums[row_sums == 0] = 1
                    synapse_strength[post_layer][pre_layer] = synapse_strength[post_layer][pre_layer]/row_sums
                if post_layer == layer_num-1 and pre_layer == 0:
                    normalized_weight = Z_sensory_response
                    # Sensory neurons directly innervate response neurons
                    for unit_ind in range(unit_num):
                        synapse_strength[post_layer][pre_layer][unit_ind,unit_ind] = normalized_weight
                if post_layer == layer_num-2 and pre_layer == layer_num-1:
                    # High-level action response neurons innervate high-level object prediction neurons
                    hl_ob_begin = ll_ob_num
                    hl_ac_begin = ll_ob_num+hl_ob_num+ll_ac_num
                    synapse_strength[post_layer][pre_layer][hl_ob_begin:hl_ob_begin+hl_ob_num,hl_ac_begin:hl_ac_begin+hl_ac_num][synapse_strength[post_layer][pre_layer][hl_ob_begin:hl_ob_begin+hl_ob_num,hl_ac_begin:hl_ac_begin+hl_ac_num] < 0] = 0
                    
                    # Normalize incoming weights to each unit
                    normalizing_factor = Z_response_prediction
                    row_sums = (synapse_strength[post_layer][pre_layer][hl_ob_begin:hl_ob_begin+hl_ob_num,hl_ac_begin:hl_ac_begin+hl_ac_num].sum(axis=1).reshape(hl_ob_num,1).repeat(hl_ac_num, axis=1))/normalizing_factor
                    row_sums = np.pad(row_sums, ((hl_ob_begin,unit_num-hl_ob_begin-hl_ob_num),(hl_ac_begin,unit_num-hl_ac_begin-hl_ac_num)), 'constant')
                    row_sums[row_sums == 0] = 1
                    synapse_strength[post_layer][pre_layer] = synapse_strength[post_layer][pre_layer]/row_sums
                    
                # Top level actions doesn't get innervated by the cortex
                '''top_level_action_begin = ll_ob_num+hl_ob_num+ll_ac_num+ml_ac_num
                synapse_strength[post_layer][pre_layer][top_level_action_begin:top_level_action_begin+tl_ac_num,:] = 0'''
        
    def calculate_winners(self, input_vec, begin_ind, node_num,stop_when_resolved=True):
        fire_history = []
        
        # Given an input, simulate the dynamics of the system, for iter_num time steps
        for _ in range(layer_num):
            cur_history = []
            for _ in range(unit_num):
                cur_history.append([])
            fire_history.append(cur_history)
        
        synched_iins = []
        winner_loser_list = [-1] * node_num
            
        t = 0
        while True:
            if not self.quiet and t % 1000 == 0:
                print('t='+str(t))
            
            self.prop_external_input(input_vec)
            for l in range(layer_num):
                for unit_ind in range(unit_num):
                    if prev_act[l][unit_ind, 0] == 1:
                        fire_history[l][unit_ind].append(t)
                 
            if len(synched_iins) == 0:       
                for first_iin in range(unit_num-iin_num, unit_num):
                    for second_iin in range(first_iin+1, unit_num):
                        # Check if iins where synched in the last sync window
                        iin_sync_window = 20
                        iin_sync_threshold = 15
                        
                        if len(fire_history[response_layer][first_iin]) < iin_sync_window or len(fire_history[response_layer][second_iin]) < iin_sync_window:
                            # Not enough firing- still can't determine synchronization
                            continue
                        intersection_of_last_window = [x for x in fire_history[response_layer][first_iin][-iin_sync_window:] if x in fire_history[response_layer][second_iin][-iin_sync_window:]]
                        if len(intersection_of_last_window) >= iin_sync_threshold:
                            synched_iins = [first_iin, second_iin]
                            break
                    if len(synched_iins) > 0:
                        sync_time_step = t
                        if not self.quiet:
                            print('sync time step: ' + str(t))
                        break
            elif t > sync_time_step+2000:
                competition_resolved = True
                
                iin_firing = set().union(*([[x for x in fire_history[response_layer][iin_ind] if x > t-self.ex_sync_window*10] for iin_ind in range(unit_num-iin_num,unit_num)]))
                for unit_ind in range(begin_ind, begin_ind+node_num):
                    winner_loser_ind = unit_ind - begin_ind
                    if winner_loser_list[winner_loser_ind] != -1:
                        continue # resolved
                    ex_intersection_of_last_window = [x for x in fire_history[response_layer][unit_ind][-self.ex_sync_window:] if x in iin_firing]
                    #ex_intersection_of_last_window = [x for x in fire_history[response_layer][unit_ind][-ex_sync_window:] if (x in iin_firing or (x+1) in iin_firing)]
                    if len(ex_intersection_of_last_window) >= self.ex_sync_threshold:
                        winner_loser_list[winner_loser_ind] = 1 # winner
                    elif len(ex_intersection_of_last_window) <= self.ex_unsync_threshold:
                        winner_loser_list[winner_loser_ind] = 0 # lower
                    else:
                        competition_resolved = False # unresolved yet
                        
                if stop_when_resolved and competition_resolved:
                    break
                
            t += 1
            if not stop_when_resolved and t == 50000:
                break
        
        if not self.quiet:
            print('winner list firing: ' + str([len(a) for a in fire_history[response_layer][begin_ind:begin_ind+node_num]]))
        
        return winner_loser_list, t, fire_history
        
    def convert_winner_list_to_action(self, winner_list):
        side_size = int(ll_ac_num / 2)
        
        right_count = np.sum(winner_list[:side_size])
        left_count = np.sum(winner_list[side_size:])
        
        action = (right_count - left_count) % ll_ac_num
        return action
    
    def update_synapse_strength_long_term(self, winning_action_list, prev_input, before_prev_input, comp_len):
        
        global synapse_strength
        
        zeta = comp_len / comp_len_zeta_ratio
        delta_input_strength = np.sum(prev_input-before_prev_input)
        max_possible_input_strength = sensory_input_strength * ll_ob_num
        normalized_delta_input_strength = delta_input_strength / max_possible_input_strength
        total_strength_change = normalized_delta_input_strength * zeta
        before_prev_input_strength = np.sum(before_prev_input)
        strength_change_vec = before_prev_input * (total_strength_change / before_prev_input_strength)
        strength_change_vec = np.pad(strength_change_vec, ((0,unit_num-ll_ob_num),(0,0)), 'constant')
        
        ll_action_begin = ll_ob_num + hl_ob_num
        
        for layer_ind in range(layer_num):
            for node_ind in range(len(winning_action_list)):
                winner_loser_factor = (winning_action_list[node_ind]-0.5)*2
                real_node_ind = node_ind+ll_action_begin
                synapse_strength[layer_ind][layer_ind][[real_node_ind],:] += winner_loser_factor*strength_change_vec.transpose()
                
        self.fix_synapse_strength()
                
    def update_synapse_strength_short_term(self):
        # Update the synapses strength according the a Hebbian learning rule
        global synapse_strength
        
        for post_layer in range(layer_num):
            for pre_layer in range(layer_num):
                post_layer_prev_act = prev_act[post_layer]
                normalizing_excitatory_vec = np.ones((unit_num-iin_num,1)) * gamma_ex
                normalizing_inhibitory_vec = np.ones((iin_num,1)) * gamma_in
                normalizing_vec = np.concatenate((normalizing_excitatory_vec, normalizing_inhibitory_vec))
                normalized_post_layer_prev_act = post_layer_prev_act - normalizing_vec
                
                pre_layer_before_prev_act = before_prev_act[pre_layer]
                
                update_mat = np.matmul(normalized_post_layer_prev_act, pre_layer_before_prev_act.transpose())
                # Strengthen inhibitory neurons weights by making them more negative (and not more positive)
                update_mat[:,-iin_num:] = (-1) * update_mat[:,-iin_num:]
                    
                synapse_strength[post_layer][pre_layer] = synapse_strength[post_layer][pre_layer] + eta * update_mat
        
        self.fix_synapse_strength()
        
    def prop_external_input(self, sensory_input_vec):
        # Simulate the dynamics of the system for a single time step
        global before_prev_act
        global prev_act
        global prev_input
        
        new_act = []
        new_input = []
        
        for post_layer in range(layer_num):
            cur_input = np.zeros((unit_num, 1))
            if post_layer == 0:
                input_from_prev_layer = np.pad(sensory_input_vec, ((0,unit_num-ll_ob_num),(0,0)), 'constant')
                cur_input = np.add(cur_input, input_from_prev_layer)
            for pre_layer in range(layer_num):
                input_from_pre_layer = np.matmul(synapse_strength[post_layer][pre_layer], prev_act[pre_layer])
                cur_input = np.add(cur_input, input_from_pre_layer)
            
            '''if post_layer == layer_num-1:
                # The response neuron of the top level action is always innervated
                top_level_action_begin = ll_ob_num+hl_ob_num+ll_ac_num+ml_ac_num
                cur_input[top_level_action_begin+active_tl_ac_node,0] = sensory_input_strength'''
                
            ''' Accumulating input and refractory period: If a neuron fired in the last time step,
            we subtract its previous input from its current input. Otherwise- we add its previous
            input to its current input. '''
            prev_input_factor = (1 - 2 * prev_act[post_layer])
            cur_input = np.add(cur_input, prev_input_factor * prev_input[post_layer])
            
            # Make sure the input is non-negative
            cur_input = np.where(cur_input>=0, cur_input, 0)
            
            cur_act = np.concatenate((self.excitatory_activation_function(cur_input[:cur_input.shape[0]-iin_num,[0]]),
                                  self.inhibitory_activation_function(cur_input[cur_input.shape[0]-iin_num:,[0]])),
                                  axis=0)
            new_act.append(cur_act)
            new_input.append(cur_input)
        
        before_prev_act = prev_act
        prev_act = new_act
        prev_input = new_input
        
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
        return 0 + (x >= excitatory_threshold)
    
    def inhibitory_activation_function(self, x):
        # Linear activation function for inhibitory neurons
        return 0 + (x >= inhibitory_threshold)

#######################
# Simulator functions #
#######################

# Parameters
OBST = 0
nose_length = 2
world_height = 50
world_length = 50

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

def generate_state_from_simulator(world, cur_player, goals):
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
    
    state_vector *= sensory_input_strength
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

#############################
# Parameter extraction tool #
#############################

class AnalysisCode(enum.Enum):
    NORMAL = 0
    ONLY_LOSERS = 1 # Need to increase sync_window and Z_ob_to_ac, and decrease sync_threshold
    ONLY_WINNERS = 2 # Need to increase Z_in, and decrease Z_ob_to_ac
    MIX = 3 # Decrease threshold-window-ratio
    NO_JUMP = 4 # Need to increase Z_in
    LOSERS_AFTER_JUMP = 5 # Decrease threshold-window-ratio
    WINNERS_BEFORE_JUMP = 6 # Need to increase sync_threshold
    MULTIPLE_JUMP = 7 # Need to increase sync_threshold

jump_factor = 2
jump_diff = 10
jump_compare_factor = 2
non_dichotomized_codes = [AnalysisCode.MIX,
                          AnalysisCode.NO_JUMP,
                          AnalysisCode.LOSERS_AFTER_JUMP,
                          AnalysisCode.WINNERS_BEFORE_JUMP,
                          AnalysisCode.MULTIPLE_JUMP]

def analyze_competition(winner_list, competitiors_fire_history):
    competitiors_fire_count = [len(x) for x in competitiors_fire_history]
    sorted_indices = sorted(range(len(competitiors_fire_count)), key=lambda k: competitiors_fire_count[k])
    finished_losers = (winner_list[sorted_indices[0]] == 1)
    
    # First pass- make sure that there's no mixture between winners and losers
    for i in range(len(sorted_indices)):
        if finished_losers and winner_list[sorted_indices[i]] == 0:
            return AnalysisCode.MIX
        if (not finished_losers) and winner_list[sorted_indices[i]] == 1:
            finished_losers = True
    
    # Handle edge cases: only losers and only winners
    if winner_list[sorted_indices[-1]] == 0:
        return AnalysisCode.ONLY_LOSERS
    elif winner_list[sorted_indices[0]] == 1:
        return AnalysisCode.ONLY_WINNERS
    
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
        return AnalysisCode.NO_JUMP
    if len(jumps) > 1:
        if jumps[-1][1] < jump_compare_factor*jumps[-2][1]:
            return AnalysisCode.MULTIPLE_JUMP
    jump_index = jumps[-1][0]
    
    # Make sure the biggest jump is in the right location
    if winner_list[sorted_indices[jump_index]] == 1:
        return AnalysisCode.WINNERS_BEFORE_JUMP
    if winner_list[sorted_indices[jump_index+1]] == 0:
        return AnalysisCode.LOSERS_AFTER_JUMP
    
    return AnalysisCode.NORMAL

def extract_parameters(configuration):
    print('Starting parameters extraction tool...')
    
    configuration['quiet'] = True
    configuration['Z_in'] = (iin_num/unit_num) * excitatory_threshold * 11
    configuration['ex_sync_window'] = 80
    configuration['ex_sync_threshold'] = 8
    configuration['ex_unsync_threshold'] = 1
    configuration['Z_ll_ob_to_ll_ac'] = 0.3 * Z_ex
    
    only_losers_count = 0
    only_winners_count = 0
    normal_count = 0
    dichotomized_count = 0
    
    max_iter_num = 100
    for i in range(max_iter_num):
        print('Starting iteration #' + str(i))
        
        starting_loc = (0,0)
        while starting_loc == (0,0):
            starting_loc = (int(np.random.rand() * world_height), int(np.random.rand() * world_length))
        world = init_world(world_height, world_length, [((0,0),1)])
        initial_player = (starting_loc[0],starting_loc[1],0)
        goals = [(24,25)]
        action_begin_loc = ll_ob_num+hl_ob_num
        
        model = ModelClass(configuration)
        model.initialize(False)
        
        input_vec = generate_state_from_simulator(world, initial_player, goals)
        winner_list, _, fire_history = model.calculate_winners(input_vec, action_begin_loc, ll_ac_num,True)
        output_code = analyze_competition(winner_list, fire_history[response_layer][action_begin_loc:action_begin_loc+ll_ac_num])
        
        print('\tOutput code: ' + str(output_code))
        
        if output_code in non_dichotomized_codes:
            only_losers_count = 0
            only_winners_count = 0
            normal_count = 0
            dichotomized_count = 0
            if output_code == AnalysisCode.MIX:
                # Decrease threshold-window-ratio
                configuration['ex_sync_threshold'] = round(0.9 * configuration['ex_sync_threshold'])
                configuration['ex_sync_window'] = round(1.1 * configuration['ex_sync_window'])
            elif output_code == AnalysisCode.NO_JUMP:
                # Need to increase Z_in
                configuration['Z_in'] = 1.1 * configuration['Z_in']
            elif output_code == AnalysisCode.LOSERS_AFTER_JUMP:
                # Decrease threshold-window-ratio
                configuration['ex_sync_threshold'] = round(0.9 * configuration['ex_sync_threshold'])
                configuration['ex_sync_window'] = round(1.1 * configuration['ex_sync_window'])
            elif output_code == AnalysisCode.WINNERS_BEFORE_JUMP:
                # Need to increase sync_threshold
                configuration['ex_sync_threshold'] = round(1.1 * configuration['ex_sync_threshold'])
            elif output_code == AnalysisCode.MULTIPLE_JUMP:
                # Need to increase sync_threshold
                configuration['ex_sync_threshold'] = round(1.1 * configuration['ex_sync_threshold'])
        else: # We got a dichotomized code
            dichotomized_count += 1
            if output_code == AnalysisCode.ONLY_LOSERS:
                only_losers_count += 1
            elif output_code == AnalysisCode.ONLY_WINNERS:
                only_winners_count += 1
            elif output_code == AnalysisCode.NORMAL:
                normal_count += 1
        
        ''' If we had a sequence of 5 dichotomized codes in a row("a window"), that means we're
        in a good region.
        We still need to make sure we don't get too much only losers/only winners.
        We define "too much" to be more than 2 in the previous window.
        We also demand that there will be at least 3 normal codes in the previous window. ''' 
        if dichotomized_count == 5:
            if only_losers_count > 2:
                # Need to increase sync_window and Z_ob_to_ac, and decrease sync_threshold
                configuration['ex_sync_threshold'] = round(0.9 * configuration['ex_sync_threshold'])
                configuration['ex_sync_window'] = round(1.1 * configuration['ex_sync_window'])
                configuration['Z_ll_ob_to_ll_ac'] = 1.1 * configuration['Z_ll_ob_to_ll_ac']
            elif only_winners_count > 2:
                # Need to increase Z_in, and decrease Z_ob_to_ac
                configuration['Z_in'] = 1.1 * configuration['Z_in']
                configuration['Z_ll_ob_to_ll_ac'] = 0.9 * configuration['Z_ll_ob_to_ll_ac']
            elif normal_count > 2:
                break
            only_losers_count = 0
            only_winners_count = 0
            normal_count = 0
            dichotomized_count = 0
            
    if i == max_iter_num:
        print('Unable to extract parameters')
        assert(False)
    
    print('Extracted parameters successfully!')

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
    goals = [(24,25)]
    
    configuration['Z_in'] = (iin_num/unit_num) * excitatory_threshold * 11
    
    model = ModelClass(configuration)
    model.initialize(load_from_file)
    
    cur_player = initial_player
    if not quiet:
        print(cur_player)
        
    shortest_path_to_target_len = get_shortest_path_to_target_len(cur_player, goals[0])
    max_rounds = 5 * shortest_path_to_target_len
    
    action_begin_loc = ll_ob_num+hl_ob_num
    
    prev_input_vec = None
    input_vec = None
    winner_list = None
    comp_len = 0
    
    for i in range(max_rounds):
        prev_input_vec = input_vec
        input_vec = generate_state_from_simulator(world, cur_player, goals)
        if not quiet:
            print('input vec: ' + str((input_vec.transpose() > 0).astype(int)))
        if i > 0:
            model.update_synapse_strength_long_term(winner_list, input_vec, prev_input_vec, comp_len)
        
        winner_list, comp_len, _ = model.calculate_winners(input_vec, action_begin_loc, ll_ac_num)
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
            prev_input_vec = input_vec
            input_vec = generate_state_from_simulator(world, cur_player, goals)
            if not quiet:
                print('input vec: ' + str((input_vec.transpose() > 0).astype(int)))
            model.update_synapse_strength_long_term(winner_list, input_vec, prev_input_vec, comp_len)
            break
    
    model.save_synapse_strength()
    score = evaluate(world, goals, cur_player, i+1, shortest_path_to_target_len, max_rounds)
    return score
    
'''scores = []
epoch_num = 60
for epoch in range(epoch_num):
    if not quiet:
        print('***')
    print('epoch ' + str(epoch))
    if not quiet:
        print('***')
    
    if epoch == 0:
        load_from_file = False
    else:
        load_from_file = True
    score = main(load_from_file, None)
    scores.append(score)
    
plt.plot(range(epoch_num), scores)
plt.savefig('res')'''

configuration = {}
extract_parameters(configuration)
print(configuration)