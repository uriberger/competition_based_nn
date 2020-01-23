import numpy as np
import math
import matplotlib.pyplot as plt
from copy import deepcopy

###################
# Model functions #
###################

# General parameters
input_size = 8
output_size = 8
excitatory_precentage = 0.8
iin_num = math.ceil((input_size+output_size) * ((1-excitatory_precentage)/excitatory_precentage))
unit_num = input_size + output_size + iin_num
competition_len = 10000

class ModelClass:
    
    default_configuration = {
        # Neuron parameters
        'excitatory_threshold' : 0.4,
        'inhibitory_threshold' : 0.2,
        'sensory_input_strength' : 0.13333,
        
        # Normalization parameter
        'Z_ex_ex_th_ratio' : 0.5,
        'Z_in_ex_th_ratio' : 2.5,
        'Z_inp_Z_ex_ratio' : 0.5,
        
        # Learning parameters
        'eta' : 0.0001,
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
        file_name += "_" + str(unit_num) + "_neurons"
        np.save(file_name, trained_strength)
    
    def init_data_structures(self, load_from_file):
        ''' Initialize the data structures.
        load_from_file- if different from 'None', holds the suffix of the file to be
        loaded. '''
        if not load_from_file:
            # Initialize random synapse strength
            self.synapse_strength = np.random.rand(unit_num, unit_num)
            self.synapse_strength[:, -iin_num:] = (-1) * self.synapse_strength[:, -iin_num:]
        else:
            file_name = "synapse_strength"
            file_name += "_" + str(unit_num) + "_neurons"
            file_name += ".npy"
            
            trained_strength = np.load(file_name)
        
            self.synapse_strength = trained_strength[0]
            
        self.prev_act = np.zeros((unit_num, 1))
        self.before_prev_act = np.zeros((unit_num, 1))
        self.prev_input_to_neurons = np.zeros((unit_num, 1))
        
        self.fix_synapse_strength()
    
    def init_normalization_parameters(self):
        # Initialize normalization parameters
        self.conf['Z_ex'] = self.conf['Z_ex_ex_th_ratio'] * self.conf['excitatory_threshold']
        self.conf['Z_iin'] = (1-excitatory_precentage)*self.conf['excitatory_threshold']*self.conf['Z_in_ex_th_ratio']
        self.conf['Z_inp'] = self.conf['Z_inp_Z_ex_ratio']*self.conf['Z_ex']
        self.conf['Z_out'] = self.conf['Z_ex'] - self.conf['Z_inp']
    
    def fix_synapse_strength(self):
        # Normalize the synapses strength, and enforce the invariants.
        excitatory_unit_num = unit_num-iin_num
        
        # Make sure excitatory weights in are all excitatory
        self.synapse_strength[:,:excitatory_unit_num][self.synapse_strength[:,:excitatory_unit_num] < 0] = 0
        
        # Normalize incoming excitatory weights to each unit    
        input_begin = 0
        output_begin = input_begin + input_size 
        
        input_to_input_row_sum = (self.synapse_strength[input_begin:input_begin+input_size,input_begin:input_begin+input_size].sum(axis=1).reshape(input_size,1).repeat(input_size, axis=1))/self.conf['Z_inp']
        output_to_input_row_sum = (self.synapse_strength[input_begin:input_begin+input_size,output_begin:output_begin+output_size].sum(axis=1).reshape(input_size,1).repeat(output_size, axis=1))/self.conf['Z_out']
        input_row_sum = np.concatenate((input_to_input_row_sum,output_to_input_row_sum),axis=1)
        
        input_to_output_row_sum = (self.synapse_strength[output_begin:output_begin+output_size,input_begin:input_begin+input_size].sum(axis=1).reshape(output_size,1).repeat(input_size, axis=1))/self.conf['Z_inp']
        output_to_output_row_sum = (self.synapse_strength[output_begin:output_begin+output_size,output_begin:output_begin+output_size].sum(axis=1).reshape(output_size,1).repeat(output_size, axis=1))/self.conf['Z_out']
        output_row_sum = np.concatenate((input_to_output_row_sum,output_to_output_row_sum),axis=1)
        
        input_to_iin_row_sum = (self.synapse_strength[-iin_num:,input_begin:input_begin+input_size].sum(axis=1).reshape(iin_num,1).repeat(input_size, axis=1))/self.conf['Z_inp']
        output_to_iin_row_sum = (self.synapse_strength[-iin_num:,output_begin:output_begin+output_size].sum(axis=1).reshape(iin_num,1).repeat(output_size, axis=1))/self.conf['Z_out']
        iin_from_ex_row_sum = np.concatenate((input_to_iin_row_sum,output_to_iin_row_sum),axis=1)
        
        excitatory_row_sums = np.concatenate((input_row_sum,output_row_sum,iin_from_ex_row_sum),axis=0)
        
        # Make sure inhibitory weights are all inhibitory
        self.synapse_strength[:,excitatory_unit_num:][self.synapse_strength[:,excitatory_unit_num:] > 0] = 0
        # Normalize incoming inhibitory weights to each unit
        inhibitory_row_sums = (-1)*((self.synapse_strength[:,excitatory_unit_num:].sum(axis=1).reshape(unit_num,1).repeat(iin_num, axis=1))/self.conf['Z_iin'])
        
        row_sums = np.concatenate((excitatory_row_sums,inhibitory_row_sums),axis=1)
        row_sums[row_sums == 0] = 1
        self.synapse_strength = self.synapse_strength/row_sums
        
    def simulate_dynamics(self, input_vec):
        # Given an input, simulate the dynamics of the system, for iter_num time steps
        fire_history = []
        for _ in range(unit_num):
            fire_history.append([])
        
        for t in range(competition_len):
            if t % 1000 == 0:
                self.my_print('t='+str(t))
            
            # Propagate external input
            self.prop_external_input(input_vec)
                
            # Document fire history
            for unit_ind in range(unit_num):
                if self.prev_act[unit_ind, 0] == 1:
                    fire_history[unit_ind].append(t)
        
        self.my_print('neurons firing: ' + str([len(a) for a in fire_history]))
        
        return fire_history
        
    def update_synapse_strength(self):
        # Update the synapses strength according the a Hebbian learning rule
        normalizing_excitatory_vec = np.ones((unit_num-iin_num,1)) * self.conf['gamma_ex']
        normalizing_inhibitory_vec = np.ones((iin_num,1)) * self.conf['gamma_in']
        normalizing_vec = np.concatenate((normalizing_excitatory_vec, normalizing_inhibitory_vec))
        normalized_prev_act = self.prev_act - normalizing_vec
        
        update_mat = np.matmul(normalized_prev_act, self.before_prev_act.transpose())
        # Strengthen inhibitory neurons weights by making them more negative (and not more positive)
        update_mat[:,-iin_num:] = (-1) * update_mat[:,-iin_num:]
            
        self.synapse_strength = self.synapse_strength + self.conf['eta'] * update_mat

        self.fix_synapse_strength()
        
    def prop_external_input(self, sensory_input_vec):
        # Simulate the dynamics of the system for a single time step
        external_input = np.pad(sensory_input_vec, ((0,unit_num-input_size),(0,0)), 'constant')
        internal_input = np.matmul(self.synapse_strength, self.prev_act)
        cur_input = np.add(external_input, internal_input)
        
        ''' Accumulating input and refractory period: If a neuron fired in the last time step,
        we subtract its previous input from its current input. Otherwise- we add its previous
        input to its current input. '''
        prev_input_factor = (1 - 2 * self.prev_act)
        cur_input = np.add(cur_input, prev_input_factor * self.prev_input_to_neurons)
        
        # Make sure the input is non-negative
        cur_input = np.where(cur_input>=0, cur_input, 0)

        cur_act = np.concatenate((self.excitatory_activation_function(cur_input[:cur_input.shape[0]-iin_num,[0]]),
                              self.inhibitory_activation_function(cur_input[cur_input.shape[0]-iin_num:,[0]])),
                              axis=0)
        
        self.before_prev_act = deepcopy(self.prev_act)
        self.prev_act = deepcopy(cur_act)
        self.prev_input_to_neurons = deepcopy(cur_input)
        
        self.update_synapse_strength()
        
    def excitatory_activation_function(self, x):
        # Linear activation function for excitatory neurons 
        return 0 + (x >= self.conf['excitatory_threshold'])
    
    def inhibitory_activation_function(self, x):
        # Linear activation function for inhibitory neurons
        return 0 + (x >= self.conf['inhibitory_threshold'])

def generate_random_sensory_input():
    sensory_input_vec = np.random.rand(input_size,1)
    return sensory_input_vec

#############
# Main code #
#############

quiet = False

configuration = {}
model = ModelClass(configuration,False,quiet)

sensory_input_vec = generate_random_sensory_input()
sensory_input_vec *= model.conf['sensory_input_strength']
if not quiet:
    print('sensory input vec: ' + str(sensory_input_vec.transpose()))

fire_history = model.simulate_dynamics(sensory_input_vec)