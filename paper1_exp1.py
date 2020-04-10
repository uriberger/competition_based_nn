import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

''' Paper 1, Experiment 1:
The purpose of this experiment is a sanity check- we want to show that when the network is given
different objects, single response neurons are activated.
'''

# General parameters
layers_size = [100]
iin_num = len(layers_size)
unit_num = sum(layers_size) + iin_num

class ModelClass:
    
    default_configuration = {
        # Neuron parameters
        'excitatory_threshold' : 1,
        'inhibitory_threshold' : 1,
        
        # Winner num
        'winner_num' : [11],
        
        # Normalization parameters
        'Z_iin_ex_th_ratio' : 1,
        'Z_intra_layer_ex_th_ratio' : 1,
        }
    
    def __init__(self, configuration, load_from_file, quiet):
        self.conf = {key : configuration[key] if key in configuration else ModelClass.default_configuration[key] for key in ModelClass.default_configuration.keys()}        
        self.quiet = quiet
        self.init_normalization_parameters()
        self.init_data_structures(load_from_file)
        
    def my_print(self, my_str):
        if not self.quiet:
            print(my_str)

    def save_synapse_strength(self, file_suffix):
        # Save the synapse strength matrix to a file.
        trained_strength = [self.synapse_strength]
        file_name = "synapse_strength"
        file_name += "_" + file_suffix
        np.save(file_name, trained_strength)
    
    def init_data_structures(self, file_suffix):
        ''' Initialize the data structures.
        load_from_file- if different from 'None', holds the suffix of the file to be
        loaded. '''
        
        if file_suffix == None:
            self.synapse_strength = np.random.rand(unit_num, unit_num)
            self.synapse_strength[:, unit_num-iin_num:] = (-1) * self.synapse_strength[:, unit_num-iin_num:]            
        else:
            file_name = "synapse_strength"
            file_name += "_" + file_suffix
            file_name += ".npy"
            
            trained_strength = np.load(file_name)
        
            self.synapse_strength = trained_strength[0]
            
        self.prev_act = np.zeros((unit_num, 1))
        self.prev_input_to_neurons = np.zeros((unit_num, 1))
        
        self.zero_matrix = np.ones((unit_num,unit_num))
        
        # Input to 1st layer
        '''Consider the N^2 input neurons as an NXN grid. Each neuron is connected only to its
        nearest neighbors. '''
        self.zero_matrix[:layers_size[0],:layers_size[0]] = 0
        N = round(layers_size[0]**0.5)
        for unit_ind in range(layers_size[0]):
            if unit_ind % N > 0:
                self.zero_matrix[unit_ind,unit_ind-1] = 1
            if unit_ind % N < N-1:
                self.zero_matrix[unit_ind,unit_ind+1] = 1
            if unit_ind >= N:
                self.zero_matrix[unit_ind,unit_ind-N] = 1
            if unit_ind < layers_size[0]-N:
                self.zero_matrix[unit_ind,unit_ind+N] = 1
        self.zero_matrix[:layers_size[0],layers_size[0]:sum(layers_size)] = 0
        
        # Input to 1st IIN
        self.zero_matrix[sum(layers_size),layers_size[0]:sum(layers_size)] = 0
        
        # Input from 1st IIN
        self.zero_matrix[layers_size[0]:,sum(layers_size)] = 0
        
        # Input to other layers
        for l in range(1,len(layers_size)):
            # Input to excitatory neurons
            self.zero_matrix[sum(layers_size[:l]):sum(layers_size[:l+1]),:sum(layers_size[:l-1])] = 0
            self.zero_matrix[sum(layers_size[:l]):sum(layers_size[:l+1]),sum(layers_size[:l]):sum(layers_size)] = 0
            
            # Input to IIN
            self.zero_matrix[sum(layers_size)+l,:sum(layers_size[:l])] = 0
            self.zero_matrix[sum(layers_size)+l,sum(layers_size[:l+1]):sum(layers_size)] = 0
            
            # Input from IIN
            self.zero_matrix[:sum(layers_size[:l]),sum(layers_size)+l] = 0
            self.zero_matrix[sum(layers_size[:l+1]):sum(layers_size),sum(layers_size)+l] = 0
            self.zero_matrix[sum(layers_size):,sum(layers_size)+l] = 0
        
        # Don't allow self loops
        np.fill_diagonal(self.zero_matrix,0)
        
        self.fix_synapse_strength()
            
    def init_normalization_parameters(self):
        # Initialize normalization parameters
        self.conf['Z_iin'] = self.conf['Z_iin_ex_th_ratio'] * self.conf['excitatory_threshold']
        self.conf['Z_intra_layer'] = self.conf['Z_intra_layer_ex_th_ratio'] * self.conf['excitatory_threshold']

        self.conf['Z_vals'] = []
        for l in range(1,len(layers_size)):
            self.conf['Z_vals'].append((self.conf['excitatory_threshold'] * layers_size[l-1])/(10*self.conf['winner_num'][l-1]))
        
        self.conf['Z_ex_to_iins'] = []
        for l in range(len(layers_size)):
            self.conf['Z_ex_to_iins'].append((self.conf['excitatory_threshold'] * layers_size[l])/(self.conf['winner_num'][l]))
        
    def reset_data_structures(self):
        self.prev_act = np.zeros((unit_num, 1))
        self.prev_input_to_neurons = np.zeros((unit_num, 1))
    
    def fix_synapse_strength(self):
        # Normalize the synapses strength, and enforce the invariants.
        excitatory_unit_num = unit_num-iin_num
        
        # For symmetry, make sure IINs receive equal input from all neurons
        self.synapse_strength[excitatory_unit_num:,:excitatory_unit_num] = 1
        
        # Also make sure intra-layer connections for input neurons are all of equal strength
        self.synapse_strength[:layers_size[0],:layers_size[0]] = 1
        
        # Enforce invariants
        self.synapse_strength = np.multiply(self.synapse_strength,self.zero_matrix)
        
        # Make sure excitatory weights are all excitatory
        self.synapse_strength[:,:excitatory_unit_num][self.synapse_strength[:,:excitatory_unit_num] < 0] = 0
        
        # Normalize incoming excitatory weights to each unit    
        
        # Input to 1st layer
        input_to_input_row_sum = (self.synapse_strength[:layers_size[0],:layers_size[0]].sum(axis=1).reshape(layers_size[0],1).repeat(layers_size[0], axis=1))/self.conf['Z_intra_layer']
        rest_to_input_row_sum = np.ones((layers_size[0],sum(layers_size)-layers_size[0]))
        input_row_sum = np.concatenate((input_to_input_row_sum,rest_to_input_row_sum),axis=1)
        
        excitatory_row_sums = input_row_sum
        
        for l in range(1,len(layers_size)):
            # Input to EINs
            input_from_earlier_layers = np.ones((layers_size[l],sum(layers_size[:l-1])))
            input_from_prev_layer = (self.synapse_strength[sum(layers_size[:l]):sum(layers_size[:l+1]),sum(layers_size[:l-1]):sum(layers_size[:l])].sum(axis=1).reshape(layers_size[l],1).repeat(layers_size[l-1], axis=1))/self.conf['Z_vals'][l-1]
            input_from_later_layers = np.ones((layers_size[l],sum(layers_size[l:])))
            row_sum = np.concatenate((input_from_earlier_layers,input_from_prev_layer,input_from_later_layers),axis=1)
            excitatory_row_sums = np.concatenate((excitatory_row_sums,row_sum),axis=0)
        
        # Input to IINs
        for l in range(len(layers_size)):
            input_from_earlier_layers = np.ones((1,sum(layers_size[:l])))
            input_from_layer = (self.synapse_strength[sum(layers_size)+l,sum(layers_size[:l]):sum(layers_size[:l+1])].sum().repeat(layers_size[l]).reshape(1,layers_size[l]))/self.conf['Z_ex_to_iins'][l]
            input_from_later_layers = np.ones((1,sum(layers_size[l+1:])))
            row_sum = np.concatenate((input_from_earlier_layers,input_from_layer,input_from_later_layers),axis=1)
            excitatory_row_sums = np.concatenate((excitatory_row_sums,row_sum),axis=0)
        
        # Make sure inhibitory weights are all inhibitory
        self.synapse_strength[:,excitatory_unit_num:][self.synapse_strength[:,excitatory_unit_num:] > 0] = 0
        # Normalize incoming inhibitory weights to each unit
        inhibitory_row_sums = (-1)*((self.synapse_strength[:,excitatory_unit_num:].sum(axis=1).reshape(unit_num,1).repeat(iin_num, axis=1))/self.conf['Z_iin'])
        
        row_sums = np.concatenate((excitatory_row_sums,inhibitory_row_sums),axis=1)
        row_sums[row_sums == 0] = 1
        self.synapse_strength = self.synapse_strength/row_sums
        
    def simulate_dynamics(self, input_vec):
        fire_history = []
        for _ in range(unit_num):
            fire_history.append([])
        
        #for t in range(T):
        t = 0
        while True:
            if t % 1000 == 0:
                self.my_print('t='+str(t))
                
            # Propagate external input
            self.prop_external_input(input_vec)
                
            # Document fire history
            for unit_ind in range(unit_num):
                if self.prev_act[unit_ind, 0] == 1:
                    fire_history[unit_ind].append(t)
                    
            t += 1
            
            all_inn_fired = True
            for l in range(len(layers_size)):
                if len(fire_history[sum(layers_size)+l]) == 0:
                    all_inn_fired = False
                    break
            if all_inn_fired:
                break
        
        self.reset_data_structures()
        
        return fire_history
        
    def prop_external_input(self, sensory_input_vec):
        # Simulate the dynamics of the system for a single time step
        cur_input = np.zeros((unit_num, 1))
        input_from_prev_layer = np.pad(sensory_input_vec, ((0,unit_num-layers_size[0]),(0,0)), 'constant')
        cur_input = np.add(cur_input, input_from_prev_layer)
        input_from_pre_layer = np.matmul(self.synapse_strength, self.prev_act)
        cur_input = np.add(cur_input, input_from_pre_layer)
        
        # Accumulating input
        cur_input = np.add(cur_input, self.prev_input_to_neurons)
        
        # Input reset for neurons that fired
        cur_input[self.prev_act == 1] = 0
        
        # Make sure the input is non-negative
        cur_input = np.where(cur_input>=0, cur_input, 0)
        
        cur_act = np.concatenate((self.excitatory_activation_function(cur_input[:unit_num-iin_num,[0]]),
                              self.inhibitory_activation_function(cur_input[unit_num-iin_num:,[0]])),
                              axis=0)
        
        self.prev_act = deepcopy(cur_act)
        self.prev_input_to_neurons = deepcopy(cur_input)
        
    def excitatory_activation_function(self, x):
        # Linear activation function for excitatory neurons 
        return 0 + (x >= self.conf['excitatory_threshold'])
    
    def inhibitory_activation_function(self, x):
        # Linear activation function for inhibitory neurons
        return 0 + (x >= self.conf['inhibitory_threshold'])
                
model = ModelClass({},None,True)
input_vec = None
fire_history = model.simulate_dynamics(input_vec)
fire_count = [len(a) for a in fire_history]
