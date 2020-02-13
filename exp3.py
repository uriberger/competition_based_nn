import numpy as np
from copy import deepcopy

''' Experiment 3:
- Two populations: Excitatory input and inhibitory

Result: Cannot predict firing rate. The problem lies in the fact that we take the input to be
non negative, which gives benefits to neurons synchronized with iins.
'''

# General parameters
input_num = 8
iin_num = 3
unit_num = input_num+iin_num
ex_num = unit_num - iin_num

# Competition parameters
T = 10000

# Configuration
SIMPLE_REFRACTORY_PERIOD = True

class ModelClass:
    
    default_configuration = {
        # Neuron parameters
        'excitatory_threshold' : 0.4,
        'inhibitory_threshold' : 0.2,
        'mean_sensory_input' : 0.13333,
        
        # Normalization parameter
        'Z_ex_ex_th_ratio' : 0.5,
        'Z_iin_ex_th_ratio' : 1,
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
            self.synapse_strength[:, ex_num:] = (-1) * self.synapse_strength[:, ex_num:]
        else:
            file_name = "synapse_strength"
            file_name += "_" + str(unit_num) + "_neurons"
            file_name += ".npy"
            
            trained_strength = np.load(file_name)
        
            self.synapse_strength = trained_strength[0]
            
        self.prev_act = np.zeros((unit_num, 1))
        self.prev_input_to_neurons = np.zeros((unit_num, 1))
        
        self.fix_synapse_strength()
    
    def init_normalization_parameters(self):
        # Initialize normalization parameters
        self.conf['Z_ex'] = self.conf['Z_ex_ex_th_ratio'] * self.conf['excitatory_threshold']
        self.conf['Z_iin'] = self.conf['Z_iin_ex_th_ratio'] * self.conf['excitatory_threshold']
    
    def fix_synapse_strength(self):
        # Normalize the synapses strength, and enforce the invariants.
        input_begin = 0
                
        excitatory_row_sums = (self.synapse_strength[:,input_begin:input_begin+input_num].sum(axis=1).reshape(unit_num,1).repeat(input_num, axis=1))/self.conf['Z_ex']
        inhibitory_row_sums = (-1)*((self.synapse_strength[:,ex_num:].sum(axis=1).reshape(unit_num,1).repeat(iin_num, axis=1))/self.conf['Z_iin'])
                    
        row_sums = np.concatenate((excitatory_row_sums,inhibitory_row_sums),axis=1)
        
        row_sums[row_sums == 0] = 1
        self.synapse_strength = self.synapse_strength/row_sums
        
    def simulate_dynamics(self, input_vec):
        # Given an input, simulate the dynamics of the system, for T time steps
        fire_history = []
        for _ in range(unit_num):
            fire_history.append([])
            
        for t in range(T):
            if t % 1000 == 0:
                self.my_print('t='+str(t))
            
            # Propagate external input
            self.prop_external_input(input_vec)
                
            # Document fire history
            for unit_ind in range(unit_num):
                if self.prev_act[unit_ind, 0] == 1:
                    fire_history[unit_ind].append(t)
                        
        self.my_print('firing: ' + str([len(a) for a in fire_history]))
        
        return fire_history
        
    def prop_external_input(self, sensory_input_vec):
        # Simulate the dynamics of the system for a single time step
        cur_input = np.zeros((unit_num, 1))
        input_from_prev_layer = np.pad(sensory_input_vec, ((0,unit_num-input_num),(0,0)), 'constant')
        cur_input = np.add(cur_input, input_from_prev_layer)
        input_from_pre_layer = np.matmul(self.synapse_strength, self.prev_act)
        cur_input = np.add(cur_input, input_from_pre_layer)
        
        ''' Accumulating input and refractory period: If a neuron fired in the last time step,
        we subtract its previous input from its current input. Otherwise- we add its previous
        input to its current input. '''
        if SIMPLE_REFRACTORY_PERIOD:
            threshold_vec = np.ones((unit_num,1))*self.conf['excitatory_threshold']
            accumulating_input = self.prev_input_to_neurons - np.multiply(threshold_vec,self.prev_act)
            cur_input = np.add(cur_input, accumulating_input)
        else:
            prev_input_factor = (1 - 2 * self.prev_act)
            cur_input = np.add(cur_input, prev_input_factor * self.prev_input_to_neurons)
        
        # Make sure the input is non-negative
        cur_input = np.where(cur_input>=0, cur_input, 0)
        
        cur_act = np.concatenate((self.excitatory_activation_function(cur_input[:ex_num,[0]]),
                                  self.inhibitory_activation_function(cur_input[ex_num:,[0]])),
                                  axis=0)
        
        self.prev_act = deepcopy(cur_act)
        self.prev_input_to_neurons = deepcopy(cur_input)
        
    def excitatory_activation_function(self, x):
        # Linear activation function for excitatory neurons 
        return 0 + (x >= self.conf['excitatory_threshold'])
    
    def inhibitory_activation_function(self, x):
        # Linear activation function for inhibitory neurons
        return 0 + (x >= self.conf['inhibitory_threshold'])
    
    def generate_random_external_input(self):
        return np.random.rand(input_num,1)*(2*self.conf['mean_sensory_input'])

iter_num = 500
fire_rate = np.zeros((unit_num,1))
for cur_iter in range(iter_num):
    model = ModelClass({},False,True)
    if cur_iter % 100 == 0:
        print('Starting iter ' + str(cur_iter))
    input_vec = model.generate_random_external_input()
    fire_history = model.simulate_dynamics(input_vec)
    fire_count = np.array([len(a) for a in fire_history]).reshape((unit_num,1))
    fire_rate += fire_count / T
fire_rate /= iter_num

mu_s = model.conf['mean_sensory_input']
ex_th = model.conf['excitatory_threshold']
in_th = model.conf['inhibitory_threshold']
Z_ex = model.conf['Z_ex']
Z_iin = model.conf['Z_iin']

ana_mu_ex = (mu_s*(in_th+Z_iin))/(ex_th * in_th + Z_iin * ex_th - Z_ex * in_th)
nume_mu_ex = np.mean(fire_rate[:input_num])
ana_mu_iin = (mu_s*Z_ex)/(ex_th * in_th + Z_iin * ex_th - Z_ex * in_th)
nume_mu_iin = np.mean(fire_rate[input_num:])

print('Analytic mu_ex: ' + str(ana_mu_ex) + ', Analytic mu_iin: ' + str(ana_mu_iin))
print('Numeric mu_ex: ' + str(nume_mu_ex) + ', Numeric mu_iin: ' + str(nume_mu_iin))
print('Fire rate for different neurons: ' + str(fire_rate.transpose()))