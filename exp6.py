import numpy as np
from copy import deepcopy

''' Experiment 6:
- A population of input excitatory neurons.
- A population of excitatory inter-neurons.
- A population of output excitatory neurons.
- Each population has a single, very strong, inhibitor, that receives the same input as this
population.
- Information flow is input <-> ein <-> output.

Result:
The EIN layer is "kind" of synchronized with the output layer. It can be easily be seen in the IINs
firing times- the output IIN always fires at the same time step or one time step after the EIN's IIN.
The reason is that once the EINs are synched with their IIN, and they fire together, a large wave of
excitation arrives in the output layer, and it becomes likely that output neurons (or the output IIN)
will fire in the following time step.
In this case, in opposed to previous cases, we have 2 types of winners:
- Winners that charge faster than the IIN: and are only phased-synchronized with their IIN.
- Winners that charge slower than the IIN: and are completely synchronized with their IIN- they fire
every few times that the IIN fires.

In previous cases a neuron that charges slower than the IIN would almost always become a loser- but
now we have more complex dynamics, with top-down input.
'''

# Confiugration
RECURRENT_CONNECTIONS = False
FEEDBACK_CONNECTIONS = True

# General parameters
input_num = 8
ein_num = 8
output_num = 8
iin_num = 3

unit_num = input_num + ein_num + output_num + iin_num

# Competition parameters
T = 10000

class ModelClass:
    
    default_configuration = {
        # Neuron parameters
        'excitatory_threshold' : 0.4,
        'inhibitory_threshold' : 0.35,
        'mean_sensory_input' : 0.13333,
        
        # Normalization parameter
        'Z_ex_ex_th_ratio' : 0.5,
        'Z_iin_ex_th_ratio' : 1,
        
        'Z_inp_to_ein_percentage' : 0.5,
        'Z_out_to_ein_percentage' : 0.5,
        
        'Z_rec_Z_ex_ratio' : 0.5,
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
            self.synapse_strength[:, unit_num-iin_num:] = (-1) * self.synapse_strength[:, unit_num-iin_num:]
        else:
            file_name = "synapse_strength"
            file_name += "_" + str(unit_num) + "_neurons"
            file_name += ".npy"
            
            trained_strength = np.load(file_name)
        
            self.synapse_strength = trained_strength[0]
            
        self.prev_act = np.zeros((unit_num, 1))
        self.prev_input_to_neurons = np.zeros((unit_num, 1))
        
        input_begin = 0
        ein_begin = input_begin + input_num
        output_begin = ein_begin + ein_num
        iin_begin = output_begin + output_num
        self.zero_matrix = np.ones((unit_num,unit_num))
        
        if not RECURRENT_CONNECTIONS:
            self.zero_matrix[input_begin:input_begin+input_num,input_begin:input_begin+input_num] = 0
        if not FEEDBACK_CONNECTIONS:
            self.zero_matrix[input_begin:input_begin+input_num,ein_begin:ein_begin+ein_num] = 0
        self.zero_matrix[input_begin:input_begin+input_num,output_begin:output_begin+output_num] = 0
        if not RECURRENT_CONNECTIONS:
            self.zero_matrix[iin_begin,input_begin:input_begin+input_num] = 0
        if not FEEDBACK_CONNECTIONS:
            self.zero_matrix[iin_begin,ein_begin:ein_begin+ein_num] = 0
        self.zero_matrix[iin_begin,output_begin:output_begin+output_num] = 0
        
        if not RECURRENT_CONNECTIONS:
            self.zero_matrix[ein_begin:ein_begin+ein_num,ein_begin:ein_begin+ein_num] = 0
        if not FEEDBACK_CONNECTIONS:
            self.zero_matrix[ein_begin:ein_begin+ein_num,output_begin:output_begin+output_num] = 0
        if not RECURRENT_CONNECTIONS:
            self.zero_matrix[iin_begin+1,ein_begin:ein_begin+ein_num] = 0
        if not FEEDBACK_CONNECTIONS:
            self.zero_matrix[iin_begin+1,output_begin:output_begin+output_num] = 0
        
        self.zero_matrix[output_begin:output_begin+output_num,input_begin:input_begin+input_num] = 0
        if not RECURRENT_CONNECTIONS:
            self.zero_matrix[output_begin:output_begin+output_num,output_begin:output_begin+output_num] = 0
        self.zero_matrix[iin_begin+2,input_begin:input_begin+input_num] = 0
        if not RECURRENT_CONNECTIONS:
            self.zero_matrix[iin_begin+2,output_begin:output_begin+output_num] = 0
        
        self.zero_matrix[ein_begin:ein_begin+ein_num,iin_begin] = 0
        self.zero_matrix[output_begin:output_begin+output_num,iin_begin] = 0
        self.zero_matrix[iin_begin:iin_begin+iin_num,iin_begin] = 0
        
        self.zero_matrix[input_begin:input_begin+input_num,iin_begin+1] = 0
        self.zero_matrix[output_begin:output_begin+output_num,iin_begin+1] = 0
        self.zero_matrix[iin_begin:iin_begin+iin_num,iin_begin+1] = 0
        
        self.zero_matrix[input_begin:input_begin+input_num,iin_begin+2] = 0
        self.zero_matrix[ein_begin:ein_begin+ein_num,iin_begin+2] = 0
        self.zero_matrix[iin_begin:iin_begin+iin_num,iin_begin+2] = 0
        
        self.fix_synapse_strength()
    
    def init_normalization_parameters(self):
        # Initialize normalization parameters
        self.conf['Z_ex'] = self.conf['Z_ex_ex_th_ratio'] * self.conf['excitatory_threshold']
        self.conf['Z_iin'] = self.conf['Z_iin_ex_th_ratio'] * self.conf['excitatory_threshold']
        
        if RECURRENT_CONNECTIONS:
            Z_other_to_ein = (1-self.conf['Z_rec_Z_ex_ratio']) * self.conf['Z_ex']
        else:
            Z_other_to_ein = self.conf['Z_ex']
        if FEEDBACK_CONNECTIONS:
            self.conf['Z_inp_to_ein'] = self.conf['Z_inp_to_ein_percentage'] * Z_other_to_ein
            self.conf['Z_out_to_ein'] = self.conf['Z_out_to_ein_percentage'] * Z_other_to_ein
        else:
            self.conf['Z_inp_to_ein'] = Z_other_to_ein
            
        self.conf['Z_rec'] = self.conf['Z_rec_Z_ex_ratio'] * self.conf['Z_ex']
    
    def fix_synapse_strength(self):
        # Normalize the synapses strength, and enforce the invariants.
        excitatory_unit_num = unit_num-iin_num
        
        # For symmetry, make sure IINs receive equal input from all neurons
        self.synapse_strength[excitatory_unit_num:,:excitatory_unit_num] = 1
        
        # Enforce invariants
        self.synapse_strength = np.multiply(self.synapse_strength,self.zero_matrix)
        
        # Make sure excitatory weights in are all excitatory
        self.synapse_strength[:,:excitatory_unit_num][self.synapse_strength[:,:excitatory_unit_num] < 0] = 0
        
        # Normalize incoming excitatory weights to each unit    
        input_begin = 0
        ein_begin = input_begin + input_num
        output_begin = ein_begin + ein_num
        iin_begin = output_begin + output_num
        
        if RECURRENT_CONNECTIONS:
            Z_non_rec = (1 - self.conf['Z_rec_Z_ex_ratio']) * self.conf['Z_ex']
        else:
            Z_non_rec = self.conf['Z_ex']
        
        # Input to input neurons
        if RECURRENT_CONNECTIONS:
            input_to_input_row_sum = (self.synapse_strength[input_begin:input_begin+input_num,input_begin:input_begin+input_num].sum(axis=1).reshape(input_num,1).repeat(input_num, axis=1))/self.conf['Z_rec']
        else:
            input_to_input_row_sum = np.ones((input_num,input_num))
        if FEEDBACK_CONNECTIONS:
            ein_to_input_row_sum = (self.synapse_strength[input_begin:input_begin+input_num,ein_begin:ein_begin+ein_num].sum(axis=1).reshape(input_num,1).repeat(ein_num, axis=1))/Z_non_rec
        else:
            ein_to_input_row_sum = np.ones((input_num,ein_num))
        output_to_input_row_sum = np.ones((input_num,output_num))
        input_row_sum = np.concatenate((input_to_input_row_sum,ein_to_input_row_sum,output_to_input_row_sum),axis=1)
        
        # Input to EINs
        input_to_ein_row_sum = (self.synapse_strength[ein_begin:ein_begin+ein_num,input_begin:input_begin+input_num].sum(axis=1).reshape(ein_num,1).repeat(input_num, axis=1))/self.conf['Z_inp_to_ein']
        if RECURRENT_CONNECTIONS:
            ein_to_ein_row_sum = (self.synapse_strength[ein_begin:ein_begin+ein_num,ein_begin:ein_begin+ein_num].sum(axis=1).reshape(ein_num,1).repeat(ein_num, axis=1))/self.conf['Z_rec']
        else:
            ein_to_ein_row_sum = np.ones((ein_num,ein_num))
        if FEEDBACK_CONNECTIONS:
            output_to_ein_row_sum = (self.synapse_strength[ein_begin:ein_begin+ein_num,output_begin:output_begin+output_num].sum(axis=1).reshape(ein_num,1).repeat(output_num, axis=1))/self.conf['Z_out_to_ein']
        else:
            output_to_ein_row_sum = np.ones((ein_num,output_num))
        ein_row_sum = np.concatenate((input_to_ein_row_sum,ein_to_ein_row_sum,output_to_ein_row_sum),axis=1)
        
        # Input to output neurons
        input_to_output_row_sum = np.ones((output_num,input_num))
        ein_to_output_row_sum = (self.synapse_strength[output_begin:output_begin+output_num,ein_begin:ein_begin+ein_num].sum(axis=1).reshape(output_num,1).repeat(ein_num, axis=1))/Z_non_rec
        if RECURRENT_CONNECTIONS:
            output_to_output_row_sum = (self.synapse_strength[output_begin:output_begin+output_num,output_begin:output_begin+output_num].sum(axis=1).reshape(output_num,1).repeat(output_num, axis=1))/self.conf['Z_rec']
        else:
            output_to_output_row_sum = np.ones((output_num,output_num))
        output_row_sum = np.concatenate((input_to_output_row_sum,ein_to_output_row_sum,output_to_output_row_sum),axis=1)
        
        # Input to input IIN
        if RECURRENT_CONNECTIONS:
            input_to_input_iin_row_sum = (self.synapse_strength[iin_begin,input_begin:input_begin+input_num].sum().repeat(input_num).reshape(1,input_num))/self.conf['Z_rec']
        else:
            input_to_input_iin_row_sum = np.ones((1,input_num))
        if FEEDBACK_CONNECTIONS:
            ein_to_input_iin_row_sum = (self.synapse_strength[iin_begin,ein_begin:ein_begin+ein_num].sum().repeat(ein_num).reshape(1,ein_num))/Z_non_rec
        else:
            ein_to_input_iin_row_sum = np.ones((1,ein_num))
        output_to_input_iin_row_sum = np.ones((1,output_num))
        input_iin_from_ex_row_sum = np.concatenate((input_to_input_iin_row_sum,ein_to_input_iin_row_sum,output_to_input_iin_row_sum),axis=1)
        
        # Input to EIN's IIN
        input_to_ein_iin_row_sum = (self.synapse_strength[iin_begin+1,input_begin:input_begin+input_num].sum().repeat(input_num).reshape(1,input_num))/self.conf['Z_inp_to_ein']
        if RECURRENT_CONNECTIONS:
            ein_to_ein_iin_row_sum = (self.synapse_strength[iin_begin+1,ein_begin:ein_begin+ein_num].sum().repeat(ein_num).reshape(1,ein_num))/self.conf['Z_rec']
        else:
            ein_to_ein_iin_row_sum = np.ones((1,ein_num))
        if FEEDBACK_CONNECTIONS:
            output_to_ein_iin_row_sum = (self.synapse_strength[iin_begin+1,output_begin:output_begin+output_num].sum().repeat(output_num).reshape(1,output_num))/self.conf['Z_out_to_ein']
        else:
            output_to_ein_iin_row_sum = np.ones((1,output_num))
        ein_iin_from_ex_row_sum = np.concatenate((input_to_ein_iin_row_sum,ein_to_ein_iin_row_sum,output_to_ein_iin_row_sum),axis=1)
        
        # Input to output IIN
        input_to_output_iin_row_sum = np.ones((1,input_num))
        ein_to_output_iin_row_sum = (self.synapse_strength[iin_begin+2,ein_begin:ein_begin+ein_num].sum().repeat(ein_num).reshape(1,ein_num))/Z_non_rec
        if RECURRENT_CONNECTIONS:
            output_to_output_iin_row_sum = (self.synapse_strength[iin_begin+2,output_begin:output_begin+output_num].sum().repeat(output_num).reshape(1,output_num))/self.conf['Z_rec']
        else:
            output_to_output_iin_row_sum = np.ones((1,output_num))
        output_iin_from_ex_row_sum = np.concatenate((input_to_output_iin_row_sum,ein_to_output_iin_row_sum,output_to_output_iin_row_sum),axis=1)
        
        excitatory_row_sums = np.concatenate((input_row_sum,ein_row_sum,output_row_sum,input_iin_from_ex_row_sum,ein_iin_from_ex_row_sum,output_iin_from_ex_row_sum),axis=0)
        
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
        
        for t in range(T):
            if t % 1000 == 0:
                self.my_print('t='+str(t))
                
            # Propagate external input
            self.prop_external_input(input_vec)
                
            # Document fire history
            for unit_ind in range(unit_num):
                if self.prev_act[unit_ind, 0] == 1:
                    fire_history[unit_ind].append(t)
        
        self.my_print('firing count: ' + str([len(a) for a in fire_history[input_num:]]))
        
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
        prev_input_factor = (1 - 2 * self.prev_act)
        cur_input = np.add(cur_input, prev_input_factor * self.prev_input_to_neurons)
        
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
    
    def generate_random_external_input(self):
        return np.random.rand(input_num,1)*(2*self.conf['mean_sensory_input'])

model = ModelClass({},False,True)
input_vec = model.generate_random_external_input()

fire_history = model.simulate_dynamics(input_vec)
fire_count = [len(a) for a in fire_history]
print('Fire count: ' + str(fire_count))