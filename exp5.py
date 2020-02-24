import numpy as np
from copy import deepcopy

''' Experiment 5:
- A population of input excitatory neurons.
- A population of output excitatory neurons.
- A single, very strong, IIN.

We're trying to understand the connection between the inhibitory threshold size and the number of
winners (among output neurons), where we define a winner to be any neuron that fired at least once.
We expect that if we average over many trials, when the inhibitory and excitatory threshold are
equal, exactly half of the neurons will become winners- since the characteristics of excitatory
input to all the neurons is equal, and a winner is exactly the neurons that charges faster than the
IIN.

Result: Our prediction is incorrect: when we set the thresholds to be equal, a bit more than half of
the neurons were winners. To get exactly half, we had to set the inhibitory threshold a bit smaller
than the excitatory threshold (in_th =~ 0.381, ex_th = 0.4). The reason: the inhibitory threshold is
not exactly dividable by the average input into the IIN. If, for example, in_th=0.4 and the average
input is 0.03, the IIN will actually fire when it's accumulated input will be 0.42 (since 0.42 is
dividable by 0.03)- after 14 time steps, on average. So all excitatory neurons that have average
input 0.029 will become winners- since they will also fire after 14 time steps (since 0.029*14=0.406
is larger than 0.4), and this is more than half of the neurons.
'''

# General parameters
input_num = 8
output_num = 8
iin_num = 1

unit_num = input_num + output_num + iin_num

# Competition parameters
T = 10000


class ModelClass:
    
    default_configuration = {
        # Neuron parameters
        'excitatory_threshold' : 0.4,
        'inhibitory_threshold' : 0.4,
        'sensory_input_strength' : 0.13333,
        
        # Normalization parameter
        'Z_ex_ex_th_ratio' : 0.5,
        'Z_iin_ex_th_ratio' : 1,
        'Z_inp_Z_ex_ratio' : 0.5,
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
            self.synapse_strength[:, unit_num - iin_num:] = (-1) * self.synapse_strength[:, unit_num - iin_num:]
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
        self.conf['Z_inp'] = self.conf['Z_inp_Z_ex_ratio'] * self.conf['Z_ex']
        self.conf['Z_out'] = self.conf['Z_ex'] - self.conf['Z_inp']
    
    def fix_synapse_strength(self):
        # Normalize the synapses strength, and enforce the invariants.
        excitatory_unit_num = unit_num - iin_num
        
        # Make sure excitatory weights in are all excitatory
        self.synapse_strength[:, :excitatory_unit_num][self.synapse_strength[:, :excitatory_unit_num] < 0] = 0
        
        # Normalize incoming excitatory weights to each unit    
        input_begin = 0
        output_begin = input_begin + input_num 
        
        input_to_input_row_sum = (self.synapse_strength[input_begin:input_begin + input_num, input_begin:input_begin + input_num].sum(axis=1).reshape(input_num, 1).repeat(input_num, axis=1)) / self.conf['Z_inp']
        output_to_input_row_sum = (self.synapse_strength[input_begin:input_begin + input_num, output_begin:output_begin + output_num].sum(axis=1).reshape(input_num, 1).repeat(output_num, axis=1)) / self.conf['Z_out']
        input_row_sum = np.concatenate((input_to_input_row_sum, output_to_input_row_sum), axis=1)
        
        input_to_output_row_sum = (self.synapse_strength[output_begin:output_begin + output_num, input_begin:input_begin + input_num].sum(axis=1).reshape(output_num, 1).repeat(input_num, axis=1)) / self.conf['Z_inp']
        output_to_output_row_sum = (self.synapse_strength[output_begin:output_begin + output_num, output_begin:output_begin + output_num].sum(axis=1).reshape(output_num, 1).repeat(output_num, axis=1)) / self.conf['Z_out']
        output_row_sum = np.concatenate((input_to_output_row_sum, output_to_output_row_sum), axis=1)
        
        input_to_iin_row_sum = (self.synapse_strength[-iin_num:, input_begin:input_begin + input_num].sum(axis=1).reshape(iin_num, 1).repeat(input_num, axis=1)) / self.conf['Z_inp']
        output_to_iin_row_sum = (self.synapse_strength[-iin_num:, output_begin:output_begin + output_num].sum(axis=1).reshape(iin_num, 1).repeat(output_num, axis=1)) / self.conf['Z_out']
        iin_from_ex_row_sum = np.concatenate((input_to_iin_row_sum, output_to_iin_row_sum), axis=1)
        
        excitatory_row_sums = np.concatenate((input_row_sum, output_row_sum, iin_from_ex_row_sum), axis=0)
        
        # Make sure inhibitory weights are all inhibitory
        self.synapse_strength[:, excitatory_unit_num:][self.synapse_strength[:, excitatory_unit_num:] > 0] = 0
        # Normalize incoming inhibitory weights to each unit
        inhibitory_row_sums = (-1) * ((self.synapse_strength[:, excitatory_unit_num:].sum(axis=1).reshape(unit_num, 1).repeat(iin_num, axis=1)) / self.conf['Z_iin'])
        
        row_sums = np.concatenate((excitatory_row_sums, inhibitory_row_sums), axis=1)
        row_sums[row_sums == 0] = 1
        self.synapse_strength = self.synapse_strength / row_sums
        
    def simulate_dynamics(self, input_vec):
        # Given an input, simulate the dynamics of the system, for iter_num time steps
        fire_history = []
        for _ in range(unit_num):
            fire_history.append([])
        
        for t in range(T):
            if t % 1000 == 0:
                self.my_print('t=' + str(t))
                
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
        input_from_prev_layer = np.pad(sensory_input_vec, ((0, unit_num - input_num), (0, 0)), 'constant')
        cur_input = np.add(cur_input, input_from_prev_layer)
        input_from_pre_layer = np.matmul(self.synapse_strength, self.prev_act)
        cur_input = np.add(cur_input, input_from_pre_layer)
            
        ''' Accumulating input and refractory period: If a neuron fired in the last time step,
        we subtract its previous input from its current input. Otherwise- we add its previous
        input to its current input. '''
        prev_input_factor = (1 - 2 * self.prev_act)
        cur_input = np.add(cur_input, prev_input_factor * self.prev_input_to_neurons)
        
        # Make sure the input is non-negative
        cur_input = np.where(cur_input >= 0, cur_input, 0)
        
        cur_act = np.concatenate((self.excitatory_activation_function(cur_input[:unit_num - iin_num, [0]]),
                              self.inhibitory_activation_function(cur_input[unit_num - iin_num:, [0]])),
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
        strength_factor = np.random.rand()
        first_active_unit = np.random.randint(input_num - 1)
        precentage_of_first_unit = np.random.rand()
        
        sensory_vec = np.zeros((input_num, 1))
        sensory_vec[first_active_unit, 0] = strength_factor * precentage_of_first_unit * self.conf['sensory_input_strength']
        sensory_vec[first_active_unit + 1, 0] = strength_factor * (1 - precentage_of_first_unit) * self.conf['sensory_input_strength']
        
        return sensory_vec


iter_num = 1000
winner_num_sum = 0
for cur_iter in range(iter_num):
    if cur_iter % 100 == 0:
        print('cur_iter=' + str(cur_iter))
    model = ModelClass({}, False, True)
    input_vec = model.generate_random_external_input()
    
    fire_history = model.simulate_dynamics(input_vec)
    fire_count = [len(a) for a in fire_history]
    winner_num = len([x for x in fire_count[input_num:input_num + output_num] if x > 0])
    winner_num_sum += winner_num
winner_num_average = winner_num_sum / iter_num
print('Winner num average: ' + str(winner_num_average))
