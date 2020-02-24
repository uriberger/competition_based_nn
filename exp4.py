import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

''' Experiment 4:
- 3 populations: Excitatory input, excitatory output, inhibitory
- Complex refractory period
- A single, very strong, inhibitory neuron (one firing is enough to reset input of all neurons) 

Result: All neurons are phase-locked with the IIN. The reason is: If they charge slower than the
IIN, they never fire. Otherwise, they fire, start charging again, and when the IIN fires the cycle
starts again.
Another possible output is if a neuron will charge twice as fast as the IIN, it will have two
phases- one of the same length as its charging rate (the charging starts after the neuron fires
after the IIN, and completes the charge before the next firing of the IIN), and one longer
(the charging starts after the neuron fires the second time after the IIN fired, and it's unable to
complete the charging).
Another important note is that we can predict who will be the winners: winners are exactly the
neurons that charge faster than the IIN. A good prediction for the charge rate is multiplying the
(sensory input divided by the excitatory threshold) by the weights from the input neurons to the
output neuron we want to predict for (sensory input divided by ex thres gives us the average firing
rate of input neurons, multiplying by the weights gives us the average input). If the average input
to an output neuron is more than (ex_th/in_th) times the average input to the IIN, it charges faster
and it will become a winner. This is not precise since we're neglecting input from other neurons,
but this is a good prediction.   
'''

# General parameters
input_num = 8
output_num = 8
iin_num = 1
unit_num = input_num+output_num+iin_num
ex_num = unit_num - iin_num

# Competition parameters
T = 10000

class ModelClass:
    
    default_configuration = {
        # Neuron parameters
        'excitatory_threshold' : 0.4,
        'inhibitory_threshold' : 0.3,
        'mean_sensory_input' : 0.13333,
        
        # Normalization parameter
        'Z_ex_ex_th_ratio' : 0.5,
        'Z_inp_Z_ex_ratio' : 0.5,
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
        self.conf['Z_inp'] = self.conf['Z_inp_Z_ex_ratio'] * self.conf['Z_ex']
        self.conf['Z_out'] = self.conf['Z_ex'] - self.conf['Z_inp']
        
        self.conf['Z_iin'] = self.conf['Z_iin_ex_th_ratio'] * self.conf['excitatory_threshold']
    
    def fix_synapse_strength(self):
        # Normalize the synapses strength, and enforce the invariants.
        input_begin = 0
        output_begin = input_begin + input_num
        iin_begin = output_begin + output_num
        
        input_to_input_row_sum = (self.synapse_strength[input_begin:input_begin+input_num,input_begin:input_begin+input_num].sum(axis=1).reshape(input_num,1).repeat(input_num, axis=1))/self.conf['Z_inp']
        output_to_input_row_sum = (self.synapse_strength[input_begin:input_begin+input_num,output_begin:output_begin+output_num].sum(axis=1).reshape(input_num,1).repeat(output_num, axis=1))/self.conf['Z_out']
        input_row_sum = np.concatenate((input_to_input_row_sum,output_to_input_row_sum),axis=1)
        
        input_to_output_row_sum = (self.synapse_strength[output_begin:output_begin+output_num,input_begin:input_begin+input_num].sum(axis=1).reshape(output_num,1).repeat(input_num, axis=1))/self.conf['Z_inp']
        output_to_output_row_sum = (self.synapse_strength[output_begin:output_begin+output_num,output_begin:output_begin+output_num].sum(axis=1).reshape(output_num,1).repeat(output_num, axis=1))/self.conf['Z_out']
        output_row_sum = np.concatenate((input_to_output_row_sum,output_to_output_row_sum),axis=1)
        
        input_to_iin_row_sum = (self.synapse_strength[iin_begin:iin_begin+iin_num,input_begin:input_begin+input_num].sum(axis=1).reshape(iin_num,1).repeat(input_num, axis=1))/self.conf['Z_inp']
        output_to_iin_row_sum = (self.synapse_strength[iin_begin:iin_begin+iin_num,output_begin:output_begin+output_num].sum(axis=1).reshape(iin_num,1).repeat(output_num, axis=1))/self.conf['Z_out']
        iin_from_ex_row_sum = np.concatenate((input_to_iin_row_sum,output_to_iin_row_sum),axis=1)
        
        excitatory_row_sums = np.concatenate((input_row_sum,output_row_sum,iin_from_ex_row_sum),axis=0)
        
        inhibitory_row_sums = (-1)*((self.synapse_strength[:,ex_num:].sum(axis=1).reshape(unit_num,1).repeat(iin_num, axis=1))/self.conf['Z_iin'])
                    
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
        active_vec = 0+(np.random.rand(input_num,1)>(3/4))
        sensory_vec = np.random.rand(input_num,1)*(2*self.conf['mean_sensory_input'])
        return np.multiply(active_vec, sensory_vec)

iter_num = 10000
hamming_dist_sum = 0
log_fp = open('log.txt','w')
for cur_iter in range(iter_num):
    if cur_iter % 100 == 0:
        log_fp.write(str(cur_iter)+'\n')
        log_fp.flush()
    model = ModelClass({},False,True)
    input_vec = model.generate_random_external_input()
    
    # Predict winners and losers
    input_units_phase_vec = (1/model.conf['excitatory_threshold'])*input_vec
    avg_input_from_input_units_vec = np.matmul(model.synapse_strength[input_num:,:input_num],input_units_phase_vec)
    ex_th_in_th_ratio = model.conf['excitatory_threshold']/model.conf['inhibitory_threshold']
    avg_input_to_iin = avg_input_from_input_units_vec[-1,0]
    predicted_winner_loser_list = [1 if avg_input_from_input_units_vec[i,0]>ex_th_in_th_ratio*avg_input_to_iin else 0 for i in range(output_num)]
    
    fire_history = model.simulate_dynamics(input_vec)
    fire_count = [len(a) for a in fire_history[input_num:input_num+output_num]]
    winner_loser_list = [1 if x>0 else 0 for x in fire_count]
    hamming_dist = len([1 for i in range(output_num) if winner_loser_list[i] != predicted_winner_loser_list[i]])
    normalized_hamming_dist = hamming_dist/output_num
    
    hamming_dist_sum += normalized_hamming_dist
average_hamming_dist = hamming_dist_sum / iter_num
print(average_hamming_dist)
log_fp.write(str(average_hamming_dist) + '\n')

A = np.zeros((unit_num,T))
A[A==0] = np.nan
for unit_ind in range(unit_num):
    for t in fire_history[unit_ind]:
        if unit_ind < input_num:
            A[unit_ind, t] = 0
        elif unit_ind < input_num+output_num:
            A[unit_ind, t] = 1
        else:
            A[unit_ind, t] = 2
plt.matshow(A[:,-200:])
plt.show()