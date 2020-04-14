import numpy as np
#import idx2numpy
from copy import deepcopy
import matplotlib.pyplot as plt

'''
'''

# General parameters
layers_size = [784,2048,2048,512,10]
iin_num = len(layers_size)
unit_num = sum(layers_size) + iin_num

class ModelClass:
    
    default_configuration = {
        # Neuron parameters
        'excitatory_threshold' : 1,
        'inhibitory_threshold' : 1,
        'response_innervation_strength' : 0.1,
        
        # Winner num
        'winner_num' : [20,10,10,3,1],
        
        # Normalization parameters
        'Z_iin_ex_th_ratio' : 1,
        'max_average_strength_ratio' : 2,
        'Z_intra_layer_ex_th_ratio' : 1,
        
        # Learning parameters
        'eta' : 0.01,
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
            #self.synapse_strength = np.random.rand(unit_num, unit_num)
            self.synapse_strength = np.ones((unit_num, unit_num))
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
        
        # Input to input neurons
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
        
        # Input to input IIN
        self.zero_matrix[sum(layers_size),layers_size[0]:sum(layers_size)] = 0
        
        # Input from input IIN
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
        
        if file_suffix == None:
            # Add some variance to the weights
            noise_reduce_factor = 10
            
            for l in range(1,len(layers_size)):
                self.synapse_strength[sum(layers_size[:l]):sum(layers_size[:l+1]),sum(layers_size[:l-1]):sum(layers_size[:l])] += np.random.normal(0,self.conf['Z_vals'][l-1]/(noise_reduce_factor*layers_size[l-1]),(layers_size[l],layers_size[l-1]))
            
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
        
        # Input to input neurons
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
        
    def simulate_dynamics(self, input_vec, innervated_response_neuron):
        fire_history = []
        for _ in range(unit_num):
            fire_history.append([])
        
        #for t in range(T):
        t = 0
        while True:
            if t % 1000 == 0:
                self.my_print('t='+str(t))
                
            # Propagate external input
            self.prop_external_input(input_vec, innervated_response_neuron)
                
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
        
    def prop_external_input(self, sensory_input_vec, innervated_response_neuron):
        # Simulate the dynamics of the system for a single time step
        cur_input = np.zeros((unit_num, 1))
        input_from_prev_layer = np.pad(sensory_input_vec, ((0,unit_num-layers_size[0]),(0,0)), 'constant')
        cur_input = np.add(cur_input, input_from_prev_layer)
        input_from_pre_layer = np.matmul(self.synapse_strength, self.prev_act)
        cur_input = np.add(cur_input, input_from_pre_layer)
        
        if innervated_response_neuron != -1:
            cur_input[sum(layers_size[:-1])+innervated_response_neuron] = self.conf['response_innervation_strength']
        
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
    
    def update_synapse_strength(self, cur_fire_count):
        for l in range(1,len(layers_size)):
            prev_layer_winners = np.array([[1 if cur_fire_count[sum(layers_size[:l-1])+i]>0.7*cur_fire_count[sum(layers_size)+l-1] else 0 for i in range(layers_size[l-1])]]).transpose()
            cur_layer_winners = np.array([[1 if cur_fire_count[sum(layers_size[:l])+i]>0.7*cur_fire_count[sum(layers_size)+l] else 0 for i in range(layers_size[l])]]).transpose()
            update_mat = np.matmul(2*(cur_layer_winners-0.5),prev_layer_winners.transpose())
            self.synapse_strength[sum(layers_size[:l]):sum(layers_size[:l+1]),sum(layers_size[:l-1]):sum(layers_size[:l])] += update_mat*self.conf['eta']*(self.conf['Z_vals'][l-1]/layers_size[l-1])
        
        self.fix_synapse_strength()
        
        # Make sure no excitatory weight is more than the maximal possible weight
        for l in range(1,len(layers_size)):
            max_val = self.conf['max_average_strength_ratio']*(self.conf['Z_vals'][l-1]/layers_size[l-1])
            self.synapse_strength[sum(layers_size[:l]):sum(layers_size[:l+1]),sum(layers_size[:l-1]):sum(layers_size[:l])][self.synapse_strength[sum(layers_size[:l]):sum(layers_size[:l+1]),sum(layers_size[:l-1]):sum(layers_size[:l])] > max_val] = max_val
        
        self.fix_synapse_strength()
        
def my_print(my_str):
    if write_to_log:
        log_fp.write(my_str + '\n')
        log_fp.flush()
    else:
        print(my_str)
        
'''def get_data(training_examples_file, test_examples_file, training_labels_file,
             test_labels_file):
    training_examples = idx2numpy.convert_from_file(training_examples_file)
    test_examples = idx2numpy.convert_from_file(test_examples_file)
    training_labels = idx2numpy.convert_from_file(training_labels_file)
    test_labels = idx2numpy.convert_from_file(test_labels_file)
    
    return (training_examples, test_examples, training_labels, test_labels)'''

def preprocess(orig_data_mat):
    data_mat = orig_data_mat.transpose(1,2,0)
    data_mat = np.reshape(data_mat, (data_mat.shape[0]*data_mat.shape[1],
                                     data_mat.shape[2]))
    data_mat = data_mat/np.max(data_mat, axis=0)
    #data_mat = data_mat - np.mean(data_mat, axis=0)
    
    return data_mat

# Load images and labels
'''orig_training_examples, orig_test_examples, training_labels, test_labels = \
        get_data('MNIST_data/train-images.idx3-ubyte',
        'MNIST_data/t10k-images.idx3-ubyte',
        'MNIST_data/train-labels.idx1-ubyte',
        'MNIST_data/t10k-labels.idx1-ubyte')
np.save('training_examples',orig_training_examples)
np.save('test_examples',orig_test_examples)
np.save('training_labels',training_labels)
np.save('test_labels',test_labels)
assert(False)'''
orig_training_examples = np.load('training_examples.npy')
orig_test_examples = np.load('test_examples.npy')
training_labels = np.load('training_labels.npy')
test_labels = np.load('test_labels.npy')

# Preprocess the images (reshape to vectors and subtract mean)
training_examples = preprocess(orig_training_examples)
test_examples = preprocess(orig_test_examples)
perm = np.random.permutation(training_examples.shape[1])
N = 1000
X = training_examples[:,perm[range(N)]]
y = training_labels[perm[range(N)]]

write_to_log = False
if write_to_log:
    log_fp = open('log.txt','w')
    
'''trained_model = ModelClass({},None,True)
for i in range(N):
    if i % 10 == 0 or (not write_to_log):
        my_print('train iter ' + str(i))
    input_vec = X[:,[i]]
    true_label = y[i]
    fire_history = trained_model.simulate_dynamics(input_vec,true_label)
    fire_count = [len(a) for a in fire_history]
    trained_model.update_synapse_strength(fire_count)
trained_model.save_synapse_strength('mnist')'''
trained_model = ModelClass({},'mnist',True)
    
my_print('Evaluating')
correct_count = 0
for i in range(N):
    if i % 10 == 0 or (not write_to_log):
        my_print('eval iter ' + str(i))
    input_vec = X[:,[i]]
    true_label = y[i]
    fire_history = trained_model.simulate_dynamics(input_vec,true_label)
    fire_count = [len(a) for a in fire_history]
    
    correct = False
    if fire_count[sum(layers_size[:-1])+true_label] > 0:
        correct = True
        for unit_ind in range(layers_size[-1]):
            if unit_ind == true_label:
                continue
            if fire_count[sum(layers_size[:-1])+unit_ind] > 0:
                correct = False
                break
        if correct:
            correct_count += 1
my_print(str(correct_count) + ' out of ' + str(N))