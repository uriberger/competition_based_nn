import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import time
from font_inputs import generate_data_sets
import math

class ModelClass:
    
    default_configuration = {
        # Neuron parameters
        'excitatory_threshold' : 1,
        'inhibitory_threshold' : 1,
        #'sensory_input_strength' : 1/100,
        'sensory_input_strength' : 1/100,
        'response_innervation_strength' : 0.01,
        'layers_size' : [441,441,512,10],
        'norm_shrink_factor' : 100,
        
        # Normalization parameters
        'Z_iin_ex_th_ratio' : [0.01,0.01,1,1,1],
        'Z_intra_layer_ex_th_ratio' : 1,
        'max_average_strength_ratio' : 1000,
        
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
        
        unit_num = sum(self.conf['layers_size']) + len(self.conf['layers_size'])
        excitatory_unit_num = sum(self.conf['layers_size'])
        
        if file_suffix == None:
            #self.synapse_strength = np.ones((unit_num, unit_num))
            self.synapse_strength = np.random.rand(unit_num, unit_num)
            self.synapse_strength[:, excitatory_unit_num:] = (-1) * self.synapse_strength[:, excitatory_unit_num:]            
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
        self.zero_matrix[:self.conf['layers_size'][0],:self.conf['layers_size'][0]] = 0
        N = round(self.conf['layers_size'][0]**0.5)
        for unit_ind in range(self.conf['layers_size'][0]):
            if unit_ind % N > 0:
                self.zero_matrix[unit_ind,unit_ind-1] = 1
            if unit_ind % N < N-1:
                self.zero_matrix[unit_ind,unit_ind+1] = 1
            if unit_ind >= N:
                self.zero_matrix[unit_ind,unit_ind-N] = 1
            if unit_ind < self.conf['layers_size'][0]-N:
                self.zero_matrix[unit_ind,unit_ind+N] = 1
        self.zero_matrix[:self.conf['layers_size'][0],self.conf['layers_size'][0]:sum(self.conf['layers_size'])] = 0
        
        # Input to 1st IIN
        self.zero_matrix[sum(self.conf['layers_size']),self.conf['layers_size'][0]:sum(self.conf['layers_size'])] = 0
        
        # Input from 1st IIN
        self.zero_matrix[self.conf['layers_size'][0]:,sum(self.conf['layers_size'])] = 0
        
        # 2nd layer- only external inputs for now, except from IIN
        self.zero_matrix[self.conf['layers_size'][0]:sum(self.conf['layers_size'][:2]),:] = 0
        self.zero_matrix[self.conf['layers_size'][0]:sum(self.conf['layers_size'][:2]),sum(self.conf['layers_size'])+1] = 1
        
        # Input to 2nd IIN
        self.zero_matrix[sum(self.conf['layers_size'])+1,:self.conf['layers_size'][0]] = 0
        self.zero_matrix[sum(self.conf['layers_size'])+1,sum(self.conf['layers_size'][:2]):sum(self.conf['layers_size'])] = 0
        
        # Input from 2nd IIN
        self.zero_matrix[:self.conf['layers_size'][0],sum(self.conf['layers_size'])+1] = 0
        self.zero_matrix[sum(self.conf['layers_size'][:2]):,sum(self.conf['layers_size'])+1] = 0
        
        # Input to other layers
        for l in range(2,len(self.conf['layers_size'])):
            # Input to excitatory neurons
            self.zero_matrix[sum(self.conf['layers_size'][:l]):sum(self.conf['layers_size'][:l+1]),:sum(self.conf['layers_size'][:l-1])] = 0
            self.zero_matrix[sum(self.conf['layers_size'][:l]):sum(self.conf['layers_size'][:l+1]),sum(self.conf['layers_size'][:l]):sum(self.conf['layers_size'])] = 0
            
            # Input to IIN
            self.zero_matrix[sum(self.conf['layers_size'])+l,:sum(self.conf['layers_size'][:l])] = 0
            self.zero_matrix[sum(self.conf['layers_size'])+l,sum(self.conf['layers_size'][:l+1]):sum(self.conf['layers_size'])] = 0
            
            # Input from IIN
            self.zero_matrix[:sum(self.conf['layers_size'][:l]),sum(self.conf['layers_size'])+l] = 0
            self.zero_matrix[sum(self.conf['layers_size'][:l+1]):sum(self.conf['layers_size']),sum(self.conf['layers_size'])+l] = 0
            self.zero_matrix[sum(self.conf['layers_size']):,sum(self.conf['layers_size'])+l] = 0
        
        # Don't allow self loops
        np.fill_diagonal(self.zero_matrix,0)
        
        self.fix_synapse_strength()
            
    def init_normalization_parameters(self):
        ''' Generate winner number list.
        For the first two, sensory layers, we don't care so we'll put 1 winner.
        For the following layers (except the last one) we want half of the neurons to be winners,
        to maximize the number of possible combinations in the next layers.
        In the last layer we want a single winner, because each input corresponds to a single class.
        '''
        self.conf['winner_num'] = [1,1]
        for l in range(2,len(self.conf['layers_size'])-1):
            self.conf['winner_num'].append(int(self.conf['layers_size'][l]/2))
        self.conf['winner_num'].append(1)
        
        # Initialize normalization parameters
        self.conf['Z_iin'] = [x * self.conf['excitatory_threshold'] for x in self.conf['Z_iin_ex_th_ratio']]
        self.conf['Z_intra_layer'] = self.conf['Z_intra_layer_ex_th_ratio'] * self.conf['excitatory_threshold']

        self.conf['Z_vals'] = []
        first = True
        for l in range(2,len(self.conf['layers_size'])):
            if first:
                norm_shrink_factor = self.conf['norm_shrink_factor']
                first = False
            else:
                norm_shrink_factor = 10
            self.conf['Z_vals'].append((self.conf['excitatory_threshold'] * self.conf['layers_size'][l-1])/(norm_shrink_factor*self.conf['winner_num'][l-1]))
        
        self.conf['Z_ex_to_iins'] = []
        for l in range(len(self.conf['layers_size'])):
            self.conf['Z_ex_to_iins'].append((self.conf['excitatory_threshold'] * self.conf['layers_size'][l])/(self.conf['winner_num'][l]))
        
    def reset_data_structures(self):
        self.prev_act = np.zeros((sum(self.conf['layers_size'])+len(self.conf['layers_size']), 1))
        self.prev_input_to_neurons = np.zeros((sum(self.conf['layers_size'])+len(self.conf['layers_size']), 1))
    
    def fix_synapse_strength(self):
        # Normalize the synapses strength, and enforce the invariants.
        excitatory_unit_num = sum(self.conf['layers_size'])
        
        # For symmetry, make sure IINs receive equal input from all neurons
        self.synapse_strength[excitatory_unit_num:,:excitatory_unit_num] = 1
        
        # Also make sure intra-layer connections for input neurons are all of equal strength
        self.synapse_strength[:self.conf['layers_size'][0],:self.conf['layers_size'][0]] = 1
        self.synapse_strength[self.conf['layers_size'][0]:sum(self.conf['layers_size'][:2]),self.conf['layers_size'][0]:sum(self.conf['layers_size'][:2])] = 1
        
        # Enforce invariants
        self.synapse_strength = np.multiply(self.synapse_strength,self.zero_matrix)
        
        # Make sure excitatory weights are all excitatory
        self.synapse_strength[:,:excitatory_unit_num][self.synapse_strength[:,:excitatory_unit_num] < 0] = 0
        
        # Make sure no excitatory weight is more than the maximal possible weight
        for l in range(2,len(self.conf['layers_size'])):
            max_allowed_val = self.conf['max_average_strength_ratio']*(self.conf['Z_vals'][l-2]/self.conf['layers_size'][l-1])
            prev_layer_begin = sum(self.conf['layers_size'][:l-1])
            prev_layer_end = sum(self.conf['layers_size'][:l])
            cur_layer_begin = sum(self.conf['layers_size'][:l])
            cur_layer_end = sum(self.conf['layers_size'][:l+1])
            max_actual_val = np.max(self.synapse_strength[cur_layer_begin:cur_layer_end,prev_layer_begin:prev_layer_end])
            if max_actual_val > max_allowed_val:
                self.synapse_strength[sum(self.conf['layers_size'][:l]):sum(self.conf['layers_size'][:l+1]),sum(self.conf['layers_size'][:l-1]):sum(self.conf['layers_size'][:l])] *= max_allowed_val/max_actual_val
        
        # Normalize incoming excitatory weights to each unit    
        
        # Input to 1st layer
        first_input_to_first_input_row_sum = (self.synapse_strength[:self.conf['layers_size'][0],:self.conf['layers_size'][0]].sum(axis=1).reshape(self.conf['layers_size'][0],1).repeat(self.conf['layers_size'][0], axis=1))/self.conf['Z_intra_layer']
        rest_to_first_input_row_sum = np.ones((self.conf['layers_size'][0],sum(self.conf['layers_size'])-self.conf['layers_size'][0]))
        first_input_row_sum = np.concatenate((first_input_to_first_input_row_sum,rest_to_first_input_row_sum),axis=1)
        
        # Input to 2nd layer
        second_input_row_sum = np.ones((self.conf['layers_size'][1],excitatory_unit_num))
        
        excitatory_row_sums = np.concatenate((first_input_row_sum,second_input_row_sum),axis=0)
        
        for l in range(2,len(self.conf['layers_size'])):
            # Input to EINs
            input_from_earlier_layers = np.ones((self.conf['layers_size'][l],sum(self.conf['layers_size'][:l-1])))
            input_from_prev_layer = (self.synapse_strength[sum(self.conf['layers_size'][:l]):sum(self.conf['layers_size'][:l+1]),sum(self.conf['layers_size'][:l-1]):sum(self.conf['layers_size'][:l])].sum(axis=1).reshape(self.conf['layers_size'][l],1).repeat(self.conf['layers_size'][l-1], axis=1))/self.conf['Z_vals'][l-2]
            input_from_later_layers = np.ones((self.conf['layers_size'][l],sum(self.conf['layers_size'][l:])))
            row_sum = np.concatenate((input_from_earlier_layers,input_from_prev_layer,input_from_later_layers),axis=1)
            excitatory_row_sums = np.concatenate((excitatory_row_sums,row_sum),axis=0)
        
        # Input to IINs
        for l in range(len(self.conf['layers_size'])):
            input_from_earlier_layers = np.ones((1,sum(self.conf['layers_size'][:l])))
            input_from_layer = (self.synapse_strength[sum(self.conf['layers_size'])+l,sum(self.conf['layers_size'][:l]):sum(self.conf['layers_size'][:l+1])].sum().repeat(self.conf['layers_size'][l]).reshape(1,self.conf['layers_size'][l]))/self.conf['Z_ex_to_iins'][l]
            input_from_later_layers = np.ones((1,sum(self.conf['layers_size'][l+1:])))
            row_sum = np.concatenate((input_from_earlier_layers,input_from_layer,input_from_later_layers),axis=1)
            excitatory_row_sums = np.concatenate((excitatory_row_sums,row_sum),axis=0)
        
        # Make sure inhibitory weights are all inhibitory
        self.synapse_strength[:,excitatory_unit_num:][self.synapse_strength[:,excitatory_unit_num:] > 0] = 0
        # Normalize incoming inhibitory weights to each unit
        unit_num = excitatory_unit_num+len(self.conf['layers_size'])
        inhibitory_row_sums = np.zeros((unit_num,0))
        for l in range(len(self.conf['layers_size'])):
            cur_layer_inhibitory_row_sums = (-1)*((self.synapse_strength[:,[excitatory_unit_num+l]].sum(axis=1).reshape(unit_num,1))/self.conf['Z_iin'][l])
            inhibitory_row_sums = np.concatenate((inhibitory_row_sums,cur_layer_inhibitory_row_sums),axis=1)
            #inhibitory_row_sums = (-1)*((self.synapse_strength[:,excitatory_unit_num:].sum(axis=1).reshape(sum(self.conf['layers_size'])+len(self.conf['layers_size']),1).repeat(len(self.conf['layers_size']), axis=1))/self.conf['Z_iin'])
        
        row_sums = np.concatenate((excitatory_row_sums,inhibitory_row_sums),axis=1)
        row_sums[row_sums == 0] = 1
        self.synapse_strength = self.synapse_strength/row_sums
        
    def update_synapse_strength(self, cur_fire_count):
        for l in range(2 ,len(self.conf['layers_size'])):
            prev_layer_winners = np.array([[1 if cur_fire_count[sum(self.conf['layers_size'][:l-1])+i]>=0.5*cur_fire_count[sum(self.conf['layers_size'])+l-1] else 0 for i in range(self.conf['layers_size'][l-1])]]).transpose()
            cur_layer_winners = np.array([[1 if cur_fire_count[sum(self.conf['layers_size'][:l])+i]>=0.5*cur_fire_count[sum(self.conf['layers_size'])+l] else 0 for i in range(self.conf['layers_size'][l])]]).transpose()
            update_mat = np.matmul(2*(cur_layer_winners-0.5),prev_layer_winners.transpose())
            self.synapse_strength[sum(self.conf['layers_size'][:l]):sum(self.conf['layers_size'][:l+1]),sum(self.conf['layers_size'][:l-1]):sum(self.conf['layers_size'][:l])] += update_mat*self.conf['eta']*(self.conf['Z_vals'][l-2]/self.conf['layers_size'][l-1])
        
        self.fix_synapse_strength()
        
    def simulate_dynamics(self, input_vec, innervated_response_neuron):
        unit_num = sum(self.conf['layers_size']) + len(self.conf['layers_size'])
        
        fire_history = []
        for _ in range(unit_num):
            fire_history.append([])
            
        sensory_input_vec = np.pad(input_vec, ((0,unit_num-self.conf['layers_size'][0]),(0,0)), 'constant')
        if innervated_response_neuron != -1:
            sensory_input_vec[sum(self.conf['layers_size'][:-1])+innervated_response_neuron,0] = self.conf['response_innervation_strength']
        
        copied_input_to_second_layer = False
        #for t in range(T):
        t = 0
        while True:
            if t % 1000 == 0:
                self.my_print('t='+str(t))
                
            if not copied_input_to_second_layer and max([len(x) for x in fire_history[:self.conf['layers_size'][0]]]) > 50:
                sensory_winner = np.argmax([len(x) for x in fire_history[:self.conf['layers_size'][0]]])
                old_sensory_input_vec = deepcopy(sensory_input_vec)
                sensory_input_vec = np.zeros((unit_num,1))
                N = round(self.conf['layers_size'][0]**0.5)
                middle_point = round((N-1)/2)
                vertical_shift = middle_point-int(sensory_winner/N)
                horizontal_shift = middle_point-(sensory_winner%N)
                for unit_ind in range(self.conf['layers_size'][0]):
                    vertical_loc = int(unit_ind/N)
                    horizontal_loc = unit_ind%N
                    if vertical_loc + vertical_shift >= N:
                        break
                    if horizontal_loc + horizontal_shift >= N:
                        continue
                    new_ind = unit_ind+vertical_shift*N+horizontal_shift
                    sensory_input_vec[self.conf['layers_size'][0]+new_ind,0] = old_sensory_input_vec[unit_ind,0]
                copied_input_to_second_layer = True
                
            # Propagate external input
            self.prop_external_input(sensory_input_vec)
                
            # Document fire history
            for unit_ind in range(unit_num):
                if self.prev_act[unit_ind, 0] == 1:
                    fire_history[unit_ind].append(t)
                    
            t += 1
            
            all_inn_fired = True
            for l in range(len(self.conf['layers_size'])):
                if len(fire_history[sum(self.conf['layers_size'])+l]) == 0:
                    all_inn_fired = False
                    break
            if all_inn_fired:
                break
        
        self.reset_data_structures()
        
        return fire_history
        
    def prop_external_input(self, sensory_input_vec):
        # Simulate the dynamics of the system for a single time step
        external_input = sensory_input_vec
        internal_input = np.matmul(self.synapse_strength, self.prev_act)
        cur_input = np.add(external_input, internal_input)
        
        # Accumulating input
        cur_input = np.add(cur_input, self.prev_input_to_neurons)
        
        # Input reset for neurons that fired
        cur_input[self.prev_act == 1] = 0
        
        # Make sure the input is non-negative
        cur_input = np.where(cur_input>=0, cur_input, 0)
        
        cur_act = np.concatenate((self.excitatory_activation_function(cur_input[:sum(self.conf['layers_size']),[0]]),
                              self.inhibitory_activation_function(cur_input[sum(self.conf['layers_size']):,[0]])),
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

N = round(ModelClass.default_configuration['layers_size'][0]**0.5)
training_set,test_set = generate_data_sets(N, ModelClass.default_configuration['sensory_input_strength'])

def plot_precision_as_a_func_of_training_epoch_num():
    training_epoch_num = 50
    
    model = ModelClass({},None,True)
    cur_time = time.time()
    
    # Training
    log_print('\tTraining...')
    for cur_training_epoch in range(training_epoch_num):
        prev_epoch_time = time.time()-cur_time
        cur_time = time.time()
        log_print('\t\tEpoch ' + str(cur_training_epoch) + ', prev epoch took ' + str(prev_epoch_time))
        cur_perm = np.random.permutation(len(training_set))
        for i in range(len(training_set)):
            log_print('\t\t\tTraining input ' + str(cur_perm[i]))
            true_label = cur_perm[i]
            input_vec = training_set[cur_perm[i]]
            
            fire_history = model.simulate_dynamics(input_vec,true_label)
            fire_count = [len(a) for a in fire_history]
            model.update_synapse_strength(fire_count)
            
    # Evaluating
    log_print('\tEvaluating...')
    correct_count = 0
    for input_ind in range(len(test_set)):
        log_print('\t\t\tEvaluating input ' + str(input_ind) + ' of ' + str(len(test_set)))
        cur_label = input_ind % len(training_set)
        input_vec = test_set[input_ind]
        
        fire_history = model.simulate_dynamics(input_vec,-1)
        fire_count = [len(a) for a in fire_history]
        
        correct = len([x for x in fire_count[sum(model.conf['layers_size'][:-1]):sum(model.conf['layers_size'])] if x > 0]) == 1 and \
            fire_count[sum(model.conf['layers_size'][:-1])+cur_label] > 0
        if correct:
            correct_count += 1
                
    log_print('\tCurrent correct count ' + str(correct_count))
    return correct_count

def log_print(my_str):
    if write_to_log:
        log_fp.write(my_str+ '\n')
        log_fp.flush()
    else:
        print(my_str)
        
write_to_log = True
if write_to_log:
    log_fp = open('log.txt','w')
    
trial_num = 10
correct_count = 0
for cur_trial in range(trial_num):
    log_print('Starting trial ' + str(cur_trial))
    correct_count += plot_precision_as_a_func_of_training_epoch_num()
log_print('Correct count average: ' + str(correct_count/trial_num))