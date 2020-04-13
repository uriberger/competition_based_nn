import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

''' Paper 1, Experiment 3:
The purpose of this experiment is to show that the network can learn to identifty objects no matter
where they are located.
'''

class ModelClass:
    
    default_configuration = {
        # Neuron parameters
        'excitatory_threshold' : 1,
        'inhibitory_threshold' : 1,
        'sensory_input_strength' : 1/3,
        'response_innervation_strength' : 0.01,
        'layers_size' : [100,512,4],
        
        # Winner num
        'winner_num' : [12,2,1],
        
        # Normalization parameters
        'Z_iin_ex_th_ratio' : 1,
        'Z_intra_layer_ex_th_ratio' : 1,
        'max_average_strength_ratio' : 1.1,
        
        # Learning parameters
        'eta' : 0.001,
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
            self.synapse_strength = np.ones((unit_num, unit_num))
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
        
        # Input to other layers
        for l in range(1,len(self.conf['layers_size'])):
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
        
        if file_suffix == None:
            # Add some variance to the weights
            noise_reduce_factor = 10
            
            for l in range(1,len(self.conf['layers_size'])):
                self.synapse_strength[sum(self.conf['layers_size'][:l]):sum(self.conf['layers_size'][:l+1]),sum(self.conf['layers_size'][:l-1]):sum(self.conf['layers_size'][:l])] += np.random.normal(0,self.conf['Z_vals'][l-1]/(noise_reduce_factor*self.conf['layers_size'][l-1]),(self.conf['layers_size'][l],self.conf['layers_size'][l-1]))
            
            self.fix_synapse_strength()
            
    def init_normalization_parameters(self):
        # Initialize normalization parameters
        self.conf['Z_iin'] = self.conf['Z_iin_ex_th_ratio'] * self.conf['excitatory_threshold']
        self.conf['Z_intra_layer'] = self.conf['Z_intra_layer_ex_th_ratio'] * self.conf['excitatory_threshold']

        self.conf['Z_vals'] = []
        for l in range(1,len(self.conf['layers_size'])):
            self.conf['Z_vals'].append((self.conf['excitatory_threshold'] * self.conf['layers_size'][l-1])/(10*self.conf['winner_num'][l-1]))
        
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
        
        # Enforce invariants
        self.synapse_strength = np.multiply(self.synapse_strength,self.zero_matrix)
        
        # Make sure excitatory weights are all excitatory
        self.synapse_strength[:,:excitatory_unit_num][self.synapse_strength[:,:excitatory_unit_num] < 0] = 0
        
        # Normalize incoming excitatory weights to each unit    
        
        # Input to 1st layer
        input_to_input_row_sum = (self.synapse_strength[:self.conf['layers_size'][0],:self.conf['layers_size'][0]].sum(axis=1).reshape(self.conf['layers_size'][0],1).repeat(self.conf['layers_size'][0], axis=1))/self.conf['Z_intra_layer']
        rest_to_input_row_sum = np.ones((self.conf['layers_size'][0],sum(self.conf['layers_size'])-self.conf['layers_size'][0]))
        input_row_sum = np.concatenate((input_to_input_row_sum,rest_to_input_row_sum),axis=1)
        
        excitatory_row_sums = input_row_sum
        
        for l in range(1,len(self.conf['layers_size'])):
            # Input to EINs
            input_from_earlier_layers = np.ones((self.conf['layers_size'][l],sum(self.conf['layers_size'][:l-1])))
            input_from_prev_layer = (self.synapse_strength[sum(self.conf['layers_size'][:l]):sum(self.conf['layers_size'][:l+1]),sum(self.conf['layers_size'][:l-1]):sum(self.conf['layers_size'][:l])].sum(axis=1).reshape(self.conf['layers_size'][l],1).repeat(self.conf['layers_size'][l-1], axis=1))/self.conf['Z_vals'][l-1]
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
        inhibitory_row_sums = (-1)*((self.synapse_strength[:,excitatory_unit_num:].sum(axis=1).reshape(sum(self.conf['layers_size'])+len(self.conf['layers_size']),1).repeat(len(self.conf['layers_size']), axis=1))/self.conf['Z_iin'])
        
        row_sums = np.concatenate((excitatory_row_sums,inhibitory_row_sums),axis=1)
        row_sums[row_sums == 0] = 1
        self.synapse_strength = self.synapse_strength/row_sums
        
    def update_synapse_strength(self, cur_fire_count):
        for l in range(1,len(self.conf['layers_size'])):
            prev_layer_winners = np.array([[1 if cur_fire_count[sum(self.conf['layers_size'][:l-1])+i]>=0.5*cur_fire_count[sum(self.conf['layers_size'])+l-1] else 0 for i in range(self.conf['layers_size'][l-1])]]).transpose()
            cur_layer_winners = np.array([[1 if cur_fire_count[sum(self.conf['layers_size'][:l])+i]>=0.5*cur_fire_count[sum(self.conf['layers_size'])+l] else 0 for i in range(self.conf['layers_size'][l])]]).transpose()
            update_mat = np.matmul(2*(cur_layer_winners-0.5),prev_layer_winners.transpose())
            self.synapse_strength[sum(self.conf['layers_size'][:l]):sum(self.conf['layers_size'][:l+1]),sum(self.conf['layers_size'][:l-1]):sum(self.conf['layers_size'][:l])] += update_mat*self.conf['eta']*(self.conf['Z_vals'][l-1]/self.conf['layers_size'][l-1])
        
        self.fix_synapse_strength()
        
        # Make sure no excitatory weight is more than the maximal possible weight
        for l in range(1,len(self.conf['layers_size'])):
            max_val = self.conf['max_average_strength_ratio']*(self.conf['Z_vals'][l-1]/self.conf['layers_size'][l-1])
            self.synapse_strength[sum(self.conf['layers_size'][:l]):sum(self.conf['layers_size'][:l+1]),sum(self.conf['layers_size'][:l-1]):sum(self.conf['layers_size'][:l])][self.synapse_strength[sum(self.conf['layers_size'][:l]):sum(self.conf['layers_size'][:l+1]),sum(self.conf['layers_size'][:l-1]):sum(self.conf['layers_size'][:l])] > max_val] = max_val
        
        self.fix_synapse_strength()
        
    def simulate_dynamics(self, input_vec, innervated_response_neuron):
        unit_num = sum(self.conf['layers_size']) + len(self.conf['layers_size'])
        
        fire_history = []
        for _ in range(unit_num):
            fire_history.append([])
            
        sensory_input_vec = np.pad(input_vec, ((0,unit_num-self.conf['layers_size'][0]),(0,0)), 'constant')
        if innervated_response_neuron != -1:
            sensory_input_vec[sum(self.conf['layers_size'][:-1])+innervated_response_neuron,0] = self.conf['response_innervation_strength']
        
        #for t in range(T):
        t = 0
        while True:
            if t % 1000 == 0:
                self.my_print('t='+str(t))
                
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

input_mat1 = np.array([
    [1,1,1,0,],
    [0,0,0,1,],
    [0,0,1,0,],
    [0,1,0,0,],
    [0,0,1,0,],
    [0,0,0,1,],
    [1,1,1,0,],
    ])
    
input_mat2 = np.array([
    [1,0,0,1,],
    [1,0,0,1,],
    [1,0,0,1,],
    [0,1,1,1,],
    [0,0,0,1,],
    [0,0,0,1,],
    [0,0,0,1,],
    ])
    
input_mat3 = np.array([
    [0,1,1,0,],
    [1,0,0,1,],
    [1,0,0,0,],
    [1,1,1,0,],
    [1,0,0,1,],
    [1,0,0,1,],
    [0,1,1,0,],
    ])
    
input_mat4 = np.array([
    [0,1,1,0,],
    [1,0,0,1,],
    [0,0,0,1,],
    [0,0,1,0,],
    [0,1,0,0,],
    [1,0,0,0,],
    [1,1,1,1,],
    ])

input_mats = [input_mat1,input_mat2,input_mat3,input_mat4]
    
N = round(ModelClass.default_configuration['layers_size'][0]**0.5)
input_height, input_width = input_mat1.shape
hori_location_num = N-input_width+1
verti_location_num = N-input_height+1
input_vecs = []
#for start_ind in range(hori_location_num*verti_location_num):
for start_ind in range(1):
    start_point_hori_location = start_ind % hori_location_num
    start_point_verti_location = int(start_ind / hori_location_num)
    after_end_hori_pad = N-start_point_hori_location-input_width
    after_end_verti_pad = N-start_point_verti_location-input_height
    for input_mat in input_mats:
        full_input_mat = np.pad(input_mat,((start_point_verti_location,after_end_verti_pad),(start_point_hori_location,after_end_hori_pad)),mode='constant')
        input_vecs.append(np.reshape(full_input_mat,(N**2,1))*ModelClass.default_configuration['sensory_input_strength'])

def plot_inputs():
    # Plot the inputs    
    fig = plt.figure()
    fig.add_subplot(2,2,1)
    a1 = plt.imshow(input_mat1,cmap='binary')
    a1.axes.get_xaxis().set_visible(False)
    a1.axes.get_yaxis().set_visible(False)
    plt.title('input1')
    fig.add_subplot(2,2,2)
    a2 = plt.imshow(input_mat2,cmap='binary')
    a2.axes.get_xaxis().set_visible(False)
    a2.axes.get_yaxis().set_visible(False)
    plt.title('input2')
    fig.add_subplot(2,2,3)
    a3 = plt.imshow(input_mat3,cmap='binary')
    a3.axes.get_xaxis().set_visible(False)
    a3.axes.get_yaxis().set_visible(False)
    plt.title('input3')
    fig.add_subplot(2,2,4)
    a4 = plt.imshow(input_mat4,cmap='binary')
    a4.axes.get_xaxis().set_visible(False)
    a4.axes.get_yaxis().set_visible(False)
    plt.title('input4')
    plt.show()

def plot_precision_as_a_func_of_training_epoch_num():
    training_epoch_num = 100
    
    experiment_iter_num = 1
    correct_sums = [0]*training_epoch_num
    for cur_experiment_iter in range(experiment_iter_num):
        if cur_experiment_iter % 10 == 0:
            print('\tIter ' + str(cur_experiment_iter))
        model = ModelClass({},None,True)
        for cur_training_epoch in range(training_epoch_num):
            print('\t\tEpoch ' + str(cur_training_epoch))
            # Training
            cur_perm = np.random.permutation(len(input_vecs))
            for i in range(len(input_vecs)):
                true_label = cur_perm[i]%4
                input_vec = input_vecs[cur_perm[i]]
                
                fire_history = model.simulate_dynamics(input_vec,true_label)
                fire_count = [len(a) for a in fire_history]
                model.update_synapse_strength(fire_count)
            # Evaluating
            correct_count = 0
            for input_ind in range(len(input_vecs)):
                cur_label = input_ind % 4
                input_vec = input_vecs[cur_label]
                
                fire_history = model.simulate_dynamics(input_vec,-1)
                fire_count = [len(a) for a in fire_history]
                
                correct = False
                if fire_count[sum(model.conf['layers_size'][:-1])+cur_label] > 0:
                    correct = True
                    for unit_ind in range(model.conf['layers_size'][-1]):
                        if unit_ind == cur_label:
                            continue
                        if fire_count[sum(model.conf['layers_size'][:-1])+unit_ind] > 0:
                            correct = False
                            break
                    if correct:
                        correct_count += 1
            correct_sums[cur_training_epoch] += correct_count
                
    precisions = [(x/experiment_iter_num)/len(input_vecs) for x in correct_sums]
    plt.plot(range(training_epoch_num), precisions)
    plt.xlabel('Epoch number')
    plt.ylabel('Average precision')
    plt.show()
    
#plot_inputs()
plot_precision_as_a_func_of_training_epoch_num()