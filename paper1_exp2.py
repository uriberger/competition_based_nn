import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

''' Paper 1, Experiment 2:
The purpose of this experiment is a simple learning task- we want to show that the network can learn
to identify several objects.
'''

class ModelClass:
    
    default_configuration = {
        # Neuron parameters
        'excitatory_threshold' : 1,
        'inhibitory_threshold' : 1,
        'sensory_input_strength' : 1/3,
        'layers_size' : [100,10],
        
        # Winner num
        'winner_num' : [9,1],
        
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
        
        unit_num = sum(self.conf['layers_size']) + len(self.conf['layers_size'])
        excitatory_unit_num = sum(self.conf['layers_size'])
        
        if file_suffix == None:
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
        
    def simulate_dynamics(self, input_vec):
        unit_num = sum(self.conf['layers_size']) + len(self.conf['layers_size'])
        
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
            for l in range(len(self.conf['layers_size'])):
                if len(fire_history[sum(self.conf['layers_size'])+l]) == 0:
                    all_inn_fired = False
                    break
            if all_inn_fired:
                break
        
        self.reset_data_structures()
        
        return fire_history
        
    def prop_external_input(self, sensory_input_vec):
        unit_num = sum(self.conf['layers_size']) + len(self.conf['layers_size'])
        
        # Simulate the dynamics of the system for a single time step
        cur_input = np.zeros((unit_num, 1))
        input_from_prev_layer = np.pad(sensory_input_vec, ((0,unit_num-self.conf['layers_size'][0]),(0,0)), 'constant')
        cur_input = np.add(cur_input, input_from_prev_layer)
        input_from_pre_layer = np.matmul(self.synapse_strength, self.prev_act)
        cur_input = np.add(cur_input, input_from_pre_layer)
        
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

input_vec1 = np.array([[
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,1,1,1,0,0,0,
    0,0,0,0,1,1,1,0,0,0,
    0,0,0,0,1,1,1,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    ]]).transpose()*ModelClass.default_configuration['sensory_input_strength']
    
input_vec2 = np.array([[
    1,1,1,1,1,1,1,1,1,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    ]]).transpose()*ModelClass.default_configuration['sensory_input_strength']
    
input_vec3 = np.array([[
    0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,
    ]]).transpose()*ModelClass.default_configuration['sensory_input_strength']
    
input_vec4 = np.array([[
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,
    1,1,1,1,1,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,
    ]]).transpose()*ModelClass.default_configuration['sensory_input_strength']
    
input_vecs = [input_vec1,input_vec2,input_vec3,input_vec4]

def plot_inputs():
    # Plot the inputs    
    fig = plt.figure()
    fig.add_subplot(2,2,1)
    a1 = plt.imshow(np.reshape(input_vec1,(10,10)),cmap='binary')
    a1.axes.get_xaxis().set_visible(False)
    a1.axes.get_yaxis().set_visible(False)
    plt.title('input1')
    fig.add_subplot(2,2,2)
    a2 = plt.imshow(np.reshape(input_vec2,(10,10)),cmap='binary')
    a2.axes.get_xaxis().set_visible(False)
    a2.axes.get_yaxis().set_visible(False)
    plt.title('input2')
    fig.add_subplot(2,2,3)
    a3 = plt.imshow(np.reshape(input_vec3,(10,10)),cmap='binary')
    a3.axes.get_xaxis().set_visible(False)
    a3.axes.get_yaxis().set_visible(False)
    plt.title('input3')
    fig.add_subplot(2,2,4)
    a4 = plt.imshow(np.reshape(input_vec4,(10,10)),cmap='binary')
    a4.axes.get_xaxis().set_visible(False)
    a4.axes.get_yaxis().set_visible(False)
    plt.title('input4')
    plt.show()

# Winner number
def plot_winner_num_as_a_func_of_response_num():
    response_nums = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
    average_winner_nums = []
    for _ in range(len(input_vecs)):
        average_winner_nums.append([])
    average_sensory_winner_nums = []
    for _ in range(len(input_vecs)):
        average_sensory_winner_nums.append([])
    
    iter_num = 1000
    for response_num in response_nums:
        print('Starting with ' + str(response_num) + ' response neurons')
        winner_num_sums = [0]*len(input_vecs)
        sensory_winner_num_sums = [0]*len(input_vecs)
        for cur_iter in range(iter_num):
            if cur_iter % 100 == 0:
                print('\tIter ' + str(cur_iter))
            for input_idx in range(len(input_vecs)):
                model = ModelClass({'layers_size' : [100,response_num]},None,True)
                fire_history = model.simulate_dynamics(input_vecs[input_idx])
                fire_count = [len(a) for a in fire_history]
                winner_num_sums[input_idx] += len([x for x in range(response_num) if fire_count[sum(model.conf['layers_size'][:-1])+x]>0])
                sensory_winner_num_sums[input_idx] += len([x for x in range(model.conf['layers_size'][0]) if fire_count[x]>0])
        for input_idx in range(len(input_vecs)):
            average_winner_num = winner_num_sums[input_idx]/iter_num
            print('Winner num average for input ' + str(input_idx) + ': ' + str(average_winner_num))
            average_winner_nums[input_idx].append(average_winner_num)
            average_sensory_winner_num = sensory_winner_num_sums[input_idx]/iter_num
            print('Sensory winner num average for input ' + str(input_idx) + ': ' + str(average_sensory_winner_num))
            average_sensory_winner_nums[input_idx].append(average_sensory_winner_num)
    for input_idx in range(len(input_vecs)):
        plt.plot(response_nums,average_winner_nums[input_idx],label='input '+str(input_idx+1))
    plt.legend()
    plt.xlabel('Number of response neurons')
    plt.ylabel('Average number of winners')
    plt.show()
    
    for input_idx in range(len(input_vecs)):
        plt.plot(response_nums,average_sensory_winner_nums[input_idx],label='input '+str(input_idx+1))
    plt.legend()
    plt.xlabel('Number of response neurons')
    plt.ylabel('Average number of sensory winners')
    plt.show()
    
def plot_winner_num_as_a_func_of_layer_num():
    response_num = 10
    
    layer_nums = [2,3,4,5]
    average_winner_nums = []
    for _ in range(len(input_vecs)):
        average_winner_nums.append([])
    
    iter_num = 1000
    for layer_num in layer_nums:
        print('Starting with ' + str(layer_num) + ' layers')
        winner_num_sums = [0]*len(input_vecs)
        for cur_iter in range(iter_num):
            if cur_iter % 100 == 0:
                print('\tIter ' + str(cur_iter))
            for input_idx in range(len(input_vecs)):
                layers_size = [100]+[response_num]*(layer_num-1)
                winner_num = [9]+[1]*(layer_num-1)
                model = ModelClass({'layers_size' : layers_size,'winner_num' : winner_num},None,True)
                fire_history = model.simulate_dynamics(input_vecs[input_idx])
                fire_count = [len(a) for a in fire_history]
                winner_num_sums[input_idx] += len([x for x in range(response_num) if fire_count[sum(model.conf['layers_size'][:-1])+x]>0])
        for input_idx in range(len(input_vecs)):
            average_winner_num = winner_num_sums[input_idx]/iter_num
            print('Winner num average for input ' + str(input_idx) + ': ' + str(average_winner_num))
            average_winner_nums[input_idx].append(average_winner_num)
    for input_idx in range(len(input_vecs)):
        plt.plot(layer_nums,average_winner_nums[input_idx],label='input '+str(input_idx+1))
    plt.legend()
    plt.xlabel('Number of layers')
    plt.ylabel('Average number of winners')
    plt.show()
    
def plot_winner_num_as_a_func_of_prev_layer_winner_num():
    response_num = 10
    
    prev_layer_winner_nums = [1,2,3,4,5,6,7,8,9,10]
    average_winner_nums = []
    for _ in range(len(input_vecs)):
        average_winner_nums.append([])
    
    iter_num = 1000
    for prev_layer_winner_num in prev_layer_winner_nums:
        print('Starting with ' + str(prev_layer_winner_num) + ' winners')
        winner_num_sums = [0]*len(input_vecs)
        for cur_iter in range(iter_num):
            if cur_iter % 100 == 0:
                print('\tIter ' + str(cur_iter))
            for input_idx in range(len(input_vecs)):
                layers_size = [100,response_num,response_num]
                winner_num = [9,prev_layer_winner_num,1]
                model = ModelClass({'layers_size' : layers_size,'winner_num' : winner_num},None,True)
                fire_history = model.simulate_dynamics(input_vecs[input_idx])
                fire_count = [len(a) for a in fire_history]
                winner_num_sums[input_idx] += len([x for x in range(response_num) if fire_count[sum(model.conf['layers_size'][:-1])+x]>0])
        for input_idx in range(len(input_vecs)):
            average_winner_num = winner_num_sums[input_idx]/iter_num
            print('Winner num average for input ' + str(input_idx) + ': ' + str(average_winner_num))
            average_winner_nums[input_idx].append(average_winner_num)
    for input_idx in range(len(input_vecs)):
        plt.plot(prev_layer_winner_nums,average_winner_nums[input_idx],label='input '+str(input_idx+1))
    plt.legend()
    plt.xlabel('Number of winners in previous layer')
    plt.ylabel('Number of winners in response layer')
    plt.show()
    
# Winner intersection
def plot_winner_intersection_as_a_func_of_response_num():
    response_nums = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
    average_ratios = []
    
    iter_num = 1000
    for response_num in response_nums:
        print('Starting with ' + str(response_num) + ' response neurons winners')
        ratio_sum = 0
        for cur_iter in range(iter_num):
            if cur_iter % 100 == 0:
                print('\tIter ' + str(cur_iter))
            winner_list = []
            for input_idx in range(len(input_vecs)):
                layers_size = [100,response_num]
                model = ModelClass({'layers_size' : layers_size},None,True)
                fire_history = model.simulate_dynamics(input_vecs[input_idx])
                fire_count = [len(a) for a in fire_history]
                winner_list += [x for x in range(response_num) if fire_count[sum(model.conf['layers_size'][:-1])+x]>0]
            union_size = len(winner_list)
            unique_winners_num = len(list(set(winner_list)))
            ratio_sum += unique_winners_num/union_size
        average_ratio = ratio_sum/iter_num
        print('Winner ratio: ' + str(average_ratio))
        average_ratios.append(average_ratio)
    plt.plot(response_nums,average_ratios)
    plt.xlabel('Number of response neurons')
    plt.ylabel('Average unique-total winner num ratio')
    plt.show()
        
#plot_inputs()
#plot_winner_num_as_a_func_of_response_num()
#plot_winner_num_as_a_func_of_layer_num()
#plot_winner_num_as_a_func_of_prev_layer_winner_num()
plot_winner_intersection_as_a_func_of_response_num()