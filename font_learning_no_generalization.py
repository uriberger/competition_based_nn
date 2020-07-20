import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import time
#from font_inputs import generate_training_set_no_generalization
from font_inputs_grayscale import generate_training_set_no_generalization

class ModelClass:
    
    default_configuration = {
        # Neuron parameters
        'excitatory_threshold' : 1,
        'inhibitory_threshold' : 1,
        'sensory_input_strength' : 1/65,
        'response_innervation_strength' : 0.01,
        'layers_size' : [400,512,62],
        
        'norm_shrink_factor' : 9000,
        
        # Normalization parameters
        'Z_iin_ex_th_ratio' : [0.01,1,1],
        'Z_intra_layer_ex_th_ratio' : 1,
        
        # Learning parameters
        'eta' : [0.01,0.01],
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
        
        if file_suffix == None:
            self.synapse_strength = []
            
            # Synapses into 1st layer
            self.synapse_strength.append([])
            self.synapse_strength[0].append(np.random.rand(self.conf['layers_size'][0], self.conf['layers_size'][0]))
            for _ in range(1,len(self.conf['layers_size'])):
                self.synapse_strength[0].append(None)
            
            # Synapses into other layers
            for l in range(1,len(self.conf['layers_size'])):
                self.synapse_strength.append([])
                for _ in range(l-1):
                    self.synapse_strength[l].append(None)
                self.synapse_strength[l].append(np.random.rand(self.conf['layers_size'][l], self.conf['layers_size'][l-1]))
                for _ in range(l,len(self.conf['layers_size'])):
                    self.synapse_strength[l].append(None)            
        else:
            file_name = "synapse_strength"
            file_name += "_" + file_suffix
            file_name += ".npy"
            
            trained_strength = np.load(file_name)
        
            self.synapse_strength = trained_strength[0]
            
        self.prev_act = np.zeros((unit_num, 1))
        self.prev_input_to_neurons = np.zeros((unit_num, 1))
        
        self.zero_matrix = np.zeros((self.conf['layers_size'][0],self.conf['layers_size'][0]))
        
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
        
        self.fix_synapse_strength()
            
    def init_normalization_parameters(self):
        ''' Generate winner number list.
        For the first two, sensory layers, we don't care so we'll put 1 winner.
        For the following layers (except the last one) we want half of the neurons to be winners,
        to maximize the number of possible combinations in the next layers.
        In the last layer we want a single winner, because each input corresponds to a single class.
        '''
        self.conf['winner_num'] = [1]
        for l in range(1,len(self.conf['layers_size'])-1):
            self.conf['winner_num'].append(90)
        self.conf['winner_num'].append(1)
        
        # Initialize normalization parameters
        self.conf['Z_iin'] = [x * self.conf['excitatory_threshold'] for x in self.conf['Z_iin_ex_th_ratio']]
        self.conf['Z_intra_layer'] = self.conf['Z_intra_layer_ex_th_ratio'] * self.conf['excitatory_threshold']

        self.conf['Z_vals'] = [self.conf['Z_intra_layer']]
        first = True
        for l in range(1,len(self.conf['layers_size'])):
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

        # Make sure intra-layer connections for input neurons are all of equal strength
        self.synapse_strength[0][0][:,:] = 1

        # Enforce invariants
        self.synapse_strength[0][0] = np.multiply(self.synapse_strength[0][0],self.zero_matrix)
        
        for post_l in range(0,len(self.conf['layers_size'])):
            for pre_l in range(0,len(self.conf['layers_size'])):
                if self.synapse_strength[post_l][pre_l] is None:
                    continue
                
                # Make sure excitatory weights are all excitatory
                self.synapse_strength[post_l][pre_l][self.synapse_strength[post_l][pre_l] < 0] = 0
        
                # Normalize incoming excitatory weights to each unit        
                weights_sums = (self.synapse_strength[post_l][pre_l].sum(axis=1).reshape(self.conf['layers_size'][post_l],1).repeat(self.conf['layers_size'][pre_l], axis=1))/self.conf['Z_vals'][post_l]
                weights_sums[weights_sums == 0] = 1
                self.synapse_strength[post_l][pre_l] = self.synapse_strength[post_l][pre_l]/weights_sums
        
    def update_synapse_strength(self, winners_losers_list, hidden_learning):
        for l in range(1 ,len(self.conf['layers_size'])):
            prev_layer_winners = winners_losers_list[sum(self.conf['layers_size'][:l-1]):sum(self.conf['layers_size'][:l]),[0]]
            cur_layer_winners = winners_losers_list[sum(self.conf['layers_size'][:l]):sum(self.conf['layers_size'][:l+1]),[0]]
            update_mat = np.matmul(cur_layer_winners,prev_layer_winners.transpose())
            if l == 1 and (not hidden_learning):
                continue
            self.synapse_strength[l][l-1] += update_mat*self.conf['eta'][l-1]*(self.conf['Z_vals'][l]/self.conf['layers_size'][l-1])
        
        self.fix_synapse_strength()
        
    def simulate_dynamics(self, input_vec, innervated_response_neuron):
        unit_num = sum(self.conf['layers_size']) + len(self.conf['layers_size'])
        
        fire_history = []
        for _ in range(unit_num):
            fire_history.append([])
            
        sensory_input_vec = np.pad(input_vec, ((0,unit_num-self.conf['layers_size'][0]),(0,0)), 'constant')
        if innervated_response_neuron != -1:
            sensory_input_vec[sum(self.conf['layers_size'][:-1])+innervated_response_neuron,0] = self.conf['response_innervation_strength']
        
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
        
        return fire_history,t
        
    def prop_external_input(self, sensory_input_vec):
        # Simulate the dynamics of the system for a single time step
        external_input = sensory_input_vec
        
        internal_input = np.zeros(external_input.shape)
        
        # Excitatory -> Excitatory
        for post_l in range(0,len(self.conf['layers_size'])):
            for pre_l in range(0,len(self.conf['layers_size'])):
                if self.synapse_strength[post_l][pre_l] is None:
                    continue
                pre_l_begin = sum(self.conf['layers_size'][:pre_l])
                pre_l_end = pre_l_begin + self.conf['layers_size'][pre_l]
                post_l_begin = sum(self.conf['layers_size'][:post_l])
                post_l_end = post_l_begin + self.conf['layers_size'][post_l]
                internal_input[post_l_begin:post_l_end,[0]] = np.matmul(self.synapse_strength[post_l][pre_l],self.prev_act[pre_l_begin:pre_l_end,[0]])
                
        for l in range(0,len(self.conf['layers_size'])):
            l_begin = sum(self.conf['layers_size'][:l])
            l_end = l_begin + self.conf['layers_size'][l]
            l_iin_index = sum(self.conf['layers_size']) + l
            
            # Inhibitory -> Excitatory
            internal_input[l_begin:l_end,[0]] += (-1)*self.conf['Z_iin'][l]*self.prev_act[l_iin_index,0]*np.ones((self.conf['layers_size'][l],1))
            
            # Excitatory -> Inhibitory
            Z_iin_from_single_ex = self.conf['Z_ex_to_iins'][l]/self.conf['layers_size'][l]
            internal_input[l_iin_index] = sum(Z_iin_from_single_ex*self.prev_act[l_begin:l_end,[0]])
        
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
    
    def calculate_winners(self, input_vec, cur_fire_count):
        # Sensory winners
        sensory_winners = np.array([[1 if input_vec[x,0]>0.5*self.conf['sensory_input_strength'] else 0 for x in range(input_vec.shape[0])]]).transpose()
        #sensory_winners = np.array([[1 if cur_fire_count[i]>=0.5*cur_fire_count[sum(self.conf['layers_size'])] else 0 for i in range(self.conf['layers_size'][0])]]).transpose()
        winner_loser_list = sensory_winners
        
        # Other layers winners
        for l in range(1,len(self.conf['layers_size'])):
            cur_layer_winners = np.array([[1 if cur_fire_count[sum(self.conf['layers_size'][:l])+i]>=0.5*cur_fire_count[sum(self.conf['layers_size'])+l] else 0 for i in range(self.conf['layers_size'][l])]]).transpose()
            winner_loser_list = np.concatenate((winner_loser_list,cur_layer_winners))
        
        return winner_loser_list
    
def plot_accuracy_as_a_func_of_training_iter():
    training_set = generate_training_set_no_generalization('calibri',ModelClass.default_configuration['sensory_input_strength'])
    test_set = training_set
    
    training_iter_num = 100
    correct_counts = []
    avg_sim_len_training = []
    avg_sim_len_test = []
    avg_sim_len_test_correct = []
    avg_sim_len_test_incorrect = []
    
    model = ModelClass({},None,True)
    cur_time = time.time()
    
    correct_count = 0
    
    # Training
    log_print('\tTraining...')
    for cur_training_iter in range(training_iter_num):
        prev_iter_time = time.time()-cur_time
        cur_time = time.time()
        log_print('\t\tTraining iter ' + str(cur_training_iter) + ', prev iter took ' + str(prev_iter_time))
        cur_perm = np.random.permutation(len(training_set))
        total_sim_len = 0
        for i in range(len(training_set)):
            true_label = cur_perm[i]
            input_vec = training_set[cur_perm[i]]
            
            fire_history,simulation_len = model.simulate_dynamics(input_vec,true_label)
            fire_count = [len(a) for a in fire_history]
            winners_losers_list = model.calculate_winners(input_vec,fire_count)
            model.update_synapse_strength(winners_losers_list, False)
            
            total_sim_len += simulation_len
            
        avg_sim_len_training.append(total_sim_len/len(training_set))
            
        # Evaluating        
        log_print('\t\tEvaluating...')
        correct_count = 0
        total_sim_len = 0
        total_sim_len_correct = 0
        total_sim_len_incorrect = 0
        confusion_matrix = np.zeros((model.conf['layers_size'][-1],model.conf['layers_size'][-1]))
        if cur_training_iter > (training_iter_num-4):
            for input_ind in range(len(test_set)):
                cur_label = input_ind % len(training_set)
                input_vec = test_set[input_ind]
                
                fire_history,simulation_len = model.simulate_dynamics(input_vec,-1)
                fire_count = [len(a) for a in fire_history]
                
                total_sim_len += simulation_len
                
                correct = len([x for x in fire_count[sum(model.conf['layers_size'][:-1]):sum(model.conf['layers_size'])] if x > 0]) == 1 and \
                    fire_count[sum(model.conf['layers_size'][:-1])+cur_label] > 0
                if correct:
                    correct_count += 1
                    total_sim_len_correct += simulation_len
                else:
                    total_sim_len_incorrect += simulation_len
                    
                winners_losers_list = model.calculate_winners(input_vec,fire_count)
                res_start = sum(model.conf['layers_size'][:-1])
                for res_unit_ind in range(confusion_matrix.shape[0]):
                    if winners_losers_list[res_start+res_unit_ind] == 1:
                        confusion_matrix[input_ind, res_unit_ind] = 1
                    
        log_print('\tCurrent correct count ' + str(correct_count))
        
        correct_counts.append(correct_count)
        avg_sim_len_test.append(total_sim_len/len(test_set))
        if correct_count > 0:
            avg_sim_len_test_correct.append(total_sim_len_correct/correct_count)
        else:
            avg_sim_len_test_correct.append(0)
        if correct_count < len(test_set):
            avg_sim_len_test_incorrect.append(total_sim_len_incorrect/(len(test_set)-correct_count))
        else:
            avg_sim_len_test_incorrect.append(0)
    plt.plot(range(training_iter_num),correct_counts)
    plt.savefig('res')
    plt.clf()
    plt.plot(range(training_iter_num),avg_sim_len_training)
    plt.savefig('res_sim_len_training')
    plt.clf()
    plt.plot(range(training_iter_num),avg_sim_len_test)
    plt.savefig('res_sim_len_test')
    plt.clf()
    plt.plot(range(training_iter_num),avg_sim_len_test_correct)
    plt.savefig('res_sim_len_test_correct')
    plt.clf()
    plt.plot(range(training_iter_num),avg_sim_len_test_incorrect)
    plt.savefig('res_sim_len_test_incorrect')
    
    plt.clf()
    plt.imshow(confusion_matrix,cmap='gray')
    plt.savefig('res_confusion')
    
    model.save_synapse_strength('good')
    
def evaluate_on_new_font(new_font_name, model_name_suffix):
    
    model = ModelClass({},model_name_suffix,True)
    test_set = generate_training_set_no_generalization(new_font_name,ModelClass.default_configuration['sensory_input_strength'])
    
    # Evaluating        
    log_print('Evaluating...')
    correct_count = 0
    for input_ind in range(len(test_set)):
        log_print('\tInput ' + str(input_ind))
        cur_label = input_ind % len(test_set)
        input_vec = test_set[input_ind]
        
        fire_history,_ = model.simulate_dynamics(input_vec,-1)
        fire_count = [len(a) for a in fire_history]
        
        correct = len([x for x in fire_count[sum(model.conf['layers_size'][:-1]):sum(model.conf['layers_size'])] if x > 0]) == 1 and \
            fire_count[sum(model.conf['layers_size'][:-1])+cur_label] > 0
        if correct:
            correct_count += 1
    
    log_print('Correct count ' + str(correct_count))
    
def print_sim_len_as_a_function_of_iter_for_ob_num(ob_num):
    training_set = generate_training_set_no_generalization('calibri',ModelClass.default_configuration['sensory_input_strength'])
    
    experiment_iter_num = 100
    training_iter_num = 50
    
    model = ModelClass({},None,True)
    
    all_experiments_sim_len_training = [0]*training_iter_num
    all_experiments_sim_len_test = [0]*training_iter_num
    all_experiments_sim_len_test_correct = [0]*training_iter_num
    all_experiments_sim_len_test_incorrect = [0]*training_iter_num
    for cur_experiment_iter in range(experiment_iter_num):
        cur_perm = np.random.permutation(len(training_set))
        reduced_training_set = [training_set[x] for x in cur_perm][:ob_num]
        reduced_test_set = reduced_training_set
        log_print('Experiment iter ' + str(cur_experiment_iter))
        for cur_training_iter in range(training_iter_num):
            total_sim_len = 0
            for i in range(len(reduced_training_set)):
                true_label = cur_perm[i]
                input_vec = training_set[cur_perm[i]]
                
                fire_history,simulation_len = model.simulate_dynamics(input_vec,true_label)
                fire_count = [len(a) for a in fire_history]
                model.update_synapse_strength(fire_count)
                
                total_sim_len += simulation_len
                
            all_experiments_sim_len_training[cur_training_iter] += total_sim_len/len(training_set)
                
            # Evaluating
            correct_count = 0
            total_sim_len = 0
            total_sim_len_correct = 0
            total_sim_len_incorrect = 0
            for input_ind in range(len(reduced_test_set)):
                cur_label = input_ind % len(reduced_training_set)
                input_vec = reduced_test_set[input_ind]
                
                fire_history,simulation_len = model.simulate_dynamics(input_vec,-1)
                fire_count = [len(a) for a in fire_history]
                
                total_sim_len += simulation_len
                
                correct = len([x for x in fire_count[sum(model.conf['layers_size'][:-1]):sum(model.conf['layers_size'])] if x > 0]) == 1 and \
                    fire_count[sum(model.conf['layers_size'][:-1])+cur_label] > 0
                if correct:
                    correct_count += 1
                    total_sim_len_correct += simulation_len
                else:
                    total_sim_len_incorrect += simulation_len
                        
            all_experiments_sim_len_test[cur_training_iter] += total_sim_len/len(reduced_test_set)
            if correct_count > 0:
                all_experiments_sim_len_test_correct[cur_training_iter] += total_sim_len_correct/correct_count
            if correct_count < len(reduced_test_set):
                all_experiments_sim_len_test_incorrect[cur_training_iter] += total_sim_len_incorrect/(len(reduced_test_set)-correct_count)
            
    print('\nAverage simulation length during training:')
    for cur_avg_sim_len in all_experiments_sim_len_training:
        print(str(cur_avg_sim_len/experiment_iter_num))
    print('\nAverage simulation length during test:')
    for cur_avg_sim_len in all_experiments_sim_len_test:
        print(str(cur_avg_sim_len/experiment_iter_num))
    print('\nAverage simulation length during test correct:')
    for cur_avg_sim_len in all_experiments_sim_len_test_correct:
        print(str(cur_avg_sim_len/experiment_iter_num))
    print('\nAverage simulation length during test incorrect:')
    for cur_avg_sim_len in all_experiments_sim_len_test_incorrect:
        print(str(cur_avg_sim_len/experiment_iter_num))

def plot_hidden_layer_sensitivity(model_name_suffix):
    model = ModelClass({},model_name_suffix,True)

    rows = 4
    cols = 4
    fig=plt.figure(figsize=(8, 8))
    for k in range(16):
        fig.add_subplot(rows, cols, k+1)
        plt.imshow(np.reshape(model.synapse_strength[1][0][10+k,:],(20,20)),cmap='gray')
    plt.subplots_adjust(hspace=0.5)
    plt.show()

def log_print(my_str):
    if write_to_log:
        log_fp.write(my_str+ '\n')
        log_fp.flush()
    else:
        print(my_str)
        
write_to_log = False
if write_to_log:
    log_fp = open('log.txt','w')
    
plot_accuracy_as_a_func_of_training_iter()
#print_sim_len_as_a_function_of_iter_for_ob_num(1)
#evaluate_on_new_font('times_new_roman','good')
#evaluate_on_new_font('calibri','good')
#plot_hidden_layer_sensitivity('good')