import numpy as np

layer_num = 1
comp_len = 20000
sample_num = 1000
normal_firing_rate = 0.005

class ModelClass:
    
    def __init__(self, configuration,init_from_file):
        # Members determined by the configuration
        self.Z_ex = configuration['Z_ex']
        self.Z_inhib = configuration['Z_inhib']
        self.Z_from_inp = configuration['Z_from_inp']
        self.Z_inter_layer = configuration['Z_inter_layer']
        self.excitatory_precentage = configuration['excitatory_precentage']
        self.inp_num = configuration['inp_num']
        self.out_num = configuration['out_num']
        self.excitatory_threshold = configuration['excitatory_threshold']
        self.inhibitory_threshold = configuration['inhibitory_threshold']
        self.eta = configuration['eta']
        self.gamma_ex = configuration['gamma_ex']
        self.gamma_in = configuration['gamma_in']
        self.quiet = configuration['quiet']
        
        # Other members
        self.ex_num = self.inp_num + self.out_num
        self.unit_num = int(round(self.ex_num/self.excitatory_precentage))
        self.iin_num = int(round((1-self.excitatory_precentage)*self.unit_num))
        self.Z_from_out = 1-configuration['Z_from_inp']
        
        # Data structures
        self.synapse_strength = []
        self.zero_matrices = []
        self.before_prev_act = []
        self.prev_act = []
        self.prev_input = []
        self.init_data_structures(init_from_file)
        
    def my_print(self, my_str):
        if not self.quiet:
            print(my_str)
    
    def init_data_structures(self,init_from_file):
        if init_from_file:
            file_name = "synapse_strength.npy"
            trained_strength = np.load(file_name)
            self.synapse_strength = trained_strength[0]
        else:
            # Initialize random synapse strength
            for post_layer in range(layer_num):
                post_layer_synapses = []
                for pre_layer in range(layer_num):
                    cur_synapse_strength = np.random.rand(self.unit_num, self.unit_num)
                    if pre_layer == post_layer:
                        # Inhibitory neurons only effect neurons in the same layer
                        cur_synapse_strength[:, -self.iin_num:] = (-1) * cur_synapse_strength[:, -self.iin_num:]
                    post_layer_synapses.append(cur_synapse_strength)
                self.synapse_strength.append(post_layer_synapses)
            
        for _ in range(layer_num):
            self.prev_act.append(np.zeros((self.unit_num, 1)))
        
        for _ in range(layer_num):
            self.before_prev_act.append(np.zeros((self.unit_num, 1)))
        
        for _ in range(layer_num):
            self.prev_input.append(np.zeros((self.unit_num, 1)))
        
        for post_layer in range(layer_num):
            self.zero_matrices.append([])
            for pre_layer in range(layer_num):
                zero_matrix = np.zeros((self.unit_num,self.unit_num))
                if pre_layer == post_layer:
                    zero_matrix[:,:] = 1
                    # No self loops
                    np.fill_diagonal(zero_matrix, 0)
                if pre_layer == post_layer-1 or pre_layer == post_layer+1:
                    ''' Each excitatory neuron innervates the corresponding excitatory neuron in
                    adjacent layers'''
                    for unit_ind in range(self.ex_num):
                        zero_matrix[unit_ind,unit_ind] = 1
                            
                self.zero_matrices[post_layer].append(zero_matrix)
        
        self.fix_synapse_strength()
    
    def fix_synapse_strength(self):
        # Normalize the synapses strength, and enforce the invariants.
        for post_layer in range(layer_num):
            for pre_layer in range(layer_num):
                self.synapse_strength[post_layer][pre_layer] = np.multiply(self.synapse_strength[post_layer][pre_layer], self.zero_matrices[post_layer][pre_layer])
                if post_layer == pre_layer+1 or post_layer == pre_layer-1:
                    normalized_weight = self.Z_inter_layer
                    for unit_ind in range(self.ex_num):
                        self.synapse_strength[post_layer][pre_layer][unit_ind,unit_ind] = normalized_weight
                if post_layer == pre_layer:
                    # Make sure excitatory intra-layer weights are all excitatory
                    self.synapse_strength[post_layer][pre_layer][:,:self.ex_num][self.synapse_strength[post_layer][pre_layer][:,:self.ex_num] < 0] = 0
                    
                    # Normalize incoming excitatory weights to each unit
                    inp_unit_begin = 0
                    out_unit_begin = inp_unit_begin + self.inp_num
                    
                    from_inp_row_sum = (self.synapse_strength[post_layer][pre_layer][:,inp_unit_begin:inp_unit_begin+self.inp_num].sum(axis=1).reshape(self.unit_num,1).repeat(self.inp_num, axis=1))/(self.Z_from_inp*self.Z_ex)
                    from_out_row_sum = (self.synapse_strength[post_layer][pre_layer][:,out_unit_begin:out_unit_begin+self.out_num].sum(axis=1).reshape(self.unit_num,1).repeat(self.out_num, axis=1))/(self.Z_from_out*self.Z_ex)
                    excitatory_row_sums = np.concatenate((from_inp_row_sum,from_out_row_sum),axis=1)
                    
                    # Make sure intra-layer inhibitory weights are all inhibitory
                    self.synapse_strength[post_layer][pre_layer][:,-self.iin_num:][self.synapse_strength[post_layer][pre_layer][:,-self.iin_num:] > 0] = 0
                    
                    # Normalize incoming inhibitory weights to each unit
                    normalizing_factor = self.Z_inhib
                    inhibitory_row_sums = (-1)*((self.synapse_strength[post_layer][pre_layer][:,-self.iin_num:].sum(axis=1).reshape(self.unit_num,1).repeat(self.iin_num, axis=1))/normalizing_factor)
                    
                    row_sums = np.concatenate((excitatory_row_sums,inhibitory_row_sums),axis=1)
                    row_sums[row_sums == 0] = 1
                    self.synapse_strength[post_layer][pre_layer] = self.synapse_strength[post_layer][pre_layer]/row_sums
                    
    def generate_random_input_vec(self):
        mean_input_value = self.excitatory_threshold/6
        std_input_value = self.excitatory_threshold/6
        input_vec = np.random.normal(mean_input_value, std_input_value, (self.inp_num,1))
        input_vec[input_vec<0] = 0
        return input_vec
    
    def simulate_dynamics(self, input_vec):
        fire_history = []
        
        # Given an input, simulate the dynamics of the system, for iter_num time steps
        for _ in range(layer_num):
            cur_history = []
            for _ in range(self.unit_num):
                cur_history.append([])
            fire_history.append(cur_history)
            
        for t in range(comp_len):
            # Propagate external input
            self.prop_external_input(input_vec)
            
            # Document fire history
            for l in range(layer_num):
                for unit_ind in range(self.unit_num):
                    if self.prev_act[l][unit_ind, 0] == 1:
                        fire_history[l][unit_ind].append(t)
        
        out_unit_start = self.inp_num
        fire_count = [len(x) for x in fire_history[-1][out_unit_start:out_unit_start+self.out_num]]
        self.my_print('\t\tfire_count: ' + str(fire_count))
        
        return fire_count
    
    def prop_external_input(self, input_vec):
        # Simulate the dynamics of the system for a single time step
        new_act = []
        new_input = []
        
        for post_layer in range(layer_num):
            cur_input = np.zeros((self.unit_num, 1))
            if post_layer == 0:
                # The first layer gets external input
                input_from_prev_layer = np.pad(input_vec, ((0,self.unit_num-self.inp_num),(0,0)), 'constant')
                cur_input = np.add(cur_input, input_from_prev_layer)
            for pre_layer in range(layer_num):
                input_from_pre_layer = np.matmul(self.synapse_strength[post_layer][pre_layer], self.prev_act[pre_layer])
                cur_input = np.add(cur_input, input_from_pre_layer)
            
            ''' Accumulating input and refractory period: If a neuron fired in the last time step,
            we subtract its previous input from its current input. Otherwise- we add its previous
            input to its current input. '''
            prev_input_factor = (1 - 2 * self.prev_act[post_layer])
            cur_input = np.add(cur_input, prev_input_factor * self.prev_input[post_layer])
            
            # Make sure the input is non-negative
            cur_input = np.where(cur_input>=0, cur_input, 0)
            
            cur_act = np.concatenate((self.excitatory_activation_function(cur_input[:cur_input.shape[0]-self.iin_num,[0]]),
                                  self.inhibitory_activation_function(cur_input[cur_input.shape[0]-self.iin_num:,[0]])),
                                  axis=0)
            new_act.append(cur_act)
            new_input.append(cur_input)
        
        self.before_prev_act = self.prev_act
        self.prev_act = new_act
        self.prev_input = new_input
        
        self.update_synapse_strength()
        
    def excitatory_activation_function(self, x):
        # Linear activation function for excitatory neurons 
        return 0 + (x >= self.excitatory_threshold)
    
    def inhibitory_activation_function(self, x):
        # Linear activation function for inhibitory neurons
        return 0 + (x >= self.inhibitory_threshold)
    
    def update_synapse_strength(self):
        # Update the synapses strength according the a Hebbian learning rule
        for post_layer in range(layer_num):
            for pre_layer in range(layer_num):
                post_layer_prev_act = self.prev_act[post_layer]
                normalizing_excitatory_vec = np.ones((self.ex_num,1)) * self.gamma_ex
                normalizing_inhibitory_vec = np.ones((self.iin_num,1)) * self.gamma_in
                normalizing_vec = np.concatenate((normalizing_excitatory_vec, normalizing_inhibitory_vec))
                normalized_post_layer_prev_act = post_layer_prev_act - normalizing_vec
                
                pre_layer_before_prev_act = self.before_prev_act[pre_layer]
                
                update_mat = np.matmul(normalized_post_layer_prev_act, pre_layer_before_prev_act.transpose())
                # Strengthen inhibitory neurons weights by making them more negative (and not more positive)
                update_mat[:,-self.iin_num:] = (-1) * update_mat[:,-self.iin_num:]
                    
                self.synapse_strength[post_layer][pre_layer] = self.synapse_strength[post_layer][pre_layer] + self.eta * update_mat
        
        self.fix_synapse_strength()
                    
def simulate_with_configuration(configuration):
    biggest_diffs = []
    cluster_diffs = []
    variances = []
    
    for iter_count in range(sample_num):
        model = ModelClass(configuration,True)
        
        if iter_count % 100 == 0:
            model.my_print('\t\tData collection iter count: ' + str(iter_count))
            
        # Generate random input
        input_vec = model.generate_random_input_vec()
        #model.my_print('\t\tinput vec: ' + str(input_vec.transpose()))
        
        # Simulate dynamics
        fire_count = model.simulate_dynamics(input_vec)
        
        # Run statistics on fire count
        biggest_diff, cluster_diff, variance = run_statistics(fire_count)
        biggest_diffs.append(biggest_diff)
        cluster_diffs.append(cluster_diff)
        variances.append(variance)
    
    biggest_diff_average = sum(biggest_diffs)/sample_num
    cluster_diff_average = sum(cluster_diffs)/sample_num
    variance_average = sum(variances)/sample_num
    return biggest_diff_average, cluster_diff_average, variance_average
    
def run_statistics(fire_count):
    # Analyze the fire count and derive different statistics measures
    
    # Biggest diff:
    ''' The biggest diff is defined as max(a[i+1]-a[i]), where a i the sorted fire counts. '''
    fire_count.sort()
    biggest_diff = max([fire_count[i+1]-fire_count[i] for i in range(len(fire_count)-1)])
    
    # Cluster diff:
    ''' Cluster diff is the difference between the mean of the winners/losers clusters. '''
    first_winner_index = np.argmax([fire_count[i+1]-fire_count[i] for i in range(0,len(fire_count)-1)])+1
    losers = fire_count[:first_winner_index]
    winners = fire_count[first_winner_index:]
    cluster_diff = np.mean(winners)-np.mean(losers)
    
    # Variance:
    ''' This is simply the variance of the fire count. '''
    variance = np.var(fire_count)
    
    return biggest_diff, cluster_diff, variance

# Give the parameters their original values
configuration = {}
configuration['excitatory_precentage'] = 0.8
configuration['inp_num'] = 8
configuration['out_num'] = 8
#configuration['excitatory_threshold'] = 0.4
configuration['excitatory_threshold'] = 0.394
configuration['inhibitory_threshold'] = 0.2
configuration['eta'] = 0.0001
configuration['gamma_ex'] = 0.05
configuration['gamma_in'] = 0.14
#configuration['Z_ex'] = 0.3
configuration['Z_ex'] = 0.25
configuration['Z_inhib'] = 0.4
configuration['Z_from_inp'] = 0.5
configuration['Z_inter_layer'] = configuration['Z_ex']
configuration['quiet'] = False

# We now list all the of the values to be tested for every parameter
tested_values_dic = {
#    'Z_ex,Z_inhib' : ([0.1+x*0.025 for x in range(13)],[0.16+x*0.042 for x in range(13)]),
    'Z_ex,excitatory_threshold' : ([0.1+x*0.025 for x in range(13)],[0.19+x*0.034 for x in range(13)])
#    'Z_inhib' : [(1-configuration['excitatory_precentage']) * configuration['excitatory_threshold'] * x for x in range(5,25)],
#    'Z_from_inp' : [x*0.03+0.2 for x in range(20)],
#    'excitatory_precentage' : [x*0.015+0.615 for x in range(20)],
#    'inp_num' : [x for x in range(5,12)],
#    'out_num' : [x for x in range(5,12)],
#    'excitatory_threshold' : [x*0.03+0.1 for x in range(20)],
#    'inhibitory_threshold' : [x*0.03+0.1 for x in range(20)],
#    'eta' : [x*0.00005 + 0.00005 for x in range(20)],
#    'gamma_ex' : [x*0.025 + 0.025 for x in range(13)],
#    'gamma_in' : [x*0.025 + 0.025 for x in range(13)],
    }
'''tested_values_dic = {
    'Z_ex' : [configuration['excitatory_threshold']*x for x in [y/5+0.6 for y in range(2)]],
    'Z_inhib' : [(1-configuration['excitatory_precentage']) * configuration['excitatory_threshold'] * x for x in range(10,12)],
    'Z_from_inp' : [x*0.03+0.47 for x in range(2)],
    'excitatory_precentage' : [x*0.015+0.785 for x in range(2)],
    'inp_num' : [x for x in range(7,9)],
    'out_num' : [x for x in range(7,9)],
    'excitatory_threshold' : [x*0.03+0.37 for x in range(2)],
    'inhibitory_threshold' : [x*0.03+0.37 for x in range(2)],
    'eta' : [x*0.00005 + 0.00005 for x in range(2)],
    'gamma_ex' : [x*0.025 + 0.025 for x in range(2)],
    'gamma_in' : [x*0.025 + 0.115 for x in range(2)],
    }'''

generate_new_synapse_strength = False
if generate_new_synapse_strength:
    #model = ModelClass(configuration,False)
    model = ModelClass(configuration,True)
    input_vec1 = model.generate_random_input_vec()
    fire_count1 = model.simulate_dynamics(input_vec1)
    model.__init__(configuration, True)
    input_vec2 = model.generate_random_input_vec()
    fire_count2 = model.simulate_dynamics(input_vec2)
    model.__init__(configuration, True)
    input_vec3 = model.generate_random_input_vec()
    fire_count3 = model.simulate_dynamics(input_vec3)
    
    '''trained_strength = [model.synapse_strength]
    file_name = "synapse_strength"
    np.save(file_name, trained_strength)'''
else:
    result_dic = {}
    
    # Finally, the actualy testing
    for tested_parameter_pair, tested_values in tested_values_dic.items():
        result_log_fp = open('result_' + tested_parameter_pair + '.txt','w')
        print('Testing ' + tested_parameter_pair)
        result_log_fp.write('Testing ' + tested_parameter_pair + '\n')
        result_log_fp.flush()
        
        tested_parameter1 = tested_parameter_pair.split(',')[0]
        tested_parameter2 = tested_parameter_pair.split(',')[1]
        
        tested_parameter1_orig_value = configuration[tested_parameter1]
        tested_parameter2_orig_value = configuration[tested_parameter2]
        
        biggest_diff_averages = []
        cluster_diff_averages = []
        variance_averages = []
        for i in range(len(tested_values[0])):
            tested_value1 = tested_values[0][i]
            tested_value2 = tested_values[1][i]
            print('\tTrying ' + tested_parameter1 + '=' + str(tested_value1) + ',' + tested_parameter2 + '=' + str(tested_value2))
            configuration[tested_parameter1] = tested_value1
            configuration[tested_parameter2] = tested_value2
            biggest_diff_average, cluster_diff_average, variance_average = simulate_with_configuration(configuration)
            print('\tbiggest diff ' + str(biggest_diff_average) + ', cluster diff ' + str(cluster_diff_average) + ', variance ' + str(variance_average))
            result_log_fp.write('\tbiggest diff ' + str(biggest_diff_average) + ', cluster diff ' + str(cluster_diff_average) + ', variance ' + str(variance_average) + '\n')
            result_log_fp.flush()
            biggest_diff_averages.append(biggest_diff_average)
            cluster_diff_averages.append(cluster_diff_average)
            variance_averages.append(variance_average)
        
        configuration[tested_parameter1] = tested_parameter1_orig_value
        configuration[tested_parameter2] = tested_parameter2_orig_value
        
        result_dic[tested_parameter_pair] = (biggest_diff_averages, cluster_diff_averages, variance_averages)
        np.save('result_'+tested_parameter_pair,[biggest_diff_averages, cluster_diff_averages, variance_averages])
            
    np.save('result_dic',[result_dic])