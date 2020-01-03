import numpy as np

layer_num = 1
comp_len = 20000
#sample_num = 100
sample_num = 3
normal_firing_rate = 0.005

class ModelClass:
    
    def __init__(self, configuration):
        # Members determined by the configuration
        self.Z_ex = configuration['Z_ex']
        self.Z_inhib = configuration['Z_inhib']
        self.Z_from_inp = configuration['Z_inp_from_inp']
        self.Z_from_out = configuration['Z_out_from_out']
        self.Z_inp_to_out = configuration['Z_inp_to_out']
        self.Z_out_to_out = configuration['Z_out_to_out']
        self.Z_inp_to_inhib = configuration['Z_inp_to_inhib']
        self.Z_out_to_inhib = configuration['Z_out_to_inhib']
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
        
        # Data structures
        self.synapse_strength = []
        self.zero_matrices = []
        self.before_prev_act = []
        self.prev_act = []
        self.prev_input = []
        self.init_data_structures()
        
    def my_print(self, my_str):
        if not self.quiet:
            print(my_str)
        
    def init_data_structures(self):
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
                    inhib_unit_begin = out_unit_begin + self.out_num
                    
                    inp_to_inp_row_sum = (self.synapse_strength[post_layer][pre_layer][inp_unit_begin:inp_unit_begin+self.inp_num,inp_unit_begin:inp_unit_begin+self.inp_num].sum(axis=1).reshape(self.inp_num,1).repeat(self.inp_num, axis=1))/(self.Z_inp_to_inp*self.Z_ex)
                    out_to_inp_row_sum = (self.synapse_strength[post_layer][pre_layer][inp_unit_begin:inp_unit_begin+self.inp_num,out_unit_begin:out_unit_begin+self.out_num].sum(axis=1).reshape(self.inp_num,1).repeat(self.out_num, axis=1))/(self.Z_out_to_inp*self.Z_ex)
                    inp_row_sum = np.concatenate((inp_to_inp_row_sum,out_to_inp_row_sum),axis=1)
                    
                    inp_to_out_row_sum = (self.synapse_strength[post_layer][pre_layer][out_unit_begin:out_unit_begin+self.out_num,inp_unit_begin:inp_unit_begin+self.inp_num].sum(axis=1).reshape(self.out_num,1).repeat(self.inp_num, axis=1))/(self.Z_inp_to_out*self.Z_ex)
                    out_to_out_row_sum = (self.synapse_strength[post_layer][pre_layer][out_unit_begin:out_unit_begin+self.out_num,out_unit_begin:out_unit_begin+self.out_num].sum(axis=1).reshape(self.out_num,1).repeat(self.out_num, axis=1))/(self.Z_out_to_out*self.Z_ex)
                    out_row_sum = np.concatenate((inp_to_out_row_sum,out_to_out_row_sum),axis=1)
                    
                    inp_to_inhib_row_sum = (self.synapse_strength[post_layer][pre_layer][inhib_unit_begin:inhib_unit_begin+self.iin_num,inp_unit_begin:inp_unit_begin+self.inp_num].sum(axis=1).reshape(self.iin_num,1).repeat(self.inp_num, axis=1))/(self.Z_inp_to_inhib*self.Z_ex)
                    out_to_inhib_row_sum = (self.synapse_strength[post_layer][pre_layer][inhib_unit_begin:inhib_unit_begin+self.iin_num,out_unit_begin:out_unit_begin+self.out_num].sum(axis=1).reshape(self.iin_num,1).repeat(self.out_num, axis=1))/(self.Z_out_to_inhib*self.Z_ex)
                    inhib_from_ex_row_sum = np.concatenate((inp_to_inhib_row_sum,out_to_inhib_row_sum),axis=1)
                    
                    excitatory_row_sums = np.concatenate((inp_row_sum,out_row_sum,inhib_from_ex_row_sum),axis=0)
                    
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
                    
def simulate_with_configuration(configuration, increase_firing_rate, decrease_firing_rate):
    biggest_diffs = []
    cluster_diffs = []
    variances = []
    
    normal_count = 0
    while normal_count < sample_num:
        model = ModelClass(configuration)
        
        # Generate random input
        input_vec = model.generate_random_input_vec()
        #model.my_print('\t\tinput vec: ' + str(input_vec.transpose()))
        
        # Simulate dynamics
        fire_count = model.simulate_dynamics(input_vec)
        average_fire_count = np.mean(fire_count)
        if average_fire_count < (normal_firing_rate-0.0025)*comp_len:
            model.my_print('\t\tFire count too low, increasing')
            increase_firing_rate(configuration)
        elif average_fire_count > (normal_firing_rate+0.0025)*comp_len:
            model.my_print('\t\tFire count too high, denreasing')
            decrease_firing_rate(configuration)
        else:
            normal_count += 1
            if normal_count % 10 == 0:
                model.my_print('Collected ' + str(normal_count) + ' out of ' + str(sample_num))
            
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
configuration['excitatory_threshold'] = 0.4
configuration['inhibitory_threshold'] = 0.2
configuration['eta'] = 0.0001
configuration['gamma_ex'] = 0.05
configuration['gamma_in'] = 0.14
configuration['Z_ex'] = 2*configuration['excitatory_threshold']
configuration['Z_inhib'] = (1-configuration['excitatory_precentage']) * configuration['excitatory_threshold'] * 10
configuration['Z_inp_to_inp'] = 0.5
configuration['Z_out_to_inp'] = 0.5
configuration['Z_inp_to_out'] = 0.5
configuration['Z_out_to_out'] = 0.5
configuration['Z_inp_to_inhib'] = 0.5
configuration['Z_out_to_inhib'] = 0.5
configuration['Z_inter_layer'] = configuration['Z_ex']
configuration['quiet'] = False

''' If we have N parameters, we go over all N choose 2 pairs of parameters and try estimating the
correlation with the different separation measures.
We start by defining, for every parameter, a function that changes this parameter, in the purpose
of increasing/decreasing the firing rate. '''

def Z_ex_increase_firing_rate(configuration):
    configuration['Z_ex'] = configuration['Z_ex'] * 1.1
    
def Z_ex_decrease_firing_rate(configuration):
    configuration['Z_ex'] = configuration['Z_ex'] * 0.9
    
def Z_inhib_increase_firing_rate(configuration):
    configuration['Z_inhib'] = configuration['Z_inhib'] * 0.9
    
def Z_inhib_decrease_firing_rate(configuration):
    configuration['Z_inhib'] = configuration['Z_inhib'] * 1.1
    
def Z_inp_to_inp_increase_firing_rate(configuration):
    configuration['Z_inhib'] = configuration['Z_inhib'] * 0.9
    
def Z_inp_to_inp_decrease_firing_rate(configuration):
    configuration['Z_inhib'] = configuration['Z_inhib'] * 1.1

# We now put all these functions into a dictionary
increase_decrease_functions_dic = {
    'Z_ex' : (Z_ex_increase_firing_rate, Z_ex_decrease_firing_rate),
    'Z_inhib' : (Z_inhib_increase_firing_rate, Z_inhib_decrease_firing_rate),
    'Z_inp_to_inp' : (Z_inhib_increase_firing_rate, Z_inhib_decrease_firing_rate),
    }

# We now list all the of the values to be tested for every parameter
tested_values_dic = {
    'Z_ex' : [configuration['excitatory_threshold']*x for x in [y/5+0.1 for y in range(20)]],
    'Z_inhib' : [(1-configuration['excitatory_precentage']) * configuration['excitatory_threshold'] * x for x in range(5,25)],
    }

# We now list the parameters to be tested
parameters_list = [
    'Z_ex',
    'Z_inhib',
    'Z_inp_to_inp',
#    'Z_out_to_inp',
#    'Z_inp_to_out',
#    'Z_out_to_out',
#    'Z_inp_to_inhib',
#    'Z_out_to_inhib',
#    'Z_inter_layer',
#    'excitatory_precentage',
#    'inp_num',
#    'out_num',
#    'excitatory_threshold',
#    'inhibitory_threshold',
#    'eta',
#    'gamma_ex',
#    'gamma_in'
    ]

result_dic = {}

# Finally, the actualy testing
for tested_parameter in parameters_list:
    for varied_parameter in parameters_list:
        if tested_parameter == varied_parameter:
            continue
        print('Testing ' + tested_parameter + ', varying ' + varied_parameter)
        increase_decrease_functions = increase_decrease_functions_dic[varied_parameter]
        tested_values = tested_values_dic[tested_parameter]
        
        tested_parameter_orig_value = configuration[tested_parameter]
        varied_parameter_orig_value = configuration[varied_parameter]
        
        biggest_diff_averages = []
        cluster_diff_averages = []
        variance_averages = []
        for tested_value in tested_values:
            print('\tTrying ' + tested_parameter + '=' + str(tested_value))
            configuration[tested_parameter] = tested_value
            biggest_diff_average, cluster_diff_average, variance_average = simulate_with_configuration(configuration, increase_decrease_functions[0],increase_decrease_functions[1])
            biggest_diff_averages.append(biggest_diff_average)
            cluster_diff_averages.append(cluster_diff_average)
            variance_averages.append(variance_average)
        
        configuration[tested_parameter] = tested_parameter_orig_value
        configuration[varied_parameter] = varied_parameter_orig_value
        
        result_dic[tested_parameter+'_'+varied_parameter] = (biggest_diff_averages, cluster_diff_averages, variance_averages)
        
print(result_dic)