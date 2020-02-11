import numpy as np
import math
import matplotlib.pyplot as plt
from copy import deepcopy

###################
# Model functions #
###################

# General parameters
input_size = 8
output_size = 8
iin_num = 2
unit_num = input_size + output_size + iin_num
competition_len = 10000

class ModelClass:
    
    default_configuration = {
        # Neuron parameters
        'excitatory_threshold' : 0.4,
        'inhibitory_threshold' : 0.2,
        'sensory_input_strength' : 0.13333,
        
        # Normalization parameter
        'Z_ex_ex_th_ratio' : 0.5,
        'Z_iin_ex_th_ratio' : 0.5,
        'Z_inp_Z_ex_ratio' : 0.5,
        
        # Learning parameters
        'eta' : 0.0001,
        'gamma_ex' : 0.05,
        'gamma_in' : 0.14,
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
            self.synapse_strength[:, -iin_num:] = (-1) * self.synapse_strength[:, -iin_num:]
        else:
            file_name = "synapse_strength"
            file_name += "_" + str(unit_num) + "_neurons"
            file_name += ".npy"
            
            trained_strength = np.load(file_name)
        
            self.synapse_strength = trained_strength[0]
            
        self.prev_act = np.zeros((unit_num, 1))
        self.before_prev_act = np.zeros((unit_num, 1))
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
        excitatory_unit_num = unit_num-iin_num
        
        # Make sure excitatory weights in are all excitatory
        self.synapse_strength[:,:excitatory_unit_num][self.synapse_strength[:,:excitatory_unit_num] < 0] = 0
        
        # Normalize incoming excitatory weights to each unit    
        input_begin = 0
        output_begin = input_begin + input_size 
        
        input_to_input_row_sum = (self.synapse_strength[input_begin:input_begin+input_size,input_begin:input_begin+input_size].sum(axis=1).reshape(input_size,1).repeat(input_size, axis=1))/self.conf['Z_inp']
        output_to_input_row_sum = (self.synapse_strength[input_begin:input_begin+input_size,output_begin:output_begin+output_size].sum(axis=1).reshape(input_size,1).repeat(output_size, axis=1))/self.conf['Z_out']
        input_row_sum = np.concatenate((input_to_input_row_sum,output_to_input_row_sum),axis=1)
        
        input_to_output_row_sum = (self.synapse_strength[output_begin:output_begin+output_size,input_begin:input_begin+input_size].sum(axis=1).reshape(output_size,1).repeat(input_size, axis=1))/self.conf['Z_inp']
        output_to_output_row_sum = (self.synapse_strength[output_begin:output_begin+output_size,output_begin:output_begin+output_size].sum(axis=1).reshape(output_size,1).repeat(output_size, axis=1))/self.conf['Z_out']
        output_row_sum = np.concatenate((input_to_output_row_sum,output_to_output_row_sum),axis=1)
        
        input_to_iin_row_sum = (self.synapse_strength[-iin_num:,input_begin:input_begin+input_size].sum(axis=1).reshape(iin_num,1).repeat(input_size, axis=1))/self.conf['Z_inp']
        output_to_iin_row_sum = (self.synapse_strength[-iin_num:,output_begin:output_begin+output_size].sum(axis=1).reshape(iin_num,1).repeat(output_size, axis=1))/self.conf['Z_out']
        iin_from_ex_row_sum = np.concatenate((input_to_iin_row_sum,output_to_iin_row_sum),axis=1)
        
        excitatory_row_sums = np.concatenate((input_row_sum,output_row_sum,iin_from_ex_row_sum),axis=0)
        
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
        
        for t in range(competition_len):
            if t % 1000 == 0:
                self.my_print('t='+str(t))
            
            # Propagate external input
            self.prop_external_input(input_vec)
                
            # Document fire history
            for unit_ind in range(unit_num):
                if self.prev_act[unit_ind, 0] == 1:
                    fire_history[unit_ind].append(t)
        
        self.my_print('neurons firing: ' + str([len(a) for a in fire_history]))
        
        return fire_history
        
    def update_synapse_strength(self):
        # Update the synapses strength according the a Hebbian learning rule
        normalizing_excitatory_vec = np.ones((unit_num-iin_num,1)) * self.conf['gamma_ex']
        normalizing_inhibitory_vec = np.ones((iin_num,1)) * self.conf['gamma_in']
        normalizing_vec = np.concatenate((normalizing_excitatory_vec, normalizing_inhibitory_vec))
        normalized_prev_act = self.prev_act - normalizing_vec
        
        update_mat = np.matmul(normalized_prev_act, self.before_prev_act.transpose())
            
        self.synapse_strength = self.synapse_strength + self.conf['eta'] * update_mat

        self.fix_synapse_strength()
        
    def prop_external_input(self, sensory_input_vec):
        # Simulate the dynamics of the system for a single time step
        external_input = np.pad(sensory_input_vec, ((0,unit_num-input_size),(0,0)), 'constant')
        internal_input = np.matmul(self.synapse_strength, self.prev_act)
        cur_input = np.add(external_input, internal_input)
        
        ''' Accumulating input and refractory period: If a neuron fired in the last time step,
        we subtract its previous input from its current input. Otherwise- we add its previous
        input to its current input. '''
        prev_input_factor = (1 - 2 * self.prev_act)
        cur_input = np.add(cur_input, prev_input_factor * self.prev_input_to_neurons)
        
        # Make sure the input is non-negative
        cur_input = np.where(cur_input>=0, cur_input, 0)

        cur_act = np.concatenate((self.excitatory_activation_function(cur_input[:cur_input.shape[0]-iin_num,[0]]),
                              self.inhibitory_activation_function(cur_input[cur_input.shape[0]-iin_num:,[0]])),
                              axis=0)
        
        self.before_prev_act = deepcopy(self.prev_act)
        self.prev_act = deepcopy(cur_act)
        self.prev_input_to_neurons = deepcopy(cur_input)
        
        #self.update_synapse_strength()
        
    def excitatory_activation_function(self, x):
        # Linear activation function for excitatory neurons 
        return 0 + (x >= self.conf['excitatory_threshold'])
    
    def inhibitory_activation_function(self, x):
        # Linear activation function for inhibitory neurons
        return 0 + (x >= self.conf['inhibitory_threshold'])

def generate_random_sensory_input():
    sensory_input_vec = np.random.rand(input_size,1)
    return sensory_input_vec

def linear_cross_correlation(fire_times_lists):
    max_lcc = (-1)*math.inf
    n = len(fire_times_lists)
    for x in range(n):
        for y in range(x+1,n):
            union_size = len(list(set(fire_times_lists[x]) | set(fire_times_lists[y])))
            if union_size == 0:
                continue
            intersection_size = len(list(set(fire_times_lists[x]) & set(fire_times_lists[y])))
            cur_lcc = intersection_size/union_size
            if cur_lcc > max_lcc:
                max_lcc = cur_lcc
    
    return max_lcc

def linear_cross_correlation_with_iins(fire_times_lists, iin_fire_times_lists):
    res = []
    n = len(fire_times_lists)
    m = len(iin_fire_times_lists)
    for x in range(n):
        max_lcc = 0
        for y in range(m):
            union_size = len(list(set(fire_times_lists[x]) | set(iin_fire_times_lists[y])))
            if union_size == 0:
                continue
            intersection_size = len(list(set(fire_times_lists[x]) & set(iin_fire_times_lists[y])))
            cur_lcc = intersection_size/union_size
            if cur_lcc > max_lcc:
                max_lcc = cur_lcc
        res.append(max_lcc)
    
    return res

def entropy(fire_indicator_vec):
    fire_prob_list = fire_indicator_vec/np.sum(fire_indicator_vec)
    fire_prob_list[fire_prob_list==0] = 1
    product_vec = np.multiply(fire_prob_list,np.log2(fire_prob_list))
    return (-1)*np.sum(product_vec)

def mutual_information(fire_times_lists):
    max_mi = (-1)*math.inf
    n = len(fire_times_lists)
    for x in range(n):
        for y in range(x+1,n):
            x_fire_indicator = np.array([1 if t in fire_times_lists[x] else 0 for t in range(competition_len)])
            y_fire_indicator = np.array([1 if t in fire_times_lists[y] else 0 for t in range(competition_len)])
            if np.sum(x_fire_indicator) == 0 or np.sum(y_fire_indicator) == 0:
                continue
            union_fire_indicator = x_fire_indicator + y_fire_indicator
            cur_mi = entropy(x_fire_indicator) + entropy(y_fire_indicator) - entropy(union_fire_indicator)
            if cur_mi > max_mi:
                max_mi = cur_mi
    
    return max_mi

def mutual_information_with_iins(fire_times_lists, iin_fire_times_lists):
    n = len(fire_times_lists)
    m = len(iin_fire_times_lists)
    res = []
    for x in range(n):
        max_mi = 0
        for y in range(m):
            x_fire_indicator = np.array([1 if t in fire_times_lists[x] else 0 for t in range(competition_len)])
            y_fire_indicator = np.array([1 if t in iin_fire_times_lists[y] else 0 for t in range(competition_len)])
            if np.sum(x_fire_indicator) == 0 or np.sum(y_fire_indicator) == 0:
                continue
            union_fire_indicator = x_fire_indicator + y_fire_indicator
            cur_mi = entropy(x_fire_indicator) + entropy(y_fire_indicator) - entropy(union_fire_indicator)
            if cur_mi > max_mi:
                max_mi = cur_mi
        res.append(max_mi)
    
    return res

def event_synchronization(fire_times_lists):
    max_es = (-1)*math.inf
    n = len(fire_times_lists)
    for x in range(n):
        for y in range(x+1,n):
            intersection_size = len(list(set(fire_times_lists[x]) & set(fire_times_lists[y])))
            m_x = len(fire_times_lists[x])
            m_y = len(fire_times_lists[y])
            if m_x == 0 or m_y == 0:
                continue
            cur_es = intersection_size/(m_x*m_y)**0.5
            if cur_es > max_es:
                max_es = cur_es
    
    return max_es

def event_synchronization_with_iins(fire_times_lists, iin_fire_times_lists):
    n = len(fire_times_lists)
    m = len(iin_fire_times_lists)
    res = []
    for x in range(n):
        max_es = 0
        for y in range(m):
            intersection_size = len(list(set(fire_times_lists[x]) & set(iin_fire_times_lists[y])))
            m_x = len(fire_times_lists[x])
            m_y = len(fire_times_lists[y])
            if m_x == 0 or m_y == 0:
                continue
            cur_es = intersection_size/(m_x*m_y)**0.5
            if cur_es > max_es:
                max_es = cur_es
        res.append(max_es)
    
    return res

def cov(x,y):
    assert(len(x) == len(y))
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    return np.mean([(x[i]-mu_x)*(y[i]-mu_y) for i in range(len(x))])

def pearson_correlation(x,y):
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    if sigma_x == 0 or sigma_y == 0:
        return 0
    return cov(x,y)/(sigma_x*sigma_y)

#############
# Main code #
#############

configuration = {}
iter_num = 5000
#Z_iin_ex_th_ratio_values = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8]
Z_iin_ex_th_ratio_values = [1.8]

lccs = []
mis = []
ess = []

lcc_fire_rate_cors = []
mi_fire_rate_cors = []
es_fire_rate_cors = []

for Z_iin_ex_th_ratio in Z_iin_ex_th_ratio_values:
    print('Trying Z_iin - ex_th ratio ' + str(Z_iin_ex_th_ratio))
    configuration['Z_iin_ex_th_ratio'] = Z_iin_ex_th_ratio
    
    lcc_sum = 0
    lcc_count = 0
    mi_sum = 0
    mi_count = 0
    es_sum = 0
    es_count = 0
    
    lcc_fire_rate_cor_sum = 0
    mi_fire_rate_cor_sum = 0
    es_fire_rate_cor_sum = 0
    
    for cur_iter in range(iter_num):
        model = ModelClass(configuration,False,True)
        if cur_iter % 100 == 0:
            print('iter ' + str(cur_iter))
        sensory_input_vec = generate_random_sensory_input()
        sensory_input_vec *= model.conf['sensory_input_strength']
        
        fire_history = model.simulate_dynamics(sensory_input_vec)
        output_start = input_size
        output_fire_history = fire_history[output_start:output_start+output_size]
        
        # Analyze results
        # Output fire history synchrony
        lcc = linear_cross_correlation(output_fire_history)
        if lcc > (-1)*math.inf:
            lcc_sum += lcc
            lcc_count += 1
        
        mi = mutual_information(output_fire_history)
        if mi > (-1)*math.inf:
            mi_sum += mi
            mi_count += 1
            
        es = event_synchronization(output_fire_history)
        if es > (-1)*math.inf:
            es_sum += es
            es_count += 1
            
        # Correlation between synchrony with IINs and fire rate
        iin_start = output_start + output_size
        iin_fire_history = fire_history[iin_start:iin_start+iin_num]
        output_fire_count = [len(x) for x in output_fire_history]
        
        lcc_with_iin = linear_cross_correlation_with_iins(output_fire_history,iin_fire_history)
        lcc_fire_rate_cor = pearson_correlation(output_fire_count, lcc_with_iin)
        lcc_fire_rate_cor_sum += lcc_fire_rate_cor
        
        mi_with_iin = mutual_information_with_iins(output_fire_history,iin_fire_history)
        mi_fire_rate_cor = pearson_correlation(output_fire_count, mi_with_iin)
        mi_fire_rate_cor_sum += mi_fire_rate_cor
        
        es_with_iin = event_synchronization_with_iins(output_fire_history,iin_fire_history)
        es_fire_rate_cor = pearson_correlation(output_fire_count, es_with_iin)
        es_fire_rate_cor_sum += es_fire_rate_cor
            
    lcc_average = lcc_sum/lcc_count
    lccs.append(lcc_average)
    print(lcc_average)
    mi_average = mi_sum/mi_count
    mis.append(mi_average)
    print(mi_average)
    es_average = es_sum/es_count
    ess.append(es_average)
    print(es_average)
    
    lcc_fire_rate_cor_average = lcc_fire_rate_cor_sum/iter_num
    lcc_fire_rate_cors.append(lcc_fire_rate_cor_average)
    print(lcc_fire_rate_cor_average)
    mi_fire_rate_cor_average = mi_fire_rate_cor_sum/iter_num
    mi_fire_rate_cors.append(mi_fire_rate_cor_average)
    print(mi_fire_rate_cor_average)
    es_fire_rate_cor_average = es_fire_rate_cor_sum/iter_num
    es_fire_rate_cors.append(es_fire_rate_cor_average)
    print(es_fire_rate_cor_average)

plt.xlabel('Z_iin')
plt.ylabel('Average maximum LCC')
plt.plot(Z_iin_ex_th_ratio_values, lccs)
plt.savefig('lcc_res')

plt.clf()
plt.xlabel('Z_iin')
plt.ylabel('Average maximum MI')
plt.plot(Z_iin_ex_th_ratio_values, mis)
plt.savefig('mi_res')

plt.clf()
plt.xlabel('Z_iin')
plt.ylabel('Average maximum ES')
plt.plot(Z_iin_ex_th_ratio_values, ess)
plt.savefig('es_res')

plt.clf()
plt.xlabel('Z_iin')
plt.ylabel('Average LCC fire rate correlation')
plt.plot(Z_iin_ex_th_ratio_values, lcc_fire_rate_cors)
plt.savefig('lcc_fire_rate_cor_res')

plt.clf()
plt.xlabel('Z_iin')
plt.ylabel('Average MI fire rate correlation')
plt.plot(Z_iin_ex_th_ratio_values, mi_fire_rate_cors)
plt.savefig('mi_fire_rate_cor_res')

plt.clf()
plt.xlabel('Z_iin')
plt.ylabel('Average ES fire rate correlation')
plt.plot(Z_iin_ex_th_ratio_values, es_fire_rate_cors)
plt.savefig('es_fire_rate_cor_res')

'''
import numpy as np
import matplotlib.pyplot as plt

Z_iin_vals = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8]
lccs = [0.121196189521357,0.15410129394931762,0.1561602328834252,0.1502111528078821,0.17392405448823514,0.22984123862488462,0.2603801398796102,0.22361717276365964,0.17345474826193807,0.13361224979206068,0.12086860917150173,0.09727947682791213,0.0897061093275939,0.08171718409102671,0.06507906579460773,0.059439297004810446,0.057729073290128594]
mis = [7.395264329879502,7.156259962412903,6.909656979834789,6.599002012470027,6.042230265375601,5.848385389561701,5.927534457918854,6.0921099313988645,6.28158377625113,6.360935164930522,6.356910177567252,6.26592134357497,6.2476090400578475,6.175460895137052,6.135200761832347,6.092745122565165,6.127732713737936]
ess = [0.21206866775034186,0.2578472661062368,0.25973150879854323,0.2514522528341769,0.2817490454485843,0.35212873568457725,0.389296088248723,0.34643218105748624,0.28544204398533957,0.22834977165281856,0.21359257094778264,0.18495913612928536,0.17940355922110388,0.16750425779669695,0.1494188000326006,0.14843879815023028,0.150120449175937]
lcc_fire_rate_cors = [0.2113407197807907,0.08227597711302606,0.0625681507100547,0.2949768450831103,0.5054141257430458,0.45813173624357206,0.37421045554855653,0.2892401310903815,0.2340664823189607,0.202892499731939,0.2008145028065447,0.1970087808945289,0.197044821000877,0.19857150926382366,0.19180837690006905,0.19514328397113032,0.1961047186029051]
mi_fire_rate_cors = [0.9236852546186145,0.9275422428515608,0.953273852386997,0.9590634029400217,0.9059786931555993,0.7294606939825223,0.5842144700660521,0.4915069903673436,0.4248255527388019,0.39670078469545667,0.3908408221732089,0.38874727882996024,0.39026907397314986,0.40564785482666776,0.3994211081541203,0.38937025234199707,0.39336339057544417]
es_fire_rate_cors = [0.18248485658530486,-0.009966868480316344,-0.13687459851934966,-0.0010318940191846,0.22238360349006986,0.23697149638122778,0.2002786583625933,0.13871259133684036,0.09851610101490223,0.0781447473398518,0.06922710774501452,0.06723425401408811,0.05976393236348759,0.05564105637765154,0.05839623753579014,0.04940388761772113,0.04882334130408871]

plt.plot(Z_iin_vals,lcc_fire_rate_cors)
plt.xlabel('Sum of strength of inhibitory synapses entering a neuron')
plt.ylabel('Correlation between LCC with IINS and firing rate')
plt.show()
'''