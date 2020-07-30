import numpy as np
from mnist_inputs import generate_reduced_data_sets
from copy import deepcopy

#write_to_log = True
# CHANGE
write_to_log = False
if write_to_log:
    log_fp = open('log.txt','w')

class ModelClass:
    
    default_configuration = {
        'sensory_input_strength' : 1/2,
        'input_size' : 784,
        'layers_size' : [900,10],
        # CHANGE
        #'layers_size' : [512,200,62],
        'init_reduce_factor' : 10,
        
        # Activation parameters
        'A' : 1.7159,
        'B' : 0.6666,
        
        # Learning parameters
        'eta' : 0.01,
        }
    
    def __init__(self, configuration, load_from_file, quiet):
        self.conf = {key : configuration[key] if key in configuration else ModelClass.default_configuration[key] for key in ModelClass.default_configuration.keys()}        
        self.quiet = quiet
        self.init_data_structures(load_from_file)
        
    def init_data_structures(self, file_suffix):
        
        if file_suffix == None:
            self.synapse_strength = []
            
            self.synapse_strength.append((np.random.rand(self.conf['layers_size'][0], self.conf['input_size'])-0.5)/self.conf['init_reduce_factor'])
            for l in range(1,len(self.conf['layers_size'])):
                self.synapse_strength.append((np.random.rand(self.conf['layers_size'][l], self.conf['layers_size'][l-1])-0.5)/self.conf['init_reduce_factor'])
                
            self.bias = []
            for l in range(len(self.conf['layers_size'])):
                self.bias.append((np.random.rand(self.conf['layers_size'][l], 1)))
        else:
            file_name = "synapse_strength"
            file_name += "_" + file_suffix
            file_name += ".npy"
            
            trained_strength = np.load(file_name)
        
            self.synapse_strength = trained_strength[0]
            self.bias = trained_strength[1]
            
        self.reset_data_structures()
        
    def reset_data_structures(self):
        self.z_cache = []
        self.a_cache = []
            
    def forward_pass(self, input_vec):
        model.reset_data_structures()
        cur_a = input_vec
        
        for l in range(len(self.conf['layers_size'])):
            cur_z = np.add(np.matmul(self.synapse_strength[l],cur_a),self.bias[l])
            if l < len(self.conf['layers_size'])-1:
                cur_a = self.hidden_activation_function(cur_z)
            else: # Output layer
                cur_a = self.output_activation_function(cur_z)
            '''if l == 0: # winner-takes-all
                sorted_a = [cur_a[x,0] for x in range(cur_a.shape[0])]
                sorted_a.sort(reverse=True)
                threshold_val = sorted_a[30]
                cur_a[cur_a < threshold_val] = 0'''
            self.z_cache.append(cur_z)
            self.a_cache.append(cur_a)
        
    def backward_pass(self, label_vec, input_vec):
        derror_dz = None
        new_synapse_strength = []
        new_bias = []
        
        for l in range(len(self.conf['layers_size'])-1, -1, -1):
            '''if l == 0: # No learning in sensory-hidden weights
                new_synapse_strength.insert(0, self.synapse_strength[l])
                new_bias.insert(0, self.bias[l])
                break'''
            if l < len(self.conf['layers_size'])-1:
                derror_dact = np.matmul(self.synapse_strength[l+1].transpose(), derror_dz)
                dact_dz = self.hidden_activation_derivative(self.z_cache[l])
                derror_dz = np.multiply(derror_dact,dact_dz)
            else:
                derror_dact = -1 * (np.multiply(label_vec, 1/self.a_cache[-1])-np.multiply((1-label_vec), 1/(1-self.a_cache[-1])))
                dact_dz = self.output_activation_derivative(self.z_cache[-1])
                derror_dz = np.matmul(dact_dz,derror_dact)
            
            if l == 0:
                dz_dw = input_vec
            else:
                dz_dw = self.a_cache[l-1]
    
            derror_dw = np.matmul(derror_dz, dz_dw.transpose())
            new_synapse_strength.insert(0, self.synapse_strength[l] - self.conf['eta'] * derror_dw)
        
            derror_db = derror_dz
            new_bias.insert(0, self.bias[l] - self.conf['eta'] * derror_db)
            
        self.synapse_strength = new_synapse_strength
        self.bias = new_bias
        self.reset_data_structures()
        
    def compare_to_weights_numeric_derivative(self, gradient_mat, layer, label_vec):
        eps = 0.0001
        max_diff_from_num_deriv = 0
        max_post_synaptic_unit = None
        max_pre_synaptic_unit = None
        
        for post_synaptic_unit in range(gradient_mat.shape[0]):
            print(post_synaptic_unit)
            for pre_synaptic_unit in range(gradient_mat.shape[1]):
                # Plus execution
                plus_synapse_strength = deepcopy(self.synapse_strength)
                plus_synapse_strength[layer][post_synaptic_unit,pre_synaptic_unit] += eps
                cur_a_plus = input_vec
                for l in range(len(self.conf['layers_size'])):
                    cur_z_plus = np.add(np.matmul(plus_synapse_strength[l],cur_a_plus),self.bias[l])
                    if l < len(self.conf['layers_size'])-1:
                        cur_a_plus = self.hidden_activation_function(cur_z_plus)
                    else: # Output layer
                        cur_a_plus = self.output_activation_function(cur_z_plus)
                loss_plus = -1 * np.sum \
                    ((np.multiply(label_vec,np.log(cur_a_plus))) + \
                    (np.multiply(1-label_vec,np.log(1-cur_a_plus))))
                    
                # Minus execution
                minus_synapse_strength = deepcopy(self.synapse_strength)
                minus_synapse_strength[layer][post_synaptic_unit,pre_synaptic_unit] -= eps
                cur_a_minus = input_vec
                for l in range(len(self.conf['layers_size'])):
                    cur_z_minus = np.add(np.matmul(minus_synapse_strength[l],cur_a_minus),self.bias[l])
                    if l < len(self.conf['layers_size'])-1:
                        cur_a_minus = self.hidden_activation_function(cur_z_minus)
                    else: # Output layer
                        cur_a_minus = self.output_activation_function(cur_z_minus)
                loss_minus = -1 * np.sum \
                    ((np.multiply(label_vec,np.log(cur_a_minus))) + \
                    (np.multiply(1-label_vec,np.log(1-cur_a_minus))))
                    
                num_deriv = (loss_plus - loss_minus)/(2*eps)
                diff_from_num_deriv = abs(num_deriv - gradient_mat[post_synaptic_unit,pre_synaptic_unit])
                if diff_from_num_deriv > max_diff_from_num_deriv:
                    max_diff_from_num_deriv = diff_from_num_deriv
                    max_post_synaptic_unit = post_synaptic_unit
                    max_pre_synaptic_unit = pre_synaptic_unit
                    
        print(max_diff_from_num_deriv,max_post_synaptic_unit,max_pre_synaptic_unit)
        
    def compare_to_bias_numeric_derivative(self, gradient_vec, layer, label_vec):
        eps = 0.0001
        max_diff_from_num_deriv = 0
        max_unit = None
        
        for unit in range(gradient_vec.shape[0]):
            # Plus execution
            plus_bias = deepcopy(self.bias)
            plus_bias[layer][unit,0] += eps
            cur_a_plus = input_vec
            for l in range(len(self.conf['layers_size'])):
                cur_z_plus = np.add(np.matmul(self.synapse_strength[l],cur_a_plus),plus_bias[l])
                if l < len(self.conf['layers_size'])-1:
                    cur_a_plus = self.hidden_activation_function(cur_z_plus)
                else: # Output layer
                    cur_a_plus = self.output_activation_function(cur_z_plus)
            loss_plus = -1 * np.sum \
                ((np.multiply(label_vec,np.log(cur_a_plus))) + \
                (np.multiply(1-label_vec,np.log(1-cur_a_plus))))
                
            # Minus execution
            minus_bias = deepcopy(self.bias)
            minus_bias[layer][unit,0] -= eps
            cur_a_minus = input_vec
            for l in range(len(self.conf['layers_size'])):
                cur_z_minus = np.add(np.matmul(self.synapse_strength[l],cur_a_minus),minus_bias[l])
                if l < len(self.conf['layers_size'])-1:
                    cur_a_minus = self.hidden_activation_function(cur_z_minus)
                else: # Output layer
                    cur_a_minus = self.output_activation_function(cur_z_minus)
            loss_minus = -1 * np.sum \
                ((np.multiply(label_vec,np.log(cur_a_minus))) + \
                (np.multiply(1-label_vec,np.log(1-cur_a_minus))))
                
            num_deriv = (loss_plus - loss_minus)/(2*eps)
            diff_from_num_deriv = abs(num_deriv - gradient_vec[unit,0])
            if diff_from_num_deriv > max_diff_from_num_deriv:
                max_diff_from_num_deriv = diff_from_num_deriv
                max_unit = unit
                    
        print(max_diff_from_num_deriv,max_unit)
    
    def hidden_activation_function(self, x):
        return self.conf['A'] * np.tanh(self.conf['B']*x)
    
    def hidden_activation_derivative(self, x):
        return self.conf['A'] * self.conf['B'] * (1 - np.power(np.tanh(self.conf['B']*x),2))
    
    def output_activation_function(self, x):
        return np.exp(x)/np.sum(np.exp(x))
    
    def output_activation_derivative(self, x):
        output_size = x.shape[0]
        
        softmax_res = self.output_activation_function(x)
        softmax_vertical = np.repeat(softmax_res, output_size, 1)
        softmax_horizontal =  np.repeat(softmax_res.transpose(), output_size, 0)
        temp_result = -1 * np.multiply(softmax_vertical, softmax_horizontal)
        
        np.fill_diagonal(temp_result, np.multiply(softmax_res,1-softmax_res))
        
        return temp_result
    
    def loss_function(self, label_vec):
        output = self.a_cache[-1]
        return -1 * np.sum \
                    ((np.multiply(label_vec,np.log(output))) + \
                    (np.multiply(1-label_vec,np.log(1-output))))
                    
    def save_synapse_strength(self, file_suffix):
        trained_strength = [self.synapse_strength]
        file_name = "synapse_strength"
        file_name += "_" + file_suffix
        np.save(file_name, trained_strength)
                    
def log_print(my_str):
    if write_to_log:
        log_fp.write(my_str+ '\n')
        log_fp.flush()
    else:
        print(my_str)
     
class_num = 10
training_examples_per_class = 600
training_examples_num = class_num*training_examples_per_class
test_examples_per_class = 60
test_examples_num = class_num*test_examples_per_class
training_set,training_label_set,test_set,test_label_set = generate_reduced_data_sets(ModelClass.default_configuration['sensory_input_strength'], training_examples_per_class, test_examples_per_class)
training_iter_num = 500
model = ModelClass({},None,True)

# Training
for cur_training_iter in range(training_iter_num):
    log_print('Training iter ' + str(cur_training_iter))
    correct_count = 0
    
    cur_perm = np.random.permutation(training_examples_num)
    total_loss = 0
    for i in range(training_examples_num):
        true_label = training_label_set[cur_perm[i]]
        input_vec = training_set[:,[cur_perm[i]]]
        
        model.forward_pass(input_vec)
        
        label_vec = np.zeros((model.conf['layers_size'][-1],1))
        label_vec[true_label] = 1
        
        total_loss += model.loss_function(label_vec)
        if np.argmax(model.a_cache[-1]) == true_label:
            correct_count += 1
        model.backward_pass(label_vec, input_vec)
                
    log_print('\tDuring training current correct count ' + str(correct_count))
    log_print('\tTotal loss ' + str(total_loss))
    
    all_z_dict = {}
    
    correct_count = 0
    for i in range(training_examples_num):
        true_label = training_label_set[i]
        input_vec = training_set[:,[i]]
        
        model.forward_pass(input_vec)
        
        if np.argmax(model.a_cache[-1]) == true_label:
            correct_count += 1
            
        all_z_dict[i] = deepcopy(model.z_cache[0])
            
    log_print('\tNot during training current correct count ' + str(correct_count))
    
    correct_count = 0
    for i in range(test_examples_num):
        true_label = test_label_set[i]
        input_vec = test_set[:,[i]]
        
        model.forward_pass(input_vec)
        
        if np.argmax(model.a_cache[-1]) == true_label:
            correct_count += 1
            
    log_print('\tTesting current correct count ' + str(correct_count))
    
model.save_synapse_strength('good')