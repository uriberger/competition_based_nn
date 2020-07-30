import numpy as np
#import idx2numpy
import os

MNIST_data_dir_path = os.path.join('..','MNIST_data')
TRAINING_IMAGE_FILENAME = 'train-images.idx3-ubyte'
TRAINING_LABEL_FILENAME = 'train-labels.idx1-ubyte'
TEST_IMAGE_FILENAME = 't10k-images.idx3-ubyte'
TEST_LABEL_FILENAME = 't10k-labels.idx1-ubyte'

CLASS_NUM = 10

'''def generate_full_data_sets(image_filename, label_filename):
    image_file_path = os.path.join(MNIST_data_dir_path,image_filename)
    examples = idx2numpy.convert_from_file(image_file_path)
    input_size = examples.shape[1]*examples.shape[2]
    m = examples.shape[0]
    examples = np.transpose(examples, axes=[1,2,0])
    examples = np.reshape(examples, (input_size,m))
    examples = examples / 255
    
    label_file_path = os.path.join(MNIST_data_dir_path,label_filename)
    labels = idx2numpy.convert_from_file(label_file_path)
    
    sorted_examples = np.zeros(examples.shape)
    sorted_labels = np.zeros(labels.shape,dtype=int)
    
    class_ind_list = [0]
    for cur_class in range(1,CLASS_NUM):
        class_ind_list.append(class_ind_list[-1]+len([x for x in labels if x == cur_class-1]))
    
    for example_ind in range(examples.shape[1]):
        cur_class = labels[example_ind]
        sorted_examples[:,class_ind_list[cur_class]] = examples[:,example_ind]
        sorted_labels[class_ind_list[cur_class]] = cur_class
        class_ind_list[cur_class] += 1
    
    return sorted_examples,sorted_labels'''

def generate_reduced_data_sets(sensory_input_strength,num_of_train_examples_for_each_class,num_of_test_examples_for_each_class):
    '''full_training_set,full_training_label_set = generate_full_data_sets(TRAINING_IMAGE_FILENAME, TRAINING_LABEL_FILENAME)
    full_test_set,full_test_label_set = generate_full_data_sets(TEST_IMAGE_FILENAME, TEST_LABEL_FILENAME)
    np.save(os.path.join('..','mnist_data'),[full_training_set,full_training_label_set,full_test_set,full_test_label_set])
    assert False'''
    
    data_list = np.load(os.path.join('..','mnist_data.npy'))
    full_training_set = data_list[0] * sensory_input_strength
    full_training_label_set = data_list[1]
    full_test_set = data_list[2] * sensory_input_strength
    full_test_label_set = data_list[3]
    
    num_of_train_examples_reduced = num_of_train_examples_for_each_class*CLASS_NUM
    reduced_training_set = np.zeros((full_training_set.shape[0],num_of_train_examples_reduced))
    reduced_training_label_set = np.zeros((num_of_train_examples_reduced,),dtype=int)
    for cur_class in range(CLASS_NUM):
        begin_ind = cur_class*num_of_train_examples_for_each_class
        end_ind = (cur_class+1)*num_of_train_examples_for_each_class
        reduced_training_set[:,begin_ind:end_ind] = full_training_set[:,full_training_label_set==cur_class][:,:num_of_train_examples_for_each_class]
        reduced_training_label_set[begin_ind:end_ind] = cur_class 
        
    num_of_test_examples_reduced = num_of_test_examples_for_each_class*CLASS_NUM
    reduced_test_set = np.zeros((full_test_set.shape[0],num_of_test_examples_reduced))
    reduced_test_label_set = np.zeros((num_of_test_examples_reduced,),dtype=int)
    for cur_class in range(CLASS_NUM):
        begin_ind = cur_class*num_of_test_examples_for_each_class
        end_ind = (cur_class+1)*num_of_test_examples_for_each_class
        reduced_test_set[:,begin_ind:end_ind] = full_test_set[:,full_test_label_set==cur_class][:,:num_of_test_examples_for_each_class]
        reduced_test_label_set[begin_ind:end_ind] = cur_class
        
    return reduced_training_set,reduced_training_label_set,reduced_test_set,reduced_test_label_set