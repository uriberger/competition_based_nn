import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

input_files_path_prefix = 'font_inputs'
N = 20

def plot_inputs():    
    # Plot the inputs
    inputs = generate_inputs()
    fig = plt.figure()
    for i in range(len(inputs)):
        fig.add_subplot(4,16,i+1)
        a1 = plt.imshow(inputs[i],cmap='gray')
        a1.axes.get_xaxis().set_visible(False)
        a1.axes.get_yaxis().set_visible(False)
    plt.show()

def generate_inputs(input_files_path_suffix):
    inputs = []
    input_file_paths = []
    
    input_files_path = input_files_path_prefix + '_' + input_files_path_suffix
    for _,_,files in os.walk(input_files_path):
        for file in files:
            cur_file_path = os.path.join(input_files_path, file)
            input_file_paths.append(cur_file_path)
            
    input_file_paths.sort()
    for cur_file_path in input_file_paths:
            image = imageio.imread(cur_file_path)
            orig_gray_image = np.reshape(image[:,:,[0]],(image.shape[0],image.shape[1]))
            gray_image = np.ones((N,N),dtype=np.int64)*255
            start_row = int((N-orig_gray_image.shape[0])/2)
            start_col = int((N-orig_gray_image.shape[1])/2)
            gray_image[start_row:start_row+orig_gray_image.shape[0],start_col:start_col+orig_gray_image.shape[1]] = orig_gray_image
            # We want black pixels in the original image to be represented by high values
            gray_image = 255 - gray_image
            
            inputs.append(gray_image)
            
    return inputs

def generate_training_set_no_generalization(input_files_path_suffix,sensory_input_strength):
    return [np.reshape(input_mat,(N**2,1))/255*sensory_input_strength for input_mat in generate_inputs(input_files_path_suffix)]

#plot_inputs()