import imageio as io
import numpy as np
import os
from utils import encode
from keras.utils import Sequence

def get_image(img_filename):
    img = io.imread(img_filename)
    return np.array(img)

class MJSynthData(Sequence):

    def __init__(self, data_folder, annotation_file, batch_size):
    	# each line is of the form './path/to/img number'
        with open(os.path.join(data_folder, annotation_file), 'r') as f:
            filenames = [line.split(' ')[0][2:] for line in f]
        self.image_filenames = filenames
        self.data_folder = data_folder

        # each filename is like 'index number_label_number' 
        self.labels = [file.split('_')[1] for file in filenames]
        self.dataset_size = len(self.image_filenames)
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(self.dataset_size/float(self.batch_size))

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx+1)*self.batch_size, self.dataset_size)

        batch_x = np.array([get_image(os.path.join(self.data_folder, file_name)) 
                        for file_name in self.image_filenames[start_idx:end_idx]])
        batch_y = np.array([encode(label) 
                        for label in self.labels[start_idx:end_idx]])

        return batch_x, batch_y