
from scipy.misc import imread,imresize
import numpy as np
import os
from keras.utils import Sequence


data_folder = '/Users/nitikasaran/hindiOCR/mjsynth_data/ramdisk/max/90kDICT32px'
train_file = 'annotation_train.txt'
test_file = 'annotation_test.txt'

def get_image(img_filename):
    # load grayscale
    img = imread(img_filename,"L")
    # resize to height 32
    width = int((32/float(img.shape[0])) * img.shape[1])
    img = imresize(img, (32, width))
    # rescale
    img = np.array(img,dtype=np.float32)
    img = (img / 255) - 0.5
    return img


class MJSynthData(Sequence):

    def __init__(self, data_folder, annotation_file,batch_size ):
    	# each line is of the form 'imgpath number' where imgpath starts with './'
    	with open(os.path.join(data_folder,annotation_file),'r') as f:
	        filenames = [line.split(' ')[0][2:] for line in f]
        self.image_filenames = filenames
        # each filename is like index number_label_number 
        self.labels = [file.split('_')[1] for file in filenames]
        self.dataset_size = len(self.image_filenames)
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil( self.dataset_size/ float(self.batch_size))

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min( (idx + 1) * self.batch_size, self.dataset_size)
        batch_x = self.image_filenames[start_idx:end_idx]
        batch_y = self.labels[start_idx:end_idx]
        images = [get_image(os.path.join(data_folder,file_name)) for file_name in batch_x]
        return images, np.array(batch_y)