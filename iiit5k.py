
from scipy.misc import imread,imresize
import numpy as np
from scipy.io import loadmat
import os
from keras.utils import Sequence


traindata_mat_file = "traindata.mat"
testdata_mat_file = "testdata.mat"

data_folder = "/Users/nitikasaran/vision/IIIT5K/"

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

class IIIT5KData(Sequence):

    def __init__(self, data_folder, mat_file,batch_size ):
        mat = loadmat(os.path.join(data_folder,mat_file))
        if 'traindata' in mat.keys():
            data = mat['traindata']
        else:
            data = mat['testdata']
        self.image_filenames = data['ImgName'][0]
        self.labels = data['GroundTruth'][0]
        self.dataset_size = len(self.image_filenames)
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil( self.dataset_size/ float(self.batch_size))

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min( (idx + 1) * self.batch_size, self.dataset_size)
        batch_x = self.image_filenames[start_idx:end_idx]
        batch_y = self.labels[start_idx:end_idx]
        images = [get_image(os.path.join(data_folder,file_name[0])) for file_name in batch_x]
        labels = [label[0] for label in batch_y]
        return images, np.array(labels)


