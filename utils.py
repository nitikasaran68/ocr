import numpy as np
import imageio as io
import imutils
from PIL import Image
from param import alphabet, rev_alphabet, max_string_len

def encode(label):
    len_label = len(label)
    ret = np.ones(max_string_len) * len(alphabet)
    for idx, char in zip(range(len_label), label):
        ret[idx] = alphabet[char]
    return ret.astype(int)

# Reverse translation of numerical classes back to characters
def decode(encoded_label):
    len_alphabet = len(rev_alphabet)
    ret = []
    for encoded_char in encoded_label:
        if encoded_char == len_alphabet:  # CTC Blank
            ret.append("")
        else:
            ret.append(rev_alphabet[encoded_char])
    return "".join(ret)

def rem_duplicates(encoded_label_dupl):
    len_label = len(encoded_label_dupl)
    len_alphabet = len(alphabet)
    encoded_label = [encoded_label_dupl[0]]
    for idx in range(1, len_label):
        if encoded_label_dupl[idx] == len_alphabet or encoded_label_dupl[idx] != encoded_label_dupl[idx-1]:
            encoded_label.append(encoded_label_dupl[idx])
    return encoded_label

def activation_to_label(y_pred):
    pred_labels = np.argmax(y_pred[0,:,:], axis=1).tolist()

    return decode(rem_duplicates(pred_labels))

def get_image(img_filename):
    img = io.imread(img_filename)
    img = np.array(img)
    return np.expand_dims(img, axis=-1)

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def process(img):
    gray_img = rgb2gray(img)
    rotated_img = imutils.rotate_bound(gray_img, 90)
    final_img = Image.fromarray(rotated_img).resize((32, 100))
    return np.array(final_img, dtype='uint8')
