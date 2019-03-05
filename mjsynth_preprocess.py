import numpy as np
import imageio as io
import os
import sys
from utils import rgb2gray, process
from pathlib import Path

def process_images(base_path, save_path, file):
    corrupt_files = open(os.path.join(save_path, 'corrupt_files.txt'), 'w')

    os.chdir(base_path)
    with open(file, 'r') as fl:
        paths = fl.read().split('\n')

    mjsynth_folder = os.path.join(save_path, 'mjsynth_processed')

    for path in paths:
        path_comp = path[2:].split(' ')[0].split('/')

        orig_img_path = os.path.join(base_path, path_comp[0], path_comp[1], path_comp[2])

        proc_img_path = os.path.join(mjsynth_folder, path_comp[0], path_comp[1])
        proc_img_name = path_comp[2]

        # Create the directory if it does not already exist.
        try:
            os.makedirs(proc_img_path)
        except FileExistsError:
            # directory already exists
            pass

        # Try to read and process the file, otherwise skip it
        try:
            img_file_path = Path(os.path.join(proc_img_path, proc_img_name))

            # Check if file already exists, if yes then skip.
            # Useful if script fails in between, due to memory limit errors
            if not img_file_path.is_file():
                orig_img = io.imread(orig_img_path)
                proc_img = process(orig_img)
                io.imwrite(img_file_path, proc_img)

            # Should prevent inodes exceeded errors
            # os.remove(orig_img_path)
        except:
            # Write the corrupt file path in a separate file to process them later.
            corrupt_files.write(path + '\n')
            pass

    corrupt_files.close()

def main(args):
    process_images(args[0], args[1], args[2])

if __name__ == '__main__':
    args = sys.argv
    if len(args) != 4:
        print('Invalid number of arguments. Correct format\n python mjsynth_preprocess.py <base_path> <save_path> <annotation_file_name>')
        exit()
    else:
        main(args[1:])