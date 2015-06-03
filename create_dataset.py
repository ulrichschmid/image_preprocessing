import os
import sys
import glob
import skimage
import skimage.io
import time
from PIL import Image
import numpy
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from logistic_sgd import LogisticRegression, load_data
import pandas as pd
import gzip, cPickle
import scipy.io

#this script processes all image files and labels from a given list of folder path tuples into one data file
#1st parameter: folder containing subfolders of the already resized images
#2nd parameter: folder containing the label .mat files
#3rd parameter: list of strings to exclude


#input parameters
output_path = "G:/machine learning/welding_data_28"
todo = [("G:/machine learning/welding_data_resized_28/P76","G:/machine learning/welding data raw/P76/logfiles/labels/", ["W11","W31","W35","W37","W92"]),
        ("G:/machine learning/welding_data_resized_28/P77","G:/machine learning/welding data raw/P77/logfiles/labels/", ["W21","W91","W92"]),
        ("G:/machine learning/welding_data_resized_28/P80","G:/machine learning/welding data raw/P80/logfiles/labels/", ["W06","W14"]),
        ("G:/machine learning/welding_data_resized_28/P81","G:/machine learning/welding data raw/P81/logfiles/labels/", ["W01","W02","W03","W21","W22","W23","W24","W25"])
         ]
train_set_size = 1.0/3.0    #part of the final set to be used for training
validation_set_size = 1.0/3.0    #part of the final set to be used for validation
#note: test_set_size = 1 - train_set_size - validation_set_size
shuffle_data_label_pairs = True    #if True, the final set will be randomly re-ordered

#helper functions
def correct_label(unchecked_label):
    """assures that a label is within a given list, and fixes the label for some cases"""
    allowed_labels = [[1],[2],[3],[4]]
    fixable_labels = [[2.5],[3.5]]
    if unchecked_label in allowed_labels:
        return unchecked_label
    print "trying to fix invalid label " + str(unchecked_label)
    assert unchecked_label in fixable_labels
    if unchecked_label == [2.5]:
        return [2]
    elif unchecked_label == [3.5]:
        return [3]

def get_dataset_from_dirs(images_location, labels_location, exclude_list):
    """ - pulls all images from a the subfolders of a given directory
        - pulls all labels from .mat files in a given directory
        - excludes folders and .mat files containing the given exclusion strings"""
    print("Processing labels from " + labels_location)

    #get images' data
    dataset = []
    for subfolder in os.listdir(images_location):
        print("Processing images from " + images_location + "/" + subfolder)
        if os.path.isdir(images_location + "/" + subfolder):
            for file_count, file_name in enumerate(sorted(glob.glob(images_location + "/" + subfolder + "/" + "*.png"),key=len)):
                exclude = False
                for excl_str in exclude_list:
                    exclude = exclude or excl_str in file_name
                if not exclude:
                    img = Image.open(file_name)
                    pixels = list(img.getdata())
                    #convert from int[0-255] to a float32[0-1]
                    pixels = [numpy.float32(float(x)/255.0) for x in pixels]
                    dataset.append(pixels)
            print("\t %s images processed"%file_count)


    #get labels, if no path specified return without adding any labels
    labels = []
    if len(labels_location) > 0:
        for file_count, file_name in enumerate(sorted(glob.glob(labels_location + "*.mat"),key=len)):
            exclude = False
            for excl_str in exclude_list:
                exclude = exclude or excl_str in file_name
            if not exclude:
                mat = scipy.io.loadmat(file_name)
                for label in list(mat["labels"]):
                    labels += list(correct_label(label))
        assert len(dataset) == len(labels)

    #return result
    return (dataset, labels)

#main script
data = []
labels = []
for path_tuple in todo:
    data_tuple = get_dataset_from_dirs(path_tuple[0],path_tuple[1],path_tuple[2])
    data += data_tuple[0]
    labels += data_tuple[1]

#shuffle data-label pair order
if shuffle_data_label_pairs:
    print "shuffling data-label pairs.."
    ordered = np.transpose(np.array([data, labels]))
    np.random.shuffle(ordered)
    unordered = np.transpose(ordered)
    data = unordered[0]
    labels = unordered[1]

train_amount = int(train_set_size*float(len(data)))
validation_amount = int(validation_set_size*float(len(data)))

train_set_x = list(data[:train_amount])
val_set_x = list(data[train_amount:train_amount+validation_amount])
test_set_x = list(data[train_amount+validation_amount:])

train_set_y = list(labels[:train_amount])
val_set_y = list(labels[train_amount:train_amount+validation_amount])
test_set_y = list(labels[train_amount+validation_amount:])

train_set = np.array(train_set_x), np.array(train_set_y)
val_set = np.array(val_set_x), np.array(val_set_y)
test_set = np.array(test_set_x), np.array(test_set_y)

dataset = [train_set, val_set, test_set]

print("dataset created")
print("train units: " + str(len(train_set_x)))
print("labels: " + str(len(train_set_y)))
print("validation units: " + str(len(val_set_x)))
print("labels: " + str(len(val_set_y)))
print("test units: " + str(len(test_set_x)))
print("labels: " + str(len(test_set_y)))

f = gzip.open(output_path + ".pkl.gz",'wb')
cPickle.dump(dataset, f, protocol=2)
f.close()

print("written data to " + output_path)
sys.exit()