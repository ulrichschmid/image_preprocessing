import gzip, cPickle
import numpy as np
import sys

#script that splits off a part from a data.pkl.gz file into a smaller subset file (not randomly)

#parameters
split = 0.01
path = "G:/machine learning/welding_data_28.pkl.gz"
output_path = "G:/machine learning/welding_data_28_smaller.pkl.gz"

#main script
#read dataset
f = gzip.open(path, 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

#split data
train_set_new = np.array(train_set[0][0:int(len(train_set[0])*split)]), np.array(train_set[1][0:int(len(train_set[1])*split)])
val_set_new = np.array(valid_set[0][0:int(len(valid_set[0])*split)]), np.array(valid_set[1][0:int(len(valid_set[1])*split)])
test_set_new = np.array(test_set[0][0:int(len(test_set[0])*split)]), np.array(test_set[1][0:int(len(test_set[1])*split)])

dataset_new = [train_set_new, val_set_new, test_set_new]

print("dataset created")
print("train units: " + str(len(train_set_new[0])))
print("labels: " + str(len(train_set_new[1])))
print("validation units: " + str(len(val_set_new[0])))
print("labels: " + str(len(val_set_new[1])))
print("test units: " + str(len(test_set_new[0])))
print("labels: " + str(len(test_set_new[1])))

#write data
f = gzip.open(output_path,'wb')
cPickle.dump(dataset_new, f, protocol=2)
f.close()

print("written data to " + output_path)
sys.exit()