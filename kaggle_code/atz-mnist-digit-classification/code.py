print("="*80)
print("Framework for 1-NN.")
print(" ")
print("Author: Dylan Pizzo")
print(" ")
print("-"*80)

# PJA: How to read spreadsheet
import numpy as np
import pandas as pd # data processing library
import os

# list current directory
print(" ")
print("Files in local directory:")
print(os.listdir("../input"))

# read the file
dataFilename = "../input/train_small.csv"
print(" ")
print("Loading data from file %s"%dataFilename)
df_train = pd.read_csv(dataFilename)

print(" ")
print("Spreadsheet columns names:")
print(df_train.columns)

label_data = df_train['label'].values
sample_labels = list(label_data)
columns = [df_train['pixel_{:0>3}'.format(i)].values for i in range(784)]
samples = list(zip(*columns))

def mydist(v1,v2):
    return max((abs(v1[i]-v2[i]) for i in range(784)))

def predict_label(v1):
    min_dist = mydist(v1, samples[0])
    label = sample_labels[0]
    ind = 0
    for v2 in samples:
        new_dist = mydist(v1, v2)
        if new_dist < min_dist:
            min_dist = new_dist
            label = sample_labels[ind]
        ind += 1
    return label

print(" ")
print("Write the prediction results as a spreadsheet:")

testdataFilename = "../input/train_full.csv"
print(" ")
print("Loading data from test file %s"%testdataFilename)
df_test = pd.read_csv(testdataFilename)

testcolumns = [df_test['pixel_{:0>3}'.format(i)].values for i in range(784)]
testvectors = list(zip(*testcolumns))

for j in range(10):
    print('SAMPLE 20000+',j)
    print("prediction:")
    print(predict_label(testvectors[20000+j]))
    test_labels = list(df_test['label'].values)
    print("supposed to be:")
    print(test_labels[20000+j])

#numPts   = np.size(x_data,0);
#array_id = np.arange(0,numPts,dtype = np.int32);
#array_x  = x_data;
#array_y  = a*x_data + b; # COMPUTE BASED ON PREDICTION

#print("numPts = " + str(numPts));
#print("array_id = " + str(array_id));
#print("array_x = " + str(array_x));
#print("array_y = " + str(array_y));

#dataDict = {'id': array_id, 'x' : array_x, 'y' : array_y}; # for now we give both predicted public and private rows the same value

#df_predict = pd.DataFrame(dataDict);  # create a data frame to store data
#print(df_predict);

#print(" ");
#print("Writing file");
#predictFilename = 'MyPrediction.csv';
#print("predictFilename = %s"%predictFilename);
#df_predict.to_csv(predictFilename,index=False);
#print(" ");

#print("-"*80);
#print("Done");
#print("="*80);