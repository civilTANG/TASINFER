print("="*80);
print("Example script to estimate regression coefficients from data set.");
print(" ");
print("Author: Paul J. Atzberger");
print(" ");
print("-"*80);

# PJA: How to read spreadsheet
import numpy as np;
import pandas as pd; # data processing library
import os;

# list current directory
print(" ");
print("Files in local directory:");
print(os.listdir("../input"));

# read the file
dataDilename = "../input/Kaggle_Training_Data.csv";
#dataDilename = "Kaggle_Test_Data.csv";
print(" ");
print("Loading data from file %s"%dataDilename);
df_train = pd.read_csv(dataDilename);

print(" ");
print("Spreadsheet data:");
print(df_train);

print(" ");
print("Spreadsheet columns names:");
print(df_train.columns);

x_data = df_train['x'].values;
y_data = df_train['y'].values;

print(" ");
print("Setup the linear system for regression.");
n = np.size(x_data,0);

A = np.zeros((2,2));
A[0,0] = np.dot(x_data,x_data);
A[0,1] = np.dot(x_data,np.ones(n));
A[1,0] = np.dot(np.ones(n),x_data.T);
A[1,1] = np.dot(np.ones(n),np.ones(n));

print(" ");
print("A = " + str(A));

RHS    = np.zeros(2);
RHS[0] = np.dot(x_data,y_data);
RHS[1] = np.dot(np.ones(n),y_data);

print(" ");
print("RHS = " + str(RHS));

q = np.linalg.solve(A,RHS);

a = q[0];
b = q[1];

print(" ");
print("a predicted = " + str(a));
print("b predicted = " + str(b));

print(" ");
print("Write the prediction results as a spreadsheet:");

numPts   = np.size(x_data,0);
array_id = np.arange(0,numPts,dtype = np.int32);
array_x  = x_data;
array_y  = a*x_data + b;

print("numPts = " + str(numPts));
print("array_id = " + str(array_id));
print("array_x = " + str(array_x));
print("array_y = " + str(array_y));

dataDict = {'id': array_id, 'x' : array_x, 'y' : array_y}; # for now we give both predicted public and private rows the same value

df_predict = pd.DataFrame(dataDict);  # create a data frame to store data
print(df_predict);

print(" ");
print("Writing file");
predictFilename = 'MyPrediction.csv';
print("predictFilename = %s"%predictFilename);
df_predict.to_csv(predictFilename,index=False);
print(" ");

print("-"*80);
print("Done");
print("="*80);