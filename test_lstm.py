# from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import Dataset
# import time
import os
import os.path
import numpy as np
import sys
import pandas as pd
import csv

import tensorflow as tf
from keras.models import Sequential, load_model

seq = int(sys.argv[1])
hyper = int(sys.argv[2])

df = pd.read_csv('./data/data_file.csv', header = None)
df_res = pd.DataFrame()

# Number of training/testing sequences
train_no = df[df.iloc[:,0] == 'training'].shape[0]
test_no = df[df.iloc[:,0] == 'testing'].shape[0]
# Load data
data = Dataset(
	seq_length=seq,
	class_limit=2,
)
X_test, y_test = data.get_all_sequences_in_memory('testing', hyper, seq)

print("-------------------------------------------------------------")
X_test = np.ravel(X_test)
X_test = X_test.reshape((test_no, seq,-1))
print("X_test.shape" ,X_test.shape)
print("y_test.shape" ,y_test.shape)
print('-------------------------------------------------------------')

# load model
rm = ResearchModels(len(data.classes),'lstm',data.seq_length, None)
rm.model = load_model('./rm.model.h5')
# testing
predictions = rm.model.predict(X_test)
loss, accuracy = rm.model.evaluate(X_test, y_test)

with open(os.path.join('data', 'data_file.csv'), 'r') as fin:
	reader = csv.reader(fin)
	data = list(reader)

filenames = []
for videos in data:
	if(videos[0] == 'testing'):
		i = 1
		while i <= int(hyper/seq):
			cnt = i*seq
			filenames.append(videos[2] + '-' + str(seq) + '-' + 'features' + str(cnt)+'.npy')
			i+=1
k = 0
for j in predictions:
	if j[0]>j[1]:
		print("Driver is alert with the confidence of",(j[0]*100),"%")
		df_res = df_res.append({'train_test': 'testing', 'Video': filenames[k], 'Accuracy': accuracy, 'Prediction': j[0],'Prediciton_class': 'Alert'},ignore_index = True)
	else:
		print("Driver is drowsy with the confidence of",(j[1]*100),"%")
		df_res = df_res.append({'train_test': 'testing', 'Video': filenames[k], 'Accuracy': accuracy, 'Prediction': j[1],'Prediciton_class': 'Drowsy'},ignore_index = True)
	k+=1
df_res.to_csv('./data/result_file.csv',index=False,header=False)