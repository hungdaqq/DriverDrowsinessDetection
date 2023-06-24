from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import Dataset
import time
import os
import os.path
import numpy as np
import sys
import pandas as pd

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
X_train, y_train = data.get_all_sequences_in_memory('training', hyper, seq)
X_test, y_test = data.get_all_sequences_in_memory('testing', hyper, seq)

checkpointer = ModelCheckpoint(filepath=os.path.join('data', 'checkpoints', 'lstm' + '-' + 'features' + '.{epoch:03d}-{val_loss:.3f}.hdf5'),verbose=1,save_best_only=True)

tb = TensorBoard(log_dir=os.path.join('data', 'logs', 'lstm'))

early_stopper = EarlyStopping(patience=5)

timestamp = time.time()

csv_logger = CSVLogger(os.path.join('data', 'logs', 'lstm' + '-' + 'training-' + str(timestamp) + '.log'))

print("-------------------------------------------------------------")
X_train = np.ravel(X_train)
X_train = X_train.reshape((train_no, seq,-1))
print("X_train.shape", X_train.shape)
print("y_train.shape", y_train.shape)

X_test = np.ravel(X_test)
X_test = X_test.reshape((test_no, seq,-1))
print("X_test.shape" ,X_test.shape)
print("y_test.shape" ,y_test.shape)
print('-------------------------------------------------------------')

batch_size = 2
nb_epoch = 20

rm = ResearchModels(len(data.classes),'lstm',data.seq_length, None)
rm.model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    validation_data=(X_test, y_test),
    verbose=1,
    callbacks=[tb, early_stopper, csv_logger],
    epochs=nb_epoch)

model_json = rm.model.to_json()
with open("rm.model.json",'w') as json_file:
	json_file.write(model_json)
rm.model.save_weights("rm.model.h5")
print("Model saved")