from keras.layers import Dense, Flatten, Dropout
from keras.layers import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from collections import deque
from keras.metrics import Recall, Precision

class ResearchModels():
    def __init__(self, nb_classes, model, seq_length, saved_model=None, features_length=2048):

        # Set defaults.
        self.seq_length = seq_length
        self.load_model = load_model
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        self.feature_queue = deque()

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy', Recall(), Precision()]
        #if self.nb_classes >= 10:
        #   metrics.append('top_k_categorical_accuracy')

        # Get the appropriate model.
        print("Loading LSTM model.")
        self.input_shape = (seq_length, features_length)
        self.model = self.lstm()
        # Now compile the network.
        optimizer = Adam(lr=0.0001)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
	
        print(self.model.summary())

    def lstm(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        model = Sequential()
        model.add(LSTM(2048, return_sequences=True, input_shape=self.input_shape,dropout=0.5))
        model.add(Flatten())
        model.add(Dense(1024, activation='elu'))
        model.add(Dense(512, activation='elu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='sigmoid'))
	
        return model