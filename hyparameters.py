from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch
from keras.models import Model
from keras.layers import Input, Embedding, Bidirectional, Concatenate, Dense, BatchNormalization, Activation, Dropout, LSTM
from keras.optimizers import Adam
import numpy as np
import json
import config

word_embedding_matrix = np.load(open(config.BASE_DIR + config.EMBEDDING_MATRIX, 'rb'))

with open(config.BASE_DIR + config.NB_WORDS, 'r') as f:
  nb_words = json.load(f)['nb_words']


class LSTM_HyperModel(HyperModel):
  def __init__(self, lstm_units, max_seq_length):
    self.lstm_units = lstm_units
    self.max_seq_length = max_seq_length 

  def build(self, hp):
    lyrics_input = Input(shape = (self.max_seq_length,))
    sim_input = Input(shape = (self.max_seq_length, ))

    lyrics = Embedding(nb_words+1,
                      config.EMBEDDING_DIM,
                      input_length = self.max_seq_length,
                      weights = [word_embedding_matrix],
                      trainable= False)(lyrics_input)
    lyrics = Bidirectional(LSTM(self.lstm_units))(lyrics)

    sim = Embedding(nb_words+1, 
                    config.EMBEDDING_DIM, 
                    input_length = self.max_seq_length,
                    weights = [word_embedding_matrix],
                    trainable = False)(sim_input)
    sim = Bidirectional(LSTM(self.lstm_units))(sim)

    merged = Concatenate()([lyrics, sim])

    for i in range(hp.Int('num_layers', 2, 6)):
      merged = Dense(hp.Choice('units', 
                              values = [32, 64, 128]))(merged)
      merged = BatchNormalization()(merged)
      merged = Activation('relu')(merged)
      merged = Dropout(config.DROPOUT)(merged)

    merged = Dense(1, activation = 'sigmoid')(merged)

    model = Model(inputs = [lyrics_input, sim_input], outputs = merged)
    opt = Adam(lr = hp.Choice('learning_rate', 
                              values = [0.01,0.001, 0.0025, 0.0001]))
    model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
    
    return model



hypermodel = LSTM_HyperModel(lstm_units = 128, max_seq_length = config.MAX_SEQ_LEN)

tuner = RandomSearch(
      hypermodel,
      objective = 'val_loss',
      max_trials = 5,
      overwrite = True
)





if __name__ == '__main__':

  print(tuner.search_space_summary())