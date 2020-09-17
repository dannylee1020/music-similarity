import config

import numpy as np
from sklearn.metrics import classification_report
from keras.models import load_model


if __name__ == '__main__':

	model = load_model(config.BASE_DIR + config.MODEL)
	lyrics_test = np.load(open(config.BASE_DIR + config.LR_TEST, 'rb'))
	sim_test = np.load(open(config.BASE_DIR + config.SIM_TEST, 'rb'))
	y_test = np.load(open(config.BASE_DIR + config.Y_TEST, 'rb'))
	y_pred =  model.predict([lyrics_test, sim_test])

	clr = classification_report(y_test, np.round(y_pred))
	print(clr)
