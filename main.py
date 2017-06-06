import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import data
#import neural_network

NUM_ROUNDS = 10

if __name__ == '__main__':

	X, y = data.get('../data/mnist_train.csv')
	X_test, y_test = data.get('../data/mnist_test.csv')


#########################################################
	# entropy
	X_train, y_train, X_remained, y_remained = data.init(X, y)

	xx_e = []
	yy_e = []
	for i in range(NUM_ROUNDS):
		xx_e.append(X_train.shape[0])
		X_train, y_train, X_remained, y_remained, score = data.active_learning(X_train, y_train, X_remained, y_remained, X_test, y_test, 'entropy')
		yy_e.append(score)

	pd.DataFrame([xx_e, yy_e]).to_csv('../results/entropy.csv')


#########################################################
	# bvsb

	X_train, y_train, X_remained, y_remained = data.init(X, y)

	xx_b = []
	yy_b = []
	for i in range(NUM_ROUNDS):
		xx_b.append(X_train.shape[0])
		X_train, y_train, X_remained, y_remained, score = data.active_learning(X_train, y_train, X_remained, y_remained, X_test, y_test, 'bvsb')
		yy_b.append(score)

	pd.DataFrame([xx_b, yy_b]).to_csv('../results/bvsb.csv')
	# plt.figure(figsize=(10, 10))
	# plt.plot(xx, yy)
	# plt.savefig('accuracy.png')

#########################################################
	# random

	X_train, y_train, X_remained, y_remained = data.init(X, y)

	xx_r = []
	yy_r = []
	for i in range(NUM_ROUNDS):
		xx_r.append(X_train.shape[0])
		X_train, y_train, X_remained, y_remained, score = data.active_learning(X_train, y_train, X_remained, y_remained, X_test, y_test, 'random')
		yy_r.append(score)

	pd.DataFrame([xx_r, yy_r]).to_csv('../results/random.csv')


	# plt.figure(figsize=(10, 10))
	# plt.plot(xx_r, yy_r, label='random')
	# plt.plot(xx_b, yy_b, label='BvSB')
	# plt.plot(xx_e, yy_e, label='entropy')
	# plt.ylabel('accuracy')
	# plt.xlabel('amount')
	# plt.gca().set_xlim(left=0)
	# plt.ylim(0, 1)
	# plt.legend()
	# plt.savefig('accuracy.png')



















