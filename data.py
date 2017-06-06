import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score

START_SIZE = 5
BATCH_SIZE = 10
NUM_CLASSES = 10

# достаем данные в явном виде
def get(path):
	data = pd.read_csv(path, header=None)

	X = data.drop(0, axis = 1).values
	y = data[0].values

	X = X.astype('float32')
	X = X / 255.0

	return([X, y])

# инициализируем стартовый набор 
def init(X, y):
	# ИСПРАВИТЬ!!! убрать одну тысячу
	X_remained = X[START_SIZE * NUM_CLASSES:5000]
	y_remained = y[START_SIZE * NUM_CLASSES:5000]
	X_train = X[:START_SIZE * NUM_CLASSES]
	y_train = y[:START_SIZE * NUM_CLASSES]

	return(X_train, y_train, X_remained, y_remained)

# вычисляем параметры A и B
def get_params(X, y):
	params = np.empty((NUM_CLASSES, NUM_CLASSES), dtype=object)

	for i in range(NUM_CLASSES):
		X_1 = X[y == i]
		y_1 = np.full(X_1.shape[0], 1, dtype=int)
		for j in range(i):
			X_2 = X[y == j]
			y_2 = np.full(X_2.shape[0], -1, dtype=int)

			X_model = np.concatenate((X_1, X_2))
			y_model = np.concatenate((y_1, y_2))
			model = get_model(X_model, y_model)
			A, B = 0, 0
			A, B = minimize(calculate_function, [A, B], args=(X_model, y_model, model))['x']
			params[i, j] = [model, A, B]

	return(params)

# строим SVM модель
def get_model(X, y):
    clf = SVC()
    model = clf.fit(X, y)
    return(model)

# вычисляем функцию, которую минимизируем
def calculate_function(params, X, y, model):
    prob_1 = 1 / (1 + np.exp(model.predict(X) * params[0] + params[1]))
    n_pos = 0
    n_neg = 0
    coefs = []
    for i in y:
        if i == 1:
            n_pos += 1
        else:
            n_neg += 1
            
    for i in y:
        if i == 1:
            coefs.append((n_pos + 1) / (n_pos + 2))
        else:
            coefs.append(1 / (n_neg + 2))
            
    coefs = np.array(coefs)
    result = -np.sum(coefs * np.log(prob_1) + (1 - coefs) * np.log(1 - prob_1))
    return(result)

# добавляем элементы рандомно
def add_random(X_train, y_train, X_remained, y_remained):
	reorder = np.random.permutation(X_remained.shape[0])
	X_train = np.concatenate((X_train, X_remained[reorder[:BATCH_SIZE]]))
	y_train = np.concatenate((y_train, y_remained[reorder[:BATCH_SIZE]]))
	X_remained = X_remained[BATCH_SIZE:]
	y_remained = y_remained[BATCH_SIZE:]

	return(X_train, y_train, X_remained, y_remained)

def get_probabilities(x, params):
	# получаем матрицу вероятностей
	probs = np.zeros(params.shape)
	for i in range(NUM_CLASSES):
		for j in range(i):
			p_ij = 1 / (1 + np.exp(params[i, j][0].predict(x.reshape(1, -1)) * params[i, j][1] + params[i, j][2]))
			probs[i, j] = p_ij
			probs[j, i] = 1 - p_ij
	return(probs)

def calculate_probabilities_function(p, probs):
	# считаем функционал, который нужно минимизировать
	result = 0
	for i in range(NUM_CLASSES):
		for j in range(NUM_CLASSES):
			if i != j:
				result += (p[i] * probs[j, i] - p[j] * probs[i, j]) ** 2
	return(result)

def get_probabilities_2(x, probs):
	p_start = np.full(NUM_CLASSES, 1 / NUM_CLASSES)
	cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})
	bnds = tuple((0,1) for x in p_start)
	p = minimize(calculate_probabilities_function, p_start, method='SLSQP', bounds=bnds, constraints=cons, args=(probs))
	return(p['x'])

def get_probabilities_2_full(X_remained, params):
	all_p = []
	for x in X_remained:
		probs = get_probabilities(x, params)
		tmp = get_probabilities_2(x, probs)
		all_p.append(tmp)
	return(all_p)

def get_indexes(all_p, strategy):
	# ищем элементы, которые будем добавлять
	strategy_values = []
	if strategy == 'bvsb':
		for p in all_p:
			tmp = strategy_bvsb(p)
			strategy_values.append(tmp)
		strategy_values = np.array(strategy_values)
	# indexes = strategy_values.argsort()[::-1][:BATCH_SIZE]
		indexes = strategy_values.argsort()[:BATCH_SIZE]

	if strategy == 'entropy':
		for p in all_p:
			tmp = strategy_entropy(p)
			strategy_values.append(tmp)
		strategy_values = np.array(strategy_values)
		indexes = strategy_values.argsort()[::-1][:BATCH_SIZE]

	print(strategy_values[indexes])
	return(indexes)

def strategy_bvsb(p):
	# стратегия лучший против второго лучшего
	indexes = p.argsort()
	return(p[indexes[-1]] - p[indexes[-2]])

def strategy_entropy(p):
	# стратегия максимизации энтропии
	H = 0
	for i in p:
	    H -= i * np.log2(i)
	return H

def get_score(X_tst, y_tst, params):
	all_p = []
	for x in X_tst:
		probs = get_probabilities(x, params)
		tmp = get_probabilities_2(x, probs)
		all_p.append(tmp)
	y_pred = np.argmax(all_p, axis=1)
	score = accuracy_score(y_tst, y_pred)
	return score

def active_learning(X_train, y_train, X_remained, y_remained, X_tst, y_tst, strategy):
	params = get_params(X_train, y_train)
	all_p = get_probabilities_2_full(X_remained, params)
	
	score = get_score(X_tst, y_tst, params)

	if strategy == 'random':
		return np.append(add_random(X_train, y_train, X_remained, y_remained), score)


	indexes = get_indexes(all_p, strategy)

	X_train = np.concatenate((X_train, X_remained[indexes]))
	y_train = np.concatenate((y_train, y_remained[indexes]))
	X_remained = np.delete(X_remained, indexes, axis=0)
	y_remained = np.delete(y_remained, indexes)

	# print(X_train.shape, X_remained.shape)

	return (X_train, y_train, X_remained, y_remained, score)
















