import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import drnn_tf


# Read training data
import_data = np.genfromtxt("training_data.csv", delimiter=' ')


X_train = np.zeros((2,import_data.shape[0]))
Y_train = np.zeros((1,import_data.shape[0]))

for i in range(0,import_data.shape[0]): # loop through rows
    data_row = import_data[:][i]
    X_train[0][i] = data_row[0]
    X_train[1][i] = data_row[1]
    Y_train[0][i] = data_row[2]

# EMPTY PARAMETER DICTIONARY WHICH WILL STORE WEIGHTS
input_features = 2
layers = [30,30,30]
output_features = 1


#parameters = drnn_tf.initialize_parameters(input_features, layers, output_features)

parameters = drnn_tf.train_model(X_train, Y_train, None, None)

#result = drnn_tf.run_model(parameters, X_train[:,0,None])


x_range = np.outer(np.ones((50,)), np.linspace(0,1,50))
x1_range = x_range
x2_range = x_range.T


def test_pred_2D(parameters, x1_range, x2_range, scatter_data):

    X_pred = np.zeros((2,x1_range.size))
    y_map  = np.zeros(x1_range.shape)
    index = 0

    for j in range(0,x1_range.shape[1]):
        for i in range(0,x1_range.shape[0]):
            X_pred[0][index] = x1_range[i][j]
            X_pred[1][index] = x2_range[i][j]
            index = index + 1

    Y_pred = drnn_tf.run_model(parameters, X_pred)

    index = 0
    for j in range(0,x1_range.shape[1]):
        for i in range(0,x1_range.shape[0]):
            y_map[i][j] = Y_pred[0,index]
            index = index + 1

    #return y_map

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x1_range, x2_range, y_map, cmap='coolwarm', alpha=0.8)
    ax.scatter(scatter_data[:,0],scatter_data[:,1],scatter_data[:,2], color='k')
    plt.show()

test_pred_2D(parameters, x1_range, x2_range, import_data)