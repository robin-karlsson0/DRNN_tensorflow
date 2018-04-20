import tensorflow as tf
import numpy as np


def create_placeholders(n_x, n_y):
    '''Store input variables as TensorFlow variables.
       Columns denotes samples.
    '''
    X = tf.placeholder(tf.float32, shape=(n_x, None), name='X')
    Y = tf.placeholder(tf.float32, shape=(n_y, None), name='Y')
    return X, Y


def initialize_parameters(input_features, h_layers, output_features):
    '''Function for initializing 'weights' and 'bias' arrays, or maps them to a parameters object if they already exist.
       - The number of hidden layers must be hard coded.

       Weight dimensions : [current layer, previous layer] (in activation units).
       
       Args:
         input_features  : Number of input features [scalar]
         h_layers        : Activations in hidden layers [array; 30,30,30]
         output_features : Number of outputs [scalar]

       Returns:
         parameters : TensorFlow objects containing all weights 'W' and biases 'b'
    '''

    # Check that number of hidden layers correspond to given input.
    hidden_layer_num = 3
    if(len(h_layers) != hidden_layer_num):
        print("ERROR : Inconsistent layers given (intiaialize_parameters)")
        return None

    tf.set_random_seed(1)
    # INPUT LAYER
    W1 = tf.get_variable('W1', [h_layers[0],input_features], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable('b1', [h_layers[0],1], initializer = tf.zeros_initializer())
    # HIDDEN LAYER 1
    W2 = tf.get_variable('W2', [h_layers[1],h_layers[0]], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable('b2', [h_layers[1],1], initializer = tf.zeros_initializer())
    # HIDDEN LAYER 2
    W3 = tf.get_variable('W3', [h_layers[2],h_layers[1]], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable('b3', [h_layers[2],1], initializer = tf.zeros_initializer())
    # OUTPUT LAYER
    W4 = tf.get_variable('W4', [output_features,h_layers[2]], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b4 = tf.get_variable('b4', [output_features,1], initializer = tf.zeros_initializer())

    # Store arrays in a dictionary object.
    parameters = {'W1':W1,
                  'b1':b1,
                  'W2':W2,
                  'b2':b2,
                  'W3':W3,
                  'b3':b3,
                  'W4':W4,
                  'b4':b4}

    return parameters


def forward_propagation(X, parameters):
    '''Forward propagate an input feature array 'X'.
       Weights and biases are stored in arrays retrieved from 'parameters'.
    '''

    # Retrieve arrays from the dictionary
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']

    Z1 = tf.add( tf.matmul(W1,X), b1 )
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add( tf.matmul(W2,A1), b2 )
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add( tf.matmul(W3,A2), b3 )
    A3 = tf.nn.relu(Z3)
    # OUTPUT LAYER
    Z4 = tf.add( tf.matmul(W4,A3), b4 )

    return Z4


def compute_cost_l2(Y, Y_label):
    '''Compute the 'L2 cost' for predicted values.
    '''
    pred = Y
    label = Y_label
    cost = tf.nn.l2_loss( tf.subtract(label, pred), name='cost_l2')
    return cost


def train_model(X_train, Y_train, X_test, Y_test, learning_rate = 0.001, num_epochs = 50000):
    '''Train the network on the given training data for the specified number of epochs, and test the trained model's
       accuracy.
       The network is stored in arrays recalled from 'parameters'.

       Args:
         X_train       : [features, samples]
         Y_train       : [labels, samples]
         X_test        : 
         Y_test        : 
         learning_rate : 
         num_epochs    : 

       Returns:
         parameters : TensorFlow objects containing all weights 'W' and biases 'b'
    '''

    print("Train model")
    print("X_train.shape = " + str(X_train.shape))
    print("Y_train.shape = " + str(Y_train.shape))
    
    #############################
    # BUILD COMPUTATIONAL GRAPH #
    #############################

    # Store 'feature' and 'sample' amount
    # n_x : input features
    # n_y : output features
    # m   : sample amount
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    # Store {X_train,Y_train} -> {X,y}
    X,Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    # Must be initializied INSIDE the computational graph for the optimizer
    # to be able to alter its values
    parameters = initialize_parameters(n_x, [30,30,30], 1)

    # Forward propagation computation
    Z4 = forward_propagation(X, parameters)

    # Cost function
    cost = compute_cost_l2(Z4, Y)

    # Backpropagation (TF optimizer)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    # Initialize all TF variables
    init = tf.global_variables_initializer()

    ###########################
    # RUN COMPUTATIONAL GRAPH #
    ###########################

    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(num_epochs):

            # How does this function work?
            _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})

            if(epoch%100 == 0):
                print("Epoch : {} ({})".format(epoch, minibatch_cost))

        # Save the parameters by mapping existing TF variables to an
        # explicit, external object
        parameters = sess.run(parameters)

        return parameters


def run_model(parameters, X_input):
    '''Compute the prediction 'Y_pred' for an input 'X_input' based on a model's 'parameters'.
       NOTE : Input feature column vector must be a 2D column vector!

       Args:
         X_in : [features, samples]
       Returns:
         Y_pred : 
    '''

    #print("Run model")
    #print("X_in.shape = " + str(X_input.shape))

    X = tf.placeholder(tf.float32, shape=(X_input.shape[0], None), name='X')

    output_layer = forward_propagation(X, parameters)

    # Initialize all TF variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        Y_pred = sess.run(output_layer, feed_dict={X: X_input})

        return Y_pred
