import tensorflow as tf #  import the tensorflow library as tf

mnist = tf.keras.datasets.mnist # handwritten digits training set

(x_train, y_train), (x_test, y_test) = mnist.load_data() # load the data into a matrix?
x_train, x_test = x_train/255.0, x_test/255.0 # convert to floating point and normalize


model = tf.keras.models.Sequential([ # build a keras sequential model
  tf.keras.layers.Flatten(input_shape=(28, 28)), # flatten the input (convert to a 1d array for input into the next layer)
  tf.keras.layers.Dense(128, activation='relu'), # Dense implements the operation output=activation(dot)input,kernel)+bias) where "activation" is the element-wise activation function passed as the activation argument. "kernel" is a weights matrix created by the layer, and the "bias" is a bias vector created by the layer. "relu" activation stands for "rectified linear unit" activation function max(x, 0)
  tf.keras.layers.Dropout(0.2), # dropout layer randomly sets input units to 0 with a frequency of rate at each step to prevent overfitting. Inputs not set to 0 are scaled by 1/(1-rate) here defined as .20.
  tf.keras.layers.Dense(10, activation='softmax') # softmax activation function converts the output vector to a probability distribution (range 0 to 1)
])

model.compile(optimizer='adam', # uses the adam algorithm which is described as a stochastic (random) gradient descent algorithm that is based on the idea of adaptive estimation of first-order and second-order moments. As quoted from Kingma et al, this algorithm is "computationally efficient, has little memory requirements, and is well suited for problems that are large in terms of data and parameters."
              loss='sparse_categorical_crossentropy', # loss functions serve to reduce the error in prediction; sparse_categorical_crossentropy is used when each dataset belongs to a single class
              metrics=['accuracy']) # metric is used to evaluate the model by way of creating two local variables: the number of correct predictions and the total number of predictions, and returning the ratio of which y_train is y_test

model.fit(x_train, y_train, epochs=10) # trains the model for a fix number of "epochs" (iterations on a dataset)

model.evaluate(x_test, y_test) # returns the loss value and metric values of the model in a test mode; batch_size undefined because w'ere using a sequential model
