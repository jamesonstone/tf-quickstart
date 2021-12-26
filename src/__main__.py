import tensorflow as tf

mnist = tf.keras.datasets.mnist # handwritten digits training set

(x_train, y_train), (x_test, y_test) = mnist.load_data() # load the data into a matrix?
x_train, x_test = x_train/255.0, x_test/255.0 # convert to floating point and normalize


model = tf.keras.models.Sequential([ # build a keras sequential model
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', # uses the adam algorithm which is described as a stochastic (random) gradient descent algorithm that is based on the idea of adaptive estimation of first-order and second-order moments. As quoted from Kingma et al, this algorithm is "computationally efficient, has little memory requirements, and is well suited for problems that are large in terms of data and parameters."
              loss='sparse_categorical_crossentropy', # loss functions serve to reduce the error in prediction; sparse_categorical_crossentropy is used when each dataset belongs to a single class
              metrics=['accuracy']) # metric is used to evaluate the model by way of creating two local variables: the number of correct predictions and the total number of predictions, and returning the ratio of which y_train is y_test

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)
