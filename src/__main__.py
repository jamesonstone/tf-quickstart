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

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)
