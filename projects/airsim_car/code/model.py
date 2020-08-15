import tensorflow as tf

# imitation learning model
class Model():
    def __init__(self, lr=0.0001, config=None):
        # placeholder
        self.image = tf.placeholder(tf.float32, shape=(None, 144, 256, 4), name="image")
        self.X = tf.placeholder(tf.float32, shape=(None, 1), name="X")
        self.y = tf.placeholder(tf.int64, shape=None, name="y")
        self.train = tf.placeholder(tf.bool, name="train")

        # network convolutional layer
        layer = tf.layers.conv2d(inputs=self.image, filters=32, kernel_size=8, strides=4, padding="valid", activation=tf.nn.relu)
        layer = tf.layers.conv2d(inputs=layer, filters=64, kernel_size=5, strides=3, padding="valid", activation=tf.nn.relu)
        layer = tf.layers.conv2d(inputs=layer, filters=64, kernel_size=3, strides=1, padding="valid", activation=tf.nn.relu)
        layer = tf.contrib.layers.flatten(layer)

        # add speed information
        self.with_data = tf.concat([layer, self.X], axis=1)

        # dense layer
        layer = tf.layers.dense(inputs=self.with_data, units=100, activation=tf.nn.relu)
        layer = tf.layers.dense(inputs=layer, units=25, activation=tf.nn.relu)
        layer = tf.layers.dense(inputs=layer, units=10, activation=tf.nn.relu)

        # network output (throttle and steering)
        self.predictions = tf.layers.dense(inputs=layer, units=2, activation=None)

        # setup adam optimizer
        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.y, predictions=self.predictions))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

        # initialize session
        self.init = tf.global_variables_initializer()
        if config is None:
            self.sess = tf.Session()
        else:
            self.sess = tf.Session(config=config)

        # setup model saver
        self.saver = tf.train.Saver(max_to_keep=5)

    # load trained model
    def load(self, file_name):
        return self.saver.restore(self.sess, file_name)

    # save trained model
    def save(self, file_name, step):
        return self.saver.save(self.sess, file_name, global_step=step)