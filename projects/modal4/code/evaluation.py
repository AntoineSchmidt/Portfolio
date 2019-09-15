import tensorflow as tf

# handel tensorboard file
class Evaluation:
    def __init__(self, store_dir):
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.tf_writer = tf.summary.FileWriter(store_dir)

        self.tf_loss = tf.placeholder(tf.float32, name="loss_summary")
        tf.summary.scalar("loss", self.tf_loss)

        self.performance_summaries = tf.summary.merge_all()

    # write data to tensorboard file
    def write_episode_data(self, episode, eval_dict):
       summary = self.sess.run(self.performance_summaries, feed_dict={self.tf_loss: eval_dict["loss"]})
       self.tf_writer.add_summary(summary, episode)
       self.tf_writer.flush()

    # close tensorboard writer
    def close_session(self):
        self.tf_writer.close()
        self.sess.close()