import os
import glob
import numpy as np
import pandas as pd

from scipy import ndimage

from evaluation import Evaluation
from model import Model


# read image data
def read_images(folder):
    files_left = sorted(glob.glob(folder + str("/img_MyCamera0_5*png")))
    images_left = np.array([ndimage.imread(file) for file in files_left])
    files_straight = sorted(glob.glob(folder + str("/img_MyCamera1_5*png")))
    images_straight = np.array([ndimage.imread(file) for file in files_straight])
    files_right = sorted(glob.glob(folder + str("/img_MyCamera2_5*png")))
    images_right = np.array([ndimage.imread(file) for file in files_right])
    return images_left, images_straight, images_right

# read text information
def read_text(file):
    df = pd.read_csv(file, sep="\t")
    labels = df[["Throttle", "Steering"]].values
    data = df[["Speed"]].values
    return labels, data


if __name__ == "__main__":
    # train parameter
    batch_size = 128
    n_minibatches = 10000

    # initialize model
    agent = Model()
    agent.sess.run(agent.init)

    # setup tensorboard files
    tensorboard_train = Evaluation("tensorboard/train")
    tensorboard_valid = Evaluation("tensorboard/valid")

    # load data
    labels, driving_data = read_text("./data/airsim_rec.txt")
    images_left, images_straight, images_right = read_images("./data/images")

    # normalize data
    driving_data /= 15
    labels[:, 1] *= 2

    # correct steering for left and right images
    labels_left = labels[:] + np.array([0., 0.2])
    labels_straight = labels[:]
    labels_right = labels[:] - np.array([0., 0.2])

    # validation and training data split
    valid_size = int(labels.shape[0] * 0.04)

    labels_train = np.concatenate([labels_left[valid_size:], labels_straight[valid_size:], labels_right[valid_size:]])
    driving_data_train = np.concatenate([driving_data[valid_size:], driving_data[valid_size:], driving_data[valid_size:]])
    images_train = np.concatenate([images_left[valid_size:], images_straight[valid_size:], images_right[valid_size:]])

    labels_valid = np.concatenate([labels_left[:valid_size], labels_straight[:valid_size], labels_right[:valid_size]])
    driving_data_valid = np.concatenate([driving_data[:valid_size], driving_data[:valid_size], driving_data[:valid_size]])
    images_valid = np.concatenate([images_left[:valid_size], images_straight[:valid_size], images_right[:valid_size]])


    # samples a minibatch
    n_datapoints = labels_train.shape[0]
    def sample_minibatch():
        indices = np.random.choice(n_datapoints, batch_size)
        return images_train[indices] / 255., driving_data_train[indices], labels_train[indices]

    # run training on n minibatches
    for i in range(n_minibatches):
        images_batch, driving_data_batch, labels_batch = sample_minibatch()
        _, c = agent.sess.run([agent.optimizer, agent.loss], feed_dict={agent.image: images_batch, agent.X: driving_data_batch, agent.y: labels_batch, agent.train: True})
        tensorboard_train.write_episode_data(i, {"loss": c})
        print(i, c)

        # calculate validation
        if i % 100 == 0 or i == n_minibatches-1:
            c_valid = agent.sess.run(agent.loss, feed_dict={agent.image: images_valid / 255., agent.X: driving_data_valid, agent.y: labels_valid, agent.train: False})
            tensorboard_valid.write_episode_data(i, {"loss": c_valid})

            agent.save("model/agent.ckpt", step=i)
            print("validation loss: ", c_valid)