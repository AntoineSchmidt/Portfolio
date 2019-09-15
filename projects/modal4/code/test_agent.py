import airsim
import math
import numpy as np
import tensorflow as tf

from model import Model


# reward algorithm (rewarding travelled distance)
def compute_reward(car_state):
    global last_position

    reward = 0
    if last_position is not None:
        pd = car_state.kinematics_estimated.position
        position = np.array([pd.x_val, pd.y_val])
        reward = math.sqrt((position[0] - last_position[0])**2 + (position[1] - last_position[1])**2)
        last_position = position
    return reward

# dynamically grow the memory used on the GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  


print("Connecting to AirSim...")
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()

print("Loading Model...")
agent = Model(config=config)
agent.sess.run(agent.init)
agent.load("model/agent.ckpt-4500")

# retrieve initial position
last_position = None
compute_reward(client.getCarState())

# evaluate model
reward_list = []
for i in range(10):
    steps = 0
    accumulated_reward = 0
    while steps < 5000:
        # gather car data, normalize speed
        car_state = np.array(client.getCarState().speed)[np.newaxis, np.newaxis, :]
        car_state /= 15
        car_image = client.simGetImages([airsim.ImageRequest("MyCamera1", airsim.ImageType.Segmentation, False, False)])[0]

        try:
            # preprocess image to 4 channel image
            img1d = np.fromstring(car_image.image_data_uint8, dtype=np.uint8)
            img_rgba = img1d.reshape(car_image.height, car_image.width, 4)
            img = np.expand_dims(img_rgba, axis=0) / 255.
        except:
            print("Fetching image failed!")
            continue

        # predict action
        action = agent.sess.run(agent.predictions, feed_dict={agent.image: img, agent.X: car_state, agent.train: False})[0]
        action[1] /= 2

        # update reward
        reward = compute_reward(client.getCarState())
        accumulated_reward += reward
        collision_info = client.simGetCollisionInfo()
        if collision_info.has_collided:
            print('Crashed', accumulated_reward)
            break
        print(i, steps, action)

        # execute action
        car_controls.throttle = float(action[0])
        car_controls.steering = float(action[1])
        client.setCarControls(car_controls)

        steps += 1

    # finish episode
    client.reset()
    reward_list.append(accumulated_reward)
    accumulated_reward = 0

# show rewards
print(reward_list)
print("mean", np.mean(reward_list), "std", np.std(reward_list))

# restore to original state
client.reset()
client.enableApiControl(False)