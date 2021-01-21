#!/usr/bin/env python3

import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import rospy
from std_msgs.msg import UInt16
import os
from tensorflow.keras.models import load_model


def process_image(img_path):
    img = load_img(img_path, target_size=(30, 30))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img


def predict_model(img_path, model):
    labels = ['20 km/h', '30 km/h', '50 km/h', '60 km/h', '70 km/h', '80 km/h', '80 km/h end', '100 km/h',
              '120 km/h', 'No overtaking', 'No overtaking for tracks', 'Crossroad with secondary way',
              'Main road', 'Give way', 'Stop', 'Road up', 'Road up for track', 'Brock', 'Other dangerous',
              'Turn left', 'Turn right', 'Winding road', 'Hollow road', 'Slippery road', 'Narrowing road',
              'Roadwork', 'Traffic light', 'Pedestrian', 'Children', 'Bike', 'Snow', 'Deer', 'End of the limits',
              'Only right', 'Only left', 'Only straight', 'Only straight and right', 'Only straight and left',
              'Take right', 'Take left', 'Circle crossroad', 'End of overtaking limit', 'End of overtaking limit for track']

    image = process_image(img_path)
    pred = np.argmax(model.predict(image), axis=1)

    prediction = labels[pred[0]]
    print(prediction)
    return prediction


def choose_way(sign):
    msg = 0
    if sign == 'Stop':
        msg = 2
        return msg
    if sign == 'Only right':
        msg = 3
        return msg
    if sign == 'Only left':
        msg = 4
        return msg
    else:
        msg = 1
        return msg


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

MODEL_PATH = '/home/adam/catkin_ws/src/robot_tsr/script/model-1.h5'

# Loading the model back:
model = load_model(MODEL_PATH)


def main():
    file_path = f'/home/adam/catkin_ws/src/robot_tsr/script/uploads/stop.jpeg'
    sign = predict_model(file_path, model)
    return sign


def talker():
    signs_publisher = rospy.Publisher('dc_motors', UInt16, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10)  # 1hz
    while not rospy.is_shutdown():
        new_sign = main()
        # Choose to right value 
        msg = choose_way(new_sign)
        rospy.loginfo(msg)
        signs_publisher.publish(msg)
        rate.sleep()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
