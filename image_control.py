import sys
import geometry_msgs.msg
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import drone_goals
import cv2
import numpy as np
import time

class TelloSubscriber(Node):
    def __init__(self):
        super().__init__('tello_subscriber')
        self.subscription = self.create_subscription(Image, 
            '/drone1/image_raw',
            self.listener_callback,
            10)
        self.bridge = CvBridge()
        self.node = rclpy.create_node('teleop_twist_keyboard')
        self.pub = self.node.create_publisher(geometry_msgs.msg.Twist, '/drone1/cmd_vel', 10)
        self.goal_generator = drone_goals.GazeboInterface()

        self.speed = 0.10 # max 0.22
        self.turn = 0.38 # max 1.82

        self.corner_number = 0

    def listener_callback(self, image_message):
        frame = self.bridge.imgmsg_to_cv2(image_message, desired_encoding='bgr8')

        cv2.imshow("Corner", frame)
        cv2.waitKey(10)

        # DETECT if the person is in the image here. This code currently just
        # generates the goal at corner 2
        person_in_image = self.corner_number == 4
        if (person_in_image):
            self.goal_generator.generate_goal_pose(self.corner_number)
            rclpy.spin(self.goal_generator)

        # Rotate robot for next corner
        if (self.corner_number > 0):
            self.rotate(5.8)
        time.sleep(1.0)
        
        self.corner_number += 1

        if (self.corner_number > 4):
            self.corner_number = 1

    def grab_frame(self):
        return self.frame

    def move_forward(self):
        twist = geometry_msgs.msg.Twist()
        twist.linear.x = 1 * self.speed
        twist.linear.y = 0 * self.speed
        twist.linear.z = 0 * self.speed
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0 * self.turn
        print(twist)
        self.pub.publish(twist)
        time.sleep(1.0)
        self.stop()
        time.sleep(1.0)

    def move_backward(self):
        twist = geometry_msgs.msg.Twist()
        twist.linear.x = -1 * self.speed
        twist.linear.y = 0 * self.speed
        twist.linear.z = 0 * self.speed
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0 * self.turn              
        print(twist)
        self.pub.publish(twist)
        time.sleep(1.0)
        self.stop()
        time.sleep(1.0)
    
    def rotate(self, degrees):
        twist = geometry_msgs.msg.Twist()
        twist.linear.x = 0 * self.speed
        twist.linear.y = 0 * self.speed
        twist.linear.z = 0 * self.speed
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.1 * degrees
        print(twist)
        self.pub.publish(twist)
        time.sleep(1.0)
        self.stop()
        time.sleep(1.0)

    def stop(self):
        twist = geometry_msgs.msg.Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0  
        print(twist)
        self.pub.publish(twist)

    def read_input(self, key):        
        if key == ord('w') or key == ord('W'):
            self.move_forward()
        elif key == ord('s') or key == ord('S'):
            self.move_backward()
        elif key == ord('A') or key == ord('a'):
            self.rotate(45.0)
        elif key == ord('D') or key == ord('d'):
            self.rotate(-45.0)
        self.stop()

'''
rclpy.init()

tello_subscriber = TelloSubscriber()
rclpy.spin(tello_subscriber)

tello_subscriber.destroy_node()
rclpy.shutdown()
cv2.destroyAllWindows()
'''