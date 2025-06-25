#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from turtlesim.srv import Spawn, Kill, KillRequest, SpawnRequest
from turtlesim.msg import Pose
from math import atan2, pi

class TurtleController:
    def __init__(self, turtle_name='turtle1', turtle2=None):
        self.turtle_name = turtle_name
        rospy.init_node('turtle_controller', anonymous=True)
        self.velocity_publisher = rospy.Publisher(f'/{self.turtle_name}/cmd_vel', Twist, queue_size=10)
        self.pose_subscriber = rospy.Subscriber(f'/{self.turtle_name}/pose', Pose, self.update_pose)
        self.pose = Pose()
        self.rate = rospy.Rate(10)
        self.turtle2 = turtle2

    def update_pose(self, data):
        self.pose = data

    def move_to_point(self, x, y):
        vel_msg = Twist()

        # Phase 1: Twisting to align with the target angle
        while not rospy.is_shutdown():
            # Calculate the angle to the target point
            target_angle = atan2(y - self.pose.y, x - self.pose.x)
            # Ensure the angle is within -pi to pi range
            while target_angle - self.pose.theta > pi:
                target_angle -= 2 * pi
            while target_angle - self.pose.theta < -pi:
                target_angle += 2 * pi

            # Set the angular velocity to turn towards the target point
            vel_msg.angular.z = 5.0 * (target_angle - self.pose.theta)

            # Publish the velocity message for twisting
            self.velocity_publisher.publish(vel_msg)

            # Check if the turtle has reached close to the target angle
            if abs(target_angle - self.pose.theta) < 0.01:
                break

            self.rate.sleep()

        # Stop the turtle's angular motion
        vel_msg.angular.z = 0
        self.velocity_publisher.publish(vel_msg)

        # Phase 2: Linear motion towards the target point
        while not rospy.is_shutdown():
            # Calculate the distance to the target point
            distance_to_target = ((x - self.pose.x) ** 2 + (y - self.pose.y) ** 2) ** 0.5

            # Set the linear velocity to move towards the target point
            if collision(self, self.turtle2):
                vel_msg.linear.x = 0
            else:
                vel_msg.linear.x = min(distance_to_target, 10.0)

            # Publish the velocity message for linear motion
            self.velocity_publisher.publish(vel_msg)

            # Check if the turtle has reached close to the target position
            if distance_to_target < 0.1:
                break

            self.rate.sleep()

        # Stop the turtle's linear motion
        vel_msg.linear.x = 0
        self.velocity_publisher.publish(vel_msg)

    def move_in_circle(self, radius):

        vel_msg = Twist()
        
        # Set the angular velocity (constant)
        angular_velocity = 0.5  # radians per second

        # Set the linear velocity such that the radius is achieved
        linear_velocity = radius * angular_velocity  # v = r * ω

        vel_msg.linear.x = linear_velocity
        vel_msg.angular.z = angular_velocity

        while not rospy.is_shutdown():
            # Publish the velocity message to move in a circle
            self.velocity_publisher.publish(vel_msg)
            self.rate.sleep()

def kill_default_turtle():
    rospy.wait_for_service('/kill')
    try:
        # Create a handle for calling the kill service
        kill_turtle = rospy.ServiceProxy('/kill', Kill)
        
        # Kill the default turtle named 'turtle1'
        kill_request = KillRequest(name='turtle1')
        kill_turtle(kill_request)
        rospy.loginfo("Killed default turtle 'turtle1'")

    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

def spawn_turtle(point, angle):
    rospy.wait_for_service('/spawn')
    try:
        # Create a handle for calling the spawn service
        spawn_turtle = rospy.ServiceProxy('/spawn', Spawn)

        spawn_request1 = SpawnRequest(x=point[0], y=point[1], theta=angle)
        response1 = spawn_turtle(spawn_request1)
        rospy.loginfo(f"Spawned turtle {response1.name} at ({point[0]}, {point[1]})")
        return response1.name

    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

class Node :
    def __init__(self, point):
        self.point = np.array(point)
        self.parent = None
        self.cost = 0.0

def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def collision(turtle1, turtle2):
    if (((turtle2.pose.x - turtle1.pose.x) ** 2 + (turtle2.pose.y - turtle1.pose.y) ** 2) ** 0.5) < 1:
        return True
    return False

# Bresenham's line algorithm to generate points along the line
def bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))

        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points

def linePossible(point1, point2, gray_image):
    # Generate points along the line using Bresenham's algorithm
    line_points = bresenham_line(point1[0], point1[1], point2[0], point2[1])
    for point in line_points:
        if gray_image[point[1], point[0]] == 0:
            return False
    return True

def rewire_tree(tree, new_node, rradius, gray_image):
    for node in tree:
        if node != new_node and distance(node.point, new_node.point) < rradius:    
            if linePossible(new_node.point, node.point, gray_image) :
                new_cost = new_node.cost + distance(node.point, new_node.point)
                if new_cost < node.cost:
                    node.parent = new_node
                    node.cost = new_cost

def rrt_star(start, goal, gray_image, max_iter=4000, radius=5.0):
    start_node = Node(start)
    tree = [start_node]

    for _ in range(max_iter):

        # Sample a random point
        random_point = np.random.uniform(0, gray_image.shape[1], 2)

        # Find the nearest node in the tree to the random point
        nearest_node = min(tree, key=lambda node: distance(node.point, random_point))

        # Steer towards the random point
        direction = random_point - nearest_node.point
        magnitude = np.linalg.norm(direction)
        if radius < magnitude :
            direction = direction * (radius / magnitude)
        else :
            direction = direction * (magnitude / radius)
        new_point = nearest_node.point + direction

        new_point = (int(new_point[0]), int(new_point[1]))

        if linePossible(new_point, nearest_node.point, gray_image) :
            new_node = Node(new_point)
            new_node.parent = nearest_node
            new_node.cost = nearest_node.cost + distance(new_point, nearest_node.point)
            tree.append(new_node)

            rewire_tree(tree, new_node, 20*radius, gray_image)
            if distance(new_point, goal) < radius:
                break
            
    # Find the path
    nearest_node = min(tree, key=lambda node: distance(node.point, goal))
    goal_node = Node(goal)
    goal_node.parent = nearest_node
    goal_node.cost = nearest_node.cost + distance(goal, nearest_node.point)
    tree.append(new_node)
    path = []
    rewire_tree(tree, goal_node, 20*radius, gray_image)
    current_node = goal_node
    while current_node is not None:
        path.append(current_node.point)
        current_node = current_node.parent

    return path[::-1]

class PathPlanner:
    def __init__(self, map_image_path):
        self.map_image = cv2.imread(map_image_path)

    def plan_path(self):
        # Convert the map image to HSV color space
        hsv_image = cv2.cvtColor(self.map_image, cv2.COLOR_BGR2HSV)

        # Define lower and upper bounds for green and red colors in HSV
        lower_green = np.array([45, 100, 100])
        upper_green = np.array([75, 255, 255])
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])

        # Threshold the HSV image to get binary masks for green and red regions
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
        red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Find contours in the binary masks
        contours_green, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get centroids of largest contours for green and red masks
        start_point = self.get_centroid_of_largest_contour(contours_green)
        goal_point = self.get_centroid_of_largest_contour(contours_red)

        # Convert the map image to grayscale
        gray_image = cv2.cvtColor(self.map_image, cv2.COLOR_BGR2GRAY)

        if start_point is None or goal_point is None:
            print("Error: Start or goal point not found.")
            return None

        path = rrt_star(start_point, goal_point, gray_image)

        if path is None:
            print("Error: Path not found.")
            return None

        return path

    def get_centroid_of_largest_contour(self, contours):
        if not contours:
            return None
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        centroid_x = int(M['m10'] / M['m00'])
        centroid_y = int(M['m01'] / M['m00'])
        return (centroid_x, centroid_y)

    def draw_path_on_map(self, path):
        # Convert path points to integers
        path = [(int(point[0]), int(point[1])) for point in path]
        # Draw the path on the map image
        for i in range(len(path) - 1):
            cv2.line(self.map_image, path[i], path[i + 1], (255, 0, 0), thickness=2)  # Blue line for the path

    def display_map_with_path(self):
        # Display the map image with the path using OpenCV
        cv2.imshow('Map with Path', self.map_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # File path to the map image
    map_image_path = 'map.png'

    # Create a PathPlanner object
    planner = PathPlanner(map_image_path)

    # Plan the path
    path = planner.plan_path()

    # Draw the path on the map image
    if path:
        planner.draw_path_on_map(path)

        # Display the map image with the path
        planner.display_map_with_path()

    try:
        kill_default_turtle()

        turtle2 = spawn_turtle((11.1/2+11.1/4, 11.1/2), pi/2)
        controller2 = TurtleController(turtle2)
        for point in path :
            turtle1 = spawn_turtle((point[0]*11.1/600, (600-point[1])*11.1/600), 0)
            break
        
        # Start moving the second turtle in a circle in a separate thread
        import threading
        threading.Thread(target=controller2.move_in_circle, args=(11.1/4,)).start()
        
        controller1 = TurtleController(turtle1, controller2)
        reached = False
        for point in path :
            controller1.move_to_point(point[0]*11.1/600, (600-point[1])*11.1/600)

    except rospy.ROSInterruptException:
        pass
