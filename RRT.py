# Standard Algorithm Implementation
# Sampling-based Algorithms RRT and RRT*

import matplotlib.pyplot as plt
import numpy as np


# Class for each tree node
class Node:
    def __init__(self, row, col):
        self.row = row        # coordinate
        self.col = col        # coordinate
        self.parent = None    # parent node
        self.cost = 0.0       # cost


# Class for RRT
class RRT:
    # Constructor
    def __init__(self, map_array, start, goal):
        self.map_array = map_array            # map array, 1->free, 0->obstacle
        self.size_row = map_array.shape[0]    # map size
        self.size_col = map_array.shape[1]    # map size

        self.start = Node(start[0], start[1]) # start node
        self.goal = Node(goal[0], goal[1])    # goal node
        self.vertices = []                    # list of nodes
        self.found = False                    # found flag


    def init_map(self):
        '''Intialize the map before each search
        '''
        self.found = False
        self.vertices = []
        self.vertices.append(self.start)


    def dis(self, node1, node2):
        '''Calculate the euclidean distance between two nodes
        arguments:
            node1 - node 1
            node2 - node 2

        return:
            euclidean distance between two nodes
        '''
        ### YOUR CODE HERE ###
        node1 = [node1.row, node1.col]
        node2 = [node2.row, node2.col]
        distance = np.sqrt((node1[0]-node2[0])**2+(node1[1]-node2[1])**2)
        return distance


    def check_collision(self, node1, node2):
        '''Check if the path between two nodes collide with obstacles
        arguments:
            node1 - node 1
            node2 - node 2

        return:
            True if the new node is valid to be connected
        '''
        ### YOUR CODE HERE ###
        nodes = np.array([[node1.row, node1.col], [node2.row, node2.col]])
        # node1_new = [node1.row, node1.col]
        # node2_new = [node2.row, node2.col]
        # node = np.array([node1_new, node2_new])
        obs_check = True
        while len(nodes) < 300 and obs_check:
            i = 0
            if self.map_array[node2.row,node2.col]==0:
                obs_check = False
                break
            while i<len(nodes) and i < 300:
                if i == 0:
                    i += 1
                    continue
                midx = int((nodes[i, 0]+nodes[i-1, 0])//2)
                midy = int((nodes[i, 1]+nodes[i-1, 1])//2)
                if self.map_array[midx, midy] == 0:
                    obs_check = False
                    break
                nodes = np.insert(nodes, i, [midx, midy], axis=0)
                i += 2
        return obs_check

    def get_new_point(self, goal_bias):
        '''Choose the goal or generate a random point
        arguments:
            goal_bias - the possibility of choosing the goal instead of a random point

        return:
            point - the new point
        '''
        ### YOUR CODE HERE ###

        bias = np.random.choice(["random","goal"], 1 , p = [1-goal_bias, goal_bias])

        if bias == "random":
            row_choice = np.random.randint(low=0, high=299, dtype = int)
            col_choice = np.random.randint(low=0, high=299, dtype = int)
            random_point = Node(row_choice, col_choice)
        else:
            random_point = Node(self.goal.row, self.goal.col)
        return random_point


    def get_nearest_node(self, point):
        '''Find the nearest node in self.vertices with respect to the new point
        arguments:
            point - the new point

        return:
            the nearest node
        '''
        ### YOUR CODE HERE ###
        distances = []
        min_val = 100000
        for i in self.vertices:
            distance = self.dis(i,point)
            # distances.append(distance)
            if distance < min_val:
                min_val = distance
                minNode = i

        return minNode


    def get_neighbors(self, new_node, neighbor_size):
        '''Get the neighbors that are within the neighbor distance from the node
        arguments:
            new_node - a new node
            neighbor_size - the neighbor distance

        return:
            neighbors - a list of neighbors that are within the neighbor distance
        '''
        ### YOUR CODE HERE ###
        neighbours = []
        for i in self.vertices:
            neighbour_distance = self.dis(i, new_node)
            if neighbour_distance <= neighbor_size:
                neighbours.append(i)
        return neighbours

    def rewire(self, new_node, neighbors, near_node):
        '''Rewire the new node and all its neighbors
        arguments:
            new_node - the new node
            neighbors - a list of neighbors that are within the neighbor distance from the node

        Rewire the new node if connecting to a new neighbor node will give least cost.
        Rewire all the other neighbor nodes.
        '''
        ### YOUR CODE HERE ###

        min = 100000
        if len(neighbors) > 1:
            minimum_node = neighbors[0]
            i = 0
            for x in neighbors:
                dist = self.dis(x, new_node)
                totalcost = x.cost + dist
                if totalcost < min:
                    min_index = i
                    min = totalcost
                    minimum_node = x
                i += 1
            new_node.parent = minimum_node
            new_node.cost = min
            self.vertices.append(new_node)
            del neighbors[min_index]
            for x in neighbors:
                if self.check_collision(x, new_node):
                    dist = self.dis(x, new_node)
                    new_cost = new_node.cost + dist
                    if new_cost < x.cost:
                        x.parent = new_node
        else:
            new_node.parent = near_node
            new_node.cost = self.dis(new_node, near_node) + new_node.parent.cost
            self.vertices.append(new_node)



    def draw_map(self):
        '''Visualization of the result
        '''
        # Create empty map
        fig, ax = plt.subplots(1)
        img = 255 * np.dstack((self.map_array, self.map_array, self.map_array))
        ax.imshow(img)

        # Draw Trees or Sample points
        for node in self.vertices[1:-1]:
            plt.plot(node.col, node.row, markersize=3, marker='o', color='y')
            plt.plot([node.col, node.parent.col], [node.row, node.parent.row], color='y')

        # Draw Final Path if found
        if self.found:
            cur = self.goal
            while cur.col != self.start.col or cur.row != self.start.row:
                plt.plot([cur.col, cur.parent.col], [cur.row, cur.parent.row], color='b')
                cur = cur.parent
                plt.plot(cur.col, cur.row, markersize=3, marker='o', color='b')

        # Draw start and goal
        plt.plot(self.start.col, self.start.row, markersize=5, marker='o', color='g')
        plt.plot(self.goal.col, self.goal.row, markersize=5, marker='o', color='r')

        # show image
        plt.show()
    def neighbours_value(self,neighbours):
        neighbours_list = []
        for x in neighbours:
            value = [x.row, x.col]
            value.append(neighbours_list)
        return neighbours_list

    def RRT(self, n_pts=10000):
        '''RRT main search function
        arguments:
            n_pts - number of points try to sample,
                    not the number of final sampled points

        In each step, extend a new node if possible, and check if reached the goal
        '''
        # Remove previous result
        self.init_map()

        ### YOUR CODE HERE ###

        # In each step,
        # get a new point,
        # get its nearest node,
        # extend the node and check collision to decide whether to add or drop,
        # if added, check if reach the neighbor region of the goal.
        i = 0
        while i < n_pts:

            point = self.get_new_point(0.05)
            near_node = self.get_nearest_node(point)
            row_val = point.row - near_node.row
            col_val = point.col - near_node.col
            ang = np.arctan2(row_val, col_val)
            step_row = int(near_node.row + 10 * np.sin(ang))
            step_col = int(near_node.col + 10 * np.cos(ang))
            step = Node(step_row, step_col)
            if self.check_collision(step, near_node):
                if [self.goal.row, self.goal.col] == [step.row, step.col]:
                    self.goal.parent = near_node
                    self.goal.cost = self.dis(step, near_node) + self.goal.parent.cost
                    self.vertices.append(self.goal)
                    self.found = True
                    break

                step.parent = near_node
                step.cost = self.dis(step, near_node) + step.parent.cost
                self.vertices.append(step)

                dis_goal = self.dis(self.goal, step)
                if dis_goal < 10 and self.check_collision(step, self.goal):
                    self.goal.parent = near_node
                    self.goal.cost = self.dis(step, near_node) + self.goal.parent.cost
                    self.vertices.append(self.goal)
                    self.found = True
                    break
            i += 1

        # Output
        if self.found:
            steps = len(self.vertices) - 2
            length = self.goal.cost
            print("It took %d nodes to find the current path" % steps)
            print("The path length is %.2f" % length)
        else:
            print("No path found")

        # Draw result
        self.draw_map()

    def RRT_star(self, n_pts=1000, neighbor_size=20):
        '''RRT* search function
        arguments:
            n_pts - number of points try to sample,
                    not the number of final sampled points
            neighbor_size - the neighbor distance

        In each step, extend a new node if possible, and rewire the node and its neighbors
        '''
        # Remove previous result
        self.init_map()

        ### YOUR CODE HERE ###

        # In each step,
        # get a new point,
        # get its nearest node,
        # extend the node and check collision to decide whether to add or drop,
        # if added, rewire the node and its neighbors,
        # and check if reach the neighbor region of the goal if the path is not found.
        i = 0
        while i < n_pts:
            point = self.get_new_point(0.05)
            near_node = self.get_nearest_node(point)
            row_val = point.row - near_node.row
            col_val = point.col - near_node.col
            ang = np.arctan2(row_val, col_val)
            step_row = int(near_node.row + 10 * np.sin(ang))
            step_col = int(near_node.col + 10 * np.cos(ang))
            step = Node(step_row, step_col)
            if self.check_collision(step, near_node):
                neighbors = self.get_neighbors(step, 30)
                self.rewire(step, neighbors, near_node)
                if [self.goal.row, self.goal.col] == [step.row, step.col]:
                    self.goal.parent = near_node
                    self.goal.cost = self.dis(step, near_node) + self.goal.parent.cost
                    self.vertices.append(self.goal)
                    self.found = True
                    break
                dis_goal = self.dis(self.goal, step)
                if dis_goal < 10:
                    if self.check_collision(step, self.goal):
                        self.goal.parent = near_node
                        self.goal.cost = self.dis(step, near_node) + self.goal.parent.cost
                        self.vertices.append(self.goal)
                        self.found = True
                        break
            i += 1
            # Output
        if self.found:
            steps = len(self.vertices) - 2
            length = self.goal.cost
            print("It took %d nodes to find the current path" % steps)
            print("The path length is %.2f" % length)
        else:
            print("No path found")

        # Draw result
        self.draw_map()
