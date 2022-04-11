# Standard Algorithm Implementation
# Sampling-based Algorithms PRM
import random

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
# from scipy.spatial import KDTree
import math
from numpy import random
from scipy import spatial



# Class for PRM
class PRM:
    # Constructor
    def __init__(self, map_array):
        self.map_array = map_array            # map array, 1->free, 0->obstacle
        self.size_row = map_array.shape[0]    # map size
        self.size_col = map_array.shape[1]    # map size

        self.samples = []                     # list of sampled points
        self.graph = nx.Graph()               # constructed graph
        self.path = []                        # list of nodes of the found path

    def check_collision(self, p1, p2):
        '''Check if the path between two points collide with obstacles
        arguments:
            p1 - point 1, [row, col]
            p2 - point 2, [row, col]

        return:
            True if there are obstacles between two points
        '''

        x = np.linspace(p1[0], p2[0]).astype(int)
        y = np.linspace(p1[1], p2[1]).astype(int)
        fullmap = zip(x, y)
        obstacle = False
        for m in fullmap:
            if self.map_array[m[0], m[1]] == 0:
                obstacle = True
        return obstacle

    def dis(self, point1, point2):
        '''Calculate the euclidean distance between two points
        arguments:
            p1 - point 1, [row, col]
            p2 - point 2, [row, col]

        return:
            euclidean distance between two points
        '''
        ### YOUR CODE HERE ###
        node1 = point1
        node2 = point2
        distance = np.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)
        return distance



    def uniform_sample(self, n_pts):
        '''Use uniform sampling and store valid points
        arguments:
            n_pts - number of points try to sample,
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()
        row_size = self.size_row
        col_size = self.size_col
        full_map = self.map_array
        row_values = np.linspace(0, row_size - 1, int(np.sqrt(n_pts))).astype(int)
        col_values = np.linspace(0, col_size - 1, int(np.sqrt(n_pts))).astype(int)
        for x in row_values:
            for y in col_values:
                if full_map[x, y] == 1:
                    self.samples.append((x, y))



    def random_sample(self, n_pts):
        '''Use random sampling and store valid points
        arguments:
            n_pts - number of points try to sample,
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()

        ### YOUR CODE HERE ###
        random_sample = []
        full_map = self.map_array
        for x in range(n_pts):
            row_choice = np.random.randint(low=0, high=300, dtype=int)
            col_choice = np.random.randint(low=0, high=300, dtype=int)
            random_sample.append((row_choice,col_choice))


        for x in range(len(random_sample)):
            rowx = random_sample[x][0]
            rowy = random_sample[x][1]
            if full_map[rowx][rowy] !=0:
                self.samples.append((rowx,rowy))








        self.samples.append((0, 0))

    def gaussian_sample(self, n_pts):
        '''Use gaussian sampling and store valid points
        arguments:
            n_pts - number of points try to sample,
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()

        ### YOUR CODE HERE ###
        self.samples.append((0, 0))
        full_map = self.map_array
        gaussian_sample = []
        for pt in range(n_pts):
            row_choice = np.random.randint(low=0, high=299, dtype=int)
            col_choice = np.random.randint(low=0, high=299, dtype=int)
            if self.map_array[row_choice , col_choice] == 0:
                gaussian_sample.append((row_choice,col_choice))
        for x in range(len(gaussian_sample)):
            obs_check = True

            while obs_check:
                x, y = int(np.random.normal(gaussian_sample[x][0], 10)), int(np.random.normal(gaussian_sample[x][1],10))
                if x< self.size_row and y < self.size_col and full_map[(x, y)] == 1:
                        obs_check = False
                        self.samples.append((x, y))

    def bridge_sample(self, n_pts):
        '''Use bridge sampling and store valid points
        arguments:
            n_pts - number of points try to sample,
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()

        ### YOUR CODE HERE ###
        self.samples.append((0, 0))

        bridge_sample = []

        for x in range(n_pts):
            # row_choice = np.random.randint(low=0, high=300, dtype=int)
            # col_choice = np.random.randint(low=0, high=300, dtype=int)
            # if self.map_array[row_choice, col_choice] == 0:
            #     bridge_sample.append((row_choice, col_choice))
            if self.map_array[random.randint(self.size_row), random.randint(self.size_col)] == 0:
                bridge_sample.append((random.randint(self.size_row), random.randint(self.size_col)))


        for x in range(len(bridge_sample)):
            pointx, pointy = bridge_sample[x][0], bridge_sample[x][1]
            secondpointx, secondpointy = int(np.random.normal(pointx, 15)), int(np.random.normal(pointy, 15))

            if secondpointx < self.size_row and secondpointy < self.size_col \
                    and self.map_array[(secondpointx,secondpointy)] == 0:
                centerx, centery = int((pointx + secondpointx) / 2), int((pointy + secondpointy) / 2)
                if self.map_array[(centerx, centery)] == 1:
                    self.samples.append((centerx, centery))


    def draw_map(self):
        '''Visualization of the result
        '''
        # Create empty map
        fig, ax = plt.subplots()
        img = 255 * np.dstack((self.map_array, self.map_array, self.map_array))
        ax.imshow(img)

        # Draw graph
        # get node position (swap coordinates)
        node_pos = np.array(self.samples)[:, [1, 0]]
        pos = dict( zip( range( len(self.samples) ), node_pos) )
        pos['start'] = (self.samples[-2][1], self.samples[-2][0])
        pos['goal'] = (self.samples[-1][1], self.samples[-1][0])

        # draw constructed graph
        nx.draw(self.graph, pos, node_size=3, node_color='y', edge_color='y' ,ax=ax)

        # If found a path
        if self.path:
            # add temporary start and goal edge to the path
            final_path_edge = list(zip(self.path[:-1], self.path[1:]))
            nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=self.path, node_size=8, node_color='b')
            nx.draw_networkx_edges(self.graph, pos=pos, edgelist=final_path_edge, width=2, edge_color='b')

        # draw start and goal
        nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=['start'], node_size=12,  node_color='g')
        nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=['goal'], node_size=12,  node_color='r')

        # show image
        plt.axis('on')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.show()

    def sample(self, n_pts=1000, sampling_method="uniform"):
        '''Construct a graph for PRM
        arguments:
            n_pts - number of points try to sample,
                    not the number of final sampled points
            sampling_method - name of the chosen sampling method

        Sample points, connect, and add nodes and edges to self.graph
        '''
        # Initialize before sampling
        self.samples = []
        self.graph.clear()
        self.path = []

        # Sample methods
        if sampling_method == "uniform":
            self.uniform_sample(n_pts)
            radius = 20
        elif sampling_method == "random":
            self.random_sample(n_pts)
            radius = 20
        elif sampling_method == "gaussian":
            self.gaussian_sample(n_pts)
            radius = 25
        elif sampling_method == "bridge":
            self.bridge_sample(n_pts)
            radius = 40
            ### YOUR CODE HERE ###

        # Find the pairs of points that need to be connected
        # and compute their distance/weight.
        # Store them as
        # pairs = [(p_id0, p_id1, weight_01), (p_id0, p_id2, weight_02),
        #          (p_id1, p_id2, weight_12) ...]
        pairs = []
        positions = np.array(self.samples)
        kdtree = spatial.KDTree(positions)
        # Creating a KD tree pairs list
        kd_pairs = list(kdtree.query_pairs(radius))

        for pt in range(len(kd_pairs)):
            collision = self.check_collision(self.samples[kd_pairs[pt][0]], self.samples[kd_pairs[pt][1]])
            if collision == False:
                euclidian = self.dis(self.samples[kd_pairs[pt][0]], self.samples[kd_pairs[pt][1]])
                if euclidian != 0:
                    indx, indy = self.samples.index(self.samples[kd_pairs[pt][0]]), self.samples.index(
                        self.samples[kd_pairs[pt][1]])
                    pairs.append((indx, indy, euclidian))

            else:
                continue

                # Use sampled points and pairs of points to build a graph.
        # To add nodes to the graph, use
        # self.graph.add_nodes_from([p_id0, p_id1, p_id2 ...])
        # To add weighted edges to the graph, use
        # self.graph.add_weighted_edges_from([(p_id0, p_id1, weight_01),
        #                                     (p_id0, p_id2, weight_02),
        #                                     (p_id1, p_id2, weight_12) ...])
        # 'p_id' here is an integer, representing the order of
        # current point in self.samples
        # For example, for self.samples = [(1, 2), (3, 4), (5, 6)],
        # p_id for (1, 2) is 0 and p_id for (3, 4) is 1.
        self.graph.add_nodes_from([])
        self.graph.add_weighted_edges_from(pairs)

        # Print constructed graph information
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        print("The constructed graph has %d nodes and %d edges" % (n_nodes, n_edges))


    def search(self, start, goal):
        '''Search for a path in graph given start and goal location
        arguments:
            start - start point coordinate [row, col]
            goal - goal point coordinate [row, col]

        Temporary add start and goal node, edges of them and their nearest neighbors
        to graph for self.graph to search for a path.
        '''
        # Clear previous path
        self.path = []

        # Temporarily add start and goal to the graph
        self.samples.append(start)
        self.samples.append(goal)
        # start and goal id will be 'start' and 'goal' instead of some integer
        self.graph.add_nodes_from(['start', 'goal'])

        ### YOUR CODE HERE ###

        # Find the pairs of points that need to be connected
        # and compute their distance/weight.
        # You could store them as
        # start_pairs = [(start_id, p_id0, weight_s0), (start_id, p_id1, weight_s1),
        #                (start_id, p_id2, weight_s2) ...]

        # Initializing an start and goal empty list
        start_nodes = []
        goal_nodes = []
        # Setting up a goal radius
        goal_radius = 60

        for x in range(len(self.samples)):
            start = self.samples[len(self.samples) - 2]
            points = self.samples[x]
            if start != points:
                obs_check = self.check_collision(start, points)
                if not obs_check:
                    length = self.dis(start, points)
                    if length != 0 :
                        if length < goal_radius:
                            start_nodes.append(('start', self.samples.index(points), length))
                else:
                    continue
        for x in range(len(self.samples)):
            sample_length = len(self.samples)-1
            start = self.samples[sample_length]

            points = self.samples[x]
            if start != points:
                obs_check = self.check_collision(start, points)
                if not obs_check:
                    distance = self.dis(start, points)
                    if distance != 0:
                        if distance < goal_radius:
                            goal_nodes.append(('goal', self.samples.index(points), distance))
                else:
                    continue
                    # Add the edge to graph
        self.graph.add_weighted_edges_from(start_nodes)
        self.graph.add_weighted_edges_from(goal_nodes)

        # Seach using Dijkstra
        try:
            self.path = nx.algorithms.shortest_paths.weighted.dijkstra_path(self.graph, 'start', 'goal')
            path_length = nx.algorithms.shortest_paths.weighted.dijkstra_path_length(self.graph, 'start', 'goal')
            print("The path length is %.2f" % path_length)
        except nx.exception.NetworkXNoPath:
            print("No path found")

        # Draw result
        self.draw_map()

        # Remove start and goal node and their edges
        self.samples.pop(-1)
        self.samples.pop(-1)
        self.graph.remove_nodes_from(['start', 'goal'])
        self.graph.remove_edges_from(start_nodes)
        self.graph.remove_edges_from(goal_nodes)
