import numpy as np
from enum import IntEnum
import itertools as iter
import random
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats

# Defining the agent grid world.

#TODO: add diagonals

class Actions(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class AgentGridworld(object):
    """
    An X by Y gridworld class for an agent in an environment with obstacles.
    """
    Actions = Actions

    def __init__(self, X, Y, obstacles, start, goals):
        """
        Params:
            X [int] -- The width of this gridworld.
            Y [int] -- The height of this gridworld.
            start [tuple] -- Starting position specified in coords (x, y).
            goals [list of tuple] -- List of goal positions specified in coords (x, y).
            obstacles [list] -- List of axis-aligned 2D boxes that represent
                obstacles in the environment for the agent. Specified in coords:
                [[(lower_x, lower_y), (upper_x, upper_y)], [...]]
        """

        assert isinstance(X, int), X
        assert isinstance(Y, int), Y
        assert X > 0
        assert Y > 0

        # Set up variables for Agent Gridworld
        self.X = X
        self.Y = Y
        self.S = X * Y
        self.A = len(Actions)
        self.start = start
        self.goals = goals

        # Set the obstacles in the environment.
        self.obstacles = obstacles


    ###########################
    #### Utility functions ####
    ###########################

    def traj_construct(self, start, goal):
        """
        Construct all trajectories between a start and goal of the shortest length.
        Params:
            start [tuple] -- Starting position specified in coords (x, y).
            goal [tuple] -- Goal position specified in coords (x, y).
        Returns:
            trajs [list] -- Trajectories between start and goal in states (s).
        """
        trajs = []
        def recurse_actions(s_curr, timestep):
            # Recursive action combo construction. Select legal combinations.
            if timestep == T-1:
                if s_curr == s_goal:
                    trajs.append(list(traj))
            else:
                rand_actions = [a for a in Actions]
                random.shuffle(rand_actions)
                for a in rand_actions:
                    s_prime, illegal = self.transition_helper(s_curr, a)
                    if not illegal:
                        traj[timestep+1] = s_prime
                        recurse_actions(s_prime, timestep+1)

        s_start = self.coor_to_state(start[0], start[1])
        s_goal = self.coor_to_state(goal[0], goal[1])
        T = abs(start[0] - goal[0]) + abs(start[1] - goal[1]) + 1
        traj = [None] * T
        traj[0] = s_start
        recurse_actions(s_start, 0)

        return trajs

    def transition_helper(self, s, a):
        """
        Given a state and action, apply the transition function to get the next state.
        Params:
            s [int] -- State.
            a [int] -- Action taken.
        Returns:
            s_prime [int] -- Next state.
            illegal [bool] -- Whether the action taken was legal or not.
        """
        x, y = self.state_to_coor(s)
        assert 0 <= a < self.A

        x_prime, y_prime = x, y
        if a == Actions.LEFT:
            x_prime = x - 1
        elif a == Actions.RIGHT:
            x_prime = x + 1
        elif a == Actions.DOWN:
            y_prime = y + 1
        elif a == Actions.UP:
            y_prime = y - 1
        # elif a == Actions.UP_LEFT:
        #     x_prime, y_prime = x - 1, y - 1
        # elif a == Actions.UP_RIGHT:
        #     x_prime, y_prime = x + 1, y - 1
        # elif a == Actions.DOWN_LEFT:
        #     x_prime, y_prime = x - 1, y + 1
        # elif a == Actions.DOWN_RIGHT:
        #     x_prime, y_prime = x + 1, y + 1
        # elif a == Actions.ABSORB:
        #     pass
        else:
            raise BaseException("undefined action {}".format(a))

        illegal = False
        if x_prime < 0 or x_prime >= self.X or y_prime < 0 or y_prime >= self.Y:
            illegal = True
            s_prime = s
        else:
            s_prime = self.coor_to_state(x_prime, y_prime)
            if self.is_blocked(s_prime):
                illegal = True
        return s_prime, illegal

    def get_action(self, s, sp):
        """
        Given two neighboring waypoints, return action between them.
        Params:
            s [int] -- First waypoint state.
            sp [int] -- Next waypoint state.
        Returns:
            a [int] -- Action taken.
        """
        x1, y1 = self.state_to_coor(s)
        x2, y2 = self.state_to_coor(sp)

        if x1 == x2:
            # if y1 == y2:
            #     return Actions.ABSORB
            # elif y1 < y2:
            #     return Actions.DOWN
            # else:
            #     return Actions.UP
            if y1 <= y2:
                return Actions.DOWN
            else:
                return Actions.UP
        elif x1 < x2:
            # if y1 == y2:
            #     return Actions.RIGHT
            # elif y1 < y2:
            #     return Actions.DOWN_RIGHT
            # else:
            #     return Actions.UP_RIGHT
            
            return Actions.RIGHT
        else:
            # if y1 == y2:
            #     return Actions.LEFT
            # elif y1 < y2:
            #     return Actions.DOWN_LEFT
            # else:
            #     return Actions.UP_LEFT
            return Actions.LEFT
    
    def is_blocked(self, s):
        """
        Returns True if s is blocked.
        By default, state-action pairs that lead to blocked states are illegal.
        """
        if self.obstacles is None:
            return False

        # Check against internal representation of boxes. 
        x, y = self.state_to_coor(s)
        for box in self.obstacles:
            if x >= box[0][0] and x <= box[1][0] and y >= box[1][1] and y <= box[0][1]:
                return True
        return False

    def visualize_grid(self):
        """
        Visualize the world with its obstacles.
        """
        self.visualize_demos([])

    def visualize_demos(self, demos):
        """
        Visualize the world with its obstacles and given demonstration.
        """
        # Create world with obstacles on the map.
        world = 0.5*np.ones((self.Y, self.X))

        # Add obstacles in the world in opaque color.
        for obstacle in self.obstacles:
            lower = obstacle[0]
            upper = obstacle[1]
            world[upper[1]:lower[1]+1, lower[0]:upper[0]+1] = 1.0

        fig1, ax1 = plt.subplots()
        plt.imshow(world, cmap='Greys', interpolation='nearest')

        # Plot markers for start and goal
        plt.scatter(self.start[0], self.start[1], c="orange", marker="o", s=100)
        for goal in self.goals:
            plt.scatter(goal[0], goal[1], c="orange", marker="x", s=300)
        
        # Plot demonstrations
        for t, demo in enumerate(demos):
            demo_x = []
            demo_y = []
            for s in demo:
                x, y = self.state_to_coor(s)
                demo_x.append(x)
                demo_y.append(y)
            step = t/float(len(demos)+1)
            col = ((1*step), (0*step), (0*step))
            plt.plot(demo_x,demo_y, c=col)

        plt.xticks(range(self.X), range(self.X))
        plt.yticks(np.arange(-0.5,self.Y+0.5),range(self.Y+1))
        ax1.set_yticklabels([])
        ax1.set_xticklabels([])
        ax1.set_yticks([])
        ax1.set_xticks([])
        ax = plt.gca()
        plt.minorticks_on
        ax.grid(True, which='both', color='black', linestyle='-', linewidth=2)
        plt.show(block=False)
        

    #################################
    # Conversion functions
    #################################
    # Helper functions convert between state number ("state") and discrete coordinates ("coor").
    #
    # State number ("state"):
    # A state `s` is an integer such that 0 <= s < self.S.
    #
    # Discrete coordinates ("coor"):
    # `x` is an integer such that 0 <= x < self.X. Increasing `x` corresponds to moving east.
    # `y` is an integer such that 0 <= y < self.Y. Increasing `y` corresponds to moving south.
    #
    #################################

    def state_to_coor(self, s):
        """
        Params:
            s [int] -- The state.
        Returns:
            x, y -- The discrete coordinates corresponding to s.
        """
        assert isinstance(s, int)
        assert 0 <= s < self.S
        y = s % self.Y
        x = s // self.Y
        return x, y

    def coor_to_state(self, x, y):
        """
        Convert discrete coordinates into a state, if that state exists.
        If no such state exists, raise a ValueError.
        Params:
            x, y [int] -- The discrete x, y coordinates of the state.
        Returns:
            s [int] -- The state.
        """

        x, y = int(x), int(y)
        if not(0 <= x < self.X):
            raise ValueError(x, self.X)
        if not (0 <= y < self.Y):
            raise ValueError(y, self.Y)

        return (x * self.Y) + (y)
