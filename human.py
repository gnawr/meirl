import numpy as np
from enum import IntEnum
import itertools as iter
import random
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats

from grid import AgentGridworld, Actions
#################################
#### Featurization functions ####
#################################

####### Obstacles Feature #######

def obstacles_feature(traj):
    """
    Compute obstacle feature values for all obstacles and the entire trajectory.
    The obstacle feature consists of distance from the obstacle.
    Params:
        traj [list] -- The trajectory.
    Returns:
        obstacle_feat [list] -- A list of obstacle features per obstacle.
    """
    obstacle_feat = np.zeros(len(gridworld.obstacles))
    for s in traj:
        obstacle_feat += np.asarray(dist_to_obstacles(s))
    return obstacle_feat.tolist()

def dist_to_obstacles(s):
    """
    Compute distance from state s to the obstacles in the environment.
    Params:
        s [int] -- The state.
    Returns:
        distances [list] -- The distance to the obstacles in the environment.
    """
    x, y = gridworld.state_to_coor(s)
    distances = []
    
    # Compute delta x and delta y in distance from obstacle
    for obstacle in gridworld.obstacles:
        # YOUR CODE HERE
        ox = (obstacle[0][0] + obstacle[1][0]) / 2
        oy = (obstacle[0][1] + obstacle[1][1]) / 2
        distances.append(np.sqrt((x - ox)**2 + (y - oy)**2))
    return distances

####### Goal Feature #######

def goals_feature(traj):
    """
    Compute goal feature values for all goals and the entire traj.
    The goal feature consists of distance from the obstacle.
    Params:
        traj [list] -- The trajectory.
    Returns:
        goal_feat [list] -- The distance to the goals in the environment.
    """
    goal_feat = np.zeros(len(gridworld.goals))
    for s in traj:
        goal_feat += np.asarray(dist_to_goals(s))
    return goal_feat.tolist()

def dist_to_goals(s):
    """
    Compute distance from state s to the goal in the environment.
    Params:
        s [int] -- The state.
    Returns:
        distance [float] -- The distance to the goals in the environment.
    """
    x, y = gridworld.state_to_coor(s)
    distances = []

    for goal in gridworld.goals:
        # YOUR CODE HERE
        gx = goal[0]
        gy = goal[1]
        distances.append(np.sqrt((x - gx)**2 + (y - gy)**2))

    return distances

####### Coordinate Features #######

def average_x_feature(traj):
    """
    Compute average x feature value for the entire trajectory.
    Params:
        traj [list] -- The trajectory.
    Returns:
        avgx_feat [float] -- The average x feature value for entire traj.
    """
    x_coords = [gridworld.state_to_coor(s)[0] for s in traj]
    return np.mean(x_coords)

def average_y_feature(traj):
    """
    Compute average y feature value for the entire trajectory.
    Params:
        traj [list] -- The trajectory.
    Returns:
        avgy_feat [float] -- The average y feature value for entire traj.
    """
    y_coords = [gridworld.state_to_coor(s)[1] for s in traj]
    return np.mean(y_coords)


####### Utils #######

def featurize(traj, feat_list, scaling_coeffs=None):
    """
    Computes the user-defined features for a given trajectory.
    Params:
        traj [list] -- A list of states the trajectory goes through.
    Returns:
        features [array] -- A list of feature values.
    """
    features = []
    for feat in range(len(feat_list)):
        if feat_list[feat] == 'goals':
            features.extend(goals_feature(traj))
        elif feat_list[feat] == 'obstacles':
            features.extend(obstacles_feature(traj))
        elif feat_list[feat] == 'avgx':
            features.append(average_x_feature(traj))
        elif feat_list[feat] == 'avgy':
            features.append(average_y_feature(traj))
    if scaling_coeffs is not None:
        for feat in range(len(features)):
            features[feat] = (features[feat] - scaling_coeffs[feat]["min"]) / (scaling_coeffs[feat]["max"] - scaling_coeffs[feat]["min"])
    return np.asarray(features)

def feat_scale_construct(feat_list):
    """
    Construct scaling constants for the features available.
    """
    # First featurize all trajectories with non-standard features.
    Phi_nonstd = np.array([featurize(xi, feat_list) for xi in SG_trajs])

    # Compute scaling coefficients depending on what feat_scaling is
    scaling_coeffs = []
    for Phi in Phi_nonstd.T:
        min_val = min(Phi)
        max_val = max(Phi)
        coeffs = {"min": min_val, "max": max_val}
        scaling_coeffs.append(coeffs)
    return scaling_coeffs

def visualize_feature(feat_vals, idx):
    """
    Visualize the world with its obstacles and given demonstration.
    """
    # Create world with obstacles on the map.
    world = np.ones((gridworld.Y, gridworld.X))
    for s in range(gridworld.S):
        x, y = gridworld.state_to_coor(s)
        world[y][x] = feat_vals[s][idx]

    # Add obstacles in the world in opaque color.
    for obstacle in gridworld.obstacles:
        lower = obstacle[0]
        upper = obstacle[1]
        world[upper[1]:lower[1]+1, lower[0]:upper[0]+1] = 10.0

    fig1, ax1 = plt.subplots()
    plt.imshow(world, cmap='Greys', interpolation='nearest')

    # Plot markers for start and goal
    plt.scatter(gridworld.start[0], gridworld.start[1], c="orange", marker="o", s=100)
    for goal in gridworld.goals:
        plt.scatter(goal[0], goal[1], c="orange", marker="x", s=300)

    plt.xticks(range(gridworld.X), range(gridworld.X))
    plt.yticks(np.arange(-0.5,gridworld.Y+0.5),range(gridworld.Y+1))
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax = plt.gca()
    plt.minorticks_on
    ax.grid(True, which='both', color='black', linestyle='-', linewidth=2)
    plt.show(block=False)


####### OBSERVATION MODEL #######

def observation_model(Phi_xi, Phi_xibar, theta, beta):
    """
    Finds observation model for given demonstrated features, using initialized model.
    Params:
        Phi_xi [array] -- The cost features for an observed trajectory.
        Phi_xibar [list] -- A list of the cost features for all trajectories in the grid.
        theta [list] -- The preference parameter.
        beta [float] -- The rationality coefficient.
    Returns:
        P_xi_bt [float] -- P(xi | theta, beta)
    """ 
    num = np.exp(-1 * beta * (np.dot(theta.T,Phi_xi)))
    denom = 0
    for phi in Phi_xibar:
        denom += np.exp(-1 * beta * (np.dot(theta.T,phi)))
    P_xi_bt = num / denom
    return P_xi_bt

def sample_demonstrations(theta, beta, samples):
    """
    Sample <samples> demonstrations for a given theta and beta.
    Params:
        theta [list] -- The preference parameter.
        beta [float] -- The rationality coefficient.
        samples [int] -- Number of demonstrations to be sampled.
    """ 
    # Generate feature values for all trajectories in the gridworld.
    Phi_xibar = [featurize(xi, feat_list, scaling_coeffs) for xi in SG_trajs]

    # Create the xi observation model for all trajectories.
    P_xi = [observation_model(Phi, Phi_xibar, theta, beta) for Phi in Phi_xibar]
    
    # Sample <samples> trajectories using this distribution.
    traj_idx = np.random.choice(len(P_xi), samples, p=P_xi)

    # Return trajectories given by traj_idx
    return [SG_trajs[i] for i in traj_idx]