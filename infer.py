import numpy as np
from enum import IntEnum
import itertools as iter
import random
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats

from grid import AgentGridworld, Actions

def inference(Phi_xis, thetas, betas):
    """
    Performs inference from given demonstrated features, using initialized model.
    Params:
        Phi_xis [list] -- A list of the cost features for observed trajectories.
        thetas [list] -- Possible theta vectors.
        betas [list] -- Possible beta values.
    Returns:
        P_bt [array] -- Posterior probability P(beta, theta | xi_1...xi_N)
    """
    prior = np.ones((len(betas), len(thetas))) / (len(betas) * len(thetas))
    
    # Generate feature values for all trajectories in the gridworld.
    Phi_xibar = [featurize(xi, feat_list, scaling_coeffs) for xi in SG_trajs]

    
    for b, beta in enumerate(betas):
        for t, theta in enumerate(thetas):
            # YOUR CODE HERE
            for phi_xi in Phi_xis:
                if isinstance(theta, list):
                    prior[b][t] *= observation_model(phi_xi, Phi_xibar, np.array(theta), beta)
                else:
                    prior[b][t] *= observation_model(phi_xi, Phi_xibar, theta, beta)

                
    P_bt = prior / np.sum(prior)
           
    return P_bt

def goal_inference(traj, goals):
    """
    Performs goal inference from given partial trajectory.
    Params:
        traj [list] -- The partial trajectory xi_SQ.
        goals [list] -- List of goals.
    Returns:
        P_g [array] -- Posterior probability P(G | xi_SQ)
    """
    prior = np.ones(len(goals)) / len(goals)
    P_g = np.ones(2)
    
    Phi_xi = featurize(traj, feat_list) # Cost from S to Q, under both G1 and G2
    for i in range(len(goals)):
        SG_trajs = gridworld.traj_construct(start, goals[i])
        QG_trajs = gridworld.traj_construct(gridworld.state_to_coor(traj[-1]), goals[i])
        Phi_xiSG = np.array([featurize(xi, feat_list)[i] for xi in SG_trajs]) # Distances from S to G_i
        Phi_xiQG = np.array([featurize(xi, feat_list)[i] for xi in QG_trajs]) # Distances from Q to G_i
        
        # YOUR CODE HERE
        P_t_goal = (np.exp(-1 * Phi_xi[i]) * np.sum(np.exp(-1 * Phi_xiQG))) / np.sum(np.exp(-1 * Phi_xiSG))
#         print('phixi ', Phi_xi)
#         print('hm ', -1 * Phi_xiSG)
        P_g[i] = P_t_goal * prior[i]
        
    P_g = P_g / np.sum(P_g)
#     print(P_g)
    return P_g

def predictable_trajectory(goal_idx):
    """
    Compute trajectory that maximizes predictability.
    Params:
        goal_idx [int] -- The goal w.r.t. we want to maximize predictability
    Returns:
        pred_traj [list] -- The trajectory that maximizes predictability.
    """
        
    # Generate feature values for all trajectories in the gridworld.
    Phi_xiSG = [featurize(xi, feat_list)[goal_idx] for xi in SG_trajs[goal_idx]]
    
    # YOUR CODE HERE
    # You want to get the index pred_idx of the most predictable trajectory.
    pred_idx = np.argmax(np.exp(-1 * np.array(Phi_xiSG)))

    pred_traj = SG_trajs[goal_idx][pred_idx]
    return pred_traj

def legible_trajectory(goals, goal_idx):
    """
    Compute trajectory that maximizes legibility.
    Params:
        goals [list] -- List of goals in the environment.
        goal_idx [int] -- The goal w.r.t. we want to maximize predictability
    Returns:
        leg_traj [list] -- The trajectory that maximizes predictability.
    """
    # max_xi P(g | xi)
    prior = np.ones(len(goals)) / len(goals)
    
    Phi_xiSGs = []
    C_xiSGs = []
    for i in range(len(goals)):
        # Generate feature values for all trajectories in the gridworld.
        Phi_xiSG = [featurize(xi, feat_list)[i] for xi in SG_trajs[i]]
        C_xiSG = [-Phi for Phi in Phi_xiSG]
        Phi_xiSGs.append(Phi_xiSG)
        C_xiSGs.append(C_xiSG)
    
    #Want to get the index leg_idx of the most legible trajectory.
    leg_idx = np.argmax(np.exp(np.array(C_xiSGs[goal_idx])) * prior[goal_idx])

    leg_traj = SG_trajs[goal_idx][leg_idx]
    return leg_traj






#################################
#### Visualization functions ####
#################################

def visualize_posterior(prob, thetas, betas):
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    matplotlib.rcParams['font.family'] = "Times New Roman"
    matplotlib.rcParams.update({'font.size': 15})

    plt.figure()
    plt.imshow(prob, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.clim(0, None)

    weights_rounded = [[round(i,2) for i in j] for j in thetas]
    plt.xticks(range(len(thetas)), weights_rounded, rotation = 'vertical')
    plt.yticks(range(len(betas)), betas)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\beta$')
    plt.title("Joint Posterior Belief")
    plt.show()

def visualize_marginal(marg, thetas):
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    matplotlib.rcParams['font.family'] = "Times New Roman"
    matplotlib.rcParams.update({'font.size': 15})

    plt.figure()
    plt.imshow([marg], cmap='Oranges', interpolation='nearest')
    plt.colorbar(ticks=[0, 0.5, 1.0])
    plt.clim(0, 1.0)

    weights_rounded = [[round(i,2) for i in j] for j in thetas]
    plt.xticks(range(len(thetas)), weights_rounded, rotation = 'vertical')
    plt.yticks([])
    plt.xlabel(r'$\theta$')
    plt.title(r'$\theta$ Marginal')
    plt.show()

def visualize_inference(prob, goals):
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    matplotlib.rcParams['font.family'] = "Times New Roman"
    matplotlib.rcParams.update({'font.size': 15})

    plt.figure()
    plt.imshow([prob], cmap='Oranges', interpolation='nearest')
    plt.colorbar(ticks=[0, 0.5, 1.0])
    plt.clim(0, 1.0)

    plt.xticks(range(len(goals)), goals, rotation = 'vertical')
    plt.yticks([])
    plt.xlabel('Goals')
    plt.title("Inference")
    plt.show()