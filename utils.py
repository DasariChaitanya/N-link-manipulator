import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import rc
from IPython.display import HTML, Image
rc('animation', html='jshtml')

# Parameters of the arm
num_links = 2
length_m = 0.5 * np.ones(num_links) 
mass_kg = 1 * np.ones(num_links) 
base_se2 = np.array([0., 0., 0.]) # Base of the arm in world frame

def forward_kinematics(theta, length_m, base_se2):
  """ Compute the location of each link in the arm given joint angles
  Args:
    theta: (N,1) array of joint angles 
    length_m: (N,1) array of arm lengths
    base_se2: (3,1) array of base frame (x,y,theta)
  Returns:
    [N+1,2] array of x,y location of each link
  """
  Tmatrix = lambda se2:  np.array([(np.cos(se2[2]), -np.sin(se2[2]), se2[0]), 
                            (np.sin(se2[2]), np.cos(se2[2]), se2[1]), 
                            (0., 0., 1.)])
  num_links = theta.shape[0]
  world_link = Tmatrix(base_se2)

  link_pos = world_link[0:2,-1]
  for idx in range(0, num_links):
    # First rotate
    world_link = world_link @ Tmatrix(np.array([0, 0, theta[idx]]))
    # Then traslate
    world_link = world_link @ Tmatrix(np.array([length_m[idx], 0, 0]))
    link_pos = np.vstack((link_pos, world_link[0:2,-1] ))
  return link_pos

def plot_arm(theta, length_m, base_se2):
  """ Plots arm given joint angles
  Args:
    theta: (N,1) array of joint angles 
    length_m: (N,1) array of arm lengths
    base_se2: (3,1) array of base frame (x,y,theta)
  """
  link_pos = forward_kinematics(theta, length_m, base_se2)
  for idx in range(1, link_pos.shape[0]):
    plt.plot(link_pos[idx-1:idx+1,0], link_pos[idx-1:idx+1,1], '-o', linewidth=5, markersize=10)


if __name__ == "__main__":
  # Let's try visualizing some arm configurations
    fig = plt.figure(figsize=(8, 8))
    plt.xlim([-1.2, 1.2])
    plt.ylim([-1.2, 1.2])
    plt.axis('off')
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')

    theta = np.random.randn(num_links, 1)
    plot_arm(theta, length_m, base_se2)
    fig.show()