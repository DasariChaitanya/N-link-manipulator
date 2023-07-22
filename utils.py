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

def animate_arm(x_traj, draw_ee=False, draw_trace=False):
  """ Animate the arm as it follows a joint angle trajectory
  Args:
    theta_traj: List of (N,1) joint angles
    draw_ee: If true, plot a trace of the end effector as it moves 
  Returns:
    animate object
  """
  fig = plt.figure()
  plt.xlim([-1.2, 1.2])
  plt.ylim([-1.2, 1.2])
  plt.axis('off')
  plt.grid()
  plt.gca().set_aspect('equal', adjustable='box')

  arm_lines = []
  for idx in range(0, length_m.shape[0]):
      arm_lines += plt.plot([0, 0], [0, 0], '-o', linewidth=5, markersize=10)

  def update_arm(i):
    theta = x_traj[i][:2]
    link_pos = forward_kinematics(theta, length_m, base_se2)
    if draw_ee: 
      plt.scatter(link_pos[-1,0], link_pos[-1,1], s=10, color='k')
    if draw_trace:
      colors = plt.cm.tab10(np.linspace(0,1,link_pos.shape[0]))
      for idx in range(1, link_pos.shape[0]):
        plt.plot(link_pos[idx-1:idx+1,0], link_pos[idx-1:idx+1,1], linewidth=5, alpha=0.1, color=colors[idx-1])
    for idx in range(1, link_pos.shape[0]):
      arm_lines[idx-1].set_data(link_pos[idx-1:idx+1,0], link_pos[idx-1:idx+1,1])

  anim = FuncAnimation(fig, update_arm, frames=len(x_traj), interval=30, blit=False, repeat=False)  
  plt.show()
  return anim



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
    plt.show()

    # Let's animate the arm moving from an initial to a final configuration
    x_0 = np.array([0., 0., 0., 0.])
    x_f = np.array([2., 2., 0., 0.])
    theta_traj = list(np.linspace(x_0, x_f, num=50, endpoint=True))
    animate_arm(theta_traj, draw_ee=True, draw_trace=True)