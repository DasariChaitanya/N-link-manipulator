import autograd.numpy as np
from autograd import grad, jacobian
import math
import matplotlib.pyplot as plt
from utils import animate_arm,manipulator_dynamics

def get_jacobians(dynamics_fun, cost_fun):
  """ Compute jacobians of dynamics and cost function
  Args:
    dynamics_fun: The dynamics function
    cost_fun: The cost function
  Returns:
    A tuple of first and second derivatives of dynamics and cost function
  """
  dyn_x = jacobian(dynamics_fun, 0)
  dyn_u = jacobian(dynamics_fun, 1)
  cost_x = jacobian(cost_fun, 0)
  cost_xx = jacobian(cost_x, 0)
  cost_u = jacobian(cost_fun, 1)
  cost_uu = jacobian(cost_u, 1)
  return dyn_x, dyn_u, cost_x, cost_xx, cost_u, cost_uu

def regulator_cost(x, u, theta_ref):
  """ Computes quadratic cost of state, action pair
  Args:
    x: Current state
    u: Current control action
    theta_ref: The reference angle to regulate the arm around
  Returns:
    Scalar cost
  """
  x1 = x[:2] - theta_ref
  x2 = x[2:]
  cost = (x1.T @ np.eye(2) @ x1) + (x2.T @ np.eye(2) @ x2)  + (u.T @ np.eye(2) @ u) 
  return cost

def lqr_input_matrices(x_ref, dyn_x, dyn_u, cost_xx, cost_uu):
  """ Compute A, B, Q, R matrices
  Args:
    x_ref: State around which to linearize around
    dyn_x: First derivative of dynamics wrt to x
    dyn_u: First derivative of dynamics wrt to u
    cost_xx: Second derivative of dynamics wrt to x
    cost_uu: Second derivative of dynamics wrt to u
  Returns:
    A, B, Q, R matrices
  """
  u_ref = np.array([0., 0.])
  A = dyn_x(x_ref,u_ref)
  B = dyn_u(x_ref,u_ref)
  R = cost_uu(x_ref,u_ref)
  Q = cost_xx(x_ref,u_ref)
  return A,B,Q,R

  
def LM(Q_uu,lamb=1):
  Q_uu_evals, Q_uu_evecs = np.linalg.eig(Q_uu)
  Q_uu_evals[Q_uu_evals < 0] = 0.0
  Q_uu_evals += lamb
  Q_uu_inv = np.dot(Q_uu_evecs, np.dot(np.diag(1.0/Q_uu_evals), Q_uu_evecs.T))
  return Q_uu_inv

def lqr_gains(A, B, Q, R, H):
  """ Run value iteration and compute V and K matrices for each time step
  Args:
    A: (2N, 2N) dynamics matrix
    B: (2N, N) dynamics matrix 
    Q: (2N, 2N) quadratic cost matrix
    R: (N, N) quadratic cost matrix
    H: Time horizon
  Returns:
    List of K matrices for each timestep
  """
  V = Q 
  K_list = [None]*H
  # K_list.append(K)
  for t in range(H-1, -1, -1):
    K = -LM(R+B.T @ V @ B) @ (B.T @ V @ A)
    V = Q + K.T @ R @ K + np.transpose(A + B @ K) @ V @ (A + B @ K)
    K_list[t] = K
  return K_list

def lqr_forward(x_0, K_list, dynamics_fun,theta_ref):
  """  Start from initial state and iteratively apply K matrices and dynamics 
  to roll out a trajectory
  Args:
    x_0: (2N, 1) initial state
    K_list: List of K matrices for each timestep
    dynamics_fun: The dynamics function
  Returns:
    List of states one for each timestep 
  """
  
  x_list = []
  x_list += [x_0]
  cost_list = []
  u_list=[]
  x_ref = np.concatenate((theta_ref,np.array([0,0])))
  for t in (range(len(K_list))):
    x_err = x_list[-1] - x_ref
    u = K_list[t] .dot(x_err)
    u_list.append(u)
    cost_list.append( regulator_cost(x_list[-1], u, theta_ref))
    x_list += [ dynamics_fun(x_list[-1],u)]
  # print(x_list[0])
  return x_list,u_list,cost_list

if __name__ == "__main__":
  # Apply LQR to regulate the arm starting from an initial state
    x_0 = np.array([0, 0., 0., 0.]) # Initial state
    x_ref = np.array([2*math.pi/3,math.pi/2, 0., 0.]) # State that we are regulating around
    H = 100 # Horizon 

    dynamics_fun = lambda x, u: manipulator_dynamics(x, u)
    cost_fun = lambda x, u: regulator_cost(x, u, x_ref[0:2])

    dyn_x, dyn_u, cost_x, cost_xx, cost_u, cost_uu = get_jacobians(dynamics_fun, cost_fun)
    A, B, Q, R = lqr_input_matrices(x_ref, dyn_x, dyn_u, cost_xx, cost_uu)
    K_list = lqr_gains(A, B, Q, R, H)

    x_traj,u_list,cost_list = lqr_forward(x_0, K_list, manipulator_dynamics,x_ref[0:2])

    animate_arm(x_traj, draw_trace=True)

  # Plot cost vs time
    plt.plot(cost_list)
    title = 'for X_0 = ' + np.array2string(x_0, precision=2) + ' and X_ref = ' + np.array2string(x_ref,precision = 2)
    plt.title(title, fontsize=14)
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.show()

  #Plot torque vs time  
    plt.plot(u_list)
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('Torque', fontsize=12)
    plt.show()

