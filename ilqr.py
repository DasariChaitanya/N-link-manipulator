import autograd.numpy as np
import math
from utils import length_m, wrapToPi, animate_arm
from lqr import get_jacobians,lqr_gains,lqr_input_matrices,manipulator_dynamics,lqr_forward,regulator_cost

#generate roll out 
H = 100 # Horizon 
x_0 = np.array([0.,0, 0., 0.])
x_ref = np.array([math.pi/20, 0, 0., 0.])
dynamics_fun = lambda x, u: manipulator_dynamics(x, u)
cost_fun = lambda x, u: regulator_cost(x, u, x_ref[0:2])

dyn_x, dyn_u, cost_x, cost_xx, cost_u, cost_uu = get_jacobians(dynamics_fun, cost_fun)
A, B, Q, R = lqr_input_matrices(x_ref, dyn_x, dyn_u, cost_xx, cost_uu)
K_list = lqr_gains(A, B, Q, R, 100)

x_ini,u_ini,_ = lqr_forward(x_0, K_list, manipulator_dynamics,x_ref[0:2])

def end_effector_cost(x, u, ee_goal):
  """ Computes cost of state, action pair
  Args:
  x: Current state
  u: Current control action
  ee_goal: The goal 
  theta_ref: The reference angle to regulate the arm around
  Returns:
  Scalar cost
  """ 
  ee_x = length_m[0] * np.cos(x[0]) + length_m[1] * np.cos(x[0]+x[1])
  ee_y = length_m[0] * np.sin(x[0]) + length_m[1] * np.sin(x[0]+x[1])

  cost = 5*((ee_x - ee_goal[0])**2 + (ee_y - ee_goal[1])**2)  + 0.5*(x[2:].T @ x[2:]) #+ (u.T @ u)
  return cost

def nearestPSD(A):
  """
  Returns:
    Nearest PSD to a matirx
  """
  C = (A + A.T)/2
  eigval, eigvec = np.linalg.eig(C)
  eigval[eigval < 0] = 0

  return eigvec.dot(np.diag(eigval)).dot(eigvec.T)

def LM(B,lamb=1):
  """
    Levenberg Marquardt algorithm
    Returns:
      Inverse of the matrix after adding lambda*I to the nearest PSD of the given matrix
  """
  B_evals, B_evecs = np.linalg.eig(B)
  B_evals[B_evals < 0] = 0.0
  B_evals += lamb
  B_inv = np.dot(B_evecs, np.dot(np.diag(1.0/B_evals), B_evecs.T))
  return B_inv

def ee_error(xe,ee_goal):
  ee_x = length_m[0] * np.cos(xe[0]) + length_m[1] * np.cos(xe[0]+xe[1])
  ee_y = length_m[0] * np.sin(xe[0]) + length_m[1] * np.sin(xe[0]+xe[1])
  ee = np.array([ee_x,ee_y])
  return np.linalg.norm(ee-ee_goal)

def ilqr_backward(x_ini,u_ini,lamb):  
  """  backward pass in an iteration of ilqr
  Args:
    x_ini,u_ini : rollout states,actions for each timesteps
    lamb: lambda for Levenberg Marquardt algorithm
  Return:
    K_list: list of gains for each timestep
    """
  K_list = [None]*H
  for t in range(H-1,-1,-1):
      A = dyn_x(x_ini[t],u_ini[t])
      A = np.vstack((np.hstack([A,np.zeros((4,1))]),np.hstack([np.zeros((1,4)),np.array([[1]])])))
      B = np.concatenate((dyn_u(x_ini[t],u_ini[t]),np.array([[0,0]])),axis = 0)
      c_off = cost_fun(x_ini[t],u_ini[t])
      q1 = np.hstack((cost_xx(x_ini[t],u_ini[t])/2 , cost_x(x_ini[t],u_ini[t])[:,None]/2))
      q2 = np.append(cost_x(x_ini[t],u_ini[t])[:,None].T/2, np.array([c_off]))
      Q = np.vstack((q1,q2))  
      Q = nearestPSD(Q)
      r1 =  np.hstack((cost_uu(x_ini[t],u_ini[t])/2 , cost_u(x_ini[t],u_ini[t])[:,None]/2))
      r2 = np.append(cost_u(x_ini[t],u_ini[t])[:,None].T/2, np.array([[0]]))
      R = np.vstack((r1,r2))
      R = np.identity(2)
      if t == H-1:
        V = Q
      K = -LM(R + B.T @ V @ B,lamb) @ (B.T @ V @ A) 
      V = Q + K.T @ R @ K + np.transpose(A + B @ K) @ V @ (A + B @ K)
      K_list[t] = K
  return K_list

def ilqr_forward(K_list,x_roll,u_roll,dynamics_fun):
  """  forward pass in an iteration of ilqr
  Args:
    K_list: list of gains for each timestep
    x_opt,u_opt : rollout states,actions for each timesteps
    dynamics_fun: The dynamics function    
  Returns:
    updated list of states,actions one for each timestep
    cost and error of trajectory
    """
  x = x_roll[0]
  u_opt = []
  x_opt = [x]    
  cost = []
  for t in range(H):
    dx = x - x_roll[t]
    dx = np.append(dx,np.array([[1]]))
    du = K_list[t] @ dx
    u = u_roll[t] + du
    u_opt.append(u)
    cost.append(cost_fun(x,u))
    x = dynamics_fun(x,u)
    x[:2] = wrapToPi(x[:2])
    x_opt.append(x)
  return x_opt,u_opt,cost


def ilqr(x_ini,u_ini,ee_goal):
  x_opt_traj,u_opt = x_ini,u_ini
  cost_list = []
  err = []
  lamb = 1
  K_list = ilqr_backward(x_opt_traj,u_opt,lamb)
  x_opt_traj,u_opt,cost = ilqr_forward(K_list,x_opt_traj,u_opt,dynamics_fun)
  cost_list.append(np.sum(cost))
  err.append(ee_error(x_opt_traj[-1],ee_goal))
  while True:
      K_list = ilqr_backward(x_opt_traj,u_opt,lamb)
      x_opt_traj,u_opt,cost = ilqr_forward(K_list,x_opt_traj,u_opt,dynamics_fun)
      cost_list.append(np.sum(cost))
      err.append(ee_error(x_opt_traj[-1],ee_goal))
      cost_conv = abs(cost_list[-1] - cost_list[-2])/cost_list[-2]
      if cost_list[-1] < cost_list[-2]:
        lamb /= 10
      else:
        lamb *= 10
      if cost_conv < 0.001 or len(cost_list) > 20 or lamb > 1000:
        break
  return x_opt_traj, err, cost_list,cost


if __name__ == "__main__":
    x_0 = np.array([math.pi/2, 0., 0., 0.])
    ee_goal = [0.0, 0.5]
    H = 100

    dynamics_fun = lambda x, u: manipulator_dynamics(x, u)
    cost_fun = lambda x, u: end_effector_cost(x, u, ee_goal)
    dyn_x, dyn_u, cost_x, cost_xx, cost_u, cost_uu = get_jacobians(dynamics_fun, cost_fun)

    x_opt_traj, err, cost_list,cost_last_iter = ilqr(x_ini,u_ini,ee_goal)
    animate_arm(x_opt_traj,draw_trace=True)