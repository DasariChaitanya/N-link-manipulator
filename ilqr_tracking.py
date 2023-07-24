import autograd.numpy as np
import math 
from utils import animate_arm,wrapToPi
from lqr import manipulator_dynamics
from ilqr import end_effector_cost,nearestPSD,ee_error,get_jacobians,LM


H = 100 # Horizon 
dynamics_fun = lambda x, u: manipulator_dynamics(x, u)
cost_fun = lambda x, u,ee_l: end_effector_cost(x, u, ee_l)


def ilqr_track_backward(x_ini,u_ini,dyn_x, dyn_u, cost_x, cost_xx, cost_u, cost_uu,ee_goal,lamb ):  
  """  backward pass in an iteration of ilqr
  Args:
    x_ini,u_ini : rollout states,actions for each timesteps
    lamb: lambda for Levenberg Marquardt algorithm
    Jacobian funcs for dynamics and cost function
  Return:
    K_list: list of gains for each timestep
    """
  H = len(x_ini)-1
  K_list = [None]*H
  scale = ee_goal.shape[0]/H
  for t in range(H-1,-1,-1):
      A = dyn_x(x_ini[t],u_ini[t])
      A = np.vstack((np.hstack([A,np.zeros((4,1))]),np.hstack([np.zeros((1,4)),np.array([[1]])])))
      B = np.concatenate((dyn_u(x_ini[t],u_ini[t]),np.array([[0,0]])),axis = 0)
      goal = ee_goal[int(np.floor(scale*t))]
      c_off = cost_fun(x_ini[t],u_ini[t],goal)
      qq = cost_x(x_ini[t],u_ini[t],goal)[:,None]
      q1 = np.hstack((cost_xx(x_ini[t],u_ini[t],goal)/2 , qq/2))
      q2 = np.append(qq.T/2, np.array([c_off]))
      Q = np.vstack((q1,q2))  
      Q = nearestPSD(Q)
      # r1 =  np.hstack((cost_uu(x_ini[t],u_ini[t])/2 , cost_u(x_ini[t],u_ini[t])[:,None]/2))
      # r2 = np.append(cost_u(x_ini[t],u_ini[t])[:,None].T/2, np.array([[0]]))
      # R = np.vstack((r1,r2))
      R = np.identity(2)
      if t == H-1:
        V = Q 
      K = -LM(R + B.T @ V @ B,lamb) @ (B.T @ V @ A) 
      V = Q + K.T @ R @ K + np.transpose(A + B @ K) @ V @ (A + B @ K)
      K_list[t] = K
  return K_list

def ilqr_track_forward(K_list,x_roll,u_roll,dynamics_fun,ee_goal):
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
    cost,err = 0,0
    scale = ee_goal.shape[0]/H
    for t in range(len(K_list)):
      dx = x - x_roll[t]
      dx = np.append(dx,np.array([[1]]))
      du = K_list[t] @ dx
      u = u_roll[t] + du
      u_opt.append(u)
      cost += cost_fun(x,u,ee_goal[int(np.floor(scale*t))])
      err += ee_error(x,ee_goal)
      x = dynamics_fun(x,u)
      x[:2] = wrapToPi(x[:2])
      x_opt.append(x)
    return x_opt,u_opt,cost,err

def simulate_ilqr_track(x_opt,u_opt,dynamics_fun, cost_fun,ee_goal,lamb):
  """  Simulates one iteration of ilqr
  Args:
    x_opt,u_opt : rollout states,actions for each timesteps
    dynamics_fun: The dynamics function
    cost_fun: Cost function
    speed_scale: Higher value denotes slower speed (set between 3 to 10)
  Returns:
    List of states,actions one for each timestep
    cost and error of trajectory 
  """
  dyn_x, dyn_u, cost_x, cost_xx, cost_u, cost_uu = get_jacobians(dynamics_fun, cost_fun)
  K_list = ilqr_track_backward(x_opt,u_opt,dyn_x, dyn_u, cost_x, cost_xx, cost_u, cost_uu,ee_goal,lamb)
  x_opt,u_opt,cost,err = ilqr_track_forward(K_list,x_opt,u_opt,dynamics_fun,ee_goal)
  return x_opt,u_opt,cost,err

def ilqr_track(ee_goal,dynamics_fun, cost_fun,speed_scale=10):  
  """  
  Args:
    ee_goal: (N, 1) end effector goal points
    K_list: List of K matrices for each timestep
    dynamics_fun: The dynamics function
    cost_fun: Cost function
    speed_scale: Higher value denotes slower speed (set between 3 to 10)
  Returns:
    List of states,actions one for each timestep
    list of trajectory cost and error for each iteration
  """
  
  cost_list,err_list = [],[]
  lamb = 1
  H = int(ee_goal.shape[0] * speed_scale)
  u_ini = np.linspace([0,0],[0.0,0.0],H)
  x_ini = [np.array([0.,0,0,0])]
  for i in range(H):
    x_ini += [dynamics_fun(x_ini[-1],u_ini[i])]
  x_opt_traj,u_opt = x_ini,u_ini
  x_opt_traj,u_opt,cost,err = simulate_ilqr_track(x_opt_traj,u_opt,dynamics_fun, cost_fun,ee_goal,lamb)
  cost_list.append(cost)
  err_list.append(err)
  while True:
      x_opt_traj,u_opt,cost,err = simulate_ilqr_track(x_opt_traj,u_opt,dynamics_fun, cost_fun,ee_goal,lamb)
      cost_list.append(cost)
      err_list.append(err)
      cost_conv = abs(cost_list[-1] - cost_list[-2])/cost_list[-2]
      if cost_list[-1] < cost_list[-2]:
        lamb /= 10
      else:
        lamb *= 10
      if cost_conv < 0.001 or len(cost_list) > 30 or lamb > 1000:
        break
  return x_opt_traj, err_list, cost_list,u_opt

if __name__ == "__main__":        
    #trajectory for letter D
    ang = np.linspace(-math.pi/2,math.pi/2, 10)
    p = np.hstack([0.5*np.cos(ang)[:,None],0.5*np.sin(ang)[:,None]+0.5])
    l = np.linspace([0,1],[0,0],10)
    p = np.vstack([l,p])

    x_opt_traj, err_list, cost_list,u_opt = ilqr_track(p,dynamics_fun, cost_fun,5)
    animate_arm(x_opt_traj,draw_ee =True,draw_trace=False)