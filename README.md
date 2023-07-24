# N-link-manipulator

In this project, I built an N-link manipulator and control it to locations of your
choice and potentially do fancier things. I chose Model Predictive Control (MPC), specifically an iterative Linear Quadratic Regulator (iLQR)! 

### In this project, the following were implemented: <br>
● A model for N-link planar manipulator <br>
● An LQR policy to regulate the manipulator about a set point <br>
● An iLQR policy to control the end effector precisely <br>
● An iLQR policy for tracking <br>

To simplify dynamics the following assumption were made:
- The mass of the link is located at a point at the end of the link.
- No friction is present at joints.

Trace of arm trajectory using LQR controller: <br>
<p align = "center">  
    <img src="/img/LQR.png" alt="lqr">
<p>
The plot of the cost function with time as it asymptotically reaches zero.
<p align = "center">  
    <img src="/img/LQR1.png" alt="lqr">
<p>
The plot of torque depicting convergence:
<p align = "center">  
    <img src="/img/LQR4.png" alt="lqr">
<p>
Trace of the arm using iLQR controller:
<p align = "center">  
    <img src="/img/ilqr.png" alt="lqr">
<p>
The plot of cost rapidly decreases and asymptotically reaches zero.
<p align = "center">  
    <img src="/img/ilqr_cost.png" alt="lqr">
<p>
The trace of the end effector tracking the letter D:
<p align = "center">  
    <img src="/img/track_slow.png" alt="lqr">
<p>
