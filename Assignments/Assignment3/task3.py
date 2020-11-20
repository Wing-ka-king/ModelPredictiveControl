import numpy as np

from model import Pendulum
from dlqr import DLQR
from simulation import EmbeddedSimEnvironment

time = 20

# Create pendulum and controller objects
pendulum = Pendulum()

# Get the system discrete-time dynamics
A, B, Bw, C = pendulum.get_discrete_system_matrices_at_eq()
ctl = DLQR(A, B, C)

### Get control gains
##K, P = ctl.get_lqr_gain(Q= np.diag([1/(10^2), 1/(5^2), (1/0.05**2), 1/(3^2)]), 
##                        R= 1/(10^2))
##
### Get feeforward gain
##lr = ctl.get_feedforward_gain(K)
##
##
##### Part I - no disturbance
##sim_env = EmbeddedSimEnvironment(model=pendulum, 
##                                 dynamics=pendulum.discrete_time_dynamics,
##                                 controller=ctl.feedfwd_feedback,
##                                 time = time )
##sim_env.set_window(time)
##t, y, u = sim_env.run([0,0,0,0])

###Part I - with disturbance
##pendulum.enable_disturbance(w=0.01)  
##sim_env_with_disturbance = EmbeddedSimEnvironment(model=pendulum, 
##                                 dynamics=pendulum.discrete_time_dynamics,
##                                 controller=ctl.feedfwd_feedback,
##                                 time = time)
##sim_env_with_disturbance.set_window(time)
##t, y, u = sim_env_with_disturbance.run([0,0,0,0])
##

##### Part II
Ai, Bi, Bwi, Ci = pendulum.get_augmented_discrete_system()
ctl.set_system(Ai, Bi, Ci)
K, P = ctl.get_lqr_gain(Q= np.diag([1/(10^2), 1/(5^2), (1/0.05**2), 1/(3^2), 1/(10.0**2)]), 
                       R= 1/(10^2))
lr = ctl.get_feedforward_gain(K)

# Get feeforward gain              
ctl.set_lr(lr)     

pendulum.enable_disturbance(w=0.1)  
##sim_env_with_disturbance = EmbeddedSimEnvironment(model=pendulum, 
##                                dynamics=pendulum.pendulum_augmented_dynamics,
##                                controller=ctl.lqr_ff_fb_integrator,
##                                time = time)
##sim_env_with_disturbance.set_window(time)
##t, y, u = sim_env_with_disturbance.run([0,0,0,0,0])

### Part III
# Output feedback
C = np.array([[1,0,0,0]])
C = np.array([[1,0,0,0],
             [0,0,1,0]])

Qp = np.eye(4)
sigma = 1
#Qp = np.array(sigma * Bi[0:4] @ Bi[0:4].T) + np.diag([0.01, 0.01, 0.01, 0.1])
Rn = np.eye(np.size(C,0))#*0.005
pendulum.set_kf_params(C,Qp,Rn)
pendulum.init_kf()

sim_env_with_disturbance_estimated = EmbeddedSimEnvironment(model=pendulum, 
                               dynamics=pendulum.pendulum_augmented_dynamics,
                               controller=ctl.lqr_ff_fb_integrator,
                               time = time)
sim_env_with_disturbance_estimated.set_estimator(True)
sim_env_with_disturbance_estimated.set_window(time)
t, y, u = sim_env_with_disturbance_estimated.run([0,0,0,0,0])
1+1
