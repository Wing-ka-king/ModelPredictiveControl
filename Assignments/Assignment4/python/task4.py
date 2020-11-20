import numpy as np
import scipy

from model import SimplePendulum, Pendulum
from mpc import MPC
from simulation import EmbeddedSimEnvironment

# Create pendulum and controller objects
pendulum = SimplePendulum()

# Get the system discrete-time dynamics
A, B, C = pendulum.get_discrete_system_matrices_at_eq()

# Solve the ARE for our system to extract the terminal weight matrix P
Q = 
R = 
P = 

# Instantiate controller
ctl = MPC(model=pendulum, 
          dynamics=pendulum.discrete_time_dynamics, 
          Q = Q , R = R, P = P,
          horizon=3,
          ulb=-20, uub=20, 
          xlb=[-np.pi/2, -np.pi/2], 
          xub=[np.pi/2, np.pi/2])

# Part II - Simple Inverted Pendulum
sim_env = EmbeddedSimEnvironment(model=pendulum, 
                                dynamics=pendulum.discrete_time_dynamics,
                                controller=ctl.mpc_controller,
                                time = 6)

t, y, u = sim_env.run([-np.pi/4,0])

# Part III - Full cart model
pendulum = Pendulum()

# Get the system discrete-time dynamics
A, B, Bw, C = pendulum.get_discrete_system_matrices_at_eq()

# Solve the ARE for our system to extract the terminal weight matrix P
Q = np.eye(4)*0
R = np.eye(1)*0
P = np.eye(4)*0

# Instantiate controller
ctl = MPC(model=pendulum, 
          dynamics=pendulum.discrete_time_dynamics, 
          horizon=7,
          Q = Q , R = R, P = P,
          ulb=-10, uub=10, 
          xlb=[-2, -10, -np.pi/2, -np.pi/2], 
          xub=[12, 10, np.pi/2, np.pi/2])

# Solve without disturbance
ctl.set_reference(x_sp=np.array([10,0,0,0]))
sim_env_full = EmbeddedSimEnvironment(model=pendulum, 
                                dynamics=pendulum.pendulum_linear_dynamics_with_disturbance,
                                controller=ctl.mpc_controller,
                                time = 6)
sim_env_full.run([0,0,0,0])

# Solve witho disturbance
pendulum.enable_disturbance(w=0.05)
ctl.set_reference(x_sp=np.array([10,0,0,0]))
sim_env_full_dist = EmbeddedSimEnvironment(model=pendulum, 
                                dynamics=pendulum.discrete_time_dynamics,
                                controller=ctl.mpc_controller,
                                time = 10)
sim_env_full_dist.set_window(5)
sim_env_full_dist.run([0,0,0,0])