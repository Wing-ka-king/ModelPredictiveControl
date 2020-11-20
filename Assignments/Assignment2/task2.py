from model import SimplePendulum, PendulumOnCart
from dp import Gridder, BiggerGridder
from simulation import EmbeddedSimEnvironment
import casadi as ca
import numpy as np
import time

##### Part 1 - Simple Pendulum model with Dynamic Programming
###
### Create pendulum and controller objects
##pendulum = SimplePendulum()
##
### Get the system discrete-time dynamics
##A, B, C = pendulum.get_discrete_system_matrices_at_eq()
##print("A: ", A, "\nB: ", B, "\nC: ", C)
##
##ctl = Gridder(model=pendulum, Q=np.diag([3000,30]), R=0.1, sim_time=5)
##start_time = time.time()
##ctl.set_linear_controller()
##time_linear = time.time() - start_time
##
### Initialize simulation environment
##sim_env = EmbeddedSimEnvironment(model=pendulum, 
##                                dynamics=pendulum.discrete_time_dynamics,
##                                controller=ctl.finite_time_dp,
##                                time = 5)
##
##t, y, u = sim_env.run([np.pi/4,0])
##energy, cost = ctl.get_energy_cost()
##print("Total Cost: ", cost)
##print("Total Energy: ", energy)
##
### Run nonlinear controller
##ctl.reset_energy_cost()
##ctl.set_weights(np.diag([3000,30]), R=0.1)
##start_time = time.time()
##ctl.set_nonlinear_controller()
##time_nonlinear = time.time() - start_time
##
### Initialize simulation environment
##sim_env = EmbeddedSimEnvironment(model=pendulum, 
##                                dynamics=pendulum.discrete_nl_dynamics,
##                                controller=ctl.finite_time_dp,
##                                time = 5)
##
##t, y, u = sim_env.run([np.pi/4,0])
##energy, cost = ctl.get_energy_cost()
##print("Nonlinear Controller Total Cost: ", cost)
##print("Nonlinear Controller Total Energy: ", energy)
##print("Linear Controller time: ", time_linear)
##print("Nonlinear Controller time: ", time_nonlinear)
##

### Part 2 - Cart model with Dynamic Programming for Tracking
#
# 
cart_pendulum = PendulumOnCart()
A, B, C = cart_pendulum.get_discrete_system_matrices_at_eq()
print("A: ", A, "\nB: ", B, "\nC: ", C)

ctl = BiggerGridder(model=cart_pendulum, 
                    Q=np.diag([ 1,1000,1,1000]), # TODO: You might want to tune this 
                    R=1,                # TODO: You might want to tune this
                    sim_time=5)
ctl.set_reference()
ctl.set_controller()


# Simulate full system
sim_env_full = EmbeddedSimEnvironment(model=cart_pendulum, 
                                dynamics=cart_pendulum.continuous_time_nonlinear_dynamics,
                                controller=ctl.finite_time_dp,
                                time = 5)

# # Also returns time and state evolution
t, y, u = sim_env_full.run([0,0,0,0])
