import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import time

class EmbeddedSimEnvironment(object):
    
    def __init__(self, model, dynamics, controller, time=5.0):
        """
        Embedded simulation environment. Simulates the syste given dynamics 
        and a control law, plots in matplotlib.

        :param model: model object
        :type model: object
        :param dynamics: system dynamics function (x, u)
        :type dynamics: casadi.DM
        :param controller: controller function (x, r)
        :type controller: casadi.DM
        :param time: total simulation time, defaults to 100 seconds
        :type time: float, optional
        """
        self.model = model
        self.dynamics = dynamics
        self.controller = controller
        self.total_sim_time = time # seconds
        self.dt = self.model.dt
        self.estimation_in_the_loop = False

        # Plotting definitions 
        self.plt_window = float("inf")    # running plot window, in seconds, or float("inf")

    def run(self, x0=[0,0,0,0]):
        """
        Run simulator with specified system dynamics and control function.
        """
        
        print("Running simulation....")
        sim_loop_length = int(self.total_sim_time/self.dt) + 1 # account for 0th
        t = np.array([0])
        y_vec = np.array([x0]).T
        u_vec = np.array([0])
        
        # Start figure
        if len(x0) == 4:
            fig, (ax1,ax2,ax3) = plt.subplots(3)
        elif len(x0) == 2:
            fig, (ax1,ax2) = plt.subplots(2)
        else:
            print("Check your state dimensions.")
            exit()
        for i in range(sim_loop_length):
            
            # Translate data to ca.DM
            x = ca.DM(np.size(y_vec,0),1).full()
            x =  np.array([y_vec[:,-1]]).T

            # Get control input and obtain next state
            try:
                u = self.controller(x, t[-1])
                time.sleep(0.5)
                x_next = self.dynamics(x, u)
            except RuntimeError as e:
                print("Uh oh, your simulator crashed due to unstable dynamics.\n \
                    Retry with new controller parameters.")
                print(e)
                exit()

            # Store data
            t = np.append(t,t[-1]+self.dt)
            y_vec = np.append(y_vec, np.array(x_next), axis=1)
            u_vec = np.append(u_vec, np.array(u))

            # Get plot window values:
            if self.plt_window != float("inf"):
                l_wnd = 0 if int(i+1 - self.plt_window/self.dt) < 1 else int(i+1 - self.plt_window/self.dt)
            else:  
                l_wnd = 0

            if len(x0) == 4:
                ax1.clear()
                ax1.set_title("Pendulum on Cart - Ref: "+str(self.model.x_d)+" [m]")
                ax1.plot( t[l_wnd:-1], y_vec[0,l_wnd:-1], 'r--', \
                          t[l_wnd:-1], y_vec[1,l_wnd:-1], 'b--')
                ax1.legend(["x1","x2"])
                ax1.set_ylabel("X1 [m] / X2 [m/s]")

                ax2.clear()
                ax2.plot(t[l_wnd:-1], y_vec[2,l_wnd:-1], 'g--', \
                         t[l_wnd:-1], y_vec[3,l_wnd:-1], 'k--')
                ax2.legend(["x3","x4"])
                ax1.set_ylabel("X3 [rad] / X4 [rad/s]")



                ax3.clear()
                ax3.plot( t[l_wnd:-1], u_vec[l_wnd:-1], 'b--')
                ax3.set_xlabel("Time [s]")
                ax3.set_ylabel("Control [u]")
            elif len(x0) == 2:
                ax1.clear()
                ax1.plot( t[l_wnd:-1], y_vec[0,l_wnd:-1], 'r--', \
                        t[l_wnd:-1], y_vec[1,l_wnd:-1], 'b--')
                
                ax1.legend(["x1","x2"])
                ax1.set_ylabel("X1 [rad] / X2 [rad/s]")
                ax1.set_title("Inverted pendulum")

                ax2.clear()
                ax2.plot( t[l_wnd:-1], u_vec[l_wnd:-1], 'b--')
                ax2.set_xlabel("Time [s]")
                ax2.set_ylabel("Control [u]")

            else:
                print("Please check your state dimensions.")
                exit()
                
            plt.pause(0.01)
            
        plt.show()
        return t, y_vec, u_vec

    def set_window(self, window):
        """
        Set the plot window length, in seconds.

        :param window: window length [s]
        :type window: float
        """
        self.plt_window = window