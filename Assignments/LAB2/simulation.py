import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import time

class EmbeddedSimEnvironment(object):
    
    def __init__(self, model, dynamics, controller, time):
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

        # Plotting definitions 
        self.plt_window = float("inf")    # running plot window, in seconds, or float("inf")

    def run(self, x0):
        """
        Run simulator with specified system dynamics and control function.
        """
        
        print("Running simulation....")
        sim_loop_length = int(self.total_sim_time/self.dt) + 1 # account for 0th
        t = np.array([0])
        y_vec = np.array([x0]).T
        u_vec = np.array([0,0,0,0]).T

        
        # Start figure
        
        #fig = plt.figure() 
        plt.rc('grid', linestyle="--", color='black')     
        fig, (ax1,ax2,ax3,ax4) = plt.subplots(4)    
        #fig, (ax3,ax4) = plt.subplots(2)
        if len(x0) >12:
            fig, (ax5) = plt.subplots(1)         

        for i in range(sim_loop_length):
            
            # Translate data to ca.DM
            x = ca.DM(np.size(y_vec,0),1).full()
            x = np.array([y_vec[:,-1]]).T
            print("Step "+str(i)+"/"+str(sim_loop_length))
            

            try:
                # Get control input and obtain next state
                u = self.controller(x)   
                #print("Control input is set as "+str(u)+" ")

                x_next = self.dynamics(x, u)
                
            except RuntimeError:
                print("Uh oh, your simulator crashed due to unstable dynamics.\n \
                       Retry with new controller parameters.")
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

            
            ax1.clear()
            #ax1.set_title("Quadcopter - Ref: "+str(self.model.x_d))
            ax1.set_title("Quadcopter")
            ax1.plot( t[l_wnd:-1], y_vec[0,l_wnd:-1], 'r-', \
                      t[l_wnd:-1], y_vec[1,l_wnd:-1], 'b--', \
                      t[l_wnd:-1], y_vec[2,l_wnd:-1], 'g-')
                        
            ax1.legend(["X position","Y position","Z position"],loc=1)
            ax1.set_ylabel("meters")
            ax1.grid()

            ax2.clear()
            ax2.plot( t[l_wnd:-1], y_vec[3,l_wnd:-1], 'r-', \
                      t[l_wnd:-1], y_vec[4,l_wnd:-1], 'b--', \
                      t[l_wnd:-1], y_vec[5,l_wnd:-1], 'g-')
                        
            ax2.legend(["X velocity","Y velocity","Z velocity"],loc=1)
            ax2.set_ylabel("m/s")
            ax2.grid()

            ax3.clear()
            ax3.plot( t[l_wnd:-1], ((y_vec[6,l_wnd:-1])*(180/np.pi)), 'r-', \
                      t[l_wnd:-1], ((y_vec[7,l_wnd:-1])*(180/np.pi)), 'b-', \
                      t[l_wnd:-1], ((y_vec[8,l_wnd:-1])*(180/np.pi)), 'g-')
                        
            ax3.legend(["roll angle","pitch angle","yaw angle"],loc=1)
            ax3.set_ylabel("degree")
            ax3.grid()

            ax4.clear()
            ax4.plot( t[l_wnd:-1], (y_vec[9,l_wnd:-1]) , 'r-', \
                      t[l_wnd:-1], (y_vec[10,l_wnd:-1]), 'b--', \
                      t[l_wnd:-1], (y_vec[11,l_wnd:-1]), 'g-')
                          
            ax4.legend(["Omega x","Omega y","Omega z"],loc=1)
            ax4.set_ylabel("rad/s")
            ax4.grid()


            if len(x0) >12:
                ax5.clear()
                ax5.plot( t[l_wnd:-1], (y_vec[12,l_wnd:-1]) , 'r-', \
                      t[l_wnd:-1], (y_vec[13,l_wnd:-1]), 'b--', \
                      t[l_wnd:-1], (y_vec[14,l_wnd:-1]), 'g-')
                        
                ax5.legend(["X error","Y error","Z error"],loc=1)
                ax5.set_ylabel("error m")
                ax5.grid()

                     
        plt.show()
        #print(y_vec)
        return t, y_vec, u_vec