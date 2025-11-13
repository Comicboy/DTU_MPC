import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, curve_fit
from scipy.linalg import expm
from scipy.linalg import eig
import scipy.linalg
import matplotlib.pyplot as plt
import control as ct
import cvxpy as cp

from MPC.functions import *
from MPC.parameters import p

# --------------------------------------------------------
# Model
# --------------------------------------------------------
class OpenLoop_ModelA(ModelSimulation):
    '''Deterministic without control, and any noise'''
    
    def dynamics(self, t, x, u, d, p):
        return FourTankSystem(t, x, u, d, p)

    def disturbance(self, t, d):
        return d  # constant
    
    def control(self, t, u):
        return u

    def step(self, t_span, x0, u, d):
        sol = solve_ivp(
            fun=lambda t, x: self.dynamics(t, x, u, d, self.p),
            t_span=t_span,
            y0=x0,
            method="BDF",
            dense_output=False
        )
        return sol.t, sol.y.T
    
    def full_output(self, T, X, U, D):
        # Helper variables
        nT, nX = X.shape
        
        # Unpack parameters
        a = self.p[0:4]     # Pipe cross-sectional areas [cm^2]
        A = self.p[4:8]     # Tank cross-sectional areas [cm^2]
        gamma = self.p[8:10] # Valve positions [-]
        g = self.p[10]      # Gravity [cm/s^2]
        rho = self.p[11]    # Density of water [g/cm^3]
        
        # Compute measured variables (liquid levels H)
        H = np.zeros((nT, nX))
        for i in range(nT):
            H[i, :] = X[i, :] / (rho * A)
            
        # Compute the flows out of each tank
        Qout = np.zeros((nT, nX))
        for i in range(nT):
            Qout[i, :] = a * np.sqrt(2 * g * H[i, :])
        # --------------------------------------------------------


        Ut = np.ones((len(self.ts),3))
        Dt = np.ones((len(self.ts),3)) 
        Ut[:,0] = self.ts
        Ut[:,1] = U[:,0]
        Ut[:,2] = U[:,1]
        Dt[:,0] = self.ts
        Dt[:,1] = D[:,0]
        Dt[:,2] = D[:,1]
        
        return T, X, H, Qout, Ut, Dt
    
    def simulate(self):
        """Main simulation loop"""
        ts = self.ts
        x_cur = self.x0
        d_cur = self.d0
        u_cur = self.u0

        Tfull = []
        Xfull = []
        Dfull = [d_cur]
        Ufull = [u_cur]
        
        y_cur = self.SystemSensor(x_cur[None,:])
        Yfull = [y_cur]

        for k in range(1, len(ts)):
            t_cur = ts[k - 1]
            t_next = ts[k]
            
            u_cur = self.control(t_cur,u_cur)
            d_cur = self.disturbance(t_cur, d_cur)

            T_segment, X_segment = self.step((t_cur, t_next), x_cur, u_cur, d_cur)

            if k == 1:
                Tfull = T_segment
                Xfull = X_segment
            else:
                Tfull = np.concatenate((Tfull, T_segment[1:]))
                Xfull = np.vstack((Xfull, X_segment[1:]))

            Dfull.append(d_cur.squeeze())
            Ufull.append(u_cur.squeeze())
            x_cur = X_segment[-1, :]
            
            y_cur = self.SystemSensor(x_cur[None,:])
            Yfull.append(y_cur)
            
        T, X, H, Qout, Ut, Dt = self.full_output(Tfull, Xfull, np.array(Ufull), np.array(Dfull))
        return T, X, H, Qout, Ut, Dt, np.array(Yfull)
# --------------------------------------------------------
# Simulation scenario
# --------------------------------------------------------

t0 = 0.0        # [s] Initial time
tf = 20 * 60    # [s] Final time
dt = 10         # sample time

ts = np.arange(t0, tf+dt, dt)

# Initial liquid masses in tanks [g]
m10 = 0.0
m20 = 0.0
m30 = 0.0
m40 = 0.0

# Flow rates from pumps [cm^3/s]
F1 = 300.0
F2 = 300.0

# Initial state vector
x0 = np.array([m10, m20, m30, m40])

# Input vector
u0 = np.array([F1, F2])
d0 = np.array([100.0, 100.0])  # disturbance starts at 100,100
# --------------------------------------------------------

ModelA = OpenLoop_ModelA(ts=ts, x0=x0, u0=u0, d0 = d0, p=p)

T, X, H, Qout, Ut, Dt, y = ModelA.simulate()
plot_results(T, X, H, Qout, Ut, Dt, plot_outputs=['H'])

plt.plot(ts, y.squeeze(), label='Measured Output')
plt.show()

us = compute_steady_state_pump_flow(H[-1,:], p)
print(us, H[-1,:])

# --------------------------------------------------------
# Simulation scenario
# --------------------------------------------------------

t0 = 0.0        # [s] Initial time
tf = 20 * 60    # [s] Final time
dt = 10         # sample time

ts = np.arange(t0, tf+dt, dt)

h10, h20, h30, h40 = H[-1,:] # heights
# Convert levels to mass [g] => m = rho * A * h
rho = p[11]
A1, A2, A3, A4 = p[4:8]

m10 = rho * A1 * h10
m20 = rho * A2 * h20
m30 = rho * A3 * h30
m40 = rho * A4 * h40

# Flow rates from pumps [cm^3/s]
F1s = 300.0
F2s = 300.0

# Initial state vector
x0s = X[-1,:]
# Input vector

u0 = np.array([F1, F2])
d0 = np.array([100.0, 100.0])  # disturbance starts at 100,100
# --------------------------------------------------------

ModelA = OpenLoop_ModelA(ts=ts, x0=x0s, u0=us, d0 = d0, p=p)

T, X, H, Qout, Ut, Dt, y = ModelA.simulate()
plot_results(T, X, H, Qout, Ut, Dt, plot_outputs=['H', 'M'])

