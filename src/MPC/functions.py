import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import inspect
from abc import ABC, abstractmethod

from MPC.parameters import p


# Cumpute steady-state pump flows for desired heights to put into controller
def compute_steady_state_pump_flow(r, p):
    # Computes the steady-state pump flows F1 and F2 for desired heights r1, r2, r3, r4
    a = p[0:4]
    A = p[4:8]
    gamma = p[8:10]
    g = p[10]
    rho = p[11]

    # Solve for F1, F2 such that inflows = outflows (done on notes)
    F1 = (a[0]*np.sqrt(2*g*r[0]) - a[2]*np.sqrt(2*g*r[2])) / gamma[0]
    F2 = (a[1]*np.sqrt(2*g*r[1]) - a[3]*np.sqrt(2*g*r[3])) / gamma[1]
    us = np.array([F1, F2])
    return us

def Pcontroller(r,y,us,Kc):
    #### UPDATE THIS FUNCTION ####
    # r: setpoints
    # y: sensor measurements
    # us: steady state inputs
    # Kc: controller gain
    # Implement all tanks???
    e = r-y
    u1 = us[0] + Kc*e[0]
    u2 = us[1] + Kc*e[1]
    u1_max = 500
    u2_max = 500
    u1 = np.clip(u1,0,u1_max)
    u2 = np.clip(u2,0,u2_max)
    return np.array([u1,u2])

def PIcontroller(r,y,us,Kc,Ki,tspan,integral_error):
    # r: setpoints
    # y: sensor measurements
    # us: steady state inputs
    # Kc: controller gain
    # Ki: integral gain
    # tspan: previous time span [t_k-1, t_k]
    # integral_error: accumulated integral error from previous step (start at (0,0))
    Ts = tspan[1] - tspan[0]  # Sampling time
    e = r - y

    # Update integral error
    integral_error_new = integral_error + np.array([e[0],e[1]]) * Ts
    # Compute raw control signal
    u = us + Kc * np.array([e[0],e[1]]) + Ki * integral_error_new
    # Apply actuator limits
    u_clipped = np.clip(u, 0, None)

    # Simple anti-windup: if saturated, don't integrate further in that direction
    for i in range(len(u)):
        if (u_clipped[i] != u[i]):  # saturation happened
            integral_error_new[i] = integral_error[i]  # freeze integrator

    return u_clipped, integral_error_new

def PIDcontroller(r,y,us,Kc,Ki,Kd,tspan,integral_error,prev_error, umin=np.array([0,0]), umax=np.array([500,500])):
    # r: setpoints
    # y: sensor measurements
    # us: steady state inputs
    # Kc: controller gain
    # Ki: integral gain
    # Kd: derivative gain
    # tspan: previous time span [t_k-1, t_k]
    # integral_error: accumulated integral error from previous step
    # prev_error: error from previous step for derivative calculation
    Ts = tspan[1] - tspan[0]  # Sampling time
    e = r - y
    e12 = np.array([e[0],e[1]])
    # Update integral error
    integral_error_new = integral_error + e12 * Ts
    derivative_error = (e12 - prev_error) / Ts if Ts > 0 else np.zeros_like(e12)  # Derivative of error

    # Compute raw control signal
    u = us + Kc * e12 + Ki * integral_error_new + Kd * derivative_error
    # Apply actuator limits
    u_clipped = np.clip(u, umin, umax)

    # Simple anti-windup: if saturated, don't integrate further in that direction
    for i in range(len(u)):
        if (u_clipped[i] != u[i]):  # saturation happened
            integral_error_new[i] = integral_error[i]  # freeze integrator

    return u_clipped, integral_error_new, derivative_error

def ScalarSampleMeanStdVar(x):
    """
    Computes the sample mean, standard deviation, and variance of a scalar random variable.

    Parameters:
    x (ndarray): Input data, shape (N,)

    Returns:
    mean (float): Sample mean
    std_dev (float): Sample standard deviation
    variance (float): Sample variance
    """
    mean = np.mean(x,axis=0)
    std_dev = np.std(x, axis=0)
    xmeanp2std = mean + 2*std_dev
    xmeann2std = mean - 2*std_dev
    return mean, std_dev, xmeann2std, xmeanp2std

def _is_direct_call():
    """Return True if this function was called directly from the top level."""
    # Get the calling function’s frame
    frame = inspect.stack()[2]  # [0] is current, [1] is _is_direct_call, [2] is caller
    # If caller's function name is '<module>', it's top-level (not nested)
    return frame.function == '<module>'

def wiener_process(T, N, Ns, seed=None, plot=True):
    """
    ScalarStdWienerProcess generates Ns realizations of a scalar standard Wiener process.

    Parameters:
    T (float): Final time
    N (int): Number of intervals
    Ns (int): Number of realizations
    seed (int, optional): Seed for random number generator

    Returns:
    W (ndarray): Standard Wiener process in [0, T], shape (Ns, N+1)
    Tw (ndarray): Time points, shape (N+1,)
    dW (ndarray): White noise used to generate the Wiener process, shape (Ns, N)
    """
    
    if seed is not None:
        np.random.seed(seed)

    dt = T / N
    dW = np.sqrt(dt) * np.random.randn(Ns, N)
    W = np.hstack((np.zeros((Ns, 1)), np.cumsum(dW, axis=1)))
    Tw = np.linspace(0, T, N + 1)

    # Only plot if the function is called directly (not from inside another function)
    if plot and _is_direct_call():
        Wmean, sW, Wmeann2std, Wmeanp2std = ScalarSampleMeanStdVar(W)
        
        plt.figure()

        plt.plot(Tw, W.T)
        plt.plot(Tw, Wmean, label='Mean', color='black', linewidth=3)
        plt.plot(Tw, Wmeann2std, color='black', linewidth=3)
        plt.plot(Tw, Wmeanp2std, color='black', linewidth=3)
        
        plt.title('Scalar Standard Wiener Process Realizations')
        plt.xlabel('Time')
        plt.ylabel('W(t)')
        
        #theoretical std deviation
        mean_theoretical = np.zeros_like(Tw)
        std_theoretical = np.sqrt(Tw)
        plt.plot(Tw, mean_theoretical, 'r', label='Theoretical Mean', linewidth=2)
        plt.plot(Tw, mean_theoretical+2*std_theoretical, 'r', label='Theoretical Std Dev', linewidth=2)
        plt.plot(Tw, mean_theoretical-2*std_theoretical, 'r', linewidth=2)
        plt.show() 

    return W, Tw, dW

def plot_results(T, X, H, Qout, U, D, sample_idxs=None, plot_outputs=['M', 'H', 'Q']):
    '''
    Plots of all outputs and inputs of the four-tank system.
    '''
    
    if sample_idxs is not None:
        T = T[sample_idxs]
        X = X[sample_idxs, :]
        H = H[sample_idxs, :]
        Qout = Qout[sample_idxs, :]
        U = U[sample_idxs, :]
        D = D[sample_idxs, :]
    
    tank_labels = ["Tank 1", "Tank 2", "Tank 3", "Tank 4"]
    if 'M' in plot_outputs:
        fig, axes = plt.subplots(2, 4, figsize=(14, 8), sharex=True)

        # --- Column 1: Control inputs ---
        axes[0, 0].step(U[:,0], U[:,1], where='post', color="C0")
        axes[0, 0].set_ylabel("Flow [cm³/s]")
        axes[0, 0].set_title("Pump 1 (F1)")
        axes[0, 0].grid(True, linestyle="--", alpha=0.7)

        axes[1, 0].step(U[:,0], U[:,2], where='post', color="C1")
        axes[1, 0].set_ylabel("Flow [cm³/s]")
        axes[1, 0].set_title("Pump 2 (F2)")
        axes[1, 0].set_xlabel("Time [s]")
        axes[1, 0].grid(True, linestyle="--", alpha=0.7)
        
        # --- Column 2: Disturbances
        axes[0, 1].step(D[:,0], D[:,1], where='post', color="C1")
        axes[0, 1].set_ylabel("Flow [cm³/s]")
        axes[0, 1].set_title("Pump 3 (F3)")
        axes[0, 1].grid(True, linestyle="--", alpha=0.7)

        axes[1, 1].step(D[:,0], D[:,2], where='post', color="C0")
        axes[1, 1].set_ylabel("Flow [cm³/s]")
        axes[1, 1].set_title("Pump 4 (F4)")
        axes[1, 1].set_xlabel("Time [s]")
        axes[1, 1].grid(True, linestyle="--", alpha=0.7)

        # Add a big column title for controls
        fig.text(0.13, 0.95, "Control Inputs", ha="center", va="center", fontsize=14, fontweight="bold")

        # # --- Columns 2-3: Tank outflows ---
        
        for idx, (i, ax, color) in enumerate(zip([2,3,0,1],axes[:, 2:].flat, ['C1','C0', 'C0','C1'])):
            ax.plot(T, Qout[:, i], label=tank_labels[i], color=color)
            ax.set_title(tank_labels[i])
            ax.set_ylabel("Outflow [cm³/s]")
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.legend()

        # Common x-labels for bottom row
        for ax in axes[1, 1:]:
            ax.set_xlabel("Time [s]")

        # Add a big title for tank outflows across cols 2–3
        fig.text(0.65, 0.95, "Tank Outflows", ha="center", va="center", fontsize=14, fontweight="bold")

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()
    
    if 'H' in plot_outputs:
        fig, axes = plt.subplots(2, 4, figsize=(14, 8), sharex=True)

        # --- Column 1: Control inputs ---
        axes[0, 0].step(U[:,0], U[:,1], where='post', color="C0")
        axes[0, 0].set_ylabel("Flow [cm³/s]")
        axes[0, 0].set_title("Pump 1 (F1)")
        axes[0, 0].grid(True, linestyle="--", alpha=0.7)

        axes[1, 0].step(U[:,0], U[:,2], where='post', color="C1")
        axes[1, 0].set_ylabel("Flow [cm³/s]")
        axes[1, 0].set_title("Pump 2 (F2)")
        axes[1, 0].set_xlabel("Time [s]")
        axes[1, 0].grid(True, linestyle="--", alpha=0.7)
        
        # --- Column 2: Disturbances
        axes[0, 1].step(D[:,0], D[:,1], where='post', color="C1")
        axes[0, 1].set_ylabel("Flow [cm³/s]")
        axes[0, 1].set_title("Pump 3 (F3)")
        axes[0, 1].grid(True, linestyle="--", alpha=0.7)

        axes[1, 1].step(D[:,0], D[:,2], where='post', color="C0")
        axes[1, 1].set_ylabel("Flow [cm³/s]")
        axes[1, 1].set_title("Pump 4 (F4)")
        axes[1, 1].set_xlabel("Time [s]")
        axes[1, 1].grid(True, linestyle="--", alpha=0.7)

        # Add a big column title for controls
        fig.text(0.13, 0.95, "Control Inputs", ha="center", va="center", fontsize=14, fontweight="bold")

        # --- Columns 2-3: Tank outflows ---
        for idx, (i, ax, color) in enumerate(zip([2,3,0,1],axes[:, 2:].flat, ['C1','C0', 'C0','C1'])):
            ax.plot(T, H[:, i], label=tank_labels[i], color=color)
            ax.set_title(tank_labels[i])
            ax.set_ylabel("Height [cm]")
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.legend()

        # Common x-labels for bottom row
        for ax in axes[1, 1:]:
            ax.set_xlabel("Time [s]")

        # Add a big title for tank outflows across cols 2–3
        fig.text(0.65, 0.95, "Tank Heights", ha="center", va="center", fontsize=14, fontweight="bold")

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()
    
    if 'Q' in plot_outputs:
        fig, axes = plt.subplots(2, 4, figsize=(14, 8), sharex=True)

        # --- Column 1: Control inputs ---
        axes[0, 0].step(U[:,0], U[:,1], where='post', color="C0")
        axes[0, 0].set_ylabel("Flow [cm³/s]")
        axes[0, 0].set_title("Pump 1 (F1)")
        axes[0, 0].grid(True, linestyle="--", alpha=0.7)

        axes[1, 0].step(U[:,0], U[:,2], where='post', color="C1")
        axes[1, 0].set_ylabel("Flow [cm³/s]")
        axes[1, 0].set_title("Pump 2 (F2)")
        axes[1, 0].set_xlabel("Time [s]")
        axes[1, 0].grid(True, linestyle="--", alpha=0.7)
        
        # --- Column 2: Disturbances
        axes[0, 1].step(D[:,0], D[:,1], where='post', color="C1")
        axes[0, 1].set_ylabel("Flow [cm³/s]")
        axes[0, 1].set_title("Pump 3 (F3)")
        axes[0, 1].grid(True, linestyle="--", alpha=0.7)

        axes[1, 1].step(D[:,0], D[:,2], where='post', color="C0")
        axes[1, 1].set_ylabel("Flow [cm³/s]")
        axes[1, 1].set_title("Pump 4 (F4)")
        axes[1, 1].set_xlabel("Time [s]")
        axes[1, 1].grid(True, linestyle="--", alpha=0.7)
        

        # Add a big column title for controls
        fig.text(0.13, 0.95, "Control Inputs", ha="center", va="center", fontsize=14, fontweight="bold")

        # --- Columns 2-3: Tank outflows ---
        for idx, (i, ax, color) in enumerate(zip([2,3,0,1],axes[:, 2:].flat, ['C1','C0', 'C0','C1'])):
            ax.plot(T, X[:, i], label=tank_labels[i], color=color)
            ax.set_title(tank_labels[i])
            ax.set_ylabel("Mass [g]")
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.legend()

        # Common x-labels for bottom row
        for ax in axes[1, 1:]:
            ax.set_xlabel("Time [s]")

        # Add a big title for tank outflows across cols 2–3
        fig.text(0.65, 0.95, "Tank Masses", ha="center", va="center", fontsize=14, fontweight="bold")

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()
    
def FourTankSystem(t, x, u, d, p):
    """
    FOURTANKSYSTEM Model dx/dt = f(t, x, u, d, p) for 4-tank system.

    Parameters
    ----------
    t : float
        Time (not explicitly used here, but included for ODE solver compatibility).
    x : array_like, shape (4,)
        States: mass of liquid in each tank [g].
    u : array_like, shape (2,)
        Inputs: flow rates in pumps [cm^3/s].
    d : array_like, shape (2,)
        Disturbances.
    p : array_like, shape (12,)
        Parameters:
            p[0:4]   = pipe cross-sectional areas [cm^2]
            p[4:8]   = tank cross-sectional areas [cm^2]
            p[8:10]  = valve positions [-]
            p[10]    = gravitational acceleration [cm/s^2]
            p[11]    = density of water [g/cm^3]

    Returns
    -------
    xdot : ndarray, shape (4,)
        Derivatives (time rates of change of masses).
    """

    # Ensure numpy arrays
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    u = np.asarray(u, dtype=float)
    d = np.asarray(d, dtype=float)

    # Unpack parameters
    a = p[0:4]     # Pipe cross-sectional areas [cm^2]
    A = p[4:8]     # Tank cross-sectional areas [cm^2]
    gamma = p[8:10] # Valve positions [-]
    g = p[10]      # Gravity [cm/s^2]
    rho = p[11]    # Density of water [g/cm^3]

    # Inflows
    F = np.zeros(4)
    F[:2] = u                     # Pumps
    F[2:] = d                     # Disturbances
    
    qin = np.zeros(4)
    qin[0] = gamma[0] * F[0]           # Pump 1 -> Tank 1
    qin[1] = gamma[1] * F[1]           # Pump 2 -> Tank 2
    qin[2] = (1 - gamma[1]) * F[1]     # Pump 2 -> Tank 3
    qin[3] = (1 - gamma[0]) * F[0]     # Pump 1 -> Tank 4

    # Outflows
    h = x / (rho * A)                  # Liquid levels [cm]
    qout = a * np.sqrt(2 * g * h)      # Outflows [cm^3/s]

    # Differential equations (mass balances)
    xdot = np.zeros(4)
    xdot[0] = rho * (qin[0] + qout[2] - qout[0])  # Tank 1
    xdot[1] = rho * (qin[1] + qout[3] - qout[1])  # Tank 2
    xdot[2] = rho * (qin[2] - qout[2]+ F[2])            # Tank 3
    xdot[3] = rho * (qin[3] - qout[3]+ F[3])            # Tank 4

    return xdot

def FourTankSystemSensor(X,p, nY=2):
    # Helper variables
    nT, nX = X.shape
    A = p[4:6] # For Tank 1 and Tank 2 only (lower tanks)
    rho = p[11]

    # Compute measured variables (liquid levels H)
    H = np.zeros((nT, nY))
    for i in range(nT):
        H[i, :] = X[i, :nY] / (rho * A) # only 1 and 2 tank are sensored
    
    y = H
    return y

def FourTankSystemOutput(X,p, nZ=2):
    # Helper variables
    nT, nX = X.shape
    A = p[4:6] # For Tank 1 and Tank 2 only (lower tanks)
    rho = p[11]
    
    # Compute measured variables (liquid levels H)
    H = np.zeros((nT, nZ))
    for i in range(nT):
        H[i, :] = X[i, :2] / (rho * A)
    
    y = H
    return y

class EulerMaruyama(ABC):
    @abstractmethod
    def ffun(self, t, x, u, d, p):
        """Drift function f(t, x, u, d, p)"""
        pass

    @abstractmethod
    def gfun(self, t, x, u, d, p):
        """Diffusion function g(t, x, u, d, p)"""
        pass

    def run(self, T, x, u, d, dt, dW, p):
        """
        Euler–Maruyama integration for dx = f dt + G dW.
        Assumes d is constant over time interval.
        """
        N = len(T) - 1
        nx = len(x)
        X = np.zeros((N + 1, nx))
        ds = np.zeros((N + 1, dW.shape[0]))  # disturbances over time

        X[0, :] = x
        ds[0, :] = d

        for k in range(N):
            dWk = dW[:, k]  # (n_noise,)
            f = self.ffun(T[k], X[k, :], u, d, p)
            G = self.gfun(T[k], X[k, :], u, d, p)
            X[k+1, :] = X[k, :] + f * dt + G @ dWk
            ds[k+1, :] = ds[k, :] + dWk

        return X, ds
     


class ModelSimulation(ABC):
    '''Main Model Structure'''
    def __init__(self, ts, x0, u0, d0, p):
        self.ts = ts
        self.x0 = x0
        self.u0 = u0
        self.d0 = d0
        self.p = p

    @abstractmethod
    def dynamics(self, t, x, u, d, p):
        """System dynamics function dx/dt = f(t, x, u, d, p)"""
        pass

    @abstractmethod
    def control(self, t, u):
        '''Update of manipulated variabls u'''
        pass
        
    @abstractmethod
    def disturbance(self, t, d, dmin=0, dmax=500):
        """Returns updated disturbance for time step"""
        pass
    

    @abstractmethod
    def step(self, t_span, x0, u, d):
        """Single integration step over interval [t_span]"""
        pass
    
    @abstractmethod
    def full_output(self, T, X, U, D):
        pass

    @abstractmethod
    def simulate(self):
        """Main simulation loop"""
        pass

    def SystemOutput(self, X):
        '''Variable z'''
        return FourTankSystemOutput(X, self.p)
    
    def SystemSensor(self, X):
        '''Sensor variable y'''
        return FourTankSystemSensor(X, self.p)