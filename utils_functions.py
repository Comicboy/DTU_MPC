import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

from abc import ABC, abstractmethod

from Modified_FourTank_parameters import p
from utils_DisturbanceModels import DisturbanceModel
from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, curve_fit
from scipy.linalg import expm
from scipy.linalg import eig
import scipy.linalg
import matplotlib.pyplot as plt
import control
import cvxpy as cp

def Modified_FourTankSystem(t, x, u, d, p):
    """
    FOURTANKSYSTEM Model dx/dt = f(t, x, u, p) for 4-tank system.

    This function implements a differential equation model
    for the 4-tank system.

    Parameters
    ----------
    t : float
        Time [s] (not used explicitly but kept for ODE solvers)
    x : array_like
        Mass of liquid in each tank [g], shape (4,)
    u : array_like
        Flow rates in pumps [cm^3/s], shape (4,)
    p : array_like
        Parameters vector containing:
            p[0:4]  -> Pipe cross sectional areas [cm^2]
            p[4:8]  -> Tank cross sectional areas [cm^2]
            p[8:10] -> Valve positions [-]
            p[10]   -> Acceleration of gravity [cm/s^2]
            p[11]   -> Density of water [g/cm^3]
    
    Returns
    -------
    xdot : ndarray
        Time derivative of mass in each tank [g/s], shape (4,)
    """

    # Unpack states, inputs, and parameters
    m = x
    F = np.zeros(4)
    F[0], F[1], F[2], F[3] = u[0], u[1], d[0], d[1]
    a = p[0:4]         # Pipe cross sectional areas [cm^2]
    A = p[4:8]         # Tank cross sectional areas [cm^2]
    gamma = p[8:10]    # Valve positions [-]
    g = p[10]          # Acceleration of gravity [cm/s^2]
    rho = p[11]        # Density of water [g/cm^3]

    # Inflows
    qin = np.zeros(4)
    qin[0] = gamma[0] * F[0]          # Valve 1 -> Tank 1
    qin[1] = gamma[1] * F[1]          # Valve 2 -> Tank 2
    qin[2] = (1 - gamma[1]) * F[1]    # Valve 2 -> Tank 3
    qin[3] = (1 - gamma[0]) * F[0]    # Valve 1 -> Tank 4

    # Outflows
    h = m / (rho * A)                 # Liquid level in each tank [cm]
    qout = a * np.sqrt(2 * g * h)     # Outflow from each tank [cm^3/s]

    # Differential equations (mass balances)
    xdot = np.zeros(4)
    xdot[0] = rho * (qin[0] + qout[2] - qout[0])         # Tank 1
    xdot[1] = rho * (qin[1] + qout[3] - qout[1])         # Tank 2
    xdot[2] = rho * (qin[2] + F[2] - qout[2])            # Tank 3
    xdot[3] = rho * (qin[3] + F[3] - qout[3])            # Tank 4

    return xdot

def find_equilibrium(f, x0_guess, u_op, d_op, p, tol=1e-9):
    """
    Find operating point x_op such that f(0, x_op, u_op, d_op, p) = 0.
    Uses scipy.optimize.fsolve (wrapped around your Modified_FourTankSystem).
    """
    def eq_fun(x):
        return f(0.0, x, u_op, d_op, p)

    x_op, info, ier, mesg = fsolve(eq_fun, x0_guess, full_output=True)
    if ier != 1:
        raise RuntimeError(f"Equilibrium search failed: {mesg}")
    return x_op

def steady_state(func, x, u, d, p):
    xs = find_equilibrium(func, x, u, d, p)
    #us = some function that computes u given setpoints
    ys = FourTankSystemSensor(xs,p)
    return xs[:,None], u[:,None], ys, d[:,None]

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

def PIDcontroller(r,y,us,Kc,Ki,Kd,tspan,integral_error,prev_error, umin=0, umax=None):
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

def FourTankSystemSensor(X, p, nY=2):
    """
    Sensor: returns liquid levels of the first nY tanks.

    Accepts:
      X shape (nx,), (nx,1), or (1,nx)

    Returns:
      Y shape (nY,1)
    """
    X = np.asarray(X, dtype=float)

    # Convert X to column vector (nx,1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim == 2:
        # If row vector (1,nx), convert to column (nx,1)
        if X.shape[0] == 1 and X.shape[1] > 1:
            X = X.T

    rho = p[11]
    A = p[4:4+nY].reshape(nY, 1)   # tank areas for measured tanks

    Y = X[:nY, :] / (rho * A)      # (nY,1)/(nY,1) -> (nY,1)
    return Y

def FourTankSystemOutput(X,p, nZ=2):
    if len(X.shape) == 1:
        X = X[None, :]  # Convert to 2D array with one row if needed
    
    # Helper variables
    rho = p[11]
    A = p[4:6]
    Z = X[:,:nZ] / (rho * A)
    return Z
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
    
def Modified_FourTankSystem_SDE(t, x, u, d, p, disturbances = Tuple[DisturbanceModel, DisturbanceModel]):
    """
    Combined model for modified four-tank system + stochastic disturbances.
    Returns f(x,u,d,p) and sigma(x,u,d,p) (To be used in Euler-Maruyama)
    
    State vector:
        x = [ m1, m2, m3, m4]
        d = [F3,F4]

    Disturbances:
        F3: DisturbanceModel to be defined
        F4: DisturbanceModel to be defined
    """

    # Unpack states
    m1, m2, m3, m4, = x

    # Unpack inputs (pump flows)
    F1, F2 = u       # deterministic pump inflows [cm^3/s]
    F3, F4 = d
    disturbanceF3, disturbanceF4 = disturbances

    # Parameters
    a = p[0:4]         # pipe areas
    A = p[4:8]         # tank areas
    gamma = p[8:10]    # valve positions
    g = p[10]          # gravity
    rho = p[11]        # density

    # Heights
    h1 = m1 / (rho * A[0])
    h2 = m2 / (rho * A[1])
    h3 = m3 / (rho * A[2])
    h4 = m4 / (rho * A[3])

    # Outflows
    q1 = a[0] * np.sqrt(2 * g * h1)
    q2 = a[1] * np.sqrt(2 * g * h2)
    q3 = a[2] * np.sqrt(2 * g * h3)
    q4 = a[3] * np.sqrt(2 * g * h4)

    # Inflows from pumps and disturbances
    qin1 = gamma[0] * F1
    qin2 = gamma[1] * F2
    qin3 = (1 - gamma[1]) * F2
    qin4 = (1 - gamma[0]) * F1

    # Disturbance inflows enter tank 3 and 4
    d3 = F3
    d4 = F4

    # --------------------------
    # DRIFT: f(x)
    # --------------------------
    f = np.zeros(6)

    # Tank mass balances
    f[0] = rho * (qin1 + q3 - q1)
    f[1] = rho * (qin2 + q4 - q2)
    f[2] = rho * (qin3 + d3 - q3)     # disturbance F3 entering the system in Tank 3
    f[3] = rho * (qin4 + d4 - q4)     # disturbance F4 entering the system in Tank 4

    # Disturbance SDEs
    f[4] = disturbanceF3.ffun(t,F3)                        # F3 drift
    f[5] = disturbanceF4.ffun(t,F4)                       # F4 drift

    # --------------------------
    # DIFFUSION: sigma(x)
    # (matrix with shape (6,2))
    # --------------------------
    sigma = np.zeros((6, 2))

    # Noise only enters F3 and F4
    sigma[4, 0] = disturbanceF3.gfun(t,F3)              # F3 diffusion
    sigma[5, 1] = disturbanceF4.gfun(t,F4)              # F4 diffusion

    return f, sigma

def qpsolver(H, g, l, u, A, bl, bu, xinit=None):

    "Implements the QP solver for problem 7"
    n = H.shape[0]
    x = cp.Variable(n)

    # Objective function
    objective = cp.Minimize(0.5 * cp.quad_form(x, H) + g.T @ x)

    # Constraints
    constraints = []
    if l is not None:
        constraints.append(x >= l)
    if u is not None:
        constraints.append(x <= u)
    if A is not None:
        if bl is not None:
            constraints.append(A @ x >= bl)
        if bu is not None:
            constraints.append(A @ x <= bu)

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    min_val = prob.solve()

    return x.value, min_val

def approx_derivative(fun, x, eps=1e-6, method='central'):
    """
    Compute Jacobian matrix of a vector-valued function fun(x).
    fun : R^n -> R^m
    Returns m×n Jacobian
    """
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(fun(x))
    m = f0.size
    n = x.size

    J = np.zeros((m, n))

    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps

        if method == 'central':
            fp = fun(x + dx)
            fm = fun(x - dx)
            J[:, i] = ((fp - fm) / (2*eps)).ravel()

        elif method == 'forward':
            fp = fun(x + dx)
            J[:, i] = ((fp - f0) / eps).ravel()

        elif method == 'backward':
            fm = fun(x - dx)
            J[:, i] = ((f0 - fm) / eps).ravel()

        else:
            raise ValueError("method must be 'central', 'forward', or 'backward'.")

    return J

def linearize_system(f, h, x_op, u_op, d_op, p, method='central'):
    """
    Linearize using scipy.optimize.approx_derivative.

    f signature: f(t, x, u, d, p) -> xdot (n,)
    h signature: h(x, p) -> y (ny,)

    Returns: A, B, Bd, C, D (continuous-time)
    """
    # wrappers to produce functions of a single vector argument
    fx = lambda x: f(0.0, x, u_op, d_op, p)
    fu = lambda u: f(0.0, x_op, u, d_op, p)
    fd = lambda d: f(0.0, x_op, u_op, d, p)
    hx = lambda x: h(x, p)

    A = approx_derivative(fx, x_op, method=method)
    B = approx_derivative(fu, u_op, method=method)
    Bd = approx_derivative(fd, d_op, method=method)

    C = approx_derivative(hx, x_op, method=method)
    # assume no direct feedthrough from u to y (modify if your h depends on u)
    D = np.zeros((C.shape[0], len(u_op)))

    return A, B, Bd, C, D

def continuous_tfs(A, B, C, D):
    """
    Build control.StateSpace and per-input-output transfer functions (MIMO TF returned by control.ss2tf)
    """
    sysc = control.ss(A, B, C, D)
    tf = control.ss2tf(sysc)
    return sysc, tf

def discretize_system(A, B, C, D, Ts, method='zoh'):
    """
    Discretize continuous-time state-space using control.c2d.
    Returns discrete system object and matrices Ad,Bd,Cd,Dd.
    """
    sysc = control.ss(A, B, C, D)
    sysd = control.c2d(sysc, Ts, method=method)
    return sysd, np.asarray(sysd.A), np.asarray(sysd.B), np.asarray(sysd.C), np.asarray(sysd.D)

def markov_parameters(Ad, Bd, Cd, Dd, N=20):
    """
    Compute discrete-time Markov parameters h[k], k=0..N-1:
      h[0] = D
      h[k] = C * A^(k-1) * B  for k>=1
    Returns H array shape (N, ny, nu).
    """
    ny, nu = Cd.shape[0], Bd.shape[1]
    H = np.zeros((N, ny, nu))
    H[0] = Dd.reshape(ny, nu)
    # compute powers iteratively for numerical stability
    A_pow = np.eye(Ad.shape[0])
    for k in range(1, N):
        A_pow = A_pow @ Ad  # A^k
        H[k] = Cd @ A_pow @ Bd
    return H

def f_jacobian(x, uk, dk, p):
    return approx_derivative(lambda x: Modified_FourTankSystem(0, x, uk, dk, p) ,x)

def g_jacobian(x, p):
    x = x.squeeze()
    return approx_derivative(lambda x: FourTankSystemSensor(x, p) ,x)

def idare(A, C, G, Rww, Rvv, Rwv, P0=None, tol=1e-9, max_iter=200):
    n = A.shape[0]
    P = np.eye(n) if P0 is None else P0.copy()

    for _ in range(max_iter):
        Re = C @ P @ C.T + Rvv
        K = (A @ P @ C.T + G @ Rwv) @ np.linalg.inv(Re)
        P_new = A @ P @ A.T + G @ Rww @ G.T - K @ Re @ K.T

        if np.linalg.norm(P_new - P) < tol:
            break
        P = P_new

    return P

def augmented_matrices(A, Bd, B, C, Cd = np.zeros((2,2))):
    nx, nd = Bd.shape
    
    A_aug = np.block([[A, Bd],[np.zeros((nd,nx)), np.identity(nd)]])
    B_aug = np.block([[B], [np.zeros((nd,nd))]])
    C_aug = np.block([C, Cd])
    
    return A_aug, B_aug, C_aug

#Disturbance Models
class DisturbanceModel(ABC):
    @abstractmethod
    def ffun(self, t, x, u, d, p):
        """Drift function f(t, x, u, d, p)"""
        pass

    @abstractmethod
    def gfun(self, t, x, u, d, p):
        """Diffusion function g(t, x, u, d, p)"""
        pass


class OU_DisturbanceModel(DisturbanceModel):
    '''
    Ornstein–Uhlenbeck (OU) process:
        dX_t = theta * (mu - X_t) dt + sigma dB_t

    Interpretation:
    - theta : Mean reversion rate. Higher = faster pull toward mu.
    - mu    : Long-term mean level that the process reverts to.
    - sigma : Noise intensity (diffusion). Controls smoothness/variance.
    '''
    
    model_type = "OU"
    
    def __init__(self, theta=0.1, mu=100.0, sigma=1.0):
        self.theta = theta      # mean reversion speed
        self.mu = mu            # long-term equilibrium level
        self.sigma = sigma      # volatility / noise strength

    def ffun(self, t, d):
        """OU drift: pulls X_t toward mu."""
        return self.theta * (self.mu - d)

    def gfun(self, t, d):
        """OU diffusion: constant noise strength."""
        return self.sigma



class BrownianMotion(DisturbanceModel):
    '''
    Standard Brownian Motion (Wiener process):
        dX_t = sigma dB_t

    Interpretation:
    - sigma : Diffusivity (noise scale). Larger sigma = rougher paths.
    - No drift term → process is a random walk around initial condition.
    '''
    
    model_type = "BM"
    
    def __init__(self, sigma=1.0, alpha = 100):
        self.sigma = sigma      # diffusion intensity
        self.alpha = alpha

    def ffun(self, t, d):
        """Zero drift: pure diffusion."""
        return 0

    def gfun(self, t, d):
        """Constant diffusion."""
        return self.sigma



class GeometricBrownianMotion(DisturbanceModel):
    ''' 
    Geometric Brownian Motion (GBM):
        dX_t = mu * X_t dt + sigma * X_t dB_t

    Interpretation:
    - mu    : Growth rate (drift). Positive = exponential growth.
    - sigma : Volatility multiplier. Noise scales with X_t → multiplicative.
    - X_t   : Always stays positive if initialized positive.
    '''
    
    model_type = "GBM"
    
    def __init__(self, mu=0.05, sigma=0.1):
        self.mu = mu            # exponential drift rate
        self.sigma = sigma      # multiplicative noise intensity

    def ffun(self, t, d):
        """GBM drift: proportional to the current value."""
        return self.mu * d

    def gfun(self, t, d):
        """GBM diffusion: multiplicative noise (sigma * X_t)."""
        return self.sigma * d   
class CoxIngersollRoss(DisturbanceModel):
    ''' 
    Cox–Ingersoll–Ross (CIR) process:
        dX_t = lambd * (xi - X_t) dt + gamma * sqrt(X_t) dB_t

    Interpretation:
    - lambd : Mean reversion speed (pulls X_t toward xi).
    - xi    : Long-term mean level (equilibrium).
    - gamma : Volatility coefficient; noise increases with sqrt(X_t).
    - Always stays non-negative.
    '''
    
    model_type = "CIR"
    
    def __init__(self, lambd=0.5, xi=100.0, gamma=1.0):
        self.lambd = lambd      # reversion speed
        self.xi = xi            # long-term mean level
        self.gamma = gamma      # diffusion scaling factor

    def ffun(self, t, d):
        """CIR drift: mean-reverting toward xi."""
        return self.lambd * (self.xi - d)

    def gfun(self, t, d):
        """CIR diffusion: gamma * sqrt(X_t), ensures non-negativity."""
        sqrt_d = np.sqrt(np.maximum(d, 0.0))
        return self.gamma * sqrt_d
    
# Kalman Filters 

class CDEKF:
    """
    Continuous–Discrete Extended Kalman Filter
    """

    def __init__(self, p, f, g, jac_f, jac_g, sigma, Rv, x0, P0):
        """
        p      : parameters
        f      : drift f(x,u,d,theta)
        g      : measurement function g(x,theta)
        jac_f  : Jacobian df/dx
        jac_g  : Jacobian dg/dx
        sigma  : diffusion matrix sigma(x,u,d,theta)
        Rv     : measurement noise covariance
        """
        self.p = p
        
        self.f = lambda x,u,d: f(0,x,u,d,self.p) 
        self.g = lambda x: g(x,self.p)
        self.jac_f = lambda x, u, d: jac_f(x,u,d, self.p)
        self.jac_g = lambda x: jac_g(x, self.p)
        self.sigma = sigma
        self.Rv = Rv
        

        self.x = np.random.multivariate_normal(mean = x0.ravel(), cov = P0)
        self.x = np.abs(self.x.reshape(-1,1))
        self.P = P0.copy()
        
        self.ek = []
        self.Re_k = []

    # -------------------------------------------------
    # Measurement update discrete at t_k
    # -------------------------------------------------
    def measurement_update(self, yk):
        # predicted output
        yhat = self.g(self.x)
        
        # sensor Jacobian
        Ck = self.jac_g(self.x)

        # innovation
        e = (yk - yhat)
        
        # innovation covariance
        Re = Ck @ self.P @ Ck.T + self.Rv
        
        self.ek.append(e)
        self.Re_k.append(Re)

        # Kalman gain
        Kk = self.P @ Ck.T @ np.linalg.inv(Re)

        # state update
        self.x = self.x + Kk @ e
        # covariance update
        I = np.eye(self.P.shape[0])
        self.P = (I - Kk @ Ck) @ self.P @ (I - Kk @ Ck).T + Kk @ self.Rv @ Kk.T

        return self.x, self.P # x[k|k], P[k|k]

    # -------------------------------------------------
    # Time update continuous between t_k and t_{k+1}
    # -------------------------------------------------
    def time_update(self, u, d, dt):
        """
        Integrates:
            xdot = f(x,u,d)
            Pdot = A P + P A^T + sigma sigma^T
        using forward Euler
        """
        u = u
        d = d

        # drift
        fx = self.f(self.x.squeeze(), u, d)
        fx = fx.reshape(-1,1)

        # linearization
        A = self.jac_f(self.x.squeeze(), u, d)

        # diffusion
        Sig = self.sigma
        # Sig = self.sigma(self.x, u, d)

        # state propagation (forward Euler)
        self.x = self.x + fx * dt

        # covariance propagation
        self.P = self.P + (
            A @ self.P +
            self.P @ A.T +
            Sig @ Sig.T
        ) * dt

        return self.x, self.P # x[k+1|k], P[k+1|k]


class StaticKalmanFilter:
    def __init__(self, A, B, C, G, Rww, Rvv, Rwv, P0, x0):

        self.A = A
        self.B = B 
        self.C = C
        self.G = G # process noise matrix

        self.Q= Rww
        self.R = Rvv
        self.S = Rwv
        self.ST = Rwv.T

        # State estimate and covariance
        self.x = abs(np.random.multivariate_normal(mean=x0.flatten(), cov = P0).reshape(-1,1))
        self.P = P0
        
        w_mean = np.zeros(A.shape[0])
        self.w = np.random.multivariate_normal(mean=w_mean, cov = Rww).reshape(-1,1)

        #compute stationary filter by solving discrete algebraic riccati equation iteratively
        P = idare(A,C,G, self.Q,self.R, self.S, P0=self.P)

        #filter gains
        Re = C@P@C.T+self.R
        Kx = P@C.T@np.linalg.inv(Re)
        Kw = (self.S) @ np.linalg.inv(Re)
        self.P = P-Kx@Re@Kx.T
        
        self.Re, self.Kx, self.Kw = Re, Kx, Kw

    def update(self, y, u):
        """
        Kalman filter update step.
        Computes: innovation, and filters x.
        Update step (to): xhat[k|k], what[k|k]
        """
        u = u.reshape(-1, 1)
        
        x_prio = self.A @ self.x + self.B @ u + self.G @ self.w
        y_prio = self.C @ x_prio
        e = y - y_prio
        
        self.x = x_prio + self.Kx @ e
        self.w = self.Kw @ e
        self.y = self.C @ self.x

    
    def j_step(self, u, N):
        ''' 
        j >= 2
        u: vector containing us upto uhat[k+j-1|k]
        One could set up the matrices for it, chose just to do it recursively.
        '''
        A, B, C, G = self.A, self.B, self.C, self.G
        
        assert u.shape == (N*2,1), f"u is of wrong shape, should be {(N*2,1)} but is {u.shape}"
        
        nx = A.shape[0]
        nu = B.shape[1]
        ny = C.shape[0]
        
        # Compute Markov parameters: H_i = C A^{i-1} B
        H = [C @ np.linalg.matrix_power(A, i) @ B for i in range(N)]
        
        # Build lifted Phix and Phiw matrices
        Phi_x = np.vstack([C @ np.linalg.matrix_power(A, i + 1) for i in range(N)])
        Phi_w = np.vstack([C @ np.linalg.matrix_power(A, i) @ G for i in range(N)])
        
        # Compute Gamma matrix (block Toeplitz from H_i)
        Gamma = np.zeros((N * ny, N * nu))
        for i in range(N):
            for j in range(i + 1):
                Gamma[i*ny:(i+1)*ny, j*nu:(j+1)*nu] = H[i - j]
        
        # # Stack inputs
        Uk = u

        # Compute lifted output prediction
        bk = Phi_x @ self.x + Phi_w @ self.w
        
        Zk = bk + Gamma @ Uk
        
        return Zk
        
class DynamicKalmanFilter:
    def __init__(self, A, B, C, G, Rww, Rvv, Rwv, P0, x0):
        """
        Dynamic Kalman filter
        """

        self.A = A
        self.B = B
        self.C = C
        self.G = G

        self.Rww = Rww
        self.Rvv = Rvv
        self.Rwv = Rwv
        self.Rvw = Rwv.T

        # State estimate and covariance
        self.x = abs(np.random.multivariate_normal(mean=x0.flatten(), cov = P0).reshape(-1,1))
        self.P = P0.copy()

        # Noise estimate and covariance
        self.w = np.zeros((A.shape[0], 1))
        self.Q = Rww.copy()
        
        # Innovation covariance
        Re = self.C @ self.P @ self.C.T + self.Rvv

        # Gains
        Kx = self.P @ self.C.T @ np.linalg.inv(Re) #filter gain in x
        Kw = self.Rwv @ np.linalg.inv(Re) #filter gain in noise w
        
        self.Re = Re
        self.Kx = Kx
        self.Kw = Kw
        
        self.ek = []
        self.Re_k = []

    def update(self, y, u):
        """
        Kalman filter update step.
        Computes: innovation, and filters x.
        Update step (to): xhat[k|k], what[k|k]
        """
        u = u.reshape(-1, 1)
        
        x_prio = self.A @ self.x + self.B @ u + self.G @ self.w # x[k|k-1]
        y_prio = self.C @ x_prio # y[k|k-1]
        P_prio = self.A @ self.P @ self.A.T + self.Q- self.A @ self.Kx @ self.Rwv.T - self.Rwv @ self.Kx.T @ self.A.T # P[k|k-1]
        Q_prio = self.Q - self.Kw @ self.Rvv @ self.Kw.T # Q[k|k-1]
        
        e = y - y_prio # one-step prediction error
        self.Re = self.C @ P_prio @ self.C.T + self.Rvv
        self.Kx = P_prio @ self.C.T @ np.linalg.inv(self.Re)
        self.Kw = self.Rwv @ np.linalg.inv(self.Re)
        
        self.x = x_prio + self.Kx @ e # x[k|k]
        self.w = self.Kw @ e # w[k|k]
        self.y = self.C @ self.x # y[k|k]
        
        self.P = P_prio - self.Kx @ self.Re @ self.Kx.T # P[k|k]
        self.Q = Q_prio - self.Kw @ self.Re @ self.Kw.T # Q[k|k]
        
        self.ek.append(e)
        self.Re_k.append(self.Re)
        
    
    def j_step(self, u, N):
        ''' 
        j >= 2
        u: vector containing us upto uhat[k+j-1|k]
        One could set up the matrices for it, chose just to do it recursively.
        '''
        A, B, C, G = self.A, self.B, self.C, self.G
        
        assert u.shape == (N*2,1), f"u is of wrong shape, should be {(N*2,1)} but is {u.shape}"
        
        nx = A.shape[0]
        nu = B.shape[1]
        ny = C.shape[0]
        
        # Compute Markov parameters: H_i = C A^{i-1} B
        H = [C @ np.linalg.matrix_power(A, i) @ B for i in range(N)]
        
        # Build lifted Phix and Phiw matrices
        Phi_x = np.vstack([C @ np.linalg.matrix_power(A, i + 1) for i in range(N)])
        Phi_w = np.vstack([C @ np.linalg.matrix_power(A, i) @ G for i in range(N)])
        
        # Compute Gamma matrix (block Toeplitz from H_i)
        Gamma = np.zeros((N * ny, N * nu))
        for i in range(N):
            for j in range(i + 1):
                Gamma[i*ny:(i+1)*ny, j*nu:(j+1)*nu] = H[i - j]
        
        # # Stack inputs
        Uk = u
        # xhat = xhat.reshape(n, 1)
        # what = what.reshape(-1, 1)

        # Compute lifted output prediction
        bk = Phi_x @ self.x + Phi_w @ self.w
        
        Zk = bk + Gamma @ Uk
        
        return Zk 
    
