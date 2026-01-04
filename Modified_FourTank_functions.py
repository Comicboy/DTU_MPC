import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, curve_fit
from scipy.optimize._numdiff import approx_derivative
from scipy.linalg import expm
from scipy.linalg import eig
import scipy.linalg
import matplotlib.pyplot as plt
import control
import cvxpy as cp

# Functions
# Translated to python from matlab from slides with ChatGPT
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


# Deterministic sensor and output functions (no noise)
def FourTankSystemSensor_Deterministic(x, p):
    # Placeholder sensor function (define your own)
    # Example: return liquid levels for all tanks (only 2 will be used though)
    rho = p[11]
    A = p[4:8]
    return x / (rho * A)

# Stochastic sensor function (with noise)
def FourTankSystemSensor(x, p, sigma = np.array([0,0,0,0]), seed=None):
    rng = np.random.default_rng(seed)  # local RNG
    # Placeholder sensor function (define your own)
    # Example: return liquid levels for all tanks (only 2 will be used though)

    g = FourTankSystemSensor_Deterministic(x, p)  # deterministic measurement

    R = np.diag(sigma**2)           # covariance
    v = rng.multivariate_normal(mean=np.zeros(len(g)), cov=R)
    return g + v

# Deterministic output function (no noise)
def FourTankSystemOutput(x, p):
    # Placeholder output function (define your own)
    # Example: return mass directly
    rho = p[11]
    A = p[4:8]
    return x / (rho * A)

def run_step(x0, t_span, u_k, d, p):
    # Simulate the system over the time span with constant input u_k and initial state x0
    sol = solve_ivp(
        fun=lambda t, x: Modified_FourTankSystem(t, x, u_k, d, p),
        t_span=t_span,
        y0=x0.flatten(),
        method='BDF',
        dense_output=False
    )

    X = sol.y.T

    # Compute heights
    a = p[0:4]
    A = p[4:8]
    g = p[10]
    rho = p[11]
    
    H = X / (rho * A)
    Qout = a * np.sqrt(2 * g * H)

    return sol.t, X, H, Qout

# Simulation for step functions with model 2.2
def sim22(times, x0, u, d, p, dt=10, noise_level=0, plot=True):
    """
    Simulation for step functions with model 2.2

    Parameters
    ----------
    times : array
        Array of start and stop for each time interval. len(times) is number of simulations
    x0 : array
        Initial state vector
    u : np.array([F1,F2]) with F1 and F2 for each interval in times
    d : np.array([F3,F4]) with F3 and F4 for each interval in times
    p : Parameters
    dt : float
        Integration substep size
    noise_level : float
        Std of noise (default 0 meaning deterministic case)
    plot : bool
        Whether to plot results
    """
    big_time = np.arange(times[0], times[-1] + dt, dt)
    N = len(big_time)
    nx = len(x0)

    # Storage
    X_all = np.zeros((0, nx))
    H_all = np.zeros((0, nx))
    T_all = np.zeros((0, 1))
    x = np.zeros((nx, N))
    y = np.zeros((nx, N))
    z = np.zeros((nx, N))
    d_all = np.zeros((2, N))
    # Initial condition
    x[:, 0] = x0

    for k in range(N-1):
        # Sensor and output functions
        y[:,k] = FourTankSystemSensor(x[:,k],p) #Height measurements for now
        z[:,k] = FourTankSystemOutput(x[:,k],p) #Height measurements
        
        x0_new = x[:,k]
        # Find interval in times k belongs to
        interval_idx = np.searchsorted(times, big_time[k], side='left') - 1
        interval_idx = np.clip(interval_idx, 0, len(times)-2)
        u_step = u[:, interval_idx]
        d_all[:,k] = np.clip(d[:, interval_idx] + np.random.normal(0, noise_level, 2), 0, None) # Adding disturbance. clips to be >=0
        # Integrate from t[k] to t[k+1]
        
        t_span=(big_time[k], big_time[k+1])
        sol_time, sol_X, sol_H, Qout = run_step(x0_new, t_span, u_step, d_all[:,k], p)
        # Take last state for next step
        x[:, k+1] = sol_X.T[:, -1]

        # Store results
        T_all = np.vstack([T_all, sol_time.reshape(-1, 1)])
        X_all = np.vstack([X_all, sol_X])
        H_all = np.vstack([H_all, sol_H])

        """
        sol = solve_ivp(
            fun=lambda t_local, x_local: FourTankSystem(t_local, x_local, u[:, k], p),
            t_span=(t[k], t[k+1]),
            y0=x[:, k],
            method='BDF'
        )

        # Take last state for next step
        x[:, k+1] = sol.y[:, -1]

        # Store results
        T_all = np.vstack([T_all, sol.t.reshape(-1, 1)])
        X_all = np.vstack([X_all, sol.y.T])
        """
    # Final sensor and output computation
    y[:, -1] = FourTankSystemSensor(x[:, -1], p)
    z[:, -1] = FourTankSystemOutput(x[:,k],p)
    d_all[:,-1] = d_all[:,-2]
    if plot == True:
        # --- Create plots ---
        fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=False)

        # --- Plot Results: Tank Levels ---
        axs[0].plot(T_all, H_all)
        axs[0].set_xlabel('Time [s]')
        axs[0].set_ylabel('Liquid Level [cm]')
        axs[0].set_title('Four Tank System Simulation')
        axs[0].legend(['Tank 1', 'Tank 2', 'Tank 3', 'Tank 4'], loc='upper right')
        axs[0].grid(True)

        # --- Plot Results: Inputs ---
        for i, label in enumerate(['F1', 'F2']):
            # Expand u into a step function over big_time
            u_expanded = np.zeros(len(big_time))
            for k in range(len(big_time)):
                interval_idx = np.searchsorted(times, big_time[k], side='right') - 1
                interval_idx = np.clip(interval_idx, 0, len(times)-2)
                u_expanded[k] = u[i, interval_idx]

            axs[1].step(big_time, u_expanded, where='post', label=label)

        for i, label in enumerate(['F3', 'F4']):
            axs[1].step(big_time, d_all[i, :], where='post', label=label)

        axs[1].set_xlabel('Time [s]')
        axs[1].set_ylabel('Input Flow [cm³/s]')
        axs[1].set_title('Input Flows to the Four Tank System')
        axs[1].legend(loc='upper right')
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()
    return x, y, z, T_all, X_all, H_all


# Generate brownian motion for model 2.3
def generate_brownian_noise(t_len, dt, sigma):
    """
    Generate noise for the last two states of the system.

    Parameters:
    - t_len: total number of time steps
    - dt: time step size
    - sigma: array of standard deviations for each noisy state

    Returns:
    - noise_array: array of shape (t_len, 2) with generated noise
    """
    domega = np.random.normal(0, np.sqrt(dt), 2 * t_len)
    noise_array = np.zeros((t_len, 2))
    for i in range(t_len):
        noise_array[i] = sigma * np.array([domega[i], domega[-i]])
    return noise_array

# Simulation for step functions with model 2.3
def sim23(times, dt, x0, u, d, p, noise_level=0, plot=True):
    A = p[4:8]
    rho = p[11]
    # noise_level default 0
    tf = times[-1]
    t = np.arange(0, tf + dt, dt)
    sigma = np.array([noise_level, noise_level])
    noise = generate_brownian_noise(len(t), dt, sigma)

    # --- Simulation Setup ---
    x = np.zeros((4, len(t)))
    #d_all = np.zeros([2,len(t)])
    x[:, 0] = x0

    # --- Simulation Loop ---
    for i in range(len(t) - 1):
        # Determine current interval
        current_time = t[i]
        interval_index = np.searchsorted(times[1:], current_time, side='right')

        # Get current inputs
        u_now = u[:, interval_index]
        d_now = d[:, interval_index]

        # Compute dynamics and apply noise to last two states (makes sure no negative heights with clip)
        dx = Modified_FourTankSystem(current_time, x[:, i], u_now, d_now, p) * dt
        x[:2, i + 1] = np.clip(x[:2, i] + dx[:2], 0, None)
        x[2:, i + 1] = np.clip(x[2:, i] + dx[2:] + 1*noise[i], 0, None)

    if plot == True:
        fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        # Subplot 1: Liquid levels (converted from states)
        for i in range(4):
            h = x[i, :] / (rho * A[i])
            axs[0].plot(t, h, label=f'Tank {i+1}')
        axs[0].set_ylabel('Liquid Level [cm]')
        axs[0].set_title('Four Tank System Simulation')
        axs[0].legend()
        axs[0].grid(True)

        # Subplot 2: Control inputs
        # --- Plot Results: Inputs ---
        # Create input vectors aligned with simulation time
        # Create input vectors aligned with simulation time
        uplot = np.zeros((2, len(t)))
        dplot = np.zeros((2, len(t)))

        for i in range(len(t)):
            interval_index = min(np.searchsorted(times[1:], t[i], side='right'), u.shape[1] - 1)
            uplot[:, i] = u[:, interval_index]
            dplot[:, i] = d[:, interval_index]

        # Plot control inputs
        for i, label in enumerate(['F1', 'F2']):
            axs[1].plot(t, uplot[i], label=label)

        for i, label in enumerate(['F3', 'F4']):
            axs[1].plot(t, dplot[i], label=label)

        axs[1].set_xlabel('Time [s]')
        axs[1].set_ylabel('Input Flow Rates')
        axs[1].legend()
        axs[1].grid(True)

    return x


# Compute steady-state pump flows for desired heights to put into controller
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


def PIDcontroller(r,y,us,Kc,Ki,Kd,tspan,integral_error,prev_error):
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
    u_clipped = np.clip(u, 0, None)

    # Simple anti-windup: if saturated, don't integrate further in that direction
    for i in range(len(u)):
        if (u_clipped[i] != u[i]):  # saturation happened
            integral_error_new[i] = integral_error[i]  # freeze integrator

    return u_clipped, integral_error_new, derivative_error


# Closed-loop simulation for model 2.2
def closed_loop_sim22(t, x0, u0, d, p, us, r, controller, noise_level=0, Kc=5,Ki=0.1,Kd=0.05, plot=True):
    """
    Closed-loop simulation for the four-tank system using provided controller.

    Args:
        t: array-like of time points (len N)
        x0: initial state vector length 4
        u0: initial control vector length 2 (u1,u2)
        d: disturbance array shape (2, N) or (2, len(t))
        p: parameters passed to fun.* calls
        us: steady-state inputs (length 2) used by controllers
        r: setpoint / reference (length >=2) - used by controllers
        controller: function object (Pcontroller, PIcontroller, or PIDcontroller)
        noise_level: std of noise for F3 and F4

    Returns:
        T_all, X_all, H_all, u, y, z
    """
    N = len(t)
    nx = len(x0)

    # Pre-allocate
    x = np.zeros((4, N))
    y = np.zeros((4, N))
    z = np.zeros((4, N))
    u = np.zeros((2, N))
    X_all = np.zeros((0, nx))    # will stack solver states (rows)
    H_all = np.zeros((0, nx))    # will stack corresponding H (rows)
    T_all = np.zeros((0, 1))     # will stack times (rows, col vector)

    # initialize
    x[:, 0] = x0
    u[:, 0] = u0

    # controller internal states
    integral_error = np.zeros(2)
    prev_error = np.zeros(2)

    # push initial state/time into storage (so plots start at t[0])
    T_all = np.vstack([T_all, np.array([[t[0]]])])
    X_all = np.vstack([X_all, x[:, 0].reshape(1, -1)])
    # if fun.run_step returns H, we don't have it for initial instant; fill with nan row for now
    H_all = np.vstack([H_all, np.full((1, nx), np.nan)])

    for k in range(N - 1):
        # Sensor and output at current state
        y[:, k] = FourTankSystemSensor(x[:, k], p)
        z[:, k] = FourTankSystemOutput(x[:, k], p)

        # compute controller output for time index k
        ctrl_name = getattr(controller, "__name__", "").lower()

        if ctrl_name == "pcontroller" or ctrl_name == "Pcontroller":
            # signature: Pcontroller(r, y, us, Kc)
            u_vals = controller(r, y[:, k], us, Kc=Kc)
            u[:, k] = np.asarray(u_vals).reshape(2,)

        elif ctrl_name == "picontroller" or ctrl_name == "PIcontroller":
            # signature: PIcontroller(r, y, us, Kc, Ki, tspan, integral_error)
            # pass current time span [t_k, t_{k+1}]
            u_vals, integral_error = controller(
                r=r,
                y=y[:, k],
                us=us,
                Kc=Kc,
                Ki=Ki,
                tspan=(t[k], t[k+1]),
                integral_error=integral_error
            )
            u[:, k] = np.asarray(u_vals).reshape(2,)

        elif ctrl_name == "pidcontroller" or ctrl_name == "PIDcontroller":
            # signature: PIDcontroller(r, y, us, Kc, Ki, Kd, tspan, integral_error, prev_error)
            u_vals, integral_error, prev_error = controller(
                r=r,
                y=y[:, k],
                us=us,
                Kc=Kc,
                Ki=Ki,
                Kd=Kd,
                tspan=(t[k], t[k+1]),
                integral_error=integral_error,
                prev_error=prev_error
            )
            u[:, k] = np.asarray(u_vals).reshape(2,)

        else:
            # assume controller is a callable that matches Pcontroller signature
            try:
                u_vals = controller(r, y[:, k], us)
                u[:, k] = np.asarray(u_vals).reshape(2,)
            except Exception:
                raise ValueError("Unknown controller signature/name. Pass P/PI/PID or a compatible callable.")

        # Add random disturbance increment to d at this time
        noise = np.random.normal(0, noise_level, 2)
        d[:, k] = d[:, k] + noise
        d_step = d[:, k]

        # Integrate one step using fun.run_step
        x0_new = x[:, k]
        u_step = u[:, k]
        t_span = (t[k], t[k+1])

        sol_time, sol_X, sol_H, Qout = run_step(x0_new, t_span, u_step, d_step, p)

        # Ensure shapes: sol_time 1D array, sol_X shape (M, nx) or (nx, M)
        sol_time = np.asarray(sol_time).reshape(-1,)
        sol_X = np.asarray(sol_X)
        # If sol_X is shaped (nx, M) transpose to (M, nx)
        if sol_X.ndim == 2 and sol_X.shape[0] == nx and sol_X.shape[1] != nx:
            sol_X = sol_X.T

        # similarly sol_H
        sol_H = np.asarray(sol_H)
        if sol_H.ndim == 2 and sol_H.shape[0] == nx and sol_H.shape[1] != nx:
            sol_H = sol_H.T

        # Take last state for next step
        x[:, k+1] = sol_X[-1, :].reshape(nx,)

        # Store solver outputs; avoid duplicating start time if sol_time[0] == t[k]
        # Append all rows of sol_time and states; if first time equals previously stored time, skip first row
        if T_all.size > 0 and np.isclose(sol_time[0], float(T_all[-1, 0])):
            start_idx = 1
        else:
            start_idx = 0

        if start_idx < len(sol_time):
            T_all = np.vstack([T_all, sol_time[start_idx:].reshape(-1, 1)])
            X_all = np.vstack([X_all, sol_X[start_idx:, :]])
            # sol_H may be empty or NaN - handle gracefully
            if sol_H.size == sol_X.shape[0] * nx:
                H_all = np.vstack([H_all, sol_H[start_idx:, :]])
            else:
                # if no sol_H available, append NaNs for the matching rows
                H_all = np.vstack([H_all, np.full((sol_X.shape[0] - start_idx, nx), np.nan)])

        # Optionally compute/update u[:, k+1] with feedforward / last known value
        # Here we simply propagate the last control forward (zero-order hold)
        u[:, k+1] = u[:, k]

    # final sensor/output at the last time index
    y[:, -1] = FourTankSystemSensor(x[:, -1], p)
    z[:, -1] = FourTankSystemOutput(x[:, -1], p)

    if plot == True:
        # Plot levels (H_all) and control inputs (u) vs time in subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        # Subplot 1: Liquid levels or states
        if H_all.shape[0] > 0:
            if H_all.shape[1] >= 4:
                axs[0].plot(T_all.flatten(), H_all[:, 0], label='Tank 1')
                axs[0].plot(T_all.flatten(), H_all[:, 1], label='Tank 2')
                axs[0].plot(T_all.flatten(), H_all[:, 2], label='Tank 3')
                axs[0].plot(T_all.flatten(), H_all[:, 3], label='Tank 4')
            else:
                axs[0].plot(T_all.flatten(), X_all[:, 0], label='State 1')
                for i in range(1, min(4, X_all.shape[1])):
                    axs[0].plot(T_all.flatten(), X_all[:, i], label=f'State {i+1}')
        else:
            axs[0].plot(t, x.T)
        axs[0].set_ylabel('Liquid Level [cm] or state')
        axs[0].set_title('Four Tank System Simulation')
        axs[0].legend()
        axs[0].grid()

        # Subplot 2: Control inputs
        axs[1].plot(t, u[0, :], label='u1')
        axs[1].plot(t, u[1, :], label='u2')
        axs[1].set_xlabel('Time [s]')
        axs[1].set_ylabel('Control Input')
        axs[1].set_title('Control Inputs Over Time')
        axs[1].legend()
        axs[1].grid()

        plt.tight_layout()
        plt.show()
    return T_all, X_all, H_all, u, y, z

# Closed-loop simulation for model 2.3
def closed_loop_sim23(t, x0, u0, d, p, us, r, controller, noise_level=0, Kc=10,Ki=1,Kd=100, plot=True):
    A = p[4:8]
    rho = p[11]

    N = len(t)
    dt = t[1]-t[0]
    nx = len(x0)
    sigma = np.array([noise_level, noise_level])
    noise = generate_brownian_noise(len(t), dt, sigma)
    
    x = np.zeros((nx, N))      # State history
    y = np.zeros((nx, N))      # Sensor history (adjust shape if needed)
    z = np.zeros((nx, N))      # Output history (adjust shape if needed)
    u = np.zeros((2, N))
    # initialize
    x[:, 0] = x0
    u[:, 0] = u0

    # controller internal states
    integral_error = np.zeros(2)
    prev_error = np.zeros(2)
    # --- Simulation Loop ---
    for k in range(N - 1):
        # Sensor and output at current state
        y[:, k] = FourTankSystemSensor(x[:, k], p)
        z[:, k] = FourTankSystemOutput(x[:, k], p)

        # compute controller output for time index k
        ctrl_name = getattr(controller, "__name__", "").lower()

        if ctrl_name == "pcontroller" or ctrl_name == "Pcontroller":
            # signature: Pcontroller(r, y, us, Kc)
            u_vals = controller(r, y[:, k], us, Kc=Kc)
            u[:, k] = np.asarray(u_vals).reshape(2,)

        elif ctrl_name == "picontroller" or ctrl_name == "PIcontroller":
            # signature: PIcontroller(r, y, us, Kc, Ki, tspan, integral_error)
            # pass current time span [t_k, t_{k+1}]
            u_vals, integral_error = controller(
                r=r,
                y=y[:, k],
                us=us,
                Kc=Kc,
                Ki=Ki,
                tspan=(t[k], t[k+1]),
                integral_error=integral_error
            )
            u[:, k] = np.asarray(u_vals).reshape(2,)

        elif ctrl_name == "pidcontroller" or ctrl_name == "PIDcontroller":
            # signature: PIDcontroller(r, y, us, Kc, Ki, Kd, tspan, integral_error, prev_error)
            u_vals, integral_error, prev_error = controller(
                r=r,
                y=y[:, k],
                us=us,
                Kc=Kc,
                Ki=Ki,
                Kd=Kd,
                tspan=(t[k], t[k+1]),
                integral_error=integral_error,
                prev_error=prev_error
            )
            u[:, k] = np.asarray(u_vals).reshape(2,)

        else:
            # assume controller is a callable that matches Pcontroller signature
            try:
                u_vals = controller(r, y[:, k], us)
                u[:, k] = np.asarray(u_vals).reshape(2,)
            except Exception:
                raise ValueError("Unknown controller signature/name. Pass P/PI/PID or a compatible callable.")


        # Get current inputs
        u_now = u[:, k]
        d_now = d[:, k]

        # Compute dynamics and apply noise to last two states (makes sure no negative heights with clip)
        dx = Modified_FourTankSystem(t[k], x[:, k], u_now, d_now, p) * dt
        x[:2, k + 1] = np.clip(x[:2, k] + dx[:2], 0, None)
        x[2:, k + 1] = np.clip(x[2:, k] + dx[2:] + noise[k], 0, None)

    # Compute final sensor readings and controller output for completeness
    y[:, -1] = FourTankSystemSensor(x[:, -1], p)
    z[:, -1] = FourTankSystemOutput(x[:, -1], p)

    # Compute the final control value
    try:
        if "pcontroller" in ctrl_name:
            u[:, -1] = controller(r, y[:, -1], us, Kc=Kc)
        elif "picontroller" in ctrl_name:
            u[:, -1], _ = controller(r, y[:, -1], us, Kc=Kc, Ki=Ki, tspan=(t[-2], t[-1]), integral_error=integral_error)
        elif "pidcontroller" in ctrl_name:
            u[:, -1], _, _ = controller(r, y[:, -1], us, Kc=Kc, Ki=Ki, Kd=Kd, tspan=(t[-2], t[-1]),
                                        integral_error=integral_error, prev_error=prev_error)
        else:
            u[:, -1] = u[:, -2]
    except Exception:
        u[:, -1] = u[:, -2]
    
    if plot == True:
        # Plot tank levels and control inputs in subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        # Subplot 1: Liquid levels (converted from states)
        for i in range(4):
            h = x[i, :] / (rho * A[i])
            axs[0].plot(t, h, label=f'Tank {i+1}')
        axs[0].set_ylabel('Liquid Level [cm]')
        axs[0].set_title('Four Tank System Simulation')
        axs[0].legend()
        axs[0].grid(True)

        # Subplot 2: Control inputs
        axs[1].plot(t, u[0, :], label='u1')
        axs[1].plot(t, u[1, :], label='u2')
        axs[1].set_xlabel('Time [s]')
        axs[1].set_ylabel('Control Input')
        axs[1].set_title('Control Inputs Over Time')
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()
    return x, y, z, u

def qpsolver(H, g, l=None, u=None, A=None, bl=None, bu=None, xinit=None):
    "Implements the QP solver for problem 7"
    "If no bounds l<=x<=u and no bounds bl<=Ax<=bu specified --> case is unconstrained"
    
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
        if bl is not None and bu is not None:
            constraints.append(A @ x >= bl)
            constraints.append(A @ x <= bu)
        elif bl is not None:
            constraints.append(A @ x >= bl)
        elif bu is not None:
            constraints.append(A @ x <= bu)

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    # min_val = prob.solve()
    min_val = prob.solve(solver=cp.OSQP, warm_start=True)

    return x.value, min_val, prob.status



def compute_AB(Ac, Bc, Cc, Dc, Ts):
    """
    Compute discrete-time A and B matrices from Ac, Bc using matrix exponential.
    """
    n = Ac.shape[0]
    m = Bc.shape[1]

    # Build augmented matrix
    aug = np.zeros((n + m, n + m))
    aug[:n, :n] = Ac
    aug[:n, n:] = Bc

    # Compute matrix exponential
    exp_aug = expm(aug * Ts)

    # Extract results
    A = exp_aug[:n, :n]
    B = exp_aug[:n, n:]
    C, D = Cc,Dc  # Doesn't change

    return A, B, C, D

def FourTankSystemLinear(t, X, U, p, A, B, C, D):
    # Unpack states, inputs, and parameters
    ap = p[0:4]         # Pipe cross sectional areas [cm^2]
    At = p[4:8]         # Tank cross sectional areas [cm^2]
    gamma = p[8:10]    # Valve positions [-]
    g = p[10]          # Acceleration of gravity [cm/s^2]
    rho = p[11]        # Density of water [g/cm^3]

    # Compute steady-state levels and time constants
    hs = ys
    T = (At / ap) * np.sqrt(2 * hs / g)

    # Reduced output matrix Cz (first two outputs)
    Cz = C[0:2, :]
    Xdot = A@X + B@U
    Y = C@X + D@U
    Z = Cz@X    
    return Xdot, Y, Z, T

# ---------------------------
# Problem 5 utilities (integrated, using built-ins)
# ---------------------------

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

def linearize_system(f, g, x_op, u_op, d_op, p, method='3-point'):
    """
    Linearize using scipy.optimize.approx_derivative.

    f signature: f(t, x, u, d, p) -> xdot (n,)
    g signature: g(x, p) -> y (ny,)  # output function

    Returns: A, B, Bd, C, D (continuous-time)
    """
    # wrappers to produce functions of a single vector argument
    fx = lambda x: f(0.0, x, u_op, d_op, p)
    fu = lambda u: f(0.0, x_op, u, d_op, p)
    fd = lambda d: f(0.0, x_op, u_op, d, p)
    gx = lambda x: g(x, p)

    A = approx_derivative(fx, x_op, method=method)
    B = approx_derivative(fu, u_op, method=method)
    Bd = approx_derivative(fd, d_op, method=method)

    C = approx_derivative(gx, x_op, method=method)
    # assume no direct feedthrough from u to y (modify if your g depends on u)
    D = np.zeros((C.shape[0], len(u_op)))

    return A, B, Bd, C, D

def continuous_tfs(A, B, C, D):
    """
    Build control.StateSpace and per-input-output transfer functions (MIMO TF returned by control.ss2tf)
    """
    sysc = control.ss(A, B, C, D)
    tf = control.ss2tf(sysc)
    return sysc, tf

def analyze_continuous_siso_tf(tf_siso):
    """
    Return DC gain (Kdc) and dominant time constant (tau) for a SISO transfer function.
    tf_siso: control.TransferFunction (SISO)
    """
    # DC gain
    try:
        Kdc = float(control.evalfr(tf_siso, 0.0))
    except Exception:
        Kdc = np.nan

    # poles
    poles = control.pole(tf_siso)
    stable_poles = [p for p in poles if np.real(p) < 0]
    if len(stable_poles) == 0:
        tau = np.nan
    else:
        # dominant pole: the one with largest real part (closest to imaginary axis)
        dom = max(stable_poles, key=lambda z: np.real(z))
        tau = -1.0 / np.real(dom)
    return Kdc, tau, poles

def _first_order_step(t, K, tau, y0=0.0):
    return y0 + K * (1 - np.exp(-t / tau))

def estimate_first_order_from_step(t, y, step_amplitude=1.0, guess=None):
    """
    Fit y(t) = y0 + K*(1 - exp(-t/tau)) to experimental step response y for step amplitude.
    Returns K_est (per unit step), tau_est, y0_est, covariance.
    """
    if guess is None:
        K0 = (y[-1] - y[0]) / max(step_amplitude, 1e-12)
        tau0 = (t[-1] - t[0]) / 3.0 if (t[-1] - t[0]) > 0 else 1.0
        y00 = y[0]
        guess = [K0, tau0, y00]

    popt, pcov = curve_fit(lambda tt, K, tau, y0: _first_order_step(tt, K, tau, y0),
                           t, y, p0=guess, maxfev=20000)
    K_est, tau_est, y0_est = popt
    # convert K_est to gain per unit input step amplitude
    K_per_unit = K_est / max(step_amplitude, 1e-12)
    return K_per_unit, tau_est, y0_est, pcov

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

def pct_error(true, approx):
    true = np.asarray(true)
    approx = np.asarray(approx)
    with np.errstate(divide='ignore', invalid='ignore'):
        return 100.0 * (approx - true) / np.where(np.abs(true) > 1e-12, true, np.nan)

def compare_gain_tau(exp_gain, exp_tau, model_gain, model_tau, label='IO'):
    print(f"Comparison for {label}:")
    print(f"  Experimental gain: {exp_gain:.6g}, model gain: {model_gain:.6g}, error: {pct_error(exp_gain, model_gain):.2f}%")
    print(f"  Experimental tau:  {exp_tau:.6g}, model tau:  {model_tau:.6g}, error: {pct_error(exp_tau, model_tau):.2f}%")

def plot_step_fit(t, y, t_model, y_model, title=None):
    plt.figure(figsize=(8, 4))
    plt.plot(t, y, 'k.', label='Experimental')
    plt.plot(t_model, y_model, '-', label='Fitted model')
    plt.xlabel('Time [s]')
    plt.ylabel('Output')
    if title:
        plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

# soptd -> second order plus time delay, although it seems like we do not have time delay yet
def soptd_step_response(t, K, T1, T2, tau, beta):

    # G(s)=K (1+beta s) e^{-tau s} / ((1+T1 s)(1+T2 s))
    # Build TF and simulate step with delay

    G = control.TransferFunction([K*beta, K], [T1*T2, T1+T2, 1]) # NOTE: we cannot include the exponential term (time delay) as it breaks the function

    # delay by shifting time axis; pad with zeros before tau
    y = control.forced_response(G, T=t[t>=0], U=np.ones_like(t[t>=0]))[1]

    # Here we can add the time delay
    y_delayed = np.interp(t, t+tau, y, left=0.0)
    return y_delayed

# foptd -> first order plus time delay (simpler model for single exponential curves)
def foptd_step_response(t, K, T, tau, beta=0.0):

    # G(s)=K (1+beta s) e^{-tau s} / (1+T s)
    # Build TF and simulate step with delay

    G = control.TransferFunction([K*beta, K], [T, 1]) 

    # delay by shifting time axis; pad with zeros before tau
    y = control.forced_response(G, T=t[t>=0], U=np.ones_like(t[t>=0]))[1]
    y_delayed = np.interp(t, t+tau, y, left=0.0)
    return y_delayed

def fit_channel_soptd(t, s, guess, speedup):
        
    if speedup:
        # --- minimal speed-up: downsample to ~0.1 s resolution ---
        t = np.asarray(t, float).ravel()
        s = np.asarray(s, float).ravel()
        dt_est = np.median(np.diff(t))
        target_dt = 0.10  # ≈ 10 Hz fit grid (change to 0.2 for even faster)
        step = max(1, int(round(target_dt / dt_est)))
        t_fit = t[::step]
        s_fit = s[::step]
        # ---------------------------------------------------------
    
    else:
        t_fit = t
        s_fit = s

    # guess = (K, T1, T2, tau, beta)
    bounds_lower = [0.0,  1e-3, 1e-3, 0.0,  -10.0]
    bounds_upper = [np.inf, 200.0, 200.0, 100.0, 10.0]
    popt, _ = curve_fit(
        soptd_step_response, t_fit, s_fit, p0=guess,
        bounds=(bounds_lower, bounds_upper), maxfev=20000
    )
    K, T1, T2, tau, beta = popt

    G = control.TransferFunction([K*beta, K], [T1*T2, T1+T2, 1])
    return G, {'K': K, 'T1': T1, 'T2': T2, 'tau': tau, 'beta': beta}


def fit_channel_foptd(t, s, guess, speedup=False):

    if speedup:
        # --- minimal speed-up: downsample to ~0.1 s resolution ---
        t = np.asarray(t, float).ravel()
        s = np.asarray(s, float).ravel()
        dt_est = np.median(np.diff(t))
        target_dt = 0.10  # ≈ 10 Hz fit grid (change to 0.2 for even faster)
        step = max(1, int(round(target_dt / dt_est)))
        t_fit = t[::step]
        s_fit = s[::step]
        # ---------------------------------------------------------
    else:
        t_fit = t
        s_fit = s

    # guess = (K, T, tau, beta)
    bounds_lower = [0.0,  1e-3, 0.0,  -10.0]
    bounds_upper = [np.inf, 200.0, 100.0, 10.0]
    popt, _ = curve_fit(
        foptd_step_response, t_fit, s_fit, p0=guess,
        bounds=(bounds_lower, bounds_upper), maxfev=8000
    )
    K, T, tau, beta = popt

    G = control.TransferFunction([K*beta, K], [T, 1])
    return G, {'K': K, 'T': T, 'tau': tau, 'beta': beta}

### Functions for Problem 8 and 9
def build_prediction_matrices(A, B, N):
    """
    Builds Phi and Gamma matrices for MPC in problem 8 and 9
    """
    nx = A.shape[0]
    nu = B.shape[1]

    Phi_x = np.zeros((N*nx, nx))
    Gamma = np.zeros((N*nx, N*nu))

    for i in range(N):
        Phi_x[i*nx:(i+1)*nx, :] = np.linalg.matrix_power(A, i+1)
        for j in range(i+1):
            H = np.linalg.matrix_power(A, i-j) @ B
            Gamma[i*nx:(i+1)*nx, j*nu:(j+1)*nu] = H

    return Phi_x, Gamma

def design_mpc(A, B, Q, R, N):
    """
    Designs MPC and returns feedback matrices.
    """
    nu = B.shape[1]

    Phi, Gamma = build_prediction_matrices(A, B, N)

    Qbar = scipy.linalg.block_diag(*([Q] * N))
    Rbar = scipy.linalg.block_diag(*([R] * N))

    H = Gamma.T @ Qbar @ Gamma + Rbar
    F = Gamma.T @ Qbar @ Phi

    # First control move extraction
    K = np.linalg.inv(H) @ F
    K0 = K[:nu, :]   # u_k = -K0 x_k

    return {
        "Phi": Phi,
        "Gamma": Gamma,
        "Qbar": Qbar,
        "Rbar": Rbar,
        "K0": K0,
        "H": H,
        "nu": nu,
        "N" : N
    }

def mpc_compute(xk, xr, mpc):
    """
    Compute unconstrained MPC control law.
    """
    u = -mpc["K0"] @ (xk - xr)
    return u

def mpc_compute_constrained(xk, xr, mpc, u_min, u_max):
    Phi = mpc["Phi"]
    Gamma = mpc["Gamma"]
    Qbar = mpc["Qbar"]
    H = mpc["H"]
    nu = mpc["nu"]
    N = mpc["N"]

    # Ensure bounds are vectors

    u_min = np.array([u_min,u_min])
    u_max = np.array([u_max,u_max])

    # Linear term
    g = Gamma.T @ Qbar @ (Phi @ (xk - xr))

    # Box constraints on U
    l = np.tile(u_min, N)
    u = np.tile(u_max, N)

    # Solve QP
    U_star, _, _ = qpsolver(H, g, l=l, u=u)

    # First control move
    u_k = U_star[:nu]

    return u_k

def closed_loop_mpc_sim_unconstrained(
    t, x0, p, d,
    mpc, xr, u_op,
    plot=True
):
    """
    Closed-loop MPC simulation unconstrained
    """

    N = len(t)
    nx = len(x0)

    # storage
    x = np.zeros((nx, N))
    u = np.zeros((2, N))
    y = np.zeros((nx, N))

    # initial condition
    x[:, 0] = x0

    for k in range(N-1):

        # --- 1. current state (THIS is x_k)
        xk = x[:, k]

        # --- 2. MPC control law
        uk = mpc_compute(xk, xr, mpc)
        u[:, k] = uk

        # --- 3. simulate nonlinear plant one step
        t_span = (t[k], t[k+1])
        sol_t, sol_X, sol_H, _ = run_step(
            xk, t_span, uk, d[:, k], p
        )

        # take last state
        x[:, k+1] = sol_X[-1, :]

        # output (heights)
        y[:, k] = FourTankSystemOutput(xk, p)

    # final output
    y[:, -1] = FourTankSystemOutput(x[:, -1], p)
    u_final = mpc_compute(xk, xr, mpc)
    u[:, -1] = u_final

    if plot:
        rho = p[11]
        A_tank = p[4:8]
        h_ref = xr / (rho*A_tank)
        h = x / (rho * A_tank[:, None])

        plt.figure(figsize=(10, 6))
        plt.plot(t, h[0], label="Tank 1")
        plt.plot(t, h[1], label="Tank 2")
        plt.plot(t, h[2], label="Tank 3")
        plt.plot(t, h[3], label="Tank 4")
        plt.axhline(h_ref[0], linestyle="--", color="k")
        plt.axhline(h_ref[1], linestyle="--", color="k")
        plt.xlabel("Time [s]")
        plt.ylabel("Level [cm]")
        plt.title("Closed-loop MPC (Nonlinear Plant)")
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.step(t, u[0], where="post", label="u1")
        plt.step(t, u[1], where="post", label="u2")
        plt.xlabel("Time [s]")
        plt.ylabel("Pump flow")
        plt.title("MPC Inputs")
        plt.legend()
        plt.grid()
        plt.show()

    return x, u, y

def closed_loop_mpc_sim_constrained(
    t, x0, p, d,
    mpc, xr, u_op, u_min, u_max,
    plot=True
):
    """
    Closed-loop MPC simulation using nonlinear four-tank plant
    """

    N = len(t)
    nx = len(x0)

    # storage
    x = np.zeros((nx, N))
    u = np.zeros((2, N))
    y = np.zeros((nx, N))

    # initial condition
    x[:, 0] = x0

    for k in range(N-1):

        # --- 1. current state (THIS is x_k)
        xk = x[:, k]

        # --- 2. MPC control law
        uk = mpc_compute_constrained(xk, xr, mpc, u_min, u_max)
        u[:, k] = uk

        # --- 3. simulate nonlinear plant one step
        t_span = (t[k], t[k+1])
        sol_t, sol_X, sol_H, _ = run_step(
            xk, t_span, uk, d[:, k], p
        )

        # take last state
        x[:, k+1] = sol_X[-1, :]

        # output (heights)
        y[:, k] = FourTankSystemOutput(xk, p)

    # final output
    y[:, -1] = FourTankSystemOutput(x[:, -1], p)
    u_final = mpc_compute_constrained(xk, xr, mpc, u_min, u_max)
    u[:, -1] = u_final
    
    if plot:
        rho = p[11]
        A_tank = p[4:8]
        h_ref = xr / (rho*A_tank)
        h = x / (rho * A_tank[:, None])

        plt.figure(figsize=(10, 6))
        plt.plot(t, h[0], label="Tank 1")
        plt.plot(t, h[1], label="Tank 2")
        plt.plot(t, h[2], label="Tank 3")
        plt.plot(t, h[3], label="Tank 4")
        plt.axhline(h_ref[0], linestyle="--", color="k")
        plt.axhline(h_ref[1], linestyle="--", color="k")
        plt.xlabel("Time [s]")
        plt.ylabel("Level [cm]")
        plt.title("Closed-loop MPC (Nonlinear Plant)")
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.step(t, u[0], where="post", label="u1")
        plt.step(t, u[1], where="post", label="u2")
        plt.xlabel("Time [s]")
        plt.ylabel("Pump flow")
        plt.title("MPC Inputs")
        plt.legend()
        plt.grid()
        plt.show()

    return x, u, y


def mpc_compute_soft_output_constrained(
    xk, xr, mpc,
    u_min, u_max,
    y_min, y_max,
    u_op=None,
    y_ref=None,
    rho_s=1e6
):
    """
    Soft ABSOLUTE output constraints with input bounds.

    Deviation model:
        dX = Phi * dx0 + Gamma * dU
        dY = Cbar * dX + Dbar * dU
        Y_abs = y_ref + dY

    Soft constraints:
        y_min <= Y_abs + s
        Y_abs - s <= y_max
        s >= 0
    """


    if "C" not in mpc or "D" not in mpc:
        raise ValueError("mpc must contain output matrices C and D (pass them into design_mpc).")

    Phi   = mpc["Phi"]
    Gamma = mpc["Gamma"]
    Qbar  = mpc["Qbar"]
    H_U   = mpc["H"]
    nu    = mpc["nu"]
    N     = mpc["N"]
    C     = mpc["C"]
    D     = mpc["D"]

    ny = C.shape[0]
    nU = N * nu
    ns = N * ny

    dx0 = np.asarray(xk) - np.asarray(xr)

    dX0 = Phi @ dx0

    Cbar = scipy.linalg.block_diag(*([C] * N))
    Dbar = scipy.linalg.block_diag(*([D] * N))

    dY0 = Cbar @ dX0
    Gy  = Cbar @ Gamma + Dbar

    gU = Gamma.T @ Qbar @ dX0

    # convert u bounds from absolute to deviation
    u_min = np.asarray(u_min, float).reshape(-1,)
    u_max = np.asarray(u_max, float).reshape(-1,)

    if u_min.size == 1: u_min = np.repeat(u_min, nu)
    if u_max.size == 1: u_max = np.repeat(u_max, nu)

    if u_op is None:
        u_op = np.zeros(nu, float)

    u_op = np.asarray(u_op, float).reshape(-1,)
    if u_op.size == 1: u_op = np.repeat(u_op, nu)

    du_min = u_min - u_op
    du_max = u_max - u_op

    # absolute output bounds in cm, stacked over horizon

    y_min = np.asarray(y_min, float).reshape(-1,)
    y_max = np.asarray(y_max, float).reshape(-1,)

    if y_min.size == 1: y_min = np.repeat(y_min, ny)
    if y_max.size == 1: y_max = np.repeat(y_max, ny)

    if y_ref is None:
        y_ref = np.zeros(ny, float)
    y_ref = np.asarray(y_ref, float).reshape(-1,)
    if y_ref.size == 1: y_ref = np.repeat(y_ref, ny)

    y_ref_stack = np.tile(y_ref, N)
    y_min_stack = np.tile(y_min, N)
    y_max_stack = np.tile(y_max, N)

    # decision w = [ΔU; s]
    S  = rho_s * np.eye(ns)
    Hw = scipy.linalg.block_diag(H_U, S)
    gw = np.hstack([gU, np.zeros(ns)])

    lw = np.hstack([np.tile(du_min, N), np.zeros(ns)])
    uw = np.hstack([np.tile(du_max, N), np.full(ns, np.inf)])

    # Y_abs = y_ref_stack + dY0 + Gy * dU
    # Gy * dU - I * s <= y_max - y_ref_stack - dY0
    # Gy * dU + I * s >= y_min - y_ref_stack - dY0

    I  = np.eye(ns)
    A1 = np.hstack([Gy, -I])
    A2 = np.hstack([Gy,  I])
    A  = np.vstack([A1, A2])

    bu = np.hstack([y_max_stack - y_ref_stack - dY0, np.full(ns,  np.inf)])
    bl = np.hstack([np.full(ns, -np.inf), y_min_stack - y_ref_stack - dY0])

    w_star, _, status = qpsolver(Hw, gw, l=lw, u=uw, A=A, bl=bl, bu=bu)

    if status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"Soft QP failed with status: {status}")

    dU_star = w_star[:nU]
    s_star  = w_star[nU:]   # stacked slack over horizon (N*ny,)

    du_k = dU_star[:nu]
    u_k  = u_op + du_k

    # Required SLACK
    # Predicted absolute outputs over horizon using the smae dU_star:
    Y_abs_pred = y_ref_stack + dY0 + Gy @ dU_star   

    # How much we violate hard bounds
    viol_low  = y_min_stack - Y_abs_pred
    viol_high = Y_abs_pred - y_max_stack
    s_req = np.maximum(0.0, np.maximum(viol_low, viol_high))  

    s_req_mat = s_req.reshape(N, ny)
    s_req_max_over_horizon = float(np.max(s_req_mat))
    s_req_per_y = np.max(s_req_mat, axis=0)  

    y_pred_next = Y_abs_pred[:ny]                          

    # return control input, slack variables, infeasibility metrics, and 1-step predicted output
    return u_k, s_star, s_req_max_over_horizon, s_req_per_y, y_pred_next


def closed_loop_mpc_sim_soft_output_constrained(
    t, x0, p, d, mpc, xr, u_op,
    u_min, u_max, y_min, y_max, rho_s=1e6,
    plot=True, debug=True
):
    """
    Closed-loop simulation on plant with soft output constraints.
    """

    t = np.asarray(t).ravel()
    N = len(t)
    nx = len(x0)

    x = np.zeros((nx, N))
    u = np.zeros((2, N))
    y = np.zeros((nx, N))
    x[:, 0] = np.asarray(x0).reshape(-1,)

    # absolute reference output (heights) at xr
    y_ref_full = FourTankSystemSensor_Deterministic(xr, p)
    ny = mpc["C"].shape[0]
    y_ref = np.asarray(y_ref_full).reshape(-1,)[:ny]

    ny = mpc["C"].shape[0]
    
    hard_feas_hist = np.zeros(N-1, dtype=int)  # 1=hard feasible, 0=infeasible
    hard_req_max_hist = np.zeros(N-1)
    hard_req_per_y    = np.zeros((N-1, ny))

    y_pred_hist = np.zeros((N-1, ny))   # predicted y_{k+1|k} for each step

    for k in range(N - 1):
        xk = x[:, k]

        # hard out constraint MPC step (only for plots)
        hard_feas_hist[k] = int(hard_feasible(
            xk=xk, xr=xr, mpc=mpc,
            u_min=u_min, u_max=u_max,
            y_min=y_min, y_max=y_max,
            u_op=u_op, y_ref=y_ref
        ))

        # soft out constraint MPC step
        uk, s_star, sreq_max, sreq_per_y, y_pred_next = mpc_compute_soft_output_constrained(
            xk=xk, xr=xr, mpc=mpc,
            u_min=u_min, u_max=u_max,
            y_min=y_min, y_max=y_max,
            u_op=u_op, y_ref=y_ref,
            rho_s=rho_s
        )

        y_pred_hist[k, :] = y_pred_next
        u[:, k] = uk

        hard_req_max_hist[k] = sreq_max
        hard_req_per_y[k, :] = sreq_per_y

        sol_t, sol_X, sol_H, _ = run_step(xk, (t[k], t[k+1]), uk, d[:, k], p)
        x[:, k+1] = sol_X[-1, :]
        y[:, k] = FourTankSystemOutput(xk, p)

    y[:, -1] = FourTankSystemOutput(x[:, -1], p)
    u[:, -1] = u[:, -2]

    if plot:
        rho = p[11]
        A_tank = p[4:8]

        # levels in cm
        h = x / (rho * A_tank[:, None])

        tk = t[:-1]

        # plot tank levels
        plt.figure(figsize=(10, 6))
        for i in range(4):
            plt.plot(t, h[i], label=f"Tank {i+1}")

        plt.plot(tk, y_pred_hist[:, 0], "--", color="tab:blue", label="Tank 1 predicted")
        plt.plot(tk, y_pred_hist[:, 1], "--", color="tab:orange", label="Tank 2 predicted")

        # constraints
        y_min_arr = np.asarray(y_min).reshape(-1,)
        y_max_arr = np.asarray(y_max).reshape(-1,)

        ny_plot = min(len(y_min_arr), len(y_max_arr), h.shape[0])

        for i in range(ny_plot):
            plt.axhline(
                y_min_arr[i],
                linestyle="--",
                color="k",
                label="Lower bound" if i == 0 else None
            )
            plt.axhline(
                y_max_arr[i],
                linestyle="--",
                color="k",
                label="Upper bound" if i == 0 else None
            )

        plt.xlabel("Time [s]")
        plt.ylabel("Level [cm]")
        plt.title("Soft Output-Constrained MPC (Nonlinear Plant)")
        plt.legend()
        plt.grid(True)
        plt.show()

        # plot inputs
        plt.figure(figsize=(10, 4))
        plt.step(t, u[0], where="post", label="u1")
        plt.step(t, u[1], where="post", label="u2")
        plt.xlabel("Time [s]")
        plt.ylabel("Pump flow")
        plt.title("MPC inputs - with soft output constraints")
        plt.legend()
        plt.grid(True)
        plt.show()

        # hard feasibility flag plot
        plt.figure(figsize=(10, 3))
        plt.step(t[:-1], hard_feas_hist, where="post")
        plt.ylim([-0.1, 1.1])
        plt.xlabel("Time [s]")
        plt.ylabel("hard feasible")
        plt.title("Hard feasibility (linear prediction QP): 1=feasible, 0=infeasible")
        plt.grid(True)
        plt.show()
        
        plt.figure(figsize=(10,4))
        plt.plot(tk, hard_req_max_hist, label="max required slack over horizon")
        plt.axhline(0.0, linestyle="--", color="k")
        plt.xlabel("Time [s]")
        plt.ylabel("required slack [cm]")
        plt.title("Hard infeasibility measure (0 = hard feasible)")
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.figure(figsize=(10,4))
        for i in range(ny):
            plt.plot(tk, hard_req_per_y[:, i], label=f"output {i+1}")
        plt.axhline(0.0, linestyle="--", color="k")
        plt.xlabel("Time [s]")
        plt.ylabel("required slack [cm]")
        plt.title("Which output causes infeasibility?")
        plt.grid(True)
        plt.legend()
        plt.show()

        tk = t[:-1]  # prediction made at time k for k+1

        plt.figure(figsize=(10, 5))

        # actual nonlinear outputs at k+1 (heights)
        plt.plot(t[1:], h[0, 1:], color="tab:blue", label="Tank 1 actual")
        plt.plot(tk, y_pred_hist[:, 0], "--", color="tab:blue", label="Tank 1 predicted")

        plt.plot(t[1:], h[1, 1:], color="tab:orange", label="Tank 2 actual")
        plt.plot(tk, y_pred_hist[:, 1], "--", color="tab:orange", label="Tank 2 predicted")


        plt.axhline(y_min[0], linestyle="--", color="k", label="Lower bound")
        plt.axhline(y_max[0], linestyle="--", color="k", label="Upper bound")

        plt.xlabel("Time [s]")
        plt.ylabel("Level [cm]")
        plt.title("One-step prediction: linear MPC vs nonlinear plant")
        plt.legend()
        plt.grid(True)
        plt.show()

    return x, u, y

def hard_feasible(xk, xr, mpc, u_min, u_max, y_min, y_max, u_op, y_ref):
    """
    Returns True if HARD output constraints are feasible for the *predicted linear MPC model*
    with input bounds, at the current step k (no slack).
    """
    Phi   = mpc["Phi"]
    Gamma = mpc["Gamma"]
    Qbar  = mpc["Qbar"]
    H_U   = mpc["H"]
    nu    = mpc["nu"]
    N     = mpc["N"]
    C     = mpc["C"]
    D     = mpc["D"]

    ny = C.shape[0]
    nU = N * nu

    dx0 = np.asarray(xk) - np.asarray(xr)
    dX0 = Phi @ dx0

    Cbar = scipy.linalg.block_diag(*([C] * N))
    Dbar = scipy.linalg.block_diag(*([D] * N))

    dY0 = Cbar @ dX0
    Gy  = Cbar @ Gamma + Dbar

    # linear term
    gU = Gamma.T @ Qbar @ dX0

    # u bounds
    u_min = np.asarray(u_min, float).reshape(-1,)
    u_max = np.asarray(u_max, float).reshape(-1,)
    if u_min.size == 1: u_min = np.repeat(u_min, nu)
    if u_max.size == 1: u_max = np.repeat(u_max, nu)

    u_op = np.asarray(u_op, float).reshape(-1,)
    if u_op.size == 1: u_op = np.repeat(u_op, nu)

    du_min = u_min - u_op
    du_max = u_max - u_op
    lU = np.tile(du_min, N)
    uU = np.tile(du_max, N)

    # y bounds
    y_min = np.asarray(y_min, float).reshape(-1,)
    y_max = np.asarray(y_max, float).reshape(-1,)

    if y_min.size == 1: y_min = np.repeat(y_min, ny)
    if y_max.size == 1: y_max = np.repeat(y_max, ny)

    y_ref = np.asarray(y_ref, float).reshape(-1,)
    if y_ref.size == 1: y_ref = np.repeat(y_ref, ny)

    y_ref_stack = np.tile(y_ref, N)
    y_min_stack = np.tile(y_min, N)
    y_max_stack = np.tile(y_max, N)

    # Hard constraints on predicted absolute outputs
    # y_min <= y_ref + dY0 + Gy*dU <= y_max
    bl = (y_min_stack - y_ref_stack - dY0)
    bu = (y_max_stack - y_ref_stack - dY0)

    # Solve hard QP via cvxpy so we can read status easy
    dU = cp.Variable(nU)
    objective = cp.Minimize(0.5 * cp.quad_form(dU, H_U) + gU.T @ dU)
    constraints = [
        dU >= lU,
        dU <= uU,
        Gy @ dU >= bl,
        Gy @ dU <= bu,
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, warm_start=True)

    return prob.status in ["optimal", "optimal_inaccurate"]


def impulse_response_markov_from_ss(Ad, Bd, Cd, Dd, n_steps=80):
    """
    Markov parameters (impulse response) for discrete-time state-space:
        h[0] = D
        h[k] = C A^(k-1) B,  k>=1
    Returns: H with shape (n_steps, ny, nu)
    """
    Ad = np.asarray(Ad, float)
    Bd = np.asarray(Bd, float)
    Cd = np.asarray(Cd, float)
    Dd = np.asarray(Dd, float)

    ny, nu = Cd.shape[0], Bd.shape[1]
    nx = Ad.shape[0]

    H = np.zeros((n_steps, ny, nu))
    H[0] = Dd.reshape(ny, nu)

    A_pow = np.eye(nx)
    for k in range(1, n_steps):
        # A_pow = A^(k-1) after this update
        if k == 1:
            A_pow = np.eye(nx)
        else:
            A_pow = A_pow @ Ad
        H[k] = Cd @ A_pow @ Bd

    return H

def plot_markov_parameters(H, Ts=1.0, title="Impulse response (Markov parameters)", channel_names=None):

    H = np.asarray(H, float)
    n_steps, ny, nu = H.shape
    k = np.arange(n_steps)
    t = k * Ts
    fig, axs = plt.subplots(ny, nu, figsize=(4.5 * nu, 3.5 * ny), sharex=True)
    axs = np.atleast_2d(axs)

    for iy in range(ny):
        for iu in range(nu):
            ax = axs[iy, iu]
            ax.stem(t, H[:, iy, iu], basefmt=" ")

            if channel_names is None:
                ax.set_title(f"F{iu+1} → Tank {iy+1}")
            else:
                ax.set_title(channel_names[iy][iu])
                
            ax.grid(True)
            if iy == ny - 1:
                ax.set_xlabel("Time [s]")
            if iu == 0:
                ax.set_ylabel("h[k]")

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
