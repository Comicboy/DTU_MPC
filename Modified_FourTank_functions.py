import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
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
def sim22(times, x0, u, d, p, noise_level=0, plot=True):
    """"
    Parameters:
    times: array of start and stop for each time interval. len(times) is number of simulations
    u: np.array([F1,F2])    with F1 and F2 for each interval in times
    d: np.array([F3,F4])
    p: Parameters
    noise_level: std of noise (default 0 meaning deterministic case)
    """
    a = p[0:4]         # Pipe cross sectional areas [cm^2]
    A = p[4:8]         # Tank cross sectional areas [cm^2]
    gamma = p[8:10]    # Valve positions [-]
    g = p[10]          # Acceleration of gravity [cm/s^2]
    rho = p[11]        # Density of water [g/cm^3]

    N = len(times)
    nx = len(x0)
    
    X_all = np.zeros((0, nx))  # All x (Will grow as we append)
    H_all = np.zeros((0, nx))  # All H (Will grow as we append)
    T_all = np.zeros((0, 1))   # All time (Will grow as we append)
    x = np.zeros((nx, N))      # State history
    y = np.zeros((nx, N))      # Sensor history (adjust shape if needed)
    z = np.zeros((nx, N))      # Output history (adjust shape if needed)

    x[:, 0] = x0  # Initial condition

    for k in range(N-1):
        # Sensor and output functions
        y[:,k] = FourTankSystemSensor(x[:,k],p) #Height measurements for now
        z[:,k] = FourTankSystemOutput(x[:,k],p) #Height measurements

        # Integrate from t[k] to t[k+1]
        x0_new = x[:,k]
        u_step = u[:, k]
        d[:, k] = np.clip(d[:, k] + np.random.normal(0, noise_level, 2), 0, None) # Adding disturbance. clips to be >=0
        t_span=(times[k], times[k+1])
        sol_time, sol_X, sol_H, Qout = run_step(x0_new, t_span, u_step, d[:,k], p)
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
            uplot = np.zeros(len(times))
            uplot[:len(times)-1] = u[i, :]
            uplot[-1] = u[i, -1]
            axs[1].step(times/60, uplot, where='post', label=label)

        for i, label in enumerate(['F3', 'F4']):
            dplot = np.zeros(len(times))
            dplot[:len(times)-1] = d[i, :]
            dplot[-1] = d[i, -1]
            axs[1].step(times/60, dplot, where='post', label=label)

        axs[1].set_xlabel('Time [min]')
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
        x[2:, i + 1] = np.clip(x[2:, i] + dx[2:] + noise[i], 0, None)

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

def qpsolver(H, g, l, u, A, bl, bu, xinit=None):
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