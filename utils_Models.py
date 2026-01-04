
import numpy as np
from scipy.integrate import solve_ivp

from utils_functions import *
from utils_Modified_FourTank_functions import Modified_FourTankSystem
from utils_DisturbanceModels import *
from typing import Tuple

class OpenLoop_ModelA(ModelSimulation):
    '''Deterministic without control, and any noise'''
    
    def dynamics(self, t, x, u, d, p):
        return Modified_FourTankSystem(t, x, u, d, p)

    def disturbance(self, t, d):
        return d  # constant
    
    def control(self, t, u):
        return u  # constant

    def step(self, t_span, x0, u, d):
        sol = solve_ivp(
            fun=lambda t, x: self.dynamics(t, x, u, d, self.p),
            t_span=t_span,
            y0=x0,
            method="BDF",
            dense_output=False
        )
        return sol.t, sol.y.T
    
    def full_output(self, T, X):
        # Helper variables
        nT, nX = X.shape
        
        # Unpack parameters
        a = self.p[0:4]     # Pipe cross-sectional areas [cm^2]
        A = self.p[4:8]     # Tank cross-sectional areas [cm^2]
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
        
        return T, X, H, Qout
    
    def simulate(self):
        """Main simulation loop"""
        ts = self.ts
        xk_minus1 = self.x0
        dk_minus1 = self.d0
        uk_minus1 = self.u0

        Tfull = [ts[0]]
        Xfull = [xk_minus1]
        Dk = [dk_minus1]
        Uk = [uk_minus1]
        Xk = [xk_minus1]
        
        y_minus1 = self.SystemSensor(xk_minus1)
        Yk = [y_minus1]

        for k in range(1, len(ts)):
            tk = ts[k]
            tk_minus1 = ts[k-1]
            
            T_segment, X_segment = self.step((tk_minus1, tk), xk_minus1, uk_minus1, dk_minus1)

            # Concatenate and extend the continuous trajectories
            Tfull = np.concatenate((Tfull, T_segment[1:]))
            Xfull = np.vstack((Xfull, X_segment[1:]))
            
            
            xk = X_segment[-1, :]
            yk = self.SystemSensor(xk)
            uk = self.control(tk,uk_minus1)
            dk = self.disturbance(tk, dk_minus1)
            
            xk_minus1 = xk
            dk_minus1 = dk.squeeze()
            uk_minus1 = uk.squeeze()
            yk_minus1 = yk
            
            Xk.append(xk)
            Yk.append(yk)
            Dk.append(dk_minus1)
            Uk.append(uk_minus1)
            
        T, X, H, Qout= self.full_output(Tfull, Xfull)
        
        Uk = np.hstack([ts[:,None], Uk])
        Dk = np.hstack([ts[:,None], Dk])
        Yk = np.array(Yk).squeeze()
        Xk = np.array(Xk)
        return T, X, H, Qout, Xk, Uk, Dk, Yk
    
class OpenLoop_ModelB(ModelSimulation):
    '''
    Model with piecewise constant disturbances for F3 and F4, and measurement noise v
    
    d_noiselevel: std for normal distribution
    v_noiselevel: variance in measurement noise covariance
    '''
    def __init__(self, ts, x0, u0, d0, p, d_noiselevel = 1, v_noiselevel=1):
        super().__init__(ts, x0, u0, d0, p)
        self.R = v_noiselevel * np.eye(2)  # measurement noise covariance
        self.d_noiselevel = d_noiselevel
    def dynamics(self, t, x, u, d):
        return  Modified_FourTankSystem(t, x, u, d, self.p)

    def disturbance(self, t, d, dmin=0, dmax=500, sigma=1):
        d_means = np.array([100,100])
        sigma = self.d_noiselevel
 
        # delta_d = np.random.normal(0, sigma, size=d.shape)
        # d = np.clip(d+delta_d, dmin, dmax)       
        
        d = np.random.normal(d_means, sigma, size=d.shape)
        d = np.clip(d, dmin, dmax)
        return d
    
    def control(self, t, u):
        return u

    def step(self, t_span, x0, u, d):
        sol = solve_ivp(
            fun=lambda t, x: self.dynamics(t, x, u, d),
            t_span=t_span,
            y0=x0,
            method="BDF",
            dense_output=False
        )
        return sol.t, sol.y.T
    
    def full_output(self, T, X):
        # Helper variables
        nT, nX = X.shape
        
        # Unpack parameters
        a = self.p[0:4]     # Pipe cross-sectional areas [cm^2]
        A = self.p[4:8]     # Tank cross-sectional areas [cm^2]
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
        
        return T, X, H, Qout

         
    def simulate(self):
        """Main simulation loop"""
        ts = self.ts
        xk_minus1 = self.x0
        dk_minus1 = self.d0
        uk_minus1 = self.u0

        Tfull = [ts[0]]
        Xfull = [xk_minus1]
        Dk = [dk_minus1]
        Uk = [uk_minus1]
        Xk = [xk_minus1]
        
        #setup for sensor measurement noise        
        v = self.sensor_noise(N=len(ts))[:,:,None]
        y_minus1 = self.SystemSensor(xk_minus1) + v[:,0]
        Yk = [y_minus1]

        for k in range(1, len(ts)):
            tk = ts[k]
            tk_minus1 = ts[k-1]
            
            T_segment, X_segment = self.step((tk_minus1, tk), xk_minus1, uk_minus1, dk_minus1)

            # Concatenate and extend the continuous trajectories
            Tfull = np.concatenate((Tfull, T_segment[1:]))
            Xfull = np.vstack((Xfull, X_segment[1:]))
            
            
            xk = X_segment[-1, :]
            yk = self.SystemSensor(xk) + v[:,k]
            uk = self.control(tk,uk_minus1)
            dk = self.disturbance(tk, dk_minus1)
            
            xk_minus1 = xk
            dk_minus1 = dk.squeeze()
            uk_minus1 = uk.squeeze()
            
            Xk.append(xk)
            Yk.append(yk)
            Dk.append(dk_minus1)
            Uk.append(uk_minus1)
            
        T, X, H, Qout= self.full_output(Tfull, Xfull)
        
        Uk = np.hstack([ts[:,None], Uk])
        Dk = np.hstack([ts[:,None], Dk])
        Yk = np.array(Yk).squeeze()
        return T, X, H, Qout, np.array(Xk), Uk, Dk, Yk
    
    def process_noise(self, N,Q=np.array([[20**2,0],[0,40**2]]), Nu = 2):
        'add to u'
        Lq = np.linalg.cholesky(Q)

        #process noise
        e =  np.random.randn(Nu,N)
        w = Lq @ e
        return w
        
    def sensor_noise(self, N, Ny=2):
        'add to y'
        Lr = np.linalg.cholesky(self.R)
        e = np.random.randn(Ny,N)
        v = Lr @ e
        return v
    
class OpenLoop_ModelC(ModelSimulation):
    def __init__(self, 
                 ts: np.array, 
                 x0: np.array, 
                 u0: np.array, 
                 d0:np.array, 
                 p:np.array, 
                 disturbances: Tuple[DisturbanceModel,DisturbanceModel], 
                 dt_small: float=0.1):
        
        super().__init__(ts, x0, u0, d0, p)
        self.disturbances = disturbances
        self.dt_small = dt_small
        
    def dynamics(self, t, x, u, d):
        return Modified_FourTankSystem_SDE(t, x, u, d, self.p, self.disturbances)
        
    def control(self, t, u):
        return u

    def disturbance(self, t, d, dmin=0, dmax=500):
        # d is stochastic here; actual d handled in solver via Brownian motion
        return d
        
    def step(self, tspan, x, u, d):
        Tvec = np.arange(tspan[0],tspan[1]+self.dt_small, self.dt_small)
        X = np.zeros((len(Tvec), len(x)))
        D = np.zeros((len(Tvec), len(d)))  # disturbance over time

        X[0, :] = x
        D[0, :] = d
        
        x_aug = np.hstack([x, d])  # shape (6,)

        for i in range(len(Tvec)-1):
            
            m = x_aug[:4]
            d = x_aug[4:]
            
            f, sigma = self.dynamics(Tvec[i], m, u, d)
            
            # Euler–Maruyama
            x_aug = x_aug + f*self.dt_small + sigma @ (np.sqrt(self.dt_small) * np.random.randn(2))
            
            X[i+1, :] = x_aug[:4]
            D[i+1, :] = np.clip(x_aug[4:],0,None)
        return Tvec, X,D
    
    def full_output(self, T, X):
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
        
        return T, X, H, Qout
    
    def simulate(self):
        """Main simulation loop"""
        ts = self.ts
        xk_minus1 = self.x0
        dk_minus1 = self.d0
        uk_minus1 = self.u0

        Tfull = [ts[0]]
        Xfull = [xk_minus1]
        Dfull = [dk_minus1]
        Uk = [uk_minus1]
        
        Xk = [xk_minus1]
        Dk = [dk_minus1]
        
        #setup for sensor
        #measurement noise
        v = self.sensor_noise(N=len(ts))
        yk_minus1 = self.SystemSensor(xk_minus1)+ v[:,0, None]
        Yk = [yk_minus1]
            
        for k in range(1, len(ts)):
            tk = ts[k]
            tk_minus1 = ts[k-1]
            
            T_segment, X_segment, D_segment = self.step([tk_minus1,tk], xk_minus1, uk_minus1, dk_minus1)

            Tfull = np.concatenate((Tfull, T_segment[1:]))
            Xfull = np.vstack((Xfull, X_segment[1:]))
            Dfull = np.vstack((Dfull, D_segment[1:]))
                
            xk = X_segment[-1, :]
            uk = self.control(tk,uk_minus1) # control update
            yk = self.SystemSensor(xk) + v[:,k, None]
            dk = D_segment[-1,:]
            
            
            #update
            xk_minus1 = xk
            uk_minus1 = uk.squeeze()
            yk_minus1 = yk
            dk_minus1 = dk
            
            #discrete samples
            Xk.append(xk)
            Dk.append(dk)
            Yk.append(yk)
            Uk.append(uk)
        
        #discrete states
        Xk = np.hstack([ts[:,None], Xk])
        Uk = np.hstack([ts[:,None], Uk])
        Dk = np.hstack([ts[:,None], Dk])
        Yk = np.array(Yk).squeeze() #messed up some dimensions here
        
        D = np.hstack([Tfull[:,None], Dfull])
        T, X, H, Qout = self.full_output(Tfull, Xfull)
        
        return T, X, H, Qout, D, Xk, Uk, Dk, Yk
        
    def sensor_noise(self, N, R=np.eye(2), Ny=2):
        'add to y'
        Lr = np.linalg.cholesky(R)
        e = np.random.randn(Ny,N)
        v = Lr @ e
        return v
    
class ClosedLoop_ModelA(ModelSimulation):    
    '''Deterministic with control, no noise
    
    us: u steady state
    r: set points of Tank 1 and 2
    '''
    def __init__(self, ts, x0, u0, d0, p, us=np.array([300,300]), r=np.array([58, 50])):
        super().__init__(ts, x0, u0, d0, p)
        self.integral_error = np.zeros(2)
        self.prev_error = np.zeros(2)
        self.tspan = ts[:2]  # time span
        self.us = us
        self.r = r # set point 
    
    def dynamics(self, t, x, u, d, p):
        return Modified_FourTankSystem(t, x, u, d, p)

    def disturbance(self, t, d):
        return d  # constant
    
    def control(self, t, y,  Kc=10,Ki=1,Kd=100):
        # PID controller
        us = self.us
        r = self.r
        y = y.squeeze()
        tspan = self.tspan
        integral_error = self.integral_error
        prev_error = self.prev_error
        u_clipped, integral_error_new, derivative_error = PIDcontroller(r,y,us,Kc,Ki,Kd,tspan,integral_error,prev_error)
        self.integral_error = integral_error_new
        self.prev_error = derivative_error
        return u_clipped

    def step(self, t_span, x0, u, d):
        sol = solve_ivp(
            fun=lambda t, x: self.dynamics(t, x, u, d, self.p),
            t_span=t_span,
            y0=x0,
            method="BDF",
            dense_output=False
        )
        return sol.t, sol.y.T
    
    def full_output(self, T, X):
        # Helper variables
        nT, nX = X.shape
        
        # Unpack parameters
        a = self.p[0:4]     # Pipe cross-sectional areas [cm^2]
        A = self.p[4:8]     # Tank cross-sectional areas [cm^2]
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
        
        return T, X, H, Qout
    
    def simulate(self):
        """Main simulation loop"""
        ts = self.ts
        xk_minus1 = self.x0
        dk_minus1 = self.d0
        uk_minus1 = self.u0

        Tfull = [ts[0]]
        Xfull = [xk_minus1]
        Dk = [dk_minus1]
        Uk = [uk_minus1]
        Xk = [xk_minus1]
        
        yk_minus1 = self.SystemSensor(xk_minus1)
        Yk = [yk_minus1]

        for k in range(1, len(ts)):
            tk = ts[k]
            tk_minus1 = ts[k-1]
            
            T_segment, X_segment = self.step((tk_minus1, tk), xk_minus1, uk_minus1, dk_minus1)

            # Concatenate and extend the continuous trajectories
            Tfull = np.concatenate((Tfull, T_segment[1:]))
            Xfull = np.vstack((Xfull, X_segment[1:]))
            
            
            xk = X_segment[-1, :]
            yk = self.SystemSensor(xk)
            uk = self.control(tk,yk_minus1)
            dk = self.disturbance(tk, dk_minus1)
            
            xk_minus1 = xk
            dk_minus1 = dk.squeeze()
            uk_minus1 = uk.squeeze()
            yk_minus1 = yk
            
            Xk.append(xk)
            Yk.append(yk)
            Dk.append(dk_minus1)
            Uk.append(uk_minus1)
            
        T, X, H, Qout= self.full_output(Tfull, Xfull)
        
        Uk = np.hstack([ts[:,None], Uk])
        Dk = np.hstack([ts[:,None], Dk])
        Yk = np.array(Yk).squeeze()
        Xk = np.array(Xk)
        return T, X, H, Qout, Xk, Uk, Dk, Yk
    
class ClosedLoop_ModelB(ModelSimulation):
    '''
    Model with piecewise constant disturbances for F3 and F4, and measurement noise v
    With PID control
    
    d_noiselevel: std for normal distribution
    v_noiselevel: variance in measurement noise covariance
    '''
        
    def __init__(self, ts, x0, u0, d0, p, d_noiselevel = 1, v_noiselevel=1, us = np.array([300,300]), r=np.array([58,50])):
        super().__init__(ts, x0, u0, d0, p)
        self.R = v_noiselevel * np.eye(2)  # measurement noise covariance
        self.d_noiselevel = d_noiselevel
        
        self.integral_error = np.zeros(2)
        self.prev_error = np.zeros(2)
        self.tspan = ts[:2] 
        self.us = us
        self.r = r # set point
        
        
    def dynamics(self, t, x, u, d):
        return  Modified_FourTankSystem(t, x, u, d, self.p)

    def disturbance(self, t, d, dmin=0, dmax=500, sigma=1):
        sigma = self.d_noiselevel
        delta_d = np.random.normal(0, sigma, size=d.shape)
        d = np.clip(d+delta_d, dmin, dmax)
        return d
    
    def control(self, t, y,  Kc=10,Ki=1,Kd=100):
        # PID controller
        us = self.us
        r = self.r
        y = y.flatten()
        tspan = self.tspan
        integral_error = self.integral_error
        prev_error = self.prev_error
        u_clipped, integral_error_new, derivative_error = PIDcontroller(r,y,us,Kc,Ki,Kd,tspan,integral_error,prev_error)
        self.integral_error = integral_error_new
        self.prev_error = derivative_error
        return u_clipped 

    def step(self, t_span, x0, u, d):
        sol = solve_ivp(
            fun=lambda t, x: self.dynamics(t, x, u, d),
            t_span=t_span,
            y0=x0,
            method="BDF",
            dense_output=False
        )
        return sol.t, sol.y.T
    
    def full_output(self, T, X):
        # Helper variables
        nT, nX = X.shape
        
        # Unpack parameters
        a = self.p[0:4]     # Pipe cross-sectional areas [cm^2]
        A = self.p[4:8]     # Tank cross-sectional areas [cm^2]
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
        
        return T, X, H, Qout

         
    def simulate(self):
        """Main simulation loop"""
        ts = self.ts
        xk_minus1 = self.x0
        dk_minus1 = self.d0
        uk_minus1 = self.u0

        Tfull = [ts[0]]
        Xfull = [xk_minus1]
        Dk = [dk_minus1]
        Uk = [uk_minus1]
        Xk = [xk_minus1]
        
        #setup for sensor measurement noise        
        v = self.sensor_noise(N=len(ts))
        yk_minus1 = self.SystemSensor(xk_minus1) + v[:,0]
        Yk = [yk_minus1]

        for k in range(1, len(ts)):
            tk = ts[k]
            tk_minus1 = ts[k-1]
            
            T_segment, X_segment = self.step((tk_minus1, tk), xk_minus1, uk_minus1, dk_minus1)

            # Concatenate and extend the continuous trajectories
            Tfull = np.concatenate((Tfull, T_segment[1:]))
            Xfull = np.vstack((Xfull, X_segment[1:]))
            
            
            xk = X_segment[-1, :]
            yk = self.SystemSensor(xk) + v[:,k]
            uk = self.control(tk,yk_minus1)
            dk = self.disturbance(tk, dk_minus1)
            
            xk_minus1 = xk
            dk_minus1 = dk.squeeze()
            uk_minus1 = uk.squeeze()
            yk_minus1 = yk
            
            Xk.append(xk)
            Yk.append(yk)
            Dk.append(dk_minus1)
            Uk.append(uk_minus1)
            
        T, X, H, Qout= self.full_output(Tfull, Xfull)
        
        Uk = np.hstack([ts[:,None], Uk])
        Dk = np.hstack([ts[:,None], Dk])
        Yk = np.array(Yk).squeeze()
        return T, X, H, Qout, np.array(Xk), Uk, Dk, Yk
    
    def process_noise(self, N,Q=np.array([[20**2,0],[0,40**2]]), Nu = 2):
        'add to u'
        Lq = np.linalg.cholesky(Q)

        #process noise
        e =  np.random.randn(Nu,N)
        w = Lq @ e
        return w
        
    def sensor_noise(self, N, Ny=2):
        'add to y'
        Lr = np.linalg.cholesky(self.R)
        e = np.random.randn(Ny,N)
        v = Lr @ e
        return v
    
class ClosedLoop_ModelC(ModelSimulation):
    def __init__(self, ts, x0, u0, d0, p, disturbances: Tuple[DisturbanceModel,DisturbanceModel], dt_small=0.1, us = np.array([300,300]), r=np.array([58,50])):
        super().__init__(ts, x0, u0, d0, p)
        self.disturbances = disturbances
        self.dt_small = dt_small
        
        self.integral_error = np.zeros(2)
        self.prev_error = np.zeros(2)
        self.tspan = ts[:2] 
        self.us = us
        self.r = r # set point
        
    def dynamics(self, t, x, u, d):
        return Modified_FourTankSystem_SDE(t, x, u, d, self.p, self.disturbances)
        
    def control(self, t, y,  Kc=10,Ki=1,Kd=100):
        # PID controller
        us = self.us
        r = self.r
        y = y.flatten()
        tspan = self.tspan
        integral_error = self.integral_error
        prev_error = self.prev_error
        u_clipped, integral_error_new, derivative_error = PIDcontroller(r,y,us,Kc,Ki,Kd,tspan,integral_error,prev_error)
        self.integral_error = integral_error_new
        self.prev_error = derivative_error
        return u_clipped 

    def disturbance(self, t, d, dmin=0, dmax=500):
        # d is solved for during Euler-Maryumama
        return d
        
    def step(self, tspan, x, u, d):
        Tvec = np.arange(tspan[0],tspan[1]+self.dt_small, self.dt_small)
        X = np.zeros((len(Tvec), len(x)))
        D = np.zeros((len(Tvec), len(d)))  # disturbance over time

        X[0, :] = x
        D[0, :] = d
        
        x_aug = np.hstack([x, d])  # shape (6,)

        for i in range(len(Tvec)-1):
            
            m = x_aug[:4]
            d = x_aug[4:]
            
            f, sigma = self.dynamics(Tvec[i], m, u, d)
            
            # Euler–Maruyama
            x_aug = x_aug + f*self.dt_small + sigma @ (np.sqrt(self.dt_small) * np.random.randn(2))
            
            X[i+1, :] = x_aug[:4]
            D[i+1, :] = np.clip(x_aug[4:],0,None)
        return Tvec, X,D
    
    def full_output(self, T, X):
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
        
        return T, X, H, Qout
    
    def simulate(self):
        """Main simulation loop"""
        ts = self.ts
        xk_minus1 = self.x0
        dk_minus1 = self.d0
        uk_minus1 = self.u0

        Tfull = [ts[0]]
        Xfull = [xk_minus1]
        Dfull = [dk_minus1]
        Uk = [uk_minus1]
        
        Xk = [xk_minus1]
        Dk = [dk_minus1]
        
        #setup for sensor
        #measurement noise
        v = self.sensor_noise(N=len(ts))
        yk_minus1 = self.SystemSensor(xk_minus1)+ v[:,0]
        Yk = [yk_minus1]
            
        for k in range(1, len(ts)):
            tk = ts[k]
            tk_minus1 = ts[k-1]
            
            T_segment, X_segment, D_segment = self.step([tk_minus1,tk], xk_minus1, uk_minus1, dk_minus1)

            Tfull = np.concatenate((Tfull, T_segment[1:]))
            Xfull = np.vstack((Xfull, X_segment[1:]))
            Dfull = np.vstack((Dfull, D_segment[1:]))
                
            xk = X_segment[-1, :]
            uk = self.control(tk,yk_minus1) # control update
            yk = self.SystemSensor(xk) + v[:,k].T
            dk = D_segment[-1,:]
            
            
            #update
            xk_minus1 = xk
            uk_minus1 = uk.squeeze()
            yk_minus1 = yk
            dk_minus1 = dk.squeeze()
            
            #discrete samples
            Xk.append(xk)
            Dk.append(dk)
            Yk.append(yk)
            Uk.append(uk)
        
        #discrete states
        Xk = np.hstack([ts[:,None], Xk])
        Uk = np.hstack([ts[:,None], Uk])
        Dk = np.hstack([ts[:,None], Dk])
        Yk = np.array(Yk).squeeze() #messed up some dimensions here
        
        D = np.hstack([Tfull[:,None], Dfull])
        T, X, H, Qout = self.full_output(Tfull, Xfull)
        
        return T, X, H, Qout, D, Xk, Uk, Dk, Yk
        
    def sensor_noise(self, N, R=np.eye(2), Ny=2):
        'add to y'
        Lr = np.linalg.cholesky(R)
        e = np.random.randn(Ny,N)
        v = Lr @ e
        return v

class SimulationModelC(ModelSimulation):
    def __init__(self, 
                 ts: np.array, 
                 x0: np.array, 
                 u0: np.array, 
                 d0:np.array, 
                 p:np.array, 
                 disturbances: Tuple[DisturbanceModel,DisturbanceModel], 
                 dt_small: float=0.1,
                 control_schedule=None):
        
        super().__init__(ts, x0, u0, d0, p)
        self.disturbances = disturbances
        self.dt_small = dt_small
        
        self.control_schedule = control_schedule
        if control_schedule != None:
            self.c_times = control_schedule[0]
            self.c_inputs = control_schedule[1]
        
        
    def dynamics(self, t, x, u, d):
        return Modified_FourTankSystem_SDE(t, x, u, d, self.p, self.disturbances)
        
    def control(self, t, u):
        if self.control_schedule == None:
            return u
        else:
            idx = np.searchsorted(self.c_times, t, side="right") - 1
            idx = np.clip(idx, 0, len(self.c_inputs) - 1)
            
        u_new = np.array(self.c_inputs[idx])
        return u_new
    def disturbance(self, t, d, dmin=0, dmax=500):
        # d is stochastic here; actual d handled in solver via Brownian motion
        return d
        
    def step(self, tspan, x, u, d):
        Tvec = np.arange(tspan[0],tspan[1]+self.dt_small, self.dt_small)
        X = np.zeros((len(Tvec), len(x)))
        D = np.zeros((len(Tvec), len(d)))  # disturbance over time

        X[0, :] = x
        D[0, :] = d
        
        x_aug = np.hstack([x, d])  # shape (6,)

        for i in range(len(Tvec)-1):
            
            m = x_aug[:4]
            d = x_aug[4:]
            
            f, sigma = self.dynamics(Tvec[i], m, u, d)
            
            # Euler–Maruyama
            x_aug = x_aug + f*self.dt_small + sigma @ (np.sqrt(self.dt_small) * np.random.randn(2))
            
            X[i+1, :] = x_aug[:4]
            D[i+1, :] = np.clip(x_aug[4:],0,None)
        return Tvec, X,D
    
    def full_output(self, T, X):
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
        
        return T, X, H, Qout
    
    def simulate(self):
        """Main simulation loop"""
        ts = self.ts
        xk_minus1 = self.x0
        dk_minus1 = self.d0
        uk_minus1 = self.u0

        Tfull = [ts[0]]
        Xfull = [xk_minus1]
        Dfull = [dk_minus1]
        Uk = [uk_minus1]
        
        Xk = [xk_minus1]
        Dk = [dk_minus1]
        
        #setup for sensor
        #measurement noise
        v = self.sensor_noise(N=len(ts))
        yk_minus1 = self.SystemSensor(xk_minus1[:,None])+ v[:,0,None]
        Yk = [yk_minus1]
            
        for k in range(1, len(ts)):
            tk = ts[k]
            tk_minus1 = ts[k-1]
            
            T_segment, X_segment, D_segment = self.step([tk_minus1,tk], xk_minus1, uk_minus1, dk_minus1)

            Tfull = np.concatenate((Tfull, T_segment[1:]))
            Xfull = np.vstack((Xfull, X_segment[1:]))
            Dfull = np.vstack((Dfull, D_segment[1:]))
                
            xk = X_segment[-1, :]
            uk = self.control(tk,uk_minus1) # control update
            yk = self.SystemSensor(xk) + v[:,k,None]
            dk = D_segment[-1,:]
            
            
            #update
            xk_minus1 = xk
            uk_minus1 = uk.squeeze()
            yk_minus1 = yk
            dk_minus1 = dk
            
            #discrete samples
            Xk.append(xk)
            Dk.append(dk)
            Yk.append(yk)
            Uk.append(uk)
        
        #discrete states
        Xk = np.hstack([ts[:,None], Xk])
        Uk = np.hstack([ts[:,None], Uk])
        Dk = np.hstack([ts[:,None], Dk])
        Yk = np.array(Yk)
        
        D = np.hstack([Tfull[:,None], Dfull])
        T, X, H, Qout = self.full_output(Tfull, Xfull)
        
        return T, X, H, Qout, D, Xk, Uk, Dk, Yk
        
    def sensor_noise(self, N, R=np.eye(2)*2, Ny=2):
        'add to y'
        Lr = np.linalg.cholesky(R)
        e = np.random.randn(Ny,N)
        v = Lr @ e
        return v