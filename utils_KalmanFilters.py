import numpy as np
from utils_functions import *
from utils_functions import linearize_system, discretize_system, steady_state, find_equilibrium
from utils_Modified_FourTank_functions import Modified_FourTankSystem, sim22

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

    def one_step(self, y, u):
        self.update(y,u)
        x_k_p1 = self.A@self.x+self.B@u+ self.G@self.w
        return x_k_p1
    
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
    
    def one_step(self, y, u):
        self.update(y,u)
        x_k_p1 = self.A@self.x+self.B@u+ self.G@self.w
        return x_k_p1
    
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
    
if __name__ == '__main__':  
    print('''
Description of what is outputted when running this script:

A nonlinear continuous-time simulation for the Modified Tank System is performed, where the disturbances are piecewise constant 
and increments away from mean [F3,F4] = [100,100] drawn from a normal distribution with variance of 100 (If disturbances are 
drawn to be negative, the value is clipped to 0)

Then the Kalman Filters, static and dynamic, are tested around the operating point using [F1,F2] = [300,300], initial value and mean [F3,F4] = [100,100].
The specific operating point is printed.


          ''')
    
    
      
    #Parameters
    a1 = 1.2272 # [cm2] Area of outlet pipe 1
    a2 = 1.2272
    a3 = 1.2272
    a4 = 1.2272

    A1 = 380.1327 #[cm2] Cross sectional area of tank 1
    A2 = 380.1327
    A3 = 380.1327
    A4 = 380.1327

    gamma1 = 0.58 # Flow distribution constant. Valve 1
    gamma2 = 0.72 # Flow distribution constant. Valve 2

    g = 981 #[cm/s2] The acceleration of gravity
    rho = 1.00 #[g/cm3] Density of water

    p = np.array([a1,a2,a3,a4, A1,A2,A3,A4, gamma1,gamma2, g, rho])
    
    # Initial liquid levels [cm]
    h10, h20, h30, h40 = 0.0, 0.0, 0.0, 0.0
    # Convert levels to mass [g] => m = rho * A * h
    m10 = rho * A1 * h10
    m20 = rho * A2 * h20
    m30 = rho * A3 * h30
    m40 = rho * A4 * h40

    x0 = np.array([m10, m20, m30, m40])  # Initial states
    u0 = np.array([300,300])
    d0 = np.array([0,0])
    
    dt = 30
    ts = np.arange(0,30*60+dt,dt)
    N = len(ts)  # number of intervals
    nx = len(x0)
    F1 = np.ones_like(ts)*300
    F2 = np.ones_like(ts)*300
    F3 = np.ones_like(ts)*100
    F4 = np.ones_like(ts)*100
    u = np.array([F1,F2])
    d = np.array([F3,F4])
    
    xs = find_equilibrium(Modified_FourTankSystem, np.array([10000, 10000, 10000, 10000]), np.array([300, 300]), np.array([100, 100]), p)
    Ac, Bc, Bdc, Cc, Dc = linearize_system(Modified_FourTankSystem, FourTankSystemSensor, xs, np.array([300, 300]), np.array([100, 100]), p)

    #combine u and d
    Bc = np.block([Bc, Bdc])
    Dc = np.zeros((2,4)) #this is zero in our case

    # #discretize 
    _, A, B, C, D = discretize_system(Ac,Bc,Cc,Dc, Ts=dt)   
    Bu = B[:,:2]
    Bd = B[:,2:] 
        
    xs, us, ys, ds = steady_state(Modified_FourTankSystem, np.array([10000, 10000, 10000, 10000]), np.array([300, 300]), np.array([100, 100]),p)

    print(f'''
    Steady-state operating point is
    xs: {xs}
    us: {us}
    ys: {ys}
    ds: {ds}''')
    
    xs = xs.squeeze()
    us = np.tile(us, (1, 361)) 
    # us[:,100:] += np.array([100,-100])[:,None]
    ds = np.tile(ds, (1, 361)) 

    u =us
    times = [0,240]
    vals = [[300,300],[500,300]]
    u = []
    for t in ts:
        idx = np.searchsorted(times, t, side="right") - 1
        idx = np.clip(idx, 0, len(vals) - 1)
        
        u.append(vals[idx])        
    u = np.array(u).T
    v_noise = 5
    x, y, z, T_all, X_all, H_all = sim22(ts, xs, u, ds, p, d_noise_level=2, v_noise_level=v_noise, plot=False)
    
    #Noise matrices (change these, especially Rww)
    temp = expm(np.block([[-Ac, Bdc@Bdc.T],[np.zeros_like(Ac),Ac.T]])*dt)
    phi11 = temp[:4,:4]
    phi12 = temp[:4,4:]
    phi22 = temp[4:,4:]

    Q = phi22.T@phi12
    
    G = np.eye(4)
    Rww = np.eye(4)*0.1  #process noise covariance
    Rvv = np.eye(2)*v_noise

    Rwv = np.zeros((4,2)) 
    Rvw = Rwv.T            # 2x4 cross covariance 

    staticKF = StaticKalmanFilter(A,Bu, C ,G,Rww,Rvv,Rwv,P0=np.eye(4)*100, x0=x0)

    # #KF predictions
    predictions = []
    ypredictions = []
    for uk, yk in zip(u.T,y.T):
        uk = uk[:,None]-us[:,0,None]
        yk = yk[:2,None]-ys
    
        staticKF.update(yk, uk) #update to estimate x[k|k]
        xhat, yhat = staticKF.x, staticKF.y #Predict x[k+1|k]
        
        predictions.append(xhat+xs[:,None])
        ypredictions.append(yhat+ys)
    static_predictions = np.array(predictions).squeeze()
    static_ypredictions = np.array(ypredictions).squeeze()

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Static KF')
    for i in range(4):
        ax = axs[i//2, i%2]     # select subplot

        ax.plot(T_all, X_all[:,i], label='continuous', color='orange')
        ax.plot(ts, x[i,:], 'x', label='discrete', color='green')
        ax.plot(ts, static_predictions[:,i], 'r.', label='KF predictions')

        ax.set_title(f"Tank {i+1}")
        # ax.set_ylim(0, None)
        ax.grid(True)
        
    # Only one legend for all subplots:
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Static KF')
    for i in range(2):
        ax = axs[i]

        ax.plot(ts, y[i,:], label='Y true', color='orange')
        ax.plot(ts, static_ypredictions[:,i], 'r.', label='KF predictions')

        ax.set_title(f"Tank {i+1}")
        # ax.set_ylim(0, None)
        ax.grid(True)
        
    # Only one legend for all subplots:
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    N = 8
    u_pred = u[:N,:N].reshape(-1,1) #u to shape (N*2,1)
    Zk = staticKF.j_step(u=u_pred, N=N)
    Zk = Zk.reshape(N, 2)
    print(Zk)

    dynamicKF = DynamicKalmanFilter(A,Bu, C ,G,Rww,Rvv,Rwv,P0=np.eye(4)*100, x0=x0)

    #KF predictions
    predictions = []
    ypredictions = []
    for uk, yk in zip(u.T,y.T):
        uk = uk[:,None]-us[:,0,None]
        yk = yk[:2,None]-ys

        dynamicKF.update(yk, uk) #update to estimate x[k|k]
        xhat, yhat = dynamicKF.x, dynamicKF.y #Predict x[k+1|k]
        
        predictions.append(xhat+xs[:,None])
        ypredictions.append(yhat+ys)
    dynamic_predictions = np.array(predictions).squeeze()
    dynamic_ypredictions = np.array(ypredictions).squeeze()

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Dynamic KF')
    for i in range(4):
        ax = axs[i//2, i%2]     # select subplot

        ax.plot(T_all, X_all[:,i], label='continuous', color='orange')
        ax.plot(ts, x[i,:], 'x', label='discrete', color='green')
        ax.plot(ts, dynamic_predictions[:,i], 'r.', label='KF predictions')

        ax.set_title(f"Tank {i+1}")
        # ax.set_ylim(0, None)
        ax.grid(True)
        
    # Only one legend for all subplots:
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Dynamic KF')
    for i in range(2):
        ax = axs[i]

        ax.plot(ts, y[i,:], label='Y true', color='orange')
        ax.plot(ts, dynamic_ypredictions[:,i], 'r.', label='KF predictions')

        ax.set_title(f"Tank {i+1}")
        # ax.set_ylim(0, None)
        ax.grid(True)
        
    # Only one legend for all subplots:
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
        
    N = 8
    u_pred = u[:N,:N].reshape(-1,1) #u to shape (N*2,1)
    Zk = dynamicKF.j_step(u=u_pred, N=N)
    Zk = Zk.reshape(N, 2)
    print(Zk)
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Static and dynamc KF')
    for i in range(2):
        ax = axs[i]

        ax.plot(ts, y[i,:], 'o', label='Y true', color='blue', markerfacecolor='none')
        ax.plot(ts, static_ypredictions[:,i], '-', label='Y static KF', color = 'red')
        ax.plot(ts, dynamic_ypredictions[:,i], '--', label='Y dynamic KF', color = 'black')
        ax.set_title(f"Tank {i+1}")
        # ax.set_ylim(0, None)
        ax.grid(True)
        
    # Only one legend for all subplots:
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()