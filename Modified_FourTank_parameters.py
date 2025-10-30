import numpy as np
# Parameters
a1, a2, a3, a4 = 1.2272, 1.2272, 1.2272, 1.2272      # [cm^2] Outlet pipe areas
A1, A2, A3, A4 = 380.1327, 380.1327, 380.1327, 380.1327  # [cm^2] Tank cross-sectional areas
g = 981.0                                            # [cm/s^2] Gravity
gamma1, gamma2 = 0.58, 0.72                          # [-] Valve positions
rho = 1.0                                            # [g/cm^3] Density of water
# Pack parameters vector p
p = np.array([a1, a2, a3, a4, A1, A2, A3, A4, gamma1, gamma2, g, rho])
