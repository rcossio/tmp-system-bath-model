

import numpy as np

# parameters
T   = 1000.0                      # in K
Kb  = 1.3806488e-23              # in J/K 
Nbids = 64
hbar = 1.0545718e-34             # in m**2*kg/s
beta = 1/(Kb*T)
beta_n = beta/Nbids
w_n = 1/(beta_n*hbar)
m   = 1061.*9.10938356e-31
w_nm = 479.94683253*100
v = 1/(beta_n*m*w_nm**2)

#1450.6475722   903.28591365  648.5259131   479.94683253
for i in range(4000):
	x = -2e0+1e-3*i
	y = np.exp(-x**2/(2*v),dtype=np.float64)/np.sqrt(2*np.pi*v)
	print "%14.6g %14.6g" %(x,y)
