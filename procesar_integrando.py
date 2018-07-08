from scipy.sparse import diags
import numpy as np
import sys


#----------------------------------
#	Renaming math functions
#----------------------------------
pi  = np.pi
sin = np.sin
exp = np.exp
log = np.log10
sqrt = np.sqrt
tanh = np.tanh
cosh = np.cosh

#----------------------------------
#	Parameters
#----------------------------------
T    = 300.0                          # in K
Kb   = 3.1672083472e-06		      # in Eh/K 
m    = 1061.0                         # in m_e
a    = 0.734                          # in a0
dt   = 48e0                            # in Eh/hbar   con 47-48 son 1.15fs  originalmente con 8 va re bien
hbar = 1.0                            # in hbar
w_c  = 500  *4.5454545454545455e-06   # in Eh
w_b  = 500  *4.5454545454545455e-06   # in Eh
V0pp = 2085 *4.5454545454545455e-06   # in Eh

nsamples = 5000
nsteps   = 250
Nbids    = 4
outfile  = sys.argv[1]
f        = 10

#---------------------------------
#	Constants
#---------------------------------
beta = 1/(Kb*T)
beta_n = beta/Nbids
w_n = 1/(beta_n*hbar)
sigmap= sqrt(m/beta_n)
S_T = sqrt(2*pi*beta*hbar**2/m)
eta_over_mwb = 0.5
eta = eta_over_mwb*m*w_b
Freq = 10

#------------------------------------------
#	Functions
#------------------------------------------
def w(i):
        return w_c*np.log((i+1-1.5)/float(f-1))

def c(i):
        return w(i)*np.sqrt((2*eta*m*w_c)/(np.pi*(f-1)))

def report(string,filename=outfile):
        output = open(filename,'a')
        output.write(string)
        output.close()

def calc_q_n(w):
        q_n =1.0
        for l in range(1,Nbids+1):
                1/sqrt(4*sin(l*pi/Nbids)**2+(beta*hbar*w)**2)
        return q_n

#-----------------------------------------
#       Normal mode transformation
#-----------------------------------------
Hessian = np.zeros((f,f),dtype=np.float64)
Hessian[0,0] = -m*w_b**2
for i in range(1,f):
        Hessian[0,i] = -c(i)
        Hessian[i,0] = -c(i)
        Hessian[i,i] = m*w(i)**2
        Hessian[0,0] += c(i)**2/m/w(i)**2
evals, C = np.linalg.eigh(Hessian)

evals[1:] = evals[1:][::-1]
C[:,1:] = C[:,1:][:,::-1]
if C[0,0] < 0.0:
        C = -C
w_nm=np.sqrt(evals/m)

#-----------------------------------------
#	Calculating Np
#-----------------------------------------
Np = 1/S_T
for k in range(1,f):
	Np *= calc_q_n(w_nm[k])

#-----------------------------------------------------
#	Calculating Qr (previo normal mode analysis)
#-----------------------------------------------------
HessianR = np.zeros((f,f),dtype=np.float64)
HessianR[0,0] = 2*m*w_b**2
for i in range(1,f):
        HessianR[0,i] = -c(i)
        HessianR[i,0] = -c(i)
        HessianR[i,i] = m*w(i)**2
        HessianR[0,0] += c(i)**2/m/w(i)**2
evals, Cr = np.linalg.eigh(HessianR)

evals = evals[::-1]
Cr = Cr[:,::-1]
w_nm_reactivo=np.sqrt(evals/m)
Qr = exp(beta*V0pp)
for k in range(f):
        Qr *= calc_q_n(w_nm_reactivo[k])


#----------------------------------------------------
#	Conseguir integrando desde archivo
#----------------------------------------------------
weighted = []
for line in open(sys.argv[2]):
	weighted.append(float(line.split()[2]))

weigthed = np.array(weighted,dtype=np.float64)
C_t = Np * np.mean (weighted)
k_t = C_t / Qr
k_CL = w_b*exp(-beta*V0pp)/pi/sqrt(2)
report("# %14.6g %14.6g %14.6g \n" %(T,eta_over_mwb,k_t/k_CL))

