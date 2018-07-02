#grep -v '#' nbids.2000.dat | awk '{n=n+1; S=S+$1}END{print S/n}'
from scipy.sparse import diags
import numpy as np
import sys


#----------------------------------
#	Renaming math functions
#----------------------------------
pi  = np.pi
exp = np.exp
log = np.log10
sqrt = np.sqrt
tanh = np.tanh
cosh = np.cosh

#----------------------------------
#	Parameters
#----------------------------------
T    = 1000.0                 # in K
Kb   = 1.3806488e-23          # in J/K 
V0   = 0.425*1.60218e-19      # in J
m    = 1061.*9.10938356e-31   # in kg
a    = 0.734*5.2918e-11       # in m
dt   = 20*1e-18               # in s
hbar = 1.0545718e-34          # in m**2*kg/s
w_c  = 500 *1e2               # in 1/m
w_b  = 500 *1e2               # in 1/m
V0pp = 2085*1e2		      # in 1/m
nsamples = 150000
Nbids    = 4
outfile  = sys.argv[1]
f=4

#---------------------------------
#	Constants
#---------------------------------
beta = 1/(Kb*T)
beta_n = beta/Nbids
w_n = 1/(beta_n*hbar)
sigmap= sqrt(m/beta_n)
S_T = sqrt(2*pi*beta*hbar**2/m)
Q = 1./S_T
N_p = 1./S_T
eta = 1.0*m*w_b

#--------------------------------
#	Functions
#--------------------------------
def w(i):
        return w_c*np.log((i+1-1.5)/float(f-1))

def c(i):
        return w(i)*np.sqrt((2*eta*m*w_c)/(np.pi*(f-1)))

def calcForce (q):
	num = 2*V0*tanh(q/a)
	den = a * (cosh(q/a))**2
	return num/den

def calc_derivada_p (q):
	derivada = calcForce(q)
	for j in range(Nbids):
		if   ( j == 0 ):
                        derivada[j] += -m*w_n**2*(2*q[j]-q[Nbids-1]-q[j+1])
		elif ( j == Nbids-1):
                        derivada[j] += -m*w_n**2*(2*q[j]-q[j-1]    -q[0]  )
		else:
			derivada[j] += -m*w_n**2*(2*q[j]-q[j-1]    -q[j+1])

	return derivada
	
	
def calcPotential (q):
	return np.sum(V0/(cosh(q/a))**2)

def heaviside_n(q):
        return np.mean(np.heaviside(q, 1.0))

def report(string):
        output = open(outfile,'a')
        output.write(string)
        output.close()

#-----------------------------------------
#	Normal mode transformation
#-----------------------------------------
Hessian = np.zeros((f,f),dtype=np.float64)
Hessian[0,0] = -m*w_b**2
for i in range(1,f):
	Hessian[0,i] = -c(i)
	Hessian[i,0] = -c(i)
	Hessian[i,i] = m*w(i)**2
evals, C = np.linalg.eigh(Hessian) 

# Comprobamos que esta bien diagonalizado
#print Hessian
#print C.dot(np.diag(evals).dot(C.T))

evals[1:] = evals[1:][::-1]
C[:,1:] = C[:,1:][:,::-1]
w_nm=np.sqrt(evals/m)

# Vemos autovalores
#print evals
#print C.T.dot(Hessian.dot(C))
#print w_nm/100.

# Vemos autovectores
#print C
#exit()
#-----------------------------------------
#	Cholesky matrix
#-----------------------------------------
CholeskyList = []
Ndim = Nbids-1
for k in range(f):
	if k == 0:
		factor = 0
	else:
		factor = (w_nm[k]/w_n)**2
	tmp = diags([-1, 2+factor, -1], [-1, 0, 1], shape=(Ndim, Ndim)).toarray()
	Qcov = np.linalg.inv(tmp)
	L = np.linalg.cholesky(Qcov)
	L /= L[0,0]
	L /= np.sqrt(beta_n*m*w_n**2)
	CholeskyList.append(L)

#-------------------------------
#	Initial sampling
#-------------------------------
z   = []
v_s = []
bf  = []
weighted = np.zeros(nsamples,dtype=np.float64)

for s in range(nsamples/2):
	#---------------------------------------
	#	Center of mass sampling
	#---------------------------------------
        cm = np.zeros(f,dtype=np.float64)
	for k in range(1,f):
		sigmacm = 1./np.sqrt(beta_n*m*w(k)**2)
		cm[k]   = np.random.normal(loc=0.0,scale=sigmacm) 
	cm = C.dot(cm)

	#------------------------------
	#	Random sampling
	#------------------------------
	q = np.zeros((f,Nbids),dtype=np.float64)
	for k in range(f):
		r = np.random.normal(loc=0.0,scale=1.0,size=Ndim)
		q[k,1:Nbids] = CholeskyList[k].dot(r)
		q[k,:] += -np.mean(q[k,:])+cm[k]
	p = np.random.normal(loc=0.0,scale=sigmap,size=Nbids*f).reshape((f,Nbids))

	print q
	exit()
	# HASTA ACA MAOMENO MIRE...
	# seguro falta todo lo del centroide
	
	#-------------------------------
	#	Sampling	
	#-------------------------------
	z.append((q,p))
	v_s.append(p[0]/m)
        deltav = calcPotential(q)
        bf.append( exp(-beta_n*deltav))

	#-------------------------------
	#	Symmetric Sampling
	#------------------------------
	p = -p
        z.append((q,p))
        v_s.append(p[0]/m)
        deltav = calcPotential(q)
        bf.append( exp(-beta_n*deltav))

v_s = np.array(v_s, dtype=np.float64)
bf  = np.array(bf,  dtype=np.float64)

exit()
#------------------------------------
#	Trajectories
#------------------------------------
for s in range(nsamples):
	h_n = 0.5
	t = 0
	while 0.0+1e-12 < h_n < 1.0-1e-12:
		
		#---------------------------------------------------
		#	Initial state
		#---------------------------------------------------
		if t == 0:
			(q,p) = z[s]

		#---------------------------------------------------
		#	Symplectic integrator - velocity verlet
		#---------------------------------------------------
		derivada_p = calc_derivada_p(q)
		p = p + 0.5 * dt * derivada_p
		q = q + 1.0 * dt * p/m
		derivada_p = calc_derivada_p(q)
		p = p + 0.5 * dt * derivada_p
		t += 1

		#---------------------------------------------------
		#	h_n(t) calculation
		#---------------------------------------------------
		if t%10 == 0:
			h_n = heaviside_n(q)

	#---------------------------------------------------------
	#	Find wight h_n
	#---------------------------------------------------------
        weighted [s] = bf[s] * v_s[s] * h_n
	report("%18.10g \n" %weighted[s])

#---------------------------------------------
#	Find transmission coefficient
#---------------------------------------------
C_t = N_p * np.mean (weighted)
k_t = C_t / Q
report("# %14.6g %14.6g %14.6g %14.6g \n" %(T,1000/T,k_t,log(k_t)))
