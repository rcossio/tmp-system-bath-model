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
T    = 200.0                          # in K
Kb   = 3.1672083472e-06		      # in Eh/K 
m    = 1061.0                         # in m_e
a    = 0.734                          # in a0
dt   = 8e0                            # in Eh/hbar
hbar = 1.0                            # in hbar
w_c  = 500  *4.5454545454545455e-06   # in Eh
w_b  = 500  *4.5454545454545455e-06   # in Eh
V0pp = 2085 *4.5454545454545455e-06   # in Eh


nsamples = 1000
nsteps = 300000
Nbids    = 4
outfile  = sys.argv[1]
f=10

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
        F = np.zeros((f,Nbids),dtype=np.float64)
	F[0,:] = m* w_b**2 *q[0,:] - m**2 *w_b**4* q[0,:]**3/(4*V0pp) 
	for k in range(1,f):
		F[0,:] += c(k)*(q[k,:]-c(k)*q[0,:]/m/w(k)**2) 
		F[k,:]  = -m* w(k)**2 *q[k,:] +c(k)*q[0,:]

#	Esto es una mentira: pongo una fuerza armonica para ver como se mueve el polimero
#	for u in range(f):
#		for v in range(Nbids):
#			F[u,v] = - m* w(1)**2 *q[u,v]
	return F

def calc_derivada_p (q):
	derivada = calcForce(q)
	for j in range(Nbids):
		if   ( j == 0 ):
                        derivada[:,j] += -m*w_n**2*(2*q[:,j]-q[:,Nbids-1]-q[:,j+1])
		elif ( j == Nbids-1):
                        derivada[:,j] += -m*w_n**2*(2*q[:,j]-q[:,j-1]    -q[:,0]  )
		else:
			derivada[:,j] += -m*w_n**2*(2*q[:,j]-q[:,j-1]    -q[:,j+1])
	return derivada
	
	
def calcPotential (q):
	V = 0.0
	for i in range(Nbids):
		V += -0.5* m* w_b**2 *q[0,i]**2 + m**2 *w_b**4* q[0,i]**4/(16*V0pp)
		for k in range(1,f):
			V +=  0.5* m* w(k)**2 *(q[k,i]-c(k)*q[0,i]/(m*w(k)**2))**2

#       Esto es una mentira: pongo una fuerza armonica para ver como se mueve el polimero
#        for v in range(Nbids):
#                for u in range(f):
#                        V +=  0.5* m* w(1)**2 *q[u,v]**2

	return V

def extendedPotential(q):
	H = calcPotential(q)
	for k in range(f):
	        for j in range(Nbids):
	                if   ( j == 0 ):
	                        H += 0.5*m*w_n**2*(q[k,j]-q[k,Nbids-1])**2
	                else:
	                        H += 0.5*m*w_n**2*(q[k,j]-q[k,j-1])**2
	return H


def deltaV(q):
	deltav = calcPotential(q)
	for k in range(1,f):
		deltav -= 0.5 * m * w_nm[k]**2 * np.linalg.norm(q[i,:])**2
	return deltav

def heaviside(q):
	qq = C.T.dot(q)
        return np.heaviside(np.mean(qq[0,:]), 1.0)

def heaviside_n(q):
        qq = C.T.dot(q)
        return np.mean(np.heaviside(qq[0,:], 1.0))

def report(string,filename=outfile):
        output = open(filename,'a')
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

evals[1:] = evals[1:][::-1]
C[:,1:] = C[:,1:][:,::-1]
if C[0,0] < 0.0:
	C = -C
w_nm=np.sqrt(evals/m)

#-----------------------------------------
#	Cholesky matrices
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
	#---------------------------------------------------------
	#	Center of mass sampling in original coordinates
	#----------------------------------------------------------
        cm = np.zeros(f,dtype=np.float64)
#	for k in range(1,f):
#		sigmacm = 1./np.sqrt(beta_n*m*w_nm[k]**2)
#		cm[k]   = np.random.normal(loc=0.0,scale=sigmacm)
#	cmqq = C.T.dot(cm)

	#------------------------------
	#	Random sampling
	#------------------------------
	qq = np.zeros((f,Nbids),dtype=np.float64)
	for k in range(f):
		r = np.random.normal(loc=0.0,scale=1.0,size=Ndim)
		qq[k,1:Nbids] = CholeskyList[k].dot(r)
		qq[k,:] += -np.mean(qq[k,:]) + cm[k] 
	p = np.random.normal(loc=0.0,scale=sigmap,size=Nbids*f).reshape((f,Nbids))

        q = C.dot(qq)

	#-------------------------------
	#	Sampling	
	#-------------------------------
	z.append((q,p))
	v_s.append(np.mean(p[0,:])/m)
        bf.append( exp(-beta_n*deltaV(q)))

	#-------------------------------
	#	Symmetric Sampling
	#------------------------------
	p = -p
        z.append((q,p))
        v_s.append(np.mean(p[0,:])/m)
        bf.append( exp(-beta_n*deltaV(q)))

v_s = np.array(v_s, dtype=np.float64)
bf  = np.array(bf,  dtype=np.float64)

#-----------------------------------
#	Trajectories
#------------------------------------
for s in range(nsamples):
	t = 0
	while t < nsteps:
		
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

		if t%10 == 0:
			this_cm = np.mean(q,axis=1)
                        report("%14.6g %14.6g %14.6g %14.6g %14.6g %14.6g %14.6g %14.6g %14.6g %14.6g \n"%tuple(this_cm))

                        this_cm = np.mean(C.T.dot(q),axis=1)
                        report("%14.6g %14.6g %14.6g %14.6g %14.6g %14.6g %14.6g %14.6g %14.6g %14.6g \n"%tuple(this_cm),filename=sys.argv[2])

			report("%14.6g %14.6g %14.6g\n" %(extendedPotential(q),np.sum(p**2/2/m),extendedPotential(q)+np.sum(p**2/2/m)),filename=sys.argv[3])


#			report("%10i %14.6g %14.6g %14.6g %14.6g %14.6g %14.6g %14.6g %14.6g %14.6g %14.6g %14.6g %14.6g \n"%(t,q[0,0],q[1,0],q[2,0],q[0,1],q[1,1],q[2,1],q[0,2],q[1,2],q[2,2],q[0,3],q[1,3],q[2,3]))
#			report("%14.6g %14.6g\n" %(heaviside_n(q),heaviside(q)))
		
	exit()
	#---------------------------------------------------------
	#	Find weight h_n
	#---------------------------------------------------------
        weighted [s] = bf[s] * v_s[s] * heaviside(q)
	report("%18.10g \n" %weighted[s])

#---------------------------------------------
#	Find transmission coefficient
#---------------------------------------------
C_t = N_p * np.mean (weighted)
k_t = C_t / Q
report("# %14.6g %14.6g %14.6g %14.6g \n" %(T,1000/T,k_t,log(k_t)))

