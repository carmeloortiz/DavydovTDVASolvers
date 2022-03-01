import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.integrate
import cmath

#--------------------------------------------------------------#
#CONSTANTS
h     = (4.136E-15) * 10**(-9) # (in GeV * s)
h_bar = h/(2*np.pi)          # (in GeV * s)
c     = 3E8                  # (in m/s)

#--------------------------------------------------------------#
#PARAMETERS

#global parameters
N = 2    #number of sites
Q = 2000 #number of modes
W = 2000 * 100 * h * c #phonon bandwidth (in GeV)

#more parameters
Ham = np.zeros((N,N),dtype = complex)  #Hamiltonian (in GeV)
w   = np.zeros((Q), dtype = complex)    #angular frequencies (in GeV)
g   = np.zeros((N,Q), dtype = complex)  #coupling strengths (dimensionless)

#Hamiltonian parameters
J     =    -100 * 100 * h * c  # (in GeV)
Delta =     100 * 100 * h * c  # (in GeV)

#spectral density parameters
gamma = (1/10) * 10**15 * h_bar #relaxation rate (in GeV)
org_energy = 100 * 100 * h * c  #reorganization energy (in GeV)

#--------------------------------------------------------------#
#FUNCTIONS

# system of ODEs corresponding to the D1 Ansatz
# returns d_lamda_n_q / dt and d_alpha_n / dt for a given time t, for each n and q
def D1_ODE_system(t, yc):
    #convention: yc is a 1D array of length N*(Q+1). The values corresponding to lambda_n_q come first, followed by the value of alpha_n.
    #convention: y is an array with shape (N,Q+1). The values corresponding to lambda_n_q come in the first Q columns. The last column is alpha_n.
    y = np.reshape(yc,(N,Q+1))

    #set 2D array
    dlambdanq_dt = np.zeros((N,Q),dtype = complex)
    dalphan_dt = np.zeros((N), dtype = complex)

    #calculating Debye-waller factor before main iteration
    S = np.zeros((N,N),dtype = complex)
    for n in range(N):
        for m in range(N):
            summand = [y[n][qq].conjugate()*y[m][qq]  - 0.5*(y[n][qq].conjugate()*y[n][qq]+y[m][qq].conjugate()*y[m][qq]) for qq in range(Q)]
            S[n][m] = cmath.exp(sum(summand))  #S[m][n] = cmath.exp(sum(summand))

    i=0
    #looping over d_lambda_n_q / dt
    n = 0
    while n < N:
        q = 0
        while q < Q:
            summand1a = [Ham[m][n]*S[n][m]* ((y[n][Q].conjugate()*y[m][Q])/(y[n][Q].conjugate()*y[n][Q])) * (y[m][q]- 0.5*y[n][q]) for m in range(N)]
            summand1b = [Ham[m][n]*S[m][n]* ((y[n][Q].conjugate()*y[m][Q])/(y[n][Q].conjugate()*y[n][Q])) * (- 0.5*y[n][q]) for m in range(N)]
            summand2a = [Ham[m][n]*S[n][m].conjugate()*y[m][Q].conjugate() for m in range(N)]
            summand2b = [Ham[m][n]*S[n][m]*y[m][Q] for m in range(N)]

            term_1a = -1j*sum(summand1a)
            term_1b = -1j*sum(summand1b)
            term_2 = -1j*w[q]*y[n][q]
            term_3 = -1j*g[n][q]*w[q]
            term_4 = (-1j/2) * (y[n][q]/(y[n][Q].conjugate()*y[n][Q])) * (y[n][Q]*sum(summand2a) - y[n][Q].conjugate()*sum(summand2b))

            dlambdanq_dt[n][q] =  term_1a + term_1b + term_2  + term_3 + term_4
            q+=1
        n+=1

    #looping over d_alpha_n / dt
    n = 0
    while n < N:
        #first term
        summand1 = [y[n][qq].conjugate()*dlambdanq_dt[n][qq] - y[n][qq]*dlambdanq_dt[n][qq].conjugate() for qq in range(Q)]
        term_1 = (-1/2)*y[n][Q]*sum(summand1) 

        #second term
        summand2 = [Ham[m][n]*y[m][Q]*S[n][m] for m in range(N)]
        term_2 = -1j*sum(summand2)

        #third term
        summand3 = [w[qq]*y[n][qq]*y[n][qq].conjugate() for qq in range(Q)]
        term_3 = -1j*y[n][Q]*sum(summand3)

        #fourth term
        summand4 = [g[n][qq]*w[qq]*(y[n][qq].conjugate() + y[n][qq]) for qq in range(Q)]
        term_4 = -1j*y[n][Q]*sum(summand4)

        #final result
        dalphan_dt[n] = term_1 + term_2 + term_3 + term_4 
        #print(' t1: {t1:.5E} t2:{t2:.5E} t3: {t3:.5E} t4: {t4:.5E}'.format(t1 = abs(term_1),t2 = abs(term_2),t3 = abs(term_3),t4 = abs(term_4)))
        #print(' t1: {t1:.5E} t2:{t2:.5E} t3: {t3:.5E} t4: {t4:.5E}'.format(t1 = term_1,t2 = term_2,t3 = term_3,t4 = term_4))
        n+=1

    #return d_lamda_n_q / dt and d_alpha_n / dt
    dydt = np.zeros((N,Q+1),dtype = complex)
    dydt[:,0:Q] = dlambdanq_dt
    dydt[:,Q] = dalphan_dt

    print('|alpha|^2: {alpha_sq:.5f} time: {time:.5f}'.format(alpha_sq = abs(y[1,Q])**2,time = (t/t_f)*100))
    return np.reshape(dydt,(N*(Q+1)))

#generates angular frequencies of modes
def spawn_w():
    delta_w = W/Q
    for q in range(Q):
        w[q] = delta_w * (q+1)
    return 0

#generates coupling strengths
def spawn_g():
    q = 0
    while q < 1000:
        g[0][q] = (C_pp(0,w[q])/w[q]**2)**0.5
        g[1][q] = 0
        q+=1
    
    while q < Q:
        g[0][q] = 0
        g[1][q] = (C_pp(1,w[q])/w[q]**2)**0.5
        q+=1
    
    return 0

#normalized spectral density function for excitonic polaron case
def C_pp(n,omega):
    #calculate normalization factor
    if n==0:
        summand = [(4*gamma**3)/(w[qq]**2 + gamma**2)**2 for qq in range(0,int(Q/2))]
    elif n==1:
        summand = [(4*gamma**3)/(w[qq]**2 + gamma**2)**2 for qq in range(int(Q/2),Q)]
    k = 1/sum(summand)

    #calculate spectral density
    C = (4*org_energy*(gamma**3)*omega)/(omega**2+gamma**2)**2

    print(k)
        
    return k*C

#--------------------------------------------------------------#
#SIMULATION

#CASE 1: Excitonic Polarons

#generate Hamiltonian, angular velocities, coupling strengths
Ham[0][0] = 0
Ham[0][1] = J
Ham[1][0] = J
Ham[1][1] = Delta
spawn_w()
spawn_g()

#initial values
t_i = 0
alpha_1 = -0.526
alpha_2 =  0.851

lambda_i = np.zeros((N,Q),dtype = complex)
#lambda_i = np.random.uniform(-1E-5,1E-5,(N,Q)).astype(complex)
alpha_i = np.array([alpha_1, alpha_2])

y_ic = np.zeros((N,Q+1), dtype = complex)
y_ic[:,0:Q] = lambda_i
y_ic[:,Q] = alpha_i
y_i = np.reshape(y_ic,(N*(Q+1)))

#time (in GeV ^-1)
t_f = 1200E-15 * (1/h_bar)
time_points = 100 * 100
t = np.arange(0,t_f + t_f/time_points,t_f/time_points)

#solve the ODE
sol = solve_ivp(D1_ODE_system, [0,t_f], y_i, t_eval = t) #, method='DOP853', rtol = 1E-11, atol = 1E-14

#--------------------------------------------------------------#
#PLOTTING

fig = plt.figure()

ax1 = fig.add_subplot(311)
ax1.plot(t*h_bar * 1E15, abs(sol.y[1*Q+0][:])**2)
ax1.set_ylabel(r"|$\alpha$₁|$^2$   ",rotation = 'horizontal')
ax1.set_xlabel("t (fs)")
ax1.set_ylim(0,1)

ax2 = fig.add_subplot(312)
ax2.plot(t*h_bar * 1E15, abs(sol.y[2*Q+1][:])**2)
ax2.set_ylabel(r"|$\alpha$₂|$^2$   ",rotation = 'horizontal')
ax2.set_xlabel("t (fs)")
ax2.set_ylim(0.55,1)

ax3 = fig.add_subplot(313)
ax3.plot(t*h_bar * 1E15, abs(sol.y[499][:])**2)
ax3.set_ylabel(r"|$\lambda$₅₀₀|$^2$       ",rotation = 'horizontal')
ax3.set_xlabel("t (fs)")
ax3.ticklabel_format(axis = 'y',style = 'sci',scilimits = (-1,1))

fig.tight_layout()

plt.show()

"""
#--------------------------------------------------------------#
TESTING

#initialize displacements with a noise factor to avoid singularity
lambda_i = np.random.uniform(-1E-5,1E-5,(N,Q)).astype(complex)
"""