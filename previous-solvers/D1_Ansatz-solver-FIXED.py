import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
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
Q = 2*1000 #number of modes per site
W = 2000 * 100 * h * c #phonon bandwidth (in GeV)

#more paramters
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
    #convention: yc is an 1D array of length N*(Q+1). The values corresponding to lambda_n_q come first, followed by the value of alpha_n.
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
            S[n][m] = cmath.exp(sum(summand))

    i=0
    #looping over d_lambda_n_q / dt
    n = 0
    while n < N:
        q = 0
        while q < Q:
            summand1  = [Ham[m][n]*S[n][m]* ((y[n][Q].conjugate()*y[m][Q])/(y[n][Q].conjugate()*y[n][Q])) * (y[m][q]- 0.5*y[n][q]) for m in range(N)]
            summand2a = [Ham[m][n]*S[n][m].conjugate()*y[m][Q].conjugate() for m in range(N)]
            summand2b = [Ham[m][n]*S[n][m]*y[m][Q] for m in range(N)]

            term_1 = -1j*sum(summand1)
            term_2 = -1j*w[q]*y[n][q]
            term_3 = -1j*g[n][q]*w[q]#+1j*g[n][q]*w[q]
            term_4 = (-1j/2) * (y[n][q]/(y[n][Q].conjugate()*y[n][Q])) * (y[n][Q]*sum(summand2a) - y[n][Q].conjugate()*sum(summand2b))

            dlambdanq_dt[n][q] =  term_1 + term_2  + term_3 + term_4
            q+=1

            """
            if(q==500):
                print(' t1: {t1:.5E} t2:{t2:.5E} t3: {t3:.5E} t4: {t4:.5E}'.format(t1 = term_1,t2 = term_2,t3 = term_3,t4 = term_4))
            """
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
        term_4 = -1j*y[n][Q]*sum(summand4)#+1j*y[n][Q]*sum(summand4)

        #final result
        dalphan_dt[n] =   term_2  + term_1  + term_3 + term_4 
        #print(' t1: {t1:.5E} t2:{t2:.5E} t3: {t3:.5E} t4: {t4:.5E}'.format(t1 = abs(term_1),t2 = abs(term_2),t3 = abs(term_3),t4 = abs(term_4)))
        #print(' t1: {t1:.5E} t2:{t2:.5E} t3: {t3:.5E} t4: {t4:.5E}'.format(t1 = term_1,t2 = term_2,t3 = term_3,t4 = term_4))
        n+=1

    #return d_lamda_n_q / dt and d_alpha_n / dt
    dydt = np.zeros((N,Q+1),dtype = complex)
    dydt[:,0:Q] = dlambdanq_dt
    dydt[:,Q] = dalphan_dt

    print('|alpha|^2:',y[1,Q]*y[1,Q].conjugate(),"time:",(t/t_f)*100)
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
        g[0][q] = (C_pp(w[q])/w[q]**2)**0.5
        q+=1
    
    while q < Q:
        g[0][q] = 0
        q+=1

    q = 0
    while q < 1000:
        g[1][q] = 0
        q+=1
    
    while q < Q:
        g[1][q] = (C_pp(w[q])/w[q]**2)**0.5
        q+=1
    
    return 0

#normalized spectral density function for excitonic polaron case
def C_pp(omega):
    #calculate normalization factor
    summand = [(4*gamma**3)/(w[qq]**2 + gamma**2)**2 for qq in range(Q)]
    k = 1/sum(summand)

    #calculate spectral density
    C = (4*org_energy*(gamma**3)*omega)/(omega**2+gamma**2)**2

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

C = np.zeros((Q),dtype = 'complex')
for q in range(Q):
    C[q] = C_pp(w[q])

#initial values
t_i = 0
alpha_1 = -0.526
alpha_2 = (1-alpha_1**2)**0.5

lambda_i = np.zeros((N,Q),dtype = complex)
alpha_i = np.array([alpha_1, alpha_2])

y_ic = np.zeros((N,Q+1), dtype = complex)
y_ic[:,0:Q] = lambda_i
y_ic[:,Q] = alpha_i
y_i = np.reshape(y_ic,(N*(Q+1)))

#time (in GeV ^-1)
t_f = 1200E-15 * (1/h_bar) * 0.2#1200E-15 * (1/h_bar)
time_points = 100 * 100
t = np.arange(0,t_f + t_f/time_points,t_f/time_points)

#solve the ODE
sol = solve_ivp(D1_ODE_system, [0,t_f], y_i, t_eval = t, method='DOP853', rtol = 1E-14, atol = 1E-17) #method='DOP853', rtol = 1E-11, atol = 1E-14

#--------------------------------------------------------------#
#PLOTTING

fig = plt.figure()
ax1 = fig.add_subplot(311)
ax1.plot(t, abs(sol.y[1*Q+0][:])**2)
ax2 = fig.add_subplot(312)
ax2.plot(t, abs(sol.y[2*Q+1][:])**2)
ax3 = fig.add_subplot(313)
ax3.plot(t, abs(sol.y[500][:])**2)
plt.show()

"""
#--------------------------------------------------------------#
#EXTRA PLOTTING
fig = plt.figure()
ax1 = fig.add_subplot(311)
ax1.plot(t, abs(sol.y[1*Q+0][:])**2)
ax2 = fig.add_subplot(312)
ax2.plot(t, abs(sol.y[2*Q+1][:])**2)
ax3 = fig.add_subplot(313)
ax3.plot(t, abs(sol.y[500][:])**2)
plt.show()
"""

"""
#--------------------------------------------------------------#
TESTING

#generate Hamiltonian, angular velocities, coupling strengths
print('w:',w[500:505])
print('g:',g[0][500:505])
print('g:',g[1][500:505])

#initial values
#checking if the reshaping works
print("alpha_1:",y_i[1*Q+0])
print("alpha_2:",y_i[2*Q+1])
print("y-shape:",np.shape(y_i))

y_it = np.reshape(y_i,(N,Q+1))
print("alpha_1:",(y_it[0,Q]))
print("alpha_2:",(y_it[1,Q]))

#final time (in GeV ^-1)
print("time (GeV):",t)
"""

"""
#--------------------------------------------------------------#
#normalized spectral density function for Drude-Lorenz case
def C_pp1(omega):
    #calculate normalization factor
    summand = [(2*gamma)/(w[qq]**2 + gamma**2) for qq in range(Q)]
    k = 1/sum(summand)

    #calculate spectral density
    C = (2*org_energy*gamma*omega)/(omega**2+gamma**2)

    return k*C
"""

"""
#generates coupling strengths
def spawn_g():
    n = 0
    while n < N:
        q = 0
        while q < Q:
            if(q%2==0):
                if(n==1):
                    g[n][q] = (C_pp(w[q])/w[q]**2)**0.5
                else:
                    g[n][q] = 0
            else:
                if(n==0):
                    g[n][q] = (C_pp(w[q])/w[q]**2)**0.5
                else:
                    g[n][q] = 0
            q+=1
        n+=1
    
    return 0
"""

"""
n = 0
while n < N:
    q = 0
    while q < Q:
        g[n][q] = (C_pp(w[q])/w[q]**2)**0.5
        q+=1
    n+=1
"""

"""
#initialize displacements with a noise factor to avoid singularity
lambda_i = np.random.uniform(-1E-5,1E-5,(N,Q)).astype(complex)
"""