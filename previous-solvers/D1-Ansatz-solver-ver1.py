import scipy
import numpy as np
import matplotlib.pyplot as plt

#--------------------------------------------------------------#
#CONSTANTS
h     = 4.136E-15 * 10**(-9) # (in GeV * s)
h_bar = h/(2*np.pi)          # (in GeV * s)
c     = 3E8                  # (in m/s)

#--------------------------------------------------------------#
#PARAMETERS

#global parameters
N = 2    #number of sites
Q = 1000 #number of modes per site
W = 2000 * 100 * h * c #phonon bandwidth (in GeV)

#more paramters
Ham = np.zeros((N,N))  #Hamiltonian (in GeV)
w   = np.zeros((Q))    #angular frequencies (in GeV)
g   = np.zeros((N,Q))  #coupling strengths (dimensionless)

#Hamiltonian parameters
J     = -100 * 100 * h * c  # (in GeV)
Delta =  100 * 100 * h * c  # (in GeV)

#spectral density parameters
gamma = (1/10) * h #relaxation rate (in GeV)
ion_energy = 100 * 100 * h * c #ionization energy (in GeV)

#--------------------------------------------------------------#
#FUNCTIONS

# system of ODEs corresponding to the D1 Ansatz
# returns d_lamda_n_q / dt and d_alpha_n / dt for a given time t, for each n and q
def D1_ODE_system(t, y):
    #convention: y is an array with shape (N,Q+1). The values corresponding to lamda_n_q come in the first Q columns. The last column is alpha_n.
    dlambdanq_dt = np.zeros((N,Q))
    dalphan_dt = np.zeros((N))

    #calculating Debye-waller factor before second loop
    S = np.zeros((N,N))
    for n in range(N):
        for m in range(N):
            summand = [y[n][qq].conjugate()*y[m][qq]  - 0.5*(y[n][qq].conjugate()*y[n][qq]+y[m][qq].conjugate()*y[m][qq]) for qq in range(Q)]
            S[n][m] = np.exp(sum(summand))

    i=0
    #looping over d_lamda_n_q / dt
    n = 0
    while n < N:
        q = 0
        while q < Q:
            summand = [Ham[m][n]*S[n][m]* ((y[n][Q].conjugate()*y[m][Q])/(y[n][Q].conjugate()*y[n][Q])) * (y[m][q]- y[n][q]) for m in range(N)]
            dlambdanq_dt[n][q] = 1j*sum(summand) - 1j*w[q]*y[n][q] + 1j*g[n][q]*w[q]
            q+=1
        n+=1

    #looping over d_alpha_n / dt
    n = 0
    while n < N:
        #first term
        summand = [y[n][qq].conjugate()*dlambdanq_dt[n][qq] - y[n][qq]*dlambdanq_dt[n][qq].conjugate() for qq in range(Q)]
        term_1 = (-1j/2)*y[n][Q]*sum(summand)

        #second term
        summand = [Ham[m][n]*y[m][Q]*S[n][m] for m in range(N)]
        term_2 = -1j*sum(summand)

        #third term
        summand = [w[qq]*y[n][qq]*y[n][qq].conjugate() for qq in range(Q)]
        term_3 = -1j*y[n][Q]*sum(summand)

        #fourth term
        summand = [g[n][qq]*w[qq]*(y[n][qq].conjugate() + y[n][qq]) for qq in range(Q)]
        term_4 = +1j*y[n][Q]*sum(summand)

        #final result
        dalphan_dt[n] = term_1 + term_2 + term_3 + term_4
        n+=1

    #return d_lamda_n_q / dt and d_alpha_n / dt
    dydt = np.zeros((N,Q+1))
    dydt[:][0:Q] = dlambdanq_dt
    dydt[:][Q] = dalphan_dt

    return dydt

#generates angular frequencies of modes
def spawn_w():
    delta_w = W/Q
    for q in range(Q):
        w[q] = delta_w * q
    return 0

#generates coupling strengths
def spawn_g():
    n = 0
    while n < N:
        q = 0
        while q < Q:
            g[n][q] = C_pp(w[q])/w[q]**2
            q+=1
        n+=1
    return 0

#normalized spectral density function for excitonic polaron case
def C_pp(omega):
    #calculate normalization factor
    summand = [(4*gamma**3)/(w[qq]**2 + gamma**2)**2 for qq in range(Q)]
    k = 1/sum(summand)

    #calculate spectral density
    C = (4*ion_energy*(gamma**3)*omega)/(omega**2+gamma**2)**2

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
lambda_i = np.zeros((N,Q))
alpha_i = np.ndarray([-0.526, 0.851])
y_i = np.zeros((N,Q+1))
y_i[:][0:Q] = lambda_i
y_i[:][Q] = alpha_i

#final time (in GeV)
t_f = 0
time_points = 20
t = np.arange(0,t_f,t_f/time_points)

#solve the ODE
sol = scipy.integrate.solve_ivp(D1_ODE_system, [0,t_f], y_i, t)

#--------------------------------------------------------------#
#PLOTTING
plt.plot(t, abs(sol.y[0][Q][:])**2)