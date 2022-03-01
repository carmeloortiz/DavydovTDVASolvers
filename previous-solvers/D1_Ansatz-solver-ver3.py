import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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
W = 2000 * 100 * h_bar * c #phonon bandwidth (in GeV)

#more paramters
Ham = np.zeros((N,N),dtype = complex)  #Hamiltonian (in GeV)
w   = np.zeros((Q), dtype = complex)    #angular frequencies (in GeV)
g   = np.zeros((N,Q), dtype = complex)  #coupling strengths (dimensionless)

#Hamiltonian parameters
J     = -100 * 100 * h_bar * c  # (in GeV)
Delta =  100 * 100 * h_bar * c  # (in GeV)

#spectral density parameters
gamma = (1/10) * 10**15 * h_bar #relaxation rate (in GeV)
ion_energy = 100 * 100 * h_bar * c #ionization energy (in GeV)

#--------------------------------------------------------------#
#FUNCTIONS

# system of ODEs corresponding to the D1 Ansatz
# returns d_lamda_n_q / dt and d_alpha_n / dt for a given time t, for each n and q
def D1_ODE_system(t, yc):
    #convention: yc is an 1D array of length N*(Q+1). The values corresponding to lambda_n_q come first, followed by the value of alpha_n.
    #convention: y is an array with shape (N,Q+1). The values corresponding to lambda_n_q come in the first Q columns. The last column is alpha_n.
    #unpacks yc
    y = np.reshape(yc,(N,Q+1))

    #set 2D arra
    dlambdanq_dt = np.zeros((N,Q),dtype = complex)
    dalphan_dt = np.zeros((N), dtype = complex)

    #calculating Debye-waller factor before main iteration
    S = np.zeros((N,N),dtype = complex)
    for n in range(N):
        for m in range(N):
            summand = [y[n][qq].conjugate()*y[m][qq]  - 0.5*(y[n][qq].conjugate()*y[n][qq]+y[m][qq].conjugate()*y[m][qq]) for qq in range(Q)]
            S[n][m] = np.exp(sum(summand),dtype = complex)

    i=0
    #looping over d_lambda_n_q / dt
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
        term_1 = (-1/2)*y[n][Q]*sum(summand)

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

        #print(' t1 ',term_1,' t2 ',term_2,' t3 ',term_3,' t4 ',term_4)
        n+=1

    #return d_lamda_n_q / dt and d_alpha_n / dt
    dydt = np.zeros((N,Q+1),dtype = complex)
    dydt[:,0:Q] = dlambdanq_dt
    dydt[:,Q] = dalphan_dt

    #print('alphas:',y[:,Q]," time:",(t/t_f)*100, " lambda:",y[0,0])
    print('alphas:',y[:,Q],)
    return np.reshape(dydt,(N*(Q+1)))

#generates angular frequencies of modes
def spawn_w():
    delta_w = W/Q
    for q in range(Q):
        w[q] = delta_w * (q+1)
    return 0

#generates coupling strengths
def spawn_g():
    n = 0
    while n < N:
        q = 0
        while q < Q:
            g[n][q] = (C_pp(w[q])/w[q]**2)**0.5
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
lambda_i = np.zeros((N,Q),dtype = complex)
alpha_i = np.array([-0.526, 0.851])
y_ic = np.zeros((N,Q+1), dtype = complex)
y_ic[:,0:Q] = lambda_i
y_ic[:,Q] = alpha_i
y_i = np.reshape(y_ic,(N*(Q+1)))

#final time (in GeV ^-1)
t_f = 1200E-15 * (1/h_bar) * 4 #CHANGE THIS
time_points = 100
t = np.arange(0,t_f + t_f/time_points,t_f/time_points)

#solve the ODE
sol = solve_ivp(D1_ODE_system, [0,t_f], y_i, t_eval = t)


#--------------------------------------------------------------#
#PLOTTING
plt.plot(t, abs(sol.y[2*Q+1][:])**2)
plt.show()




"""
#--------------------------------------------------------------#
TESTING "REPOSITORY"

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