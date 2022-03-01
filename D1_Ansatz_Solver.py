import numpy as np
import cmath
from numba import jit
from scipy.integrate import solve_ivp

#--------------------------------------------------------------#
#FUNCTIONS

#runs the simulation according to the parameters and initial conditions given
def simulate(Ham, w, g, y_i, t):
    #solve the system of ODE's using python's solve_ivp
    y = solve_ivp(D1_solve_ODE, [0,t[-1]], y_i, args = [Ham, w, g, t[-1]], t_eval = t)

    return y

#intermediary function used to print out the progress towards completion
def D1_solve_ODE(t, yc, Ham, w, g, t_f):
    #returns dy/dt for each variational parameter
    dydt = D1_ODE_system(t, yc, Ham, w, g)

    #prints out the second site probability
    Q = len(w)
    print('|alpha_2|^2: {alpha_sq}, completion (%): {time}'.format(alpha_sq = abs(yc[2*Q+1])**2,time = (abs(t)/t_f)*100))

    return dydt

# system of ODEs corresponding to the D1 Ansatz
# returns d_lamda_n_q / dt and d_alpha_n / dt for a given time t, for each n and q
@jit(nopython=True, parallel=True)
def D1_ODE_system(t, yc, Ham, w, g):
    N = len(Ham[0]) #number of sites
    Q = len(w)      #number of phonon modes

    #convention: yc is a 1D array of length N*(Q+1). The values corresponding to lambda_n_q come first, followed by the value of alpha_n.
    #convention: y is an array with shape (N,Q+1). The values corresponding to lambda_n_q come in the first Q columns. The last column is alpha_n.
    y = np.reshape(yc,(N,Q+1))

    #initialize time derivatives
    dlambdanq_dt = np.zeros((N,Q), dtype = np.complex128)
    dalphan_dt = np.zeros((N), dtype = np.complex128)

    #calculating Debye-waller factor before main iteration
    S = np.zeros((N,N),dtype = np.complex128)
    for n in range(N):
        for m in range(N):
            summand = [y[n][qq].conjugate()*y[m][qq]  - 0.5*(y[n][qq].conjugate()*y[n][qq]+y[m][qq].conjugate()*y[m][qq]) for qq in range(Q)]
            S[n][m] = cmath.exp(sum(summand))

    #looping over d_lambda_n_q / dt
    n = 0
    while n < N:
        q = 0
        while q < Q:
            #first term
            summand1  = [Ham[m][n]*S[n][m]* ((y[n][Q].conjugate()*y[m][Q])/(y[n][Q].conjugate()*y[n][Q])) * (y[m][q]- y[n][q]) for m in range(N)] 
            term_1 = -1j*sum(summand1) 
            
            #second term
            term_2 = -1j*w[q]*y[n][q]

            #third term
            term_3 = -1j*g[n][q]

            #final result
            dlambdanq_dt[n][q] = term_1  + term_2 + term_3
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
        summand4 = [g[n][qq]*(y[n][qq].conjugate() + y[n][qq]) for qq in range(Q)]
        term_4 = -1j*y[n][Q]*sum(summand4)

        #final result
        dalphan_dt[n] = term_1 + term_2 + term_3 + term_4
        n+=1

    #return d_lamda_n_q / dt and d_alpha_n / dt
    dydt = np.zeros((N,Q+1),dtype = np.complex128)
    dydt[:,0:Q] = dlambdanq_dt
    dydt[:,Q] = dalphan_dt

    return np.reshape(dydt,(N*(Q+1)))