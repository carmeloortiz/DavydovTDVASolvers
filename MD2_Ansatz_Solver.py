import numpy as np
import cmath
from numba import jit
from scipy.integrate import solve_ivp

#--------------------------------------------------------------#
#FUNCTIONS

#runs the simulation according to the parameters and initial conditions given
def simulate(M, Ham, w, g, y_i, t):
    #solve the system of ODE's using python's solve_ivp
    y = solve_ivp(MD2_solve_ODE, [0,t[-1]], y_i, args = [M, Ham, w, g, t[-1]], t_eval = t)

    return y

#intermediary function used to print out the progress towards completion
def MD2_solve_ODE(t, yc, M, Ham, w, g, t_f):
    #returns matrix A and constant vector b in the standard form Ax = b, x := dydt
    Ab = MD2_ODE_system(t, yc, M, Ham, w, g)
    A = np.matrix(Ab[0])
    b = Ab[1]

    #calculates a Moore-Penrose pseudoinverse
    pseudo_inv = np.linalg.pinv(A)
    A_pinv = np.array(pseudo_inv)

    #matrix multiplication to determine dydt
    dydt = A_pinv.dot(b)

    #prints out the second site probability
    Q = len(w)
    print('|alpha_2|^2: {alpha_sq}, completion (%): {time}'.format(alpha_sq = abs(sum(yc[M:2*M]))**2,time = (abs(t)/t_f)*100))

    return dydt[0:len(b)]

# system of ODEs corresponding to the Multi-D2 Ansatz
@jit(nopython=True, parallel=True)
def MD2_ODE_system(t, yc, M, Ham, w, g):
    N = len(Ham[0]) #number of sites
    Q = len(w)      #number of phonon modes

    #convention: yc is a 1D array of length (N+Q)*M. The values corresponding to A_m_k come first, followed by the values of lambda_q_k.
    #convention: y is an array with shape (N+Q,M). The values corresponding to A_m_k come in the first N rows. The last Q rows are lambda_q_k.
    y = np.reshape(yc,(N+Q,M))

    #initialize time derivatives
    dAnk_dt = np.zeros((N,M),dtype = np.complex128)
    dlambdaqk_dt = np.zeros((Q,M), dtype = np.complex128)

    #calculating Debye-waller factor before main iteration
    S = np.zeros((M,M),dtype = np.complex128)
    for k in range(M):
        for p in range(M):
            summand = [y[N+qq][k].conjugate()*y[N+qq][p]  - 0.5*(y[N+qq][k].conjugate()*y[N+qq][k]+y[N+qq][p].conjugate()*y[N+qq][p]) for qq in range(Q)]
            S[k][p] = cmath.exp(sum(summand))

    #matrix representing ODE system
    #first N columns : Amk, next Q columns : lambdaqk, next Q columns: complex conjugate of lambdaqk
    MATRIX = np.zeros(((N+Q)*M,(N+2*Q)*M), dtype = np.complex128)

    
    #assigning values to the matrix. For an explanation, see derivation
    for k in range(M):
        for m in range(N): 
            for p in range(M):
                for mm in range(N):
                    if mm == m:
                        MATRIX[M*m + k][M*mm + p] = 1j*S[k][p]
                    else:
                        MATRIX[M*m + k][M*mm + p] = 0
                for q in range(Q):
                        MATRIX[M*m + k][M*(N+q) + p] = 1j*y[m][p]*S[k][p]*(y[N+q][k].conjugate() - 0.5*y[N+q][p].conjugate()) 
                for q in range(Q):  
                        MATRIX[M*m + k][M*(N+Q+q) + p] = -0.5j*y[m][p]*S[k][p]*y[N+q][p]
        for q in range(Q):
            for p in range(M):
                for m in range(N):
                        MATRIX[M*(N+q) + k][M*m + p] = 1j*y[m][k].conjugate()*y[N+q][p]*S[k][p]
                for qq in range(Q):
                    if qq == q:
                        summand1 = [y[m][k].conjugate()*y[m][p]*S[k][p] for m in range(N)]
                        summand2 = [y[m][k].conjugate()*y[m][p]*S[k][p]*y[N+q][p]*(y[N+qq][k].conjugate() - 0.5*y[N+qq][p].conjugate()) for m in range(N)]

                        MATRIX[M*(N+q) + k][M*(N+qq) + p] = 1j*sum(summand1) + 1j*sum(summand2)
                    else:
                        summand2 = [y[m][k].conjugate()*y[m][p]*S[k][p]*y[N+q][p]*(y[N+qq][k].conjugate() - 0.5*y[N+qq][p].conjugate()) for m in range(N)]

                        MATRIX[M*(N+q) + k][M*(N+qq) + p] = 1j*sum(summand2)
                for qq in range(Q):
                        summand = [y[m][k].conjugate()*y[m][p]*y[N+q][p]*S[k][p]*y[N+qq][p]]
                        
                        MATRIX[M*(N+q) + k][M*(N+Q+q) + p] = -0.5j*sum(summand) 
    
    #b-vector
    #first N columns : Amk, next Q columns : lambdaqk
    b = np.zeros(((N+Q)*M), dtype = np.complex128)

    #assigning values to the b-vector. For an explanation, see derivation
    for k in range(M):
        for m in range(N): 
            #first sum
            summand1 =  sum([Ham[m][mm]*
                        sum([y[mm][p]*S[k][p] 
                        for p  in range(M)])
                        for mm in range(N)])  

            #second sum
            summand2 =  sum([y[m][p]*S[k][p]*
                        sum([w[q]*y[N+q][k].conjugate()*y[N+q][p] 
                        for q in range(Q)])                
                        for p in range(M)])

            #third sum
            summand3 =  sum([
                        sum([y[mm][p]*S[k][p]*
                        sum([g[m][mm][q]*(y[N+q][p] + y[N+q][k].conjugate())
                        for q  in range(Q)])    
                        for p  in range(M)])
                        for mm in range(N)])
            
            b[M*m + k] = summand1 + summand2 + summand3


        for q in range(Q):
            #first sum
            summand1 =  sum([ 
                        sum([
                        sum([Ham[m][mm]*y[m][k].conjugate()*y[mm][p]*y[N+q][p]*S[k][p]
                        for p  in range(M)])
                        for mm in range(N)])
                        for m  in range(N)])

            #second sum
            summand2 =  sum([
                        sum([y[m][k].conjugate()*y[m][p]*w[k]*y[N+q][p]*S[k][p]
                        for p in range(M)])
                        for m in range(N)])

            #third sum
            summand3 =  sum([
                        sum([y[N+q][p]*S[k][p]*
                        sum([y[m][k].conjugate()*y[m][p]*w[qq]*y[N+qq][k].conjugate()*y[N+qq][p]
                        for qq in range(Q)])
                        for p  in range(M)])
                        for m  in range(N)])

            #fourth sum
            summand4 =  sum([
                        sum([
                        sum([y[m][k].conjugate()*y[mm][p]*g[m][mm][q]*S[k][p]
                        for p  in range(M)])
                        for mm in range(N)])
                        for m  in range(N)])

            #fifth sum
            summand5 =  sum([
                        sum([
                        sum([y[N+q][p]*S[k][p]*
                        sum([y[m][k].conjugate()*y[mm][p]*g[m][mm][qq]*(y[N+qq][p]+y[N+qq][k].conjugate())
                        for qq in range(Q)])
                        for p  in range(M)])
                        for mm in range(N)])                    
                        for m  in range(N)])

            b[M*(N+q) + k] = summand1 + summand2 + summand3 + summand4 + summand5

    return MATRIX, b