import numpy as np

#Two-Level System for the Multi-D2 Ansatz
#Initialization of System and Variational Parameters

#--------------------------------------------------------------#
#CONSTANTS
h     = (4.136E-15) * 10**(-9) # (in GeV * s)
h_bar = h/(2*np.pi)            # (in GeV * s)
c     = 3E8                    # (in m/s)

#--------------------------------------------------------------#
#PARAMETERS

#system parameters
N = 2                           #number of sites
M = 2                           #number of superpositions
Q = 20                         #number of phonon modes
W = 1000 * 100 * h * c          #phonon bandwidth       (in GeV)

#spectral density parameters
Gamma = (1/10) * 10**15 * h_bar #relaxation rate        (in GeV)
Org_Energy = 100 * 100 * h * c  #reorganization energy  (in GeV)
Alpha = 0.1                     #Kondo parameter        (in Gev)

#Hamiltonian parameters
J     =    100 * 100 * h * c    #off-diagonal coupling  (in GeV)
Delta =    0                    #site energy difference (in GeV)

#--------------------------------------------------------------#
#FUNCTIONS

#----------------------------------------#
#defining the system

def spawn_Ham(): #defines the Hamiltonian
    
    #Hamiltonian (in GeV)
    Ham = np.zeros((N,N),dtype = np.complex128)  

    #assign values to Hamiltonian
    Ham[0][0] = 0
    Ham[0][1] = J
    Ham[1][0] = J
    Ham[1][1] = Delta

    return Ham

def spawn_w(ref = "linear"): #generates frequencies of phonon modes

    #frequencies (in GeV)
    w = np.zeros((Q), dtype = np.complex128) 

    #assign values to frequencies
    if   ref == "linear": #linear dispersion
        delta_w = W/Q
        for q in range(Q):
            w[q] = delta_w * (q+1)
    elif ref == "logarithmic": #logarithmic dispersion
        for q in range(Q):
            w[q] = -W*np.log(1-(q/(Q+1)))
            
    return w

def spawn_g(w, gamma = Gamma, org_energy = Org_Energy, alpha = Alpha, ref = "QOBO"): #generates coupling strengths

    #coupling strengths (dimensionless)
    g = np.zeros((N,N,Q), dtype = np.complex128)  

    #assign values to coupling strengths
    #assume each phonon mode is coupled only to a single site
    q = 0
    while q < int(Q/2):
        g[0,0,q] = 0#(SD(w[q],gamma,org_energy,alpha,ref)/w[q]**2)**0.5
        g[1,1,q] = 0
        q+=1
    
    while q < Q:
        g[0,0,q] = 0
        g[1,1,q] = 0#(SD(w[q],gamma,org_energy,alpha,ref)/w[q]**2)**0.5
        q+=1
    
    return g

def SD(omega, gamma, org_energy, alpha, ref): #returns spectral density at a particular frequency
    
    if   ref == "QOBO":
        C_pp = (4*org_energy*(gamma**3)*omega)/(omega**2+gamma**2)**2
    elif ref == "Drude":
        C_pp = (2*org_energy*gamma*omega)/(omega**2+gamma**2)
    elif ref == "Ohmic":
        C_pp = (np.pi/2)*Alpha*omega*np.exp(-omega/W)
        
    return C_pp

#----------------------------------------#
#initializing variational parameters

def spawn_params(): #generate variational parameters
    A_i      = np.random.uniform(-1E-5,1E-5,(N,M)).astype(np.complex128)   #probability amplitudes
    lambda_i = np.random.uniform(-1E-5,1E-5,(Q,M)).astype(np.complex128)   #phonon mode displacements

    #assign values to probability amplitudes
    A_i[0][0]+=0
    A_i[0][1]+=0
    A_i[1][0]+=0.4
    A_i[1][1]+=0.6

    #combine into one ndarray y_i
    y_ic        = np.zeros((N+Q,M), dtype = np.complex128)
    y_ic[0:N,:] = A_i
    y_ic[N:,:]  = lambda_i
    y_i = np.reshape(y_ic,((N+Q)*M))

    return y_i, M

def spawn_time(): #generate time range
    t_i = 0                            #(in GeV ^-1)
    t_f = 1001E-15 * (1/h_bar)         #(in GeV ^-1)
    time_points = 100 * 100

    #initialize time interval array
    t = np.arange(0,t_f + t_f/time_points,t_f/time_points)

    return t
    