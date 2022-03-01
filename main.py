import numpy as np
import matplotlib.pyplot as plt
from   scipy.integrate import solve_ivp

#main ODE solvers
import  D1_Ansatz_Solver as  D1
import MD2_Ansatz_Solver as MD2

#files for initializing parameters and Hamiltonian
import  D1_2LS_initialization as  D1_2LS
import MD2_2LS_initialization as MD2_2LS

#files for plotting and visualization
import  D1_2LS_visualization  as  D1_2LS_vis
import MD2_2LS_visualization  as MD2_2LS_vis
import DMEF_comparison        as DMEF

#--------------------------------------------------------------#
#CONSTANTS
h     = (4.136E-15) * 10**(-9) # (in GeV * s)
h_bar = h/(2*np.pi)            # (in GeV * s)
c     = 3E8                    # (in m/s)

#--------------------------------------------------------------#
#SIMULATION

def simulate_D1_2LS(plotting = True):  #Two-Level System with the D1 Ansatz
    print("Solving D1 ODE System: Initializing Parameters...")
    #------------------------------#
    #generate Hamiltonian, frequencies, and coupling strengths
    Ham = D1_2LS.spawn_Ham()                    #Hamiltonian (in Gev), returns array with shape (N,N)
    w   = D1_2LS.spawn_w()                      #frequencies (in GeV), returns array with shape (Q)
    g   = D1_2LS.spawn_g(w)                     #coupling strengths, returns array with shape (N,Q)

    #------------------------------#
    #initialize variational parameters
    y_i = D1_2LS.spawn_params()

    #------------------------------#
    #specify time range of simulation
    t   = D1_2LS.spawn_time()

    #------------------------------#
    #solve the ODE system
    y   = D1.simulate(Ham, w, g, y_i, t)

    #------------------------------#
    #plot the results
    if plotting == True:
        D1_2LS_vis.plotresults(t, y, w)

    return t, y, w

def simulate_MD2_2LS(plotting = True):  #Two-Level System with the Multi-D2 Ansatz
    print("Solving Multi-D2 ODE System: Initializing Parameters...")
    #------------------------------#
    #generate Hamiltonian, frequencies, and coupling strengths
    Ham = MD2_2LS.spawn_Ham()                    #Hamiltonian (in Gev), returns array with shape (N,N)
    w   = MD2_2LS.spawn_w()                      #frequencies (in GeV), returns array with shape (Q)
    g   = MD2_2LS.spawn_g(w)                     #coupling strengths, returns array with shape (N,N,Q)

    #------------------------------#
    #initialize variational parameters
    y_i, M = MD2_2LS.spawn_params()

    #------------------------------#
    #specify time range of simulation
    t      = MD2_2LS.spawn_time()

    #------------------------------#
    #solve the ODE system
    y      = MD2.simulate(M, Ham, w, g, y_i, t)

    #------------------------------#
    #plot the results
    if plotting == True:
        MD2_2LS_vis.plotresults(t, y, w, M)

    return t, y, w, M

#--------------------------------------------------------------#
#MAIN FUNCTION

def main():
    D1_data  = simulate_D1_2LS(plotting = False)
    MD2_data = simulate_MD2_2LS(plotting = False)

    DMEF.plot_comparison(D1_data, MD2_data)

if __name__ == "__main__":
    main()