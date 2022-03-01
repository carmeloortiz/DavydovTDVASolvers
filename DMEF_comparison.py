import numpy as np
import matplotlib.pyplot as plt

#Two-Level System
#Comparison With Ehrenfest Dynamics and Density Matrix
#Visualization and Plotting

#--------------------------------------------------------------#
#CONSTANTS
h     = (4.136E-15) * 10**(-9) # (in GeV * s)
h_bar = h/(2*np.pi)            # (in GeV * s)
c     = 3E8                    # (in m/s)

def plot_comparison(D1_Data, MD2_Data):
    
    #unpack arrays
    D1_t = D1_Data[0]
    D1_y = D1_Data[1]
    D1_w = D1_Data[2]

    MD2_t = MD2_Data[0]
    MD2_y = MD2_Data[1]
    MD2_w = MD2_Data[2]
    MD2_M = MD2_Data[3]

    #modulus squared of second site probability amplitude
    D1_Q = len(D1_w)
    D1_alpha_2_sq = abs(D1_y.y[2*D1_Q+1][:])**2

    MD2_Q = len(MD2_w)
    MD2_alpha_2_sq = abs(sum([MD2_y.y[MD2_M+i][:] for i in range(MD2_M)]))**2
    
    """
    #open file containing results from Ehrenfest and Density Matrix methods
    DM_file = open("JCP_HEOM_Data\\lam2delt0density.dat", "r")
    data = DM_file.read()
    data = data.split('\n')
    DM_t     =  [float(row.split(' ')[0]) for row in data]
    DM_alpha =  [float(row.split(' ')[1]) for row in data]
    DM_file.close()

    EF_file = open("JCP_HEOM_Data\\lam2delt0ehrenfest.dat", "r")
    data = EF_file.read()
    data = data.split('\n')
    EF_t     =  [float(row.split(' ')[0]) for row in data]
    EF_alpha =  [float(row.split(' ')[1]) for row in data]
    EF_file.close()
    """

    #horizontal line at the 0.5 mark on the y-axis
    point5 = [0.5 for i in D1_t]

    #plotting
    fig = plt.figure()

    ax2 = fig.add_subplot(111)
    ax2.plot( D1_t * h_bar * 1E15, D1_alpha_2_sq  ,color = 'green' ,linewidth = 1, label = 'D1')
    ax2.plot(MD2_t * h_bar * 1E15, MD2_alpha_2_sq ,color = 'yellow',linewidth = 1, label = 'Multi_D2')
    ax2.plot( D1_t * h_bar * 1E15, point5         ,color = 'blue'  ,linewidth = 1, linestyle = 'dashed')
    #ax2.plot(DM_t, DM_alpha ,color = 'red'    ,linewidth = 1, linestyle = 'dashed', label = 'Density Matrix')
    #ax2.plot(EF_t, EF_alpha ,color = 'orange' ,linewidth = 1, linestyle = 'dashed', label = 'Ehrenfest')
    ax2.set_ylabel(r"|$\alpha$₂|$^2$   ",rotation = 'horizontal')
    ax2.set_xlabel("t (fs)")
    ax2.set_ylim(-0.1,1.1)
    ax2.set_xlim(D1_t[0] * h_bar * 1E15,D1_t[-1] * h_bar * 1E15)
    ax2.set_title("2LS at very low reorganization energies")
    ax2.legend(loc = 'lower right')

    fig.tight_layout()
    plt.show()