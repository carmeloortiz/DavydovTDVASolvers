import numpy as np
import matplotlib.pyplot as plt

#Two-Level System for the D1 Ansatz
#Visualization and Plotting

#--------------------------------------------------------------#
#CONSTANTS
h     = (4.136E-15) * 10**(-9) # (in GeV * s)
h_bar = h/(2*np.pi)            # (in GeV * s)
c     = 3E8                    # (in m/s)

#--------------------------------------------------------------#
#FUNCTIONS

#----------------------------------------#
#defining the system

def plotresults(t, sol, w): #plots the results of the simulation

    Q = len(w)
    #modulus squared of second site probability amplitude
    alpha_2_sq = abs(sol.y[2*Q+1][:])**2

    #horizontal line at the 0.5 mark on the y-axis
    point5 = [0.5 for i in t]

    #plotting
    fig = plt.figure()

    ax2 = fig.add_subplot(111)
    ax2.plot(t * h_bar * 1E15, alpha_2_sq ,color = 'green',linewidth = 1)
    ax2.plot(t * h_bar * 1E15, point5     ,color = 'blue' ,linewidth = 1, linestyle = 'dashed')
    ax2.set_ylabel(r"|$\alpha$₂|$^2$   ",rotation = 'horizontal')
    ax2.set_xlabel("t (fs)")
    ax2.set_ylim(-0.1,1.1)
    ax2.set_xlim(t[0] * h_bar * 1E15,t[-1] * h_bar * 1E15)

    fig.tight_layout()
    plt.show()

    return True








