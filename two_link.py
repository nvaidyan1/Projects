# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 13:09:34 2018

@author: Natarajan Vaidyanathan
"""

# two joint arm in a vertical plane, with gravity
import numpy as  np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint

# forward dynamics equations of our passive two-joint arm
def twojointarm(state,t,aparams):
	
	a1,a2,a1d,a2d = state
	l1,l2 = aparams['l1'], aparams['l2']
	m1,m2 = aparams['m1'], aparams['m2']
	i1,i2 = aparams['i1'], aparams['i2']
	r1,r2 = aparams['r1'], aparams['r2']
	g = 9.81
	M11 = i1 + i2 + (m1*r1*r1) + (m2*((l1*l1) + (r2*r2) + (2*l1*r2*np.cos(a2))))
	M12 = i2 + (m2*((r2*r2) + (l1*r2*np.cos(a2))))
	M21 = M12
	M22 = i2 + (m2*r2*r2)
	M = np.matrix([[M11,M12],[M21,M22]])
	C1 = -(m2*l1*a2d*a2d*r2*np.sin(a2)) - (2*m2*l1*a1d*a2d*r2*np.sin(a2))
	C2 = m2*l1*a1d*a1d*r2*np.sin(a2)
	C = np.matrix([[C1],[C2]])
	G1 = (g*np.sin(a1)*((m2*l1)+(m1*r1))) + (g*m2*r2*np.sin(a1+a2))
	G2 = g*m2*r2*np.sin(a1+a2)
	G = np.matrix([[G1],[G2]]);U =[[2.6],[2.5]]
	ACC = np.linalg.inv(M) * (U-C-G)
	a1dd,a2dd = ACC[0,0],ACC[1,0]
	return [a1d, a2d, a1dd, a2dd]

# parameters of the arm
aparams = {
	'l1' : 0.3384, # metres
	'l2' : 0.4554,
	'r1' : 0.1692,
	'r2' : 0.2277,
	'm1' : 2.10,   # kg
	'm2' : 1.65,
	'i1' : 0.025,  # kg*m*m
	'i2' : 0.075
}

# forward kinematics
def joints_to_hand(A,aparams):

	l1 = aparams['l1']
	l2 = aparams['l2']
	n = np.shape(A)[0]
	E = np.zeros((n,2))
	H = np.zeros((n,2))
	for i in range(n):
		E[i,0] = l1 * np.cos(A[i,0])
		E[i,1] = l1 * np.sin(A[i,0])
		H[i,0] = E[i,0] + (l2 * np.cos(A[i,0]+A[i,1]))
		H[i,1] = E[i,1] + (l2 * np.sin(A[i,0]+A[i,1]))
	return H,E


 #---------------------------------------------------------------------------------------
fig = plt.figure()
l1,l2 = aparams['l1'], aparams['l2']
ax = fig.add_subplot(111, autoscale_on=False, xlim=([-l1-l2, l1+l2]), ylim=([-l1-l2, l1+l2]))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
 
def init():
    """initialize animation"""
    line.set_data([], [])
    plt.xlim([-l1-l2, l1+l2])
    plt.ylim([-l1-l2, l1+l2])
    time_text.set_text('')
    
    return line, time_text

def animatearm(i):
    dt=0.05
    thisx = [0, E[i,0], H[i,0]]
    thisy = [0, E[i,1], H[i,1]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i*dt))
    return line, time_text

	
		
state0 = [2*np.pi/180, 0*np.pi/180, 0, 0] # initial joint angles and vels
t = np.arange(2001.)/200                # 10 seconds at 200 Hz
state = odeint(twojointarm, state0, t, args=(aparams,))
#plt.plot(t,state)
A = state[:,[0,1]]
A[:,0] = A[:,0] - (np.pi/2)
H,E = joints_to_hand(A,aparams)
temp=H.shape[0]
ani=animation.FuncAnimation(fig, animatearm,np.arange(1,temp,5),
                            interval=25, blit=True, init_func=init)

                            
                            
                            
                            
                            
                            