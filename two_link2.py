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

def get_Jee(state,aparams):
    l1 = aparams['l1'];l2 = aparams['l2']
    a1,a2,a1d,a2d = state
    Jee = np.array([[-l1*np.sin(a1)-l2*np.sin(a1+a2),-l2*np.sin(a1+a2)],[l1*np.cos(a1)+l2*np.cos(a1+a2),l2*np.cos(a1+a2)]]) 
    return Jee

def get_Mxee(state,aparams):
   a1,a2,a1d,a2d = state;
   l1,l2 = aparams['l1'], aparams['l2']
   m1,m2 = aparams['m1'], aparams['m2']
   M11 = (m1+m2)*l1**2 + m2*l2**2 + 2*m2*l1*l2*np.cos(a2)
   M12 = m2*l2**2 + m2*l1*l2*np.cos(a2);M21 = M12
   M22 = m2*l2**2
   M = np.matrix([[M11,M12],[M21,M22]])
   Jee=get_Jee(state,aparams)
   return(np.linalg.pinv(Jee @  np.linalg.pinv(M) @Jee.T))
   
def get_G(state,aparams):
    a1,a2,a1d,a2d = state;
    l1,l2 = aparams['l1'], aparams['l2']
    m1,m2 = aparams['m1'], aparams['m2']; g = aparams['g']
    G1 = (m1+m2)*g*l1*np.cos(a1) + m2*g*l2*np.cos(a1+a2)
    G2 = m2*g*l2*np.cos(a1+a2)
    G = np.matrix([[G1],[G2]]);
    return(G)

#Mxee=get_Mxee(statw, aparams)
#Jee=get_Jee(statw, aparams)
#u = Jee.T @ Mxee @(kp*np.array([[xf[0]-xo[0]],[xf[1]-xo[1]]]) )
#%%
    
# forward dynamics equations of our passive two-joint arm
def get_ctrl(state,aparams,kp,kd):
	a1,a2,a1d,a2d = state
	l1,l2 = aparams['l1'], aparams['l2']
	m1,m2 = aparams['m1'], aparams['m2'];g = aparams['g']
	G1 = (m1+m2)*g*l1*np.cos(a1) + m2*g*l2*np.cos(a1+a2)
	G2 = m2*g*l2*np.cos(a1+a2)
	G = np.matrix([[G1],[G2]]);U =np.matrix([[kp*(xd[0,0]-a1)-kd*a1d],[.5*kp*(xd[1,0]-a2)-kd*a2d]])+G;return U
    
def get_ctrl2(state,aparams,kp,kd):
    """This is in task space"""
    a1,a2,a1d,a2d = state
    l1,l2 = aparams['l1'], aparams['l2']
    m1,m2 = aparams['m1'], aparams['m2'];g = aparams['g']
    G1 = (m1+m2)*g*l1*np.cos(a1) + m2*g*l2*np.cos(a1+a2)
    G2 = m2*g*l2*np.cos(a1+a2)
    G = np.matrix([[G1],[G2]]);
    Mxee=get_Mxee(state, aparams)
    Jee=get_Jee(state, aparams)
    pa=np.matrix([a1,a2])
    pw=np.matrix([a1d,a2d])
    px,_ = joints_to_hand(pa,aparams)
    pv = Jee@pw.T
    U = Jee.T @ Mxee @(kp*np.array([[xf[0]-px[0,0]],[xf[1]-px[0,1]]]) -kd*pv)+G
    return U

def twojointarm(state,t,aparams):
	a1,a2,a1d,a2d = state
	l1,l2 = aparams['l1'], aparams['l2']
	m1,m2 = aparams['m1'], aparams['m2']
#	i1,i2 = aparams['i1'], aparams['i2']
#	r1,r2 = aparams['r1'], aparams['r2']
	M11 = (m1+m2)*l1**2 + m2*l2**2 + 2*m2*l1*l2*np.cos(a2)
	M12 = m2*l2**2 + m2*l1*l2*np.cos(a2)
	M21 = M12
	M22 = m2*l2**2
	M = np.matrix([[M11,M12],[M21,M22]])
	C1 = -m2*l1*l2*(2*a1d*a2d+a2d**2)*np.sin(a2)
	C2 = m2*l1*l2*a1d**2*np.sin(a2)
	C = np.matrix([[C1],[C2]]);g = aparams['g']
	G1 = (m1+m2)*g*l1*np.cos(a1) + m2*g*l2*np.cos(a1+a2)
	G2 = m2*g*l2*np.cos(a1+a2)
	G = np.matrix([[G1],[G2]]);U = get_ctrl2(state,aparams,kp,kd) ##############################
	ACC = np.linalg.inv(M) * (U-C-G)
	a1dd,a2dd = ACC[0,0],ACC[1,0]
	return [a1d, a2d, a1dd, a2dd]

# parameters of the arm
aparams = {
	'l1' : 1.0, # metres
	'l2' : 1.0,
	'r1' : 0.5,
	'r2' : 0.5,
	'm1' : 1.0,   # kg
	'm2' : 1.0,
	'i1' : 0.025,  # kg*m*m
	'i2' : 0.025,
    'g'  : 10
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

def hand2joint(xo,aparams):
    l1 = aparams['l1'];	l2 = aparams['l2']
    r2 = xo[0]**2 + xo[1]**2
    C = (r2 - (l1**2 +l2**2))/2*l1*l2
    D = np.sqrt(1-C**2)
    q2=np.arctan2(D,C)
    q1=np.arctan2(xo[1],xo[0])-np.arctan2(l2*np.sin(q2),l1+l2*np.cos(q2))
    
    return (q1,q2)
    

#%%---------------------------------------------------------------------------------------

 
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


kp=50; kd=20
#%%
# xo=np.array([1.0,1.0]);xf=np.array([2.0,0])
xo=np.array([0.7,1.5]);xf=np.array([0.04,1.6])
q1,q2=hand2joint(xo,aparams);q3,q4=hand2joint(xf,aparams)
to_animate=True
to_plot=True
#%%
"""
#%%
plt.plot(0,0,'kx')
print(np.rad2deg(q1));print(np.rad2deg(q2))
plt.plot(l1*np.cos(q1),l1*np.sin(q1),'b.');plt.plot(l1*np.cos(q3),l1*np.sin(q3),'g.');
plt.axis('equal');np.rad2deg([q3,q4])
plt.plot(xo[0],xo[1],'bx');plt.plot(xf[0],xf[1],'gx')                  
ang=np.arange(-np.pi,np.pi,0.01)                          
plt.plot((l1+l2)*np.cos(ang),(l1+l2)*np.sin(ang),'b');np.rad2deg([q3,q4])
#%%
"""
state0 = [q1, q2, 0, 0] # initial joint angles and vels
xd=np.matrix([[q3],[q4]])
t = np.arange(4001.)/1000                # 10 seconds at 200 Hz
state = odeint(twojointarm, state0, t, args=(aparams,))
#plt.plot(t,state)
A = state[:,[0,1]]
#A[:,0] = A[:,0] - (np.pi/2)
H,E = joints_to_hand(A,aparams)
temp=H.shape[0]
#%%
fig = plt.figure(figsize=(5,5))
l1,l2 = aparams['l1'], aparams['l2'];
ang=np.arange(-np.pi,np.pi,0.01)                      
ax = fig.add_subplot(111, autoscale_on=False, xlim=([-l1-l2, l1+l2]), ylim=([-l1-l2, l1+l2]))
ax.grid()
ax.plot(xo[0],xo[1],'bx'); ax.plot(xf[0],xf[1],'gx')                     
ax.plot((l1+l2)*np.cos(ang),(l1+l2)*np.sin(ang),'b');
line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

#%%
if to_animate:ani=animation.FuncAnimation(fig, animatearm,np.arange(1,temp,5),interval=25, blit=True, init_func=init)
if to_plot:
    thisx = [0, E[0,0], H[0,0]]
    thisy = [0, E[0,1], H[0,1]]
    endx = [0, E[-1,0], H[-1,0]]
    endy = [0, E[-1,1], H[-1,1]]
    plt.plot(thisx, thisy,'b-o');plt.plot(endx, endy,'g-o');plt.plot(H[:,0],H[:,1],'k--')
    plt.plot(0,0,'ko')
    plt.axis('equal');                
    ang=np.arange(-np.pi,np.pi,0.01)                          
    plt.plot((l1+l2)*np.cos(ang),(l1+l2)*np.sin(ang),'b');np.rad2deg([q3,q4])

                         
#%%                            
#plt.plot(t,state)                            
                            
                            