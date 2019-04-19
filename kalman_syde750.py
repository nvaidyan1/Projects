import nengo
import numpy as np
nengo.rc.set('progress', 'progress_bar', 'nengo.utils.progress.TerminalProgressBar')



class mass_dyn(object):
    def __init__(self, mass=1.0, xo=0,yo=0,dt=0.001, seed=None):
        self.mass = mass
        self.dt = dt
        self.xo=xo
        self.yo=yo
        self.x=np.zeros([4,1])
        self.u=np.zeros([2,1])
        self.reset(seed)
        self.A = np.matrix([[1, 0, self.dt, 0],
               [0, 1, 0, self.dt],
               [0, 0, 1, 0],
               [0, 0, 0 ,1]]); #change to expm(A)
        self.B = np.matrix([[0,0],
                            [0,0],
                            [self.dt/self.mass,0],
                            [0,self.dt/self.mass]]);
        
    def reset(self, seed):
        self.rng = np.random.RandomState(seed=seed)
        self.x[0,0] = self.xo
        self.x[1,0] = self.yo
        self.x[2,0] = 0
        self.x[3,0] = 0
        
    def step(self, ux,uy):
        self.u[0,0],self.u[1,0] = ux, uy
        self.x = self.A*self.x + self.B* self.u 
        
        
    def generate_html(self, xd,yd):
        x=50+self.x[0,0]
        y=50-self.x[1,0]
        xd_plot=50+xd
        yd_plot=50-yd
        return '''
        <svg width="100%" height="100%" viewbox="30 30 40 40">
            <line x1="0" y1="50" x2="100" y2="50" style="stroke:grey;stroke-width:0.25" />
            <line x1="50" y1="0" x2="50" y2="100" style="stroke:grey;stroke-width:0.25" />
            <circle cx="{x}" cy="{y}" r="1" style="stroke:blue;stroke-width:0.25"" fill="blue"/>
            <circle cx="{xd_plot}" cy="{yd_plot}" r="1" style="stroke:black;stroke-width:0.25"" fill="none"/>
        </svg>
        '''.format(**locals())
        
class CtrlNetwork(nengo.Network):
    def __init__(self, label=None, **kwargs):
        super(CtrlNetwork, self).__init__(label=label)
        
        
        with self:

            
            tau=0.001
            self.xd = nengo.Ensemble(n_neurons=1000, dimensions=1, radius=10)
            self.yd = nengo.Ensemble(n_neurons=1000, dimensions=1, radius=10)
            self.xdot_d = nengo.Node(None, size_in=1)
            self.ydot_d = nengo.Node(None, size_in=1)
            
            self.u = nengo.Ensemble(n_neurons=8000, dimensions=2, radius=20)
            #nengo.Connection(self.uy, self.mass_dyn[3], synapse=0.0)
            
            self.x = nengo.Ensemble(n_neurons=1000, dimensions=1, radius=10)
            self.y = nengo.Ensemble(n_neurons=1000, dimensions=1, radius=10)
            
            self.x_diff = nengo.Ensemble(n_neurons=1000, dimensions=1, radius=10)
            nengo.Connection(self.xd, self.x_diff, synapse=tau)
            nengo.Connection(self.x, self.x_diff, synapse=tau, transform=-1)
            
            self.y_diff = nengo.Ensemble(n_neurons=1000, dimensions=1, radius=10)
            nengo.Connection(self.yd, self.y_diff, synapse=tau)
            nengo.Connection(self.y, self.y_diff, synapse=tau, transform=-1)
            
            
            Kp = 1.5
            nengo.Connection(self.x_diff, self.u[0], transform=Kp, synapse=tau)
            nengo.Connection(self.y_diff, self.u[1], transform=Kp, synapse=tau)
            
            self.dx = nengo.Ensemble(n_neurons=1000, dimensions=1, radius=10)
            self.dy = nengo.Ensemble(n_neurons=1000, dimensions=1, radius=10)
        
    
            self.xd_diff = nengo.Ensemble(n_neurons=1000, dimensions=1,radius=10)
            nengo.Connection(self.xdot_d, self.xd_diff, synapse=tau)
            nengo.Connection(self.dx, self.xd_diff, synapse=tau, transform=-1)
    
            self.yd_diff = nengo.Ensemble(n_neurons=1000, dimensions=1,radius=10)
            nengo.Connection(self.ydot_d, self.yd_diff, synapse=tau)
            nengo.Connection(self.dy, self.yd_diff, synapse=tau, transform=-1)
    
            Kd = 1
            nengo.Connection(self.xd_diff, self.u[0], transform=Kd, synapse=tau)
            nengo.Connection(self.yd_diff, self.u[1], transform=Kd, synapse=tau)

class MassNetwork(nengo.Network):
    def __init__(self, label=None, **kwargs):
        super(MassNetwork, self).__init__(label=label)
        self.env = mass_dyn(**kwargs)
        
        with self:
            def func(t, x):
                self.env.step(x[2],x[3])
                func._nengo_html_ = self.env.generate_html(x[0],x[1])
                return (self.env.x[0,0],self.env.x[1,0], self.env.x[2,0],self.env.x[3,0])
            self.mass_dyn = nengo.Node(func, size_in=4)
            
            self.xd = nengo.Node(None, size_in=1)
            nengo.Connection(self.xd, self.mass_dyn[0], synapse=0.0)

            self.yd = nengo.Node(None, size_in=1)
            nengo.Connection(self.yd, self.mass_dyn[1], synapse=0.0)
            
            self.ux = nengo.Ensemble(n_neurons=500, dimensions=1, radius=20)
            nengo.Connection(self.ux, self.mass_dyn[2], synapse=0.0)
            
            self.uy = nengo.Ensemble(n_neurons=500, dimensions=1, radius=20)
            nengo.Connection(self.uy, self.mass_dyn[3], synapse=0.0)
            
            
            self.x = nengo.Node(None, size_in=1)
            self.y = nengo.Node(None, size_in=1)
            self.dx = nengo.Node(None, size_in=1)
            self.dy = nengo.Node(None, size_in=1)
            nengo.Connection(self.mass_dyn[0], self.x, synapse=0.0)
            nengo.Connection(self.mass_dyn[1], self.y, synapse=0.0) 
            nengo.Connection(self.mass_dyn[2], self.dx, synapse=0.00)
            nengo.Connection(self.mass_dyn[3], self.dy, synapse=0.0)
         

        
class Estimate_Network(nengo.Network):
    def __init__(self, label=None, **kwargs):
        super(Estimate_Network, self).__init__(label=label)
        tau=0.001; taup = 0.1; dt=0.001
        with self:
            self.mass=1
            
            self.u = nengo.Ensemble(n_neurons=6000, dimensions=2, radius=10) #u
            self.y =  nengo.Ensemble(n_neurons=6000, dimensions=4, radius=10)
            self.pred = nengo.Ensemble(n_neurons=12000,dimensions=4,radius=10)
            
            
            self.Q_est=0.1
            self.R_est=0.1
            
            
            A =[[1, 0, taup, 0],[0, 1, 0, taup],[0, 0, 1, 0],[0, 0, 0 ,1]]
            B=[[0,0],[0,0],[m_gain*taup/self.mass,0],[0,m_gain*taup/self.mass]]
            
            nengo.Connection(self.u,self.pred,transform=B,synapse=taup)
            nengo.Connection(self.pred,self.pred,transform=A,synapse=taup)
            def gain_func(x):
                return(x[0]+x[8]*(x[4]-x[0])+x[9]*(x[6]-x[2]),
                x[1]+x[8]*(x[5]-x[1])+x[9]*(x[7]-x[3]),
                x[2]+x[9]*(x[4]-x[0])+x[10]*(x[6]-x[2]),
                x[3]+x[9]*(x[5]-x[1])+x[10]*(x[7]-x[3]))
            self.upd2 = nengo.Ensemble(n_neurons=10000,dimensions=7,radius=10)
            

            def gain_func3(x):
                return(x[4]*x[0]+x[5]*x[2],
                x[4]*x[1]+x[5]*x[3],
                x[5]*x[0]+x[6]*x[2],
                x[5]*x[1]+x[6]*x[3])
                
                
            nengo.Connection(self.pred,self.upd2[:4],transform=-1,synapse=tau)
            nengo.Connection(self.y,self.upd2[:4],synapse=tau)

            
            def prod3(x):
                l,m = x[0], x[1]; #Q's values
                i,j = x[2], x[3]; #R's values
                a,b,c=x[4],x[5],x[6] #P's elements
                
                p = a + 2*b*dt + c*dt**2 + l ; #P_ = APA' +Q
                q = b + c*dt
                r = c + m
                
                s = p + i  #P_+R
                t = q 
                u = r + j
                
                l = -u/(t**2-s*u); 
                m = t/(t**2-s*u);
                n = -s/(t**2-s*u) #inv(P+R)
                
                k1 = l*p + m*q
                k2 = m*p + n*q
                k3 = m*q + n*r
                
                return(k1,k2,k3)
                

            
            
            def getP(x):
                l,m = x[0], x[1]; #Q's values
                i,j = x[2], x[3]; #R's values
                a,b,c=x[4],x[5],x[6] #P's elements
                
                p = a + 2*b*dt + c*dt**2 + l ; #P_ = APA' +Q
                q = b + c*dt
                r = c + m
                
                s = p + i  #P_+R
                t = q 
                u = r + j
                
                l = -u/(t**2-s*u); 
                m = t/(t**2-s*u);
                n = -s/(t**2-s*u) #inv(P+R)
                
                k1 = l*p + m*q
                k2 = m*p + n*q
                k3 = m*q + n*r
                return(-q*k2+p*(1-k1), -r*k2+q*(1-k1), -q*k2+r*(1-k3))
                
                            
            # self.P = nengo.Ensemble(n_neurons=7000, dimensions=7, radius=0.5,neuron_type=nengo.Direct()) #This node works with direct mode
            Peval=np.abs(np.random.normal(loc=0.0, scale=0.1, size=[3000,7]))
            self.P = nengo.Ensemble(n_neurons=10000, dimensions=7, radius=0.5,eval_points=Peval)
            self.Ex_n=nengo.Node(self.Q_est);self.Ez_n=nengo.Node(self.R_est);
                
            
            self.ONE=nengo.Node(output=lambda t:[t<0.002,0,t<0.002]); 
            nengo.Connection(self.ONE,self.P[4:],synapse=None);
            nengo.Connection(self.Ex_n,self.P[:2],transform=[[1],[1]],synapse=None);
            nengo.Connection(self.Ez_n,self.P[2:4],transform=[[1],[1]],synapse=None);
            
            nengo.Connection(self.P,self.P[4:],function=getP,synapse=taup)
            nengo.Connection(self.P,self.upd2[4:],transform=30,function=prod3,synapse=taup)
            nengo.Connection(self.upd2,self.pred,transform=(1/30)*np.eye(4),function=gain_func3,synapse=taup)
            

model = nengo.Network(seed=0)
model.config[nengo.Ensemble].neuron_type = nengo.Direct() # to run with direct mode
#model.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
m_gain=2; #Scaling constant =2

with model:
    tau=0.001
    controller = CtrlNetwork(seed=2)
    env = MassNetwork(mass=1,seed=1,xo=5,yo=0) #xo, yo are the initial values of the mass 
    
    x_target = nengo.Node(0.0) # Target x value
    y_target = nengo.Node(7.0) # Target y value
    nengo.Connection(x_target, controller.xd, synapse=None)
    nengo.Connection(y_target, controller.yd, synapse=None)

    xd_target = nengo.Node(None, size_in=1)
    yd_target = nengo.Node(None, size_in=1)
    
    nengo.Connection(xd_target,controller.xdot_d,synapse=tau)
    nengo.Connection(yd_target,controller.ydot_d,synapse=tau)
    nengo.Connection(x_target, env.xd, synapse=None)
    nengo.Connection(y_target, env.yd, synapse=None)
    
    nengo.Connection(controller.u[0],env.ux,transform=m_gain,synapse=None)
    nengo.Connection(controller.u[1],env.uy,transform=m_gain,synapse=None)
    
    nengo.Connection(env.x,controller.x,synapse=None)
    nengo.Connection(env.y,controller.y,synapse=None)
    nengo.Connection(env.dx,controller.dx,synapse=None)
    nengo.Connection(env.dy,controller.dy,synapse=None)
    
    #%% Kalman Filter
    
    kf = Estimate_Network(seed=2) 
    noise = nengo.Node(nengo.processes.WhiteSignal(1, high=25,rms=0.25, seed=0), size_out=4)

    
    nengo.Connection(noise,kf.y,synapse=tau);
    nengo.Connection(controller.u,kf.u,synapse=None)
    nengo.Connection(env.x, kf.y[0], synapse=tau)
    nengo.Connection(env.y, kf.y[1], synapse=tau)
    nengo.Connection(env.dx, kf.y[2], synapse=tau)
    nengo.Connection(env.dy, kf.y[3], synapse=tau)
 
#%%
    xp = nengo.Probe(env.x)
    yp = nengo.Probe(env.y)
    dxp = nengo.Probe(env.dx)
    dyp = nengo.Probe(env.dy)
    
    predp = nengo.Probe(kf.pred,synapse=0.01) 

    upp = nengo.Probe(kf.upd2,synapse=0.01)
    obsp = nengo.Probe(kf.y,synapse=0.01)
    ctrlp = nengo.Probe(controller.u,synapse=0.01)
    ctrlkf = nengo.Probe(kf.u,synapse=0.01)
    check = nengo.Probe(kf.P,synapse=0.01)

with nengo.Simulator(model) as sim:
    sim.run(1.5)
#%%
import matplotlib.pyplot as plt
plt.figure()
plt.title('model probe')
plt.plot(sim.trange(),sim.data[xp],'b',alpha=0.5)
plt.plot(sim.trange(),sim.data[yp],'r',alpha=0.5)
plt.plot(sim.trange(),sim.data[dxp],'g',alpha=0.5)
plt.plot(sim.trange(),sim.data[dyp],'k',alpha=0.5)

plt.title('predict probe')
plt.plot(sim.trange(), sim.data[obsp][:,0],'b',alpha=0.5)
plt.plot(sim.trange(), sim.data[obsp][:,1],'r',alpha=0.5)
plt.plot(sim.trange(), sim.data[obsp][:,2],'g',alpha=0.5)
plt.plot(sim.trange(), sim.data[obsp][:,3],'k',alpha=0.5)
"""
"""
#plt.figure()
plt.title('States')
plt.plot(sim.trange(), sim.data[predp][:,0],'b',label='x')
plt.plot(sim.trange(), sim.data[predp][:,1],'r',label='y')
plt.plot(sim.trange(), sim.data[predp][:,2],'g',label='xvel')
plt.plot(sim.trange(), sim.data[predp][:,3],'k',label='yvel');
plt.legend();
plt.show()
#%%
plt.figure()
plt.title('Control')
plt.plot(sim.trange(), sim.data[ctrlp][:,0],'b')
plt.plot(sim.trange(), sim.data[ctrlp][:,1],'r')

plt.plot(sim.trange(), sim.data[ctrlkf][:,0],'b',alpha=0.5)
plt.plot(sim.trange(), sim.data[ctrlkf][:,1],'r',alpha=0.5)
plt.show()
#%%
"""
plt.figure()
plt.title('upd2')
plt.plot(sim.trange(), sim.data[upp][:,0],'b--')
plt.plot(sim.trange(), sim.data[upp][:,1],'r--')
plt.plot(sim.trange(), sim.data[upp][:,2],'g--')
plt.plot(sim.trange(), sim.data[upp][:,3],'k--')

plt.figure()
plt.title('Kgains')
plt.plot(sim.trange(), sim.data[upp][:,4],'b')
plt.plot(sim.trange(), sim.data[upp][:,5],'r')
plt.plot(sim.trange(), sim.data[upp][:,6],'g')

plt.figure()
plt.plot(sim.trange(), sim.data[check][:,4],'b')
plt.plot(sim.trange(), sim.data[check][:,5],'r')
plt.plot(sim.trange(), sim.data[check][:,6],'g')

plt.plot(sim.trange(), sim.data[check][:,:4],'b')
#"""


#%% XY prane trajectory plot
plt.figure(figsize=(6,6))
plt.plot(0,0,'ko')
plt.title("Initialized at origin")
plt.plot(sim.data[xp],sim.data[yp],label='Actual')
plt.plot(sim.data[obsp][:,0],sim.data[obsp][:,1],label='Observation')
plt.plot(sim.data[predp][:,0],sim.data[predp][:,1],label='Prediction')
plt.legend();plt.show();plt.axis('equal')