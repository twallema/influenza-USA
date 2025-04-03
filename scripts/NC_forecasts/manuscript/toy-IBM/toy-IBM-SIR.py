##############
## PACKAGES ##
##############

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use(['dark_background'])
sns_c = sns.color_palette(palette='deep')
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Set global plot parameters. 
plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams['figure.dpi'] = 80

##################
## DEFINE MODEL ##
##################

class Body():

    def __init__(self, x0, y0, box_size, sigma, recov_rate, serorev_rate, is_S=True):
        
        # position and movement parameters
        self.x = x0
        self.y = y0
        self.loc_history = [(x0, y0)]
        self.box_size = box_size
        self.sigma = sigma

        # disease parameters
        self.recov_rate = recov_rate
        self.serorev_rate = serorev_rate

        # disease state
        self.is_S = is_S
        self.is_I = False
        self.is_R = not is_S

        # disease history
        self.S_history = [is_S]
        self.I_history = [False]
        self.R_history = [not is_S]

        pass
        

    def add_to_loc_history(self, x, y):
        self.loc_history.append((x, y))
    
    def add_to_S_history(self, x):
        self.S_history.append(x)
        
    def add_to_I_history(self, x):
        self.I_history.append(x)
        
    def add_to_R_history(self, x):
        self.R_history.append(x)
        
    @staticmethod
    def new_position(z, dz, box_size):
        zhat = abs(z + dz)

        if(zhat > box_size):
            zhat = 2*box_size - zhat
        return zhat

    def move(self):
        sigma = self.sigma * self.box_size
        dx = np.random.normal(loc=0.0, scale=sigma)
        dy = np.random.normal(loc=0.0, scale=sigma)
        self.x = Body.new_position(z=self.x, dz=dx, box_size=self.box_size)
        self.y = Body.new_position(z=self.y, dz=dy, box_size=self.box_size)
        self.add_to_loc_history(self.x, self.y)

    def distance(self, other): 
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def seroreverse(self):
        if (self.is_R):
            prob = np.exp(-1/self.serorev_rate)
            change_status = np.random.binomial(n=1, p=prob)
            if (change_status):
                self.is_S = True
                self.is_I = False
                self.is_R = False
    
    def recover(self):
        if (self.is_I):
            prob = np.exp(-1/self.recov_rate)
            change_status = np.random.binomial(n=1, p=prob)
            if (change_status):
                self.is_S = False
                self.is_I = False
                self.is_R = True    
    
    def infect(self, other, min_dist):

        # compute distance
        dist = self.distance(other)

        if dist <= min_dist:
            # infection event
            if ((self.is_I) & (other.is_S)):
                    prob = 1 #np.exp(-(1/self.transmission_coefficient)*dist)
                    change_status = np.random.binomial(n=1, p=prob)
                    if (change_status):
                        other.is_S = False
                        other.is_I = True
                        other.is_R = False        
            elif ((self.is_S) & (other.is_I)):
                prob = 1 #np.exp(-(1/self.transmission_coefficient)*dist)
                change_status = np.random.binomial(n=1, p=prob)
                if (change_status):
                    self.is_S = False
                    self.is_I = True
                    self.is_R = False

class Simulation():

    def __init__(self, N=1000, I_0=1, f_R=0, box_size=200, sigma=0.005, recov_rate=0.50, serorev_rate=1e-9):
        
        # initialise variables
        self.N = N
        self.I_0 = I_0
        self.f_R = f_R
        self.box_size = box_size
        
        # initialize first timestep
        ## Total population
        self.bodies = [
            Body(
                x0=np.random.uniform(low=0.0, high=self.box_size),
                y0=np.random.uniform(low=0.0, high=self.box_size),
                box_size=self.box_size,
                sigma=sigma,
                recov_rate=recov_rate,
                serorev_rate=serorev_rate,
                is_S=is_S,
            )
            for is_S in np.random.binomial(1, 1-f_R, N).astype(bool)
        ]

        ## Initial infected
        IDs = np.random.randint(0, N, size=I_0)
        for ID in IDs:
            self.bodies[ID].is_S = False
            self.bodies[ID].is_I = True
            self.bodies[ID].is_R = False
            self.bodies[ID].S_history = [False]
            self.bodies[ID].I_history = [True]
            self.bodies[ID].R_history = [False]

            
    def run(self, n_iter, t_intervention=None, strength_intervention=0.50, min_dist=1):
        self.n_iter = n_iter
        self.t_intervention = t_intervention
        
        for i in range(n_iter+1):
            # intervention
            if t_intervention:
                if i == t_intervention:
                    for b in self.bodies:
                        b.sigma *= 0.5*strength_intervention

            # recover
            for j, b1 in enumerate(self.bodies):
                b1.recover()
            # seroreverse
            for j, b1 in enumerate(self.bodies):
                b1.seroreverse()
            # infect
            for j, b1 in enumerate(self.bodies):
                for k, b2 in enumerate(self.bodies):
                    if (j < k):
                        b1.infect(b2, min_dist)
            # update health status
            [b.add_to_S_history(b.is_S) for b in self.bodies]
            [b.add_to_I_history(b.is_I) for b in self.bodies]
            [b.add_to_R_history(b.is_R) for b in self.bodies]
            # move
            [b.move() for b in self.bodies]
            print(f'Working on timestep: {i}')
        print('done simulating')
        
    def animate(self):
        fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(8, 4))
        # bodies
        ax=axes[0]
        ax.set(title='', xlim=(0, self.box_size), ylim=(0, self.box_size))
        ax.set_xticks([])
        ax.set_yticks([])
        scat = ax.scatter(x=[], y=[], c='black', s=4, marker='o')
        
        # SIR plot
        axes[1].set_xlim([0, self.n_iter])
        axes[1].set_ylim([0, 100])
        axes[1].set_xlabel('Time', fontsize=14)
        axes[1].set_ylabel('Fraction of population (%)', fontsize=14)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)

        time = []
        S_lst = []
        I_lst = []
        R_lst = []
        line_S, = axes[1].plot([], [], color=sns_c[2], linewidth=4)
        line_I, = axes[1].plot([], [], color=sns_c[3], linewidth=4)
        line_R, = axes[1].plot([], [], color=sns_c[7], linewidth=4)
        axes[1].legend(['S', 'I', 'R'], fontsize=14, loc=1)
        fig.tight_layout()
        
        if self.t_intervention:
            axes[1].axvline(x=self.t_intervention, ymin=0, ymax=1, color='white', linewidth=2, linestyle='--')

        def update(i):
            
            # scatterplot
            x = [b.loc_history[i][0] for b in self.bodies]
            y = [b.loc_history[i][1] for b in self.bodies]
            x = np.array(x).reshape(-1, 1)
            y = np.array(y).reshape(-1, 1)
            scat.set_offsets(np.concatenate((x, y), axis=1))
            # determine right color
            colors=[]
            for b in self.bodies:
                if b.I_history[i]: 
                    colors.append(sns_c[3])
                elif b.R_history[i]:
                    colors.append(sns_c[7])
                else:
                    colors.append(sns_c[2])
            scat.set_edgecolors(np.array(colors))
            scat.set_facecolors(np.array(colors))

            # disease progression plot
            S = sum([b.S_history[i] for b in self.bodies])
            I = sum([b.I_history[i] for b in self.bodies])
            R = sum([b.R_history[i] for b in self.bodies])

            time.append(i)
            S_lst.append(S/self.N*100)
            I_lst.append(I/self.N*100)
            R_lst.append(R/self.N*100)
            
            line_S.set_data(time[:i],S_lst[:i])
            line_I.set_data(time[:i],I_lst[:i])
            line_R.set_data(time[:i],R_lst[:i])
                
            return scat, line_S, line_I, line_R

        anim = FuncAnimation(fig, update, frames=n_iter, interval=100, blit=True)
        return anim
    
###############
## RUN MODEL ##
###############

# System setup
L = 200                 # box size
N = 2000                # individuals
I_0 = 1                 # initial infectees
f_R = 0.0               # initial recovered
# Disease parameters
min_dist = 5.5
recov_rate = 1/2
serorev_rate = 1e-9
# Simulation parameters
n_iter = 70             # timesteps
t_intervention = None
strength_intervention = 0.50
# Run simulation
sim = Simulation(N=N, box_size=L, I_0=I_0, f_R=f_R,
                 recov_rate=recov_rate, serorev_rate=serorev_rate)
sim.run(n_iter=n_iter, t_intervention=t_intervention, strength_intervention=strength_intervention, min_dist=min_dist)
# Animate
anim = sim.animate() 
HTML(anim.to_jshtml())
anim.save('animation.gif', writer='imagemagick', fps=8)
