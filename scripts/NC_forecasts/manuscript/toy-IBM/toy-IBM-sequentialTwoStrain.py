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

    def __init__(self, x0, y0, box_size, transmission_coefficient_1, transmission_coefficient_2, sigma, recov_rate, is_S=True, is_R1=False, is_R2=False):
        
        # position and movement parameters
        self.x = x0
        self.y = y0
        self.loc_history = [(x0, y0)]
        self.box_size = box_size
        self.sigma = sigma

        # disease parameters
        self.recov_rate = recov_rate
        self.transmission_coefficient_1 = transmission_coefficient_1
        self.transmission_coefficient_2 = transmission_coefficient_2

        # disease state
        self.is_S = is_S
        self.is_I1 = False
        self.is_I2 = False
        self.is_R1 = is_R1
        self.is_R2 = is_R2
        self.is_I12 = False
        self.is_I21 = False
        self.is_R = False

        # disease history
        self.S_history = [is_S]
        self.I1_history = [False]
        self.I2_history = [False]
        self.R1_history = [is_R1]
        self.R2_history = [is_R2]
        self.I12_history = [False]
        self.I21_history = [False]
        self.R_history = [False]

        pass
        
    @staticmethod
    def add_to_loc_history(history, x, y):
        history.append((x, y))

    @staticmethod
    def add_to_history(history, x):
        history.append(x)

    @staticmethod
    def update_position(z, dz, box_size):
        zhat = abs(z + dz)
        if(zhat > box_size):
            zhat = 2*box_size - zhat
        return zhat

    def move(self):
        sigma = self.sigma * self.box_size
        dx = np.random.normal(loc=0.0, scale=sigma)
        dy = np.random.normal(loc=0.0, scale=sigma)
        self.x = Body.update_position(z=self.x, dz=dx, box_size=self.box_size)
        self.y = Body.update_position(z=self.y, dz=dy, box_size=self.box_size)
        self.add_to_loc_history(self.loc_history, self.x, self.y)

    def distance(self, other): 
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def recover(self):
        if ((self.is_I1) | (self.is_I2) | (self.is_I12) | (self.is_I21)):
            prob = np.exp(-1/self.recov_rate)
            change_status = np.random.binomial(n=1, p=prob)
            if (change_status):
                if self.is_I1:
                    self.is_I1 = False
                    self.is_R1 = True    
                elif self.is_I2:
                    self.is_I2 = False
                    self.is_R2 = True
                elif self.is_I12:
                    self.is_I12 = False
                    self.is_R = True
                elif self.is_I21:
                    self.is_I21 = False
                    self.is_R = True

    def infect_strain1(self, other, min_dist):

        # compute distance
        dist = self.distance(other)

        if dist <= min_dist:
                
            # infection event: S with strain 1
            if (((self.is_I1) | (self.is_I21)) & (other.is_S)):
                    prob = 1 #np.exp(-(1/self.transmission_coefficient_1)*dist)
                    change_status = np.random.binomial(n=1, p=prob)
                    if (change_status):
                        other.is_S = False
                        other.is_I1 = True
            elif ((self.is_S) & ((other.is_I1) & (other.is_I21))):
                prob = 1 #np.exp(-(1/self.transmission_coefficient_1)*dist)
                change_status = np.random.binomial(n=1, p=prob)
                if (change_status):
                    self.is_S = False
                    self.is_I1 = True

            # infection event: R2 with strain 1
            if (((self.is_I1) | (self.is_I21)) & (other.is_R2)):
                    prob = 1 #np.exp(-(1/self.transmission_coefficient_1)*dist)
                    change_status = np.random.binomial(n=1, p=prob)
                    if (change_status):
                        other.is_S = False
                        other.is_I21 = True
            elif ((self.is_R2) & ((other.is_I1) & (other.is_I21))):
                prob = 1 #np.exp(-(1/self.transmission_coefficient_1)*dist)
                change_status = np.random.binomial(n=1, p=prob)
                if (change_status):
                    self.is_S = False
                    self.is_I21 = True

    def infect_strain2(self, other, min_dist):

        # compute distance
        dist = self.distance(other)

        if dist <= min_dist:

            # infection event: S with strain 2
            if (((self.is_I2) | (self.is_I12)) & (other.is_S)):
                prob = 1 #np.exp(-(1/self.transmission_coefficient_2)*dist)
                change_status = np.random.binomial(n=1, p=prob)
                if (change_status):
                    other.is_S = False
                    other.is_I2 = True
            elif ((self.is_S) & ((other.is_I2) & (other.is_I12))):
                prob = 1 #np.exp(-(1/self.transmission_coefficient_2)*dist)
                change_status = np.random.binomial(n=1, p=prob)
                if (change_status):
                    self.is_S = False
                    self.is_I2 = True
        
            # infection event: R1 with strain 2
            if (((self.is_I2) | (self.is_I12)) & (other.is_R1)):
                prob = 1 #np.exp(-(1/self.transmission_coefficient_2)*dist)
                change_status = np.random.binomial(n=1, p=prob)
                if (change_status):
                    other.is_S = False
                    other.is_I12 = True
            elif ((self.is_R1) & ((other.is_I1) & (other.is_I12))):
                prob = 1 #np.exp(-(1/self.transmission_coefficient_2)*dist)
                change_status = np.random.binomial(n=1, p=prob)
                if (change_status):
                    self.is_S = False
                    self.is_I12 = True


class Simulation():

    def __init__(self, N=1000, I1_0=1, I2_0=1, f_R1=0, f_R2=0, box_size=200, transmission_coefficient_1=2, transmission_coefficient_2=2, sigma=0.005, recov_rate=0.50):
        
        # initialise variables
        self.N = N
        self.box_size = box_size

        # initialize first timestep
        ## Total population
        self.bodies=[]
        draws = np.random.multinomial(1, [1-f_R1-f_R2, f_R1, f_R2], size=N).astype(bool)
        for i in range(N):
            is_S, is_R1, is_R2 = list(draws[i,:])
            self.bodies.append(
                Body(
                    x0=np.random.uniform(low=0.0, high=self.box_size),
                    y0=np.random.uniform(low=0.0, high=self.box_size),
                    box_size=self.box_size,
                    transmission_coefficient_1=transmission_coefficient_1,
                    transmission_coefficient_2=transmission_coefficient_2,
                    sigma=sigma,
                    recov_rate=recov_rate,
                    is_S=is_S,
                    is_R1=is_R1,
                    is_R2=is_R2,
                    )   
            )

        ## Initial infected strain 1
        IDs = np.random.randint(0, N, size=I1_0)
        for ID in IDs:
            self.bodies[ID].is_S = False
            self.bodies[ID].is_I1 = True
            self.bodies[ID].is_R1 = False
            self.bodies[ID].is_R2 = False
            self.bodies[ID].S_history = [False]
            self.bodies[ID].I1_history = [True]
            self.bodies[ID].R1_history = [False]
            self.bodies[ID].R2_history = [False]

        ## initial infected strain 2
        IDs = np.random.randint(0, N, size=I2_0)
        for ID in IDs:
            self.bodies[ID].is_S = False
            self.bodies[ID].is_I2 = True
            self.bodies[ID].is_R1 = False
            self.bodies[ID].is_R2 = False
            self.bodies[ID].S_history = [False]
            self.bodies[ID].I2_history = [True]
            self.bodies[ID].R1_history = [False]
            self.bodies[ID].R2_history = [False]

            
    def run(self, n_iter, t_intervention=None, strength_intervention=0.50, min_dist=1):

        self.n_iter = n_iter
        self.t_intervention = t_intervention
        
        for i in range(n_iter+1):

            # intervention
            if t_intervention:
                if i == t_intervention:
                    for b in self.bodies:
                        b.sigma *= 0.5*strength_intervention
                        b.transmission_coefficient_1 *= strength_intervention
                        b.transmission_coefficient_2 *= strength_intervention
            # recover
            for j, b1 in enumerate(self.bodies):
                b1.recover()
            # infect with strain 1
            for j, b1 in enumerate(self.bodies):
                for k, b2 in enumerate(self.bodies):
                    if (j < k):
                        b1.infect_strain1(b2, min_dist)
            # infect with strain 2
            for j, b1 in enumerate(self.bodies):
               for k, b2 in enumerate(self.bodies):
                   if (j < k):
                       b1.infect_strain2(b2, min_dist)

            # update health status
            [b.add_to_history(b.S_history, b.is_S) for b in self.bodies]
            [b.add_to_history(b.I1_history, b.is_I1) for b in self.bodies]
            [b.add_to_history(b.I2_history, b.is_I2) for b in self.bodies]
            [b.add_to_history(b.R1_history, b.is_R1) for b in self.bodies]
            [b.add_to_history(b.R2_history, b.is_R2) for b in self.bodies]
            [b.add_to_history(b.I12_history, b.is_I12) for b in self.bodies]
            [b.add_to_history(b.I21_history, b.is_I21) for b in self.bodies]
            [b.add_to_history(b.R_history, b.is_R) for b in self.bodies]
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
        I1_lst = []
        I2_lst = []
        R1_lst = []
        R2_lst = []
        R_lst = []
        line_S, = axes[1].plot([], [], color=sns_c[2], linewidth=4)
        line_I1, = axes[1].plot([], [], color=sns_c[3], linewidth=4)
        line_I2, = axes[1].plot([], [], color=sns_c[4], linewidth=4)
        line_R1, = axes[1].plot([], [], color=sns_c[5], linewidth=4)
        line_R2, = axes[1].plot([], [], color=sns_c[6], linewidth=4)
        line_R, = axes[1].plot([], [], color=sns_c[7], linewidth=4)
        axes[1].legend(['S', 'I1', 'I2', 'R1', 'R2', 'R'], fontsize=14, loc=1)
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
                if ((b.I1_history[i]) | (b.I21_history[i])): 
                    colors.append(sns_c[3])
                elif ((b.I2_history[i]) | (b.I12_history[i])): 
                    colors.append(sns_c[4])
                elif b.R1_history[i]: 
                    colors.append(sns_c[5])
                elif b.R2_history[i]: 
                    colors.append(sns_c[6])                    
                elif b.R_history[i]:
                    colors.append(sns_c[7])
                else:
                    colors.append(sns_c[2])
            scat.set_edgecolors(np.array(colors))
            scat.set_facecolors(np.array(colors))

            # disease progression plot
            S = sum([b.S_history[i] for b in self.bodies])
            I1 = sum([b.I1_history[i] for b in self.bodies])
            I2 = sum([b.I2_history[i] for b in self.bodies])
            I12 = sum([b.I12_history[i] for b in self.bodies])
            I21 = sum([b.I21_history[i] for b in self.bodies])
            R1 = sum([b.R1_history[i] for b in self.bodies])
            R2 = sum([b.R2_history[i] for b in self.bodies])
            R = sum([b.R_history[i] for b in self.bodies])

            time.append(i)
            S_lst.append(S/self.N*100)
            I1_lst.append((I1+I21)/self.N*100)
            I2_lst.append((I2+I12)/self.N*100)
            R1_lst.append(R1/self.N*100)
            R2_lst.append(R2/self.N*100)
            R_lst.append(R/self.N*100)
            
            line_S.set_data(time[:i],S_lst[:i])
            line_I1.set_data(time[:i],I1_lst[:i])
            line_I2.set_data(time[:i],I2_lst[:i])
            line_R1.set_data(time[:i],R1_lst[:i])
            line_R2.set_data(time[:i],R2_lst[:i])
            line_R.set_data(time[:i],R_lst[:i])
                
            return scat, line_S, line_I1, line_I2, line_R1, line_R2, line_R

        anim = FuncAnimation(fig, update, frames=n_iter, interval=100, blit=True)
        return anim
    
###############
## RUN MODEL ##
###############

# System setup
L = 200                 # box size
N = 2000                # individuals
I_0 = 1                 # initial infectees
f_R1 = 0.0               # initial recovered
f_R2 = 0.0               # initial recovered
# Disease parameters
sigma = 0.005
min_dist = 5.5
recov_rate = 1/2
transmission_coefficient_1 = transmission_coefficient_2 = 10000
# Simulation parameters
n_iter = 60             # timesteps
t_intervention = None
strength_intervention = 0.50
# Run simulation
sim = Simulation(N=N, box_size=L, I1_0=I_0, I2_0=I_0, f_R1=f_R1, f_R2=f_R2, transmission_coefficient_1=transmission_coefficient_1,
                 transmission_coefficient_2=transmission_coefficient_2, recov_rate=recov_rate, sigma=sigma)
sim.run(n_iter=n_iter, t_intervention=t_intervention, strength_intervention=strength_intervention, min_dist=min_dist)
# Animate
anim = sim.animate() 
HTML(anim.to_jshtml())
anim.save('animation.gif', writer='imagemagick', fps=8)
