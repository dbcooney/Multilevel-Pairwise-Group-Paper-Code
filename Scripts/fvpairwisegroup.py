"""
This script gives the baseline numerical simulation for the upwind finite volume scheme
for our PDE model of multilevel selection with pairwise competition between groups. This
script is used to generate Figures 6.1, 6.5, and 6.9 for the respective cases of PD, HD, 
and SH games, plotting snapshots of the numerical solution at various points in time.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmap


from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


"""
Defining the average payoff of group members in terms of the game-theoretic parameters.
"""

def G(x,gamma,alpha,P):
	return P + gamma * x + alpha * (x ** 2.0)
	
def G_j(j,N,gamma,alpha,P):
	return P + N * (0.5 * gamma * (2.0 * j + 1.0) * (N ** -2.0) + alpha * (1.0 / (3.0 * (N** 3.0))) * (3.0 * (j ** 2.0)  + 3.0 * j + 1.0) )
	
N = 800
time_step = 0.003
time_length = 9600




"""
Picking the group-level victory probability used in the simulation.
"""

group_type = "payoff"
#group_type = "Fermi"
#group_type = "local normalization"
#group_type = "Tullock"



"""
Choosing the payoff parameters for the game. 
"""

#PD shadow payoffs
gamma =1.5
alpha = -1.
beta = -1.

#HD shadow payoffs
"""
gamma = 3.5
alpha = -2.
beta = 1.
"""

#SH Payoffs
"""
gamma = 0.
alpha = 2.
beta = -1.
"""

#SH payoff
#P = 2.

#PD and HD Payoff
P = 1.


"""
Setting up other parameters for the numerical simulation.
"""

s = 1.
lamb = 14.

if gamma == 0:
	time_length = 250
elif beta > 0: 
	if lamb == 0.1:
		time_length = 1600
	elif lamb == 14.:
		time_length = 1200
elif beta < 0: 
	if lamb == 0.1:
		time_length = 1000
	elif lamb == 14.:
		time_length = 1400

def theta_init(j,N,theta):
	return N ** (1.0 - theta) * (((N - j) ** theta) - ((N - j - 1.0) ** theta) )
	
theta_vec = np.vectorize(theta_init)

G_vec = np.vectorize(G)
Gj_vec = np.vectorize(G_j)

index_holder = np.zeros(N)
for j in range(N):
	index_holder[j] = j
	
f_j = np.ones(N)
f_j = theta_vec(index_holder,N,1.0)

index_vec = np.zeros(N)
for j in range(N):
	index_vec[j] = float(j) / N

#f_j = np.zeros(N)
"""
for j in range(N):
	if j < 50:
		f_j[j] = 2.0
"""
	
print(index_holder)
print(Gj_vec(index_holder,N,2.0,1.0,P))



"""
Defining fluxes across volume boundaries (corresponding to the effects of individual-
level replication events) and characterizing how these fluxes impact the within-group
replicator dynamics. 
"""



def flux_right(j,N,beta,alpha):
	return ((j+1.0) / N) * (1.0 - (j+1.0) / N) * (beta + alpha * ((j+1.0)/N))
	
def flux_left(j,N,beta,alpha):
	return ((np.float(j)) / N) * (1.0 - (np.float(j)) / N) * (beta + alpha * ((np.float(j))/N))
	



flux_right_vec = np.vectorize(flux_right)
flux_left_vec = np.vectorize(flux_left)

def within_group(f,N,alpha,beta,index_holder):
	left_roll = np.roll(f,-1)
	left_roll[-1] = 0.
	right_roll = np.roll(f,1)
	right_roll[0] = 0.
	
	upper_flux = flux_right_vec(index_holder,N,beta,alpha)
	lower_flux = flux_left_vec(index_holder,N,beta,alpha)
	
	upper_flux_up = np.where(upper_flux < 0.0,1.0,0.0)
	upper_flux_down = np.where(upper_flux > 0.0,1.0,0.0)
	
	lower_flux_up = np.where(lower_flux < 0.0,1.0,0.0)
	lower_flux_down = np.where(lower_flux > 0.0,1.0,0.0)
	
	
	upper_half = upper_flux_up * upper_flux * left_roll + upper_flux_down * upper_flux * f 
	lower_half = lower_flux_up * lower_flux * f + lower_flux_down * lower_flux * right_roll
	return N*(-upper_half + lower_half)
	
	
	

def peak_minus(lamb,gamma,theta):
	sqrt_term = (lamb * gamma)**(2) - 4. * (3. + lamb) * (lamb * gamma - lamb - 2. * theta - 1.0)
	return (lamb * gamma - np.sqrt(sqrt_term)) / (6. + 2. * lamb)
	
def pd_peak(lamb,gamma,alpha,beta,theta):
	denom = -2.0 * (3.0 + lamb) * alpha
	radicand = (lamb * gamma - 2.0 * alpha - 2.0 * np.abs(beta)) ** 2.0
	radicand += 4.0 * (3.0 + lamb) * alpha * (lamb * (gamma + alpha) - (np.abs(beta) - alpha) * theta - np.abs(beta) )
	num = (lamb * gamma - 2.0 * alpha - 2.0 * np.abs(beta)) - np.sqrt(radicand)
	return num / denom
	
def hd_peak(lamb,gamma,alpha,beta,theta):
	denom = 2.0 * (3.0 + lamb) * np.abs(alpha)
	radicand = (lamb * gamma + 2.0 * beta + 2.0 * np.abs(alpha)) ** 2.0
	radicand -= 4.0 * (3.0 + lamb) * np.abs(alpha) * (lamb * (gamma - np.abs(alpha)) - (np.abs(alpha) - beta) * theta + beta)
	num = lamb * gamma + 2.0 * beta + 2.0 * np.abs(alpha) - np.sqrt(radicand)
	return num / denom
	
	

"""
Defining terms used to describe between-group competition. 
"""		
	
	
def group_function(x,group_type,alpha,gamma,P):
	
	if group_type == "coop":
		return x
	elif group_type == "payoff":
		return G(x,gamma,alpha,P)
		
def group_switch_prob(x,u,s,group_type,alpha,gamma,P):
	
	focal_group = group_function(x,group_type,alpha,gamma,P)
	role_group = group_function(u,group_type,alpha,gamma,P)
	return 0.5 + 0.5 * np.tanh(s * (focal_group - role_group))
	#return 0.5 + 0.5 * (focal_group - role_group)
	
	
	
"""
Calculating average group-level victory probability for z-punisher groups over u-punisher
groups for (z,u) \in [i/N,(i+1)/N] \times [j/N,(j+1)/N] using the trapezoidal rule and
our finite volume assumption that the density is a piecewise-constant function taking
constant values on each grid volume.
"""	

	
def group_switch_terms(j,k,N,s,group_type,alpha,gamma,P):
	
	ll = group_switch_prob(float(j)/N,float(k)/N,s,group_type,alpha,gamma,P)
	lr = group_switch_prob((j+1.)/N,float(k)/N,s,group_type,alpha,gamma,P)
	ul = group_switch_prob(float(j)/N,(k+1.)/N,s,group_type,alpha,gamma,P)
	ur = group_switch_prob((j+1.)/N,(k+1.)/N,s,group_type,alpha,gamma,P)
	
	return 0.25 * (ll + lr + ul + ur) 
	
	
"""
Further characterizing the group-level victory probabilities for each grid volume, and
using these calculations to describe the effect of pairwise group-level competition on
the dynamics of multilevel selection.
"""	
	
def group_switch_matrix(N,s,group_type,alpha,gamma,P):
	
	matrix = np.zeros((N,N))
	for j in range(N):
		for k in range(N):
			matrix[j,k] = group_switch_terms(j,k,N,s,group_type,alpha,gamma,P)
	return matrix
	
print(group_switch_matrix(N,s,"coop",alpha,gamma,P))
A = group_switch_matrix(N,s,"coop",alpha,gamma,P)
print(A + np.transpose(A))
print(np.sum(A + np.transpose(A)))
		

print(group_function(1.,"coop",alpha,gamma,P))
print(group_switch_prob(0.,1.,s,"coop",alpha,gamma,P))
print(group_switch_terms(N,N,N,s,"coop",alpha,gamma,P))
group_matrix = group_switch_matrix(N,s,group_type,alpha,gamma,P)


def between_group_term(f,N,s,type,group_matrix,alpha,gamma,P):
	
	return (1. / N) * f * ( np.dot(group_matrix,f) - np.dot(np.transpose(group_matrix),f))
	#return (1. / N) * f * np.dot(group_matrix,f)	
def group_reproduction_rate(f,N,s,group_type,group_matrix,alpha,gamma,P):
	group_matrix = group_switch_matrix(N,s,group_type,alpha,gamma,P)
	return (1. / N) * np.dot(group_matrix,f) 
	
	


peak_holder = [float(np.argmax(f_j))/N]


Z = [[0,0],[0,0]]
levels = np.arange(0.,time_step * time_length+ time_step,time_step)
CS3 = plt.contourf(Z, levels, cmap=cmap.get_cmap('viridis_r'))
plt.clf()



"""
Running the finite volume simulations for our model of multilevel selection with
pairwise group-level competition, and plotting sample snapshots obtained from the numerical
solution obtained at different points of time.  
"""

for time in range(time_length):
	
	#if time % 100 == 0:
	if time % 100 == 0:
		plt.plot(np.arange(0.5/N,1.0+0.5/N,1.0/N),f_j, color = cmap.viridis_r((np.float(time) / time_length)**1), lw = 3.)
	
	elif gamma == 0. and time % 50 == 0:
		plt.plot(np.arange(0.5/N,1.0+0.5/N,1.0/N),f_j, color = cmap.viridis_r((np.float(time) / time_length)**1), lw = 3.)
	
	between_group_effect = between_group_term(f_j,N,s,group_type,group_matrix,alpha,gamma,P)
	within_group_effect = within_group(f_j,N,alpha,beta,index_holder)
	righthandside = lamb * between_group_effect + within_group_effect
	print(np.sum(righthandside))
	f_j = f_j + time_step * righthandside

	
	print((1.0 / N) * np.sum(f_j))
	peak_holder.append(float(np.argmax(f_j))/N)
	

	


plt.xlabel(r"Fraction of Cooperators ($x$)", fontsize = 20.)
plt.ylabel(r"Probability Density ($f(t,x)$)", fontsize = 20.)

plt.colorbar(CS3) 

if 0. < -beta / alpha < 1.:
	plt.axvline(x = -beta / alpha, lw = 4., ls = '--', color = 'k', alpha = 0.8)
	if lamb == 0.1:
		xeq_label_height = 12.
	else:
		xeq_label_height = 6.
	plt.annotate(r"$x_{\mathrm{eq}}$", xy = (- beta / alpha - 0.1,xeq_label_height), fontsize = 16.)
	
if gamma <= 0:
	plt.axis([0.,1.,0.,10.])
if gamma > 0:
	if beta > 0:
		if lamb == 0.1:
			plt.axis([0.,1.,0.,14.])
		elif lamb == 14.:
			plt.axis([0.,1.,0.,8.])
	 
		



plt.tight_layout()


"""
Saving figure with plot of time-dependent solutions.
"""

if gamma == 0. and lamb == 0.:
	plt.savefig("SH_pairwise_lamb_0.png")
	
elif gamma == 0. and lamb == 1.:
	plt.savefig("SH_pairwise_lamb_1.png")
	
elif gamma == 3.5 and alpha == -2. and beta == 1.:
	if lamb == 0.1:
		plt.savefig("HD_pairwise_trajectories_lamb_0p1.png")
	elif lamb == 14.:
		plt.savefig("HD_pairwise_trajectories_lamb_14.png")
		
elif gamma == 1.5 and alpha == -1. and beta == -1.:
	if lamb == 0.1:
		plt.savefig("PD_pairwise_trajectories_lamb_0p1.png")
	elif lamb == 14.:
		plt.savefig("PD_pairwise_trajectories_lamb_14.png")
	



a = np.array([0.0,1.0,2.0])
print(np.roll(a,-1))
b = 0.0000
#print(b == 0.0)
c = flux_right_vec(index_holder,N,-1.,-1.)
print(np.where(c < 0.0,1.0,0.0))



print(np.dot(f_j,index_vec) / np.sum(f_j))

print(group_reproduction_rate(f_j,N,s,group_type,group_matrix,alpha,gamma,P))
if lamb > 0:
	print(0.5 + (0.5 / lamb) * (-beta - alpha))
print(0.5 + 0.5 * np.tanh(s * (G(1.,gamma,alpha,P) - G(0.,gamma,alpha,P))))


plt.show()


