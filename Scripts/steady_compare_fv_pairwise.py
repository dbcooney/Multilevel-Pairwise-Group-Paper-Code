"""
Script used to generate Figures 6.2, 6.6, and B.1, displaying sample densities achieved 
under numerical simulations of our PDE model of multilevel selection after 9,600 timesteps
for different strengths \lambda of group-level competition. The scenarios considered 
include the PD game with the Fermi group-level update rule (Figure 6.2), the HD game
with the Fermi rule (Figure 6.6), and the PD game with the pairwise local and Tullock
group-level update rules (Figure 6.1). 
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import os


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
	
N = 200
time_step = 0.003
time_length = 9600




"""
Picking the group-level victory probability used in the simulation.
"""

type = "payoff"
group_type = "Fermi"
#group_type = "local normalization"
#group_type = "Tullock"



"""
Choosing the payoff parameters for the game. 
"""

#PD Payoffs no shadw
"""
gamma = 2.
alpha = -1.
beta = -1.
"""

P = 1.


#PD Payoffs shadow
"""
gamma = 1.5
alpha = -1.
beta = -1.
"""

#HD Payoffs shadow
gamma = 3.5
alpha = -2.
beta = 1.


#HD Payoffs no shadow

gamma = 4.
alpha = -2.
beta = 1.


"""
Setting up other parameters for the numerical simulation.
"""

s = 1.
lamb = 2.
inv_a = 2.
lamb = 6.

if gamma + 2. * alpha >= 0:
	lamb1 = 4.
	lamb2 = 8.
	lamb3 = 24.
	lamb4 = 48.
	lamb5 = 96.
	
elif gamma + 2. * alpha < 0:
	if beta > 0:
		lamb1 = 12.
		lamb2 = 24.
		lamb3 = 48.
		lamb4 = 96.
		lamb5 = 192.
	elif beta <= 0.:
		lamb1 = 24.
		lamb2 = 48.
		lamb3 = 96.
		lamb4 = 192.
		lamb5 = 384.
	
		

"""
if gamma + 2. * alpha >= 0:
	lamb1 = 4.
	lamb2 = 8.
	lamb3 = 24.
	lamb4 = 48.
	lamb5 = 96.

elif gamma + 2. * alpha < 0:
	lamb1 = 24.
	lamb2 = 48.
	lamb3 = 96.
	lamb4 = 192.
	lamb5 = 384.
"""






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
	
	
	upper_half = upper_flux_up * upper_flux * left_roll + upper_flux_down * lower_flux * f 
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
	
		return G(x,gamma,alpha,P)
		
		
"""
Defining the victory probability of a focal group featuring a fraction x of cooperators
when paired against a u-cooperator group in a group-level conflict.
""" 		
		
def group_switch_prob(x,u,s,inv_a,group_type,alpha,gamma,P):
	
	focal_group = group_function(x,group_type,alpha,gamma,P)
	role_group = group_function(u,group_type,alpha,gamma,P)
	if group_type == "Fermi":
		return 0.5 + 0.5 * np.tanh(s * (focal_group - role_group))
	elif group_type == "local normalization":
		if np.abs(focal_group) == 0. and np.abs(role_group) == 0.:
			return 0.5
		else:
			return 0.5 + 0.5 * ((focal_group - role_group) / (np.abs(focal_group) + np.abs(role_group)))
	#return 0.5 + 0.5 * (focal_group - role_group)
	elif group_type == "Tullock":
		if focal_group == 0. and role_group == 0.:
			return 0.5
		else:
			G_min = 0.
			num = (focal_group - (G_min))**(inv_a)
			denom = (focal_group - (G_min))**(inv_a) + (role_group - (G_min))**(inv_a)
			#print(denom)
			return num / denom
			
	
"""
Calculating average group-level victory probability for x-cooperator groups over u-cooperator
groups for (x,u) \in [i/N,(i+1)/N] \times [j/N,(j+1)/N] using the trapezoidal rule and
our finite volume assumption that the density is a piecewise-constant function taking
constant values on each grid volume.
"""
def group_switch_terms(j,k,N,s,inv_a,type,alpha,gamma,P):
	
	ll = group_switch_prob(float(j)/N,float(k)/N,s,inv_a,group_type,alpha,gamma,P)
	lr = group_switch_prob((j+1.)/N,float(k)/N,s,inv_a,group_type,alpha,gamma,P)
	ul = group_switch_prob(float(j)/N,(k+1.)/N,s,inv_a,group_type,alpha,gamma,P)
	ur = group_switch_prob((j+1.)/N,(k+1.)/N,s,inv_a,group_type,alpha,gamma,P)
	
	return 0.25 * (ll + lr + ul + ur) 
	
	
	
"""
Further characterizing the group-level victory probabilities for each grid volume, and
using these calculations to describe the effect of pairwise group-level competition on
the dynamics of multilevel selection.
"""
		
	
def group_switch_matrix(N,s,inv_a,group_type,alpha,gamma,P):
	
	matrix = np.zeros((N,N))
	for j in range(N):
		for k in range(N):
			matrix[j,k] = group_switch_terms(j,k,N,s,inv_a,group_type,alpha,gamma,P)
	return matrix
	

group_matrix = group_switch_matrix(N,s,inv_a,group_type,alpha,gamma,P)




def between_group_term(f,N,s,inv_a,group_type,group_matrix,alpha,gamma,P):
	return (1. / N) * f * ( np.dot(group_matrix,f) - np.dot(np.transpose(group_matrix),f))

	
	
	
#A = righthand(f_j, Gj_vec(index_holder,N,2.0,1.0),N)

peak_holder = [float(np.argmax(f_j))/N]


Z = [[0,0],[0,0]]
levels = np.arange(0.,time_step * time_length+ time_step,time_step)
CS3 = plt.contourf(Z, levels, cmap=cmap.get_cmap('viridis'))
plt.clf()

f_j1 = theta_vec(index_holder,N,1.0)
f_j2 = theta_vec(index_holder,N,1.0)
f_j3 = theta_vec(index_holder,N,1.0)
f_j4 = theta_vec(index_holder,N,1.0)
f_j5 = theta_vec(index_holder,N,1.0)



"""
Running the finite volume simulations for our model of multilevel selection with
pairwise group-level competition for each relative strength \lambda of group-level  
competition considered. We use these simulations to generate plots of the densities 
achieved for each value of \lambda after 9,600 time-steps.
"""

for time in range(time_length):
	
	
	
	between_group_effect1 = between_group_term(f_j1,N,s,inv_a,group_type,group_matrix,alpha,gamma,P)
	within_group_effect1 = within_group(f_j1,N,alpha,beta,index_holder)
	righthandside1 = lamb1 * between_group_effect1 + within_group_effect1
	f_j1 = f_j1 + time_step * righthandside1
	
	between_group_effect2 = between_group_term(f_j2,N,s,inv_a,group_type,group_matrix,alpha,gamma,P)
	within_group_effect2 = within_group(f_j2,N,alpha,beta,index_holder)
	righthandside2 = lamb2 * between_group_effect2 + within_group_effect2
	f_j2 = f_j2 + time_step * righthandside2
	
	between_group_effect3 = between_group_term(f_j3,N,s,inv_a,group_type,group_matrix,alpha,gamma,P) 
	within_group_effect3 = within_group(f_j3,N,alpha,beta,index_holder)
	righthandside3 = lamb3 * between_group_effect3 + within_group_effect3
	f_j3 = f_j3 + time_step * righthandside3
	
	between_group_effect4 = between_group_term(f_j4,N,s,inv_a,group_type,group_matrix,alpha,gamma,P)
	within_group_effect4 = within_group(f_j4,N,alpha,beta,index_holder)
	righthandside4 = lamb4 * between_group_effect4 + within_group_effect4
	f_j4 = f_j4 + time_step * righthandside4
	
	between_group_effect5 = between_group_term(f_j5,N,s,inv_a,group_type,group_matrix,alpha,gamma,P)
	within_group_effect5 = within_group(f_j5,N,alpha,beta,index_holder)
	righthandside5 = lamb5 * between_group_effect5 + within_group_effect5
	f_j5 = f_j5 + time_step * righthandside5
	
	#print (1.0 / N) * np.sum(f_j)


if gamma == 1.5 and alpha == -1.0 and beta == -1.0:
	plt.plot(np.arange(0.5/N,1.0+0.5/N,1.0/N),f_j1, color = plt.cm.YlOrRd(0.2), lw = 6., label = r"$\lambda = 24$")
	plt.plot(np.arange(0.5/N,1.0+0.5/N,1.0/N),f_j2, color = plt.cm.YlOrRd(0.4), lw = 6., label = r"$\lambda = 48$")
	plt.plot(np.arange(0.5/N,1.0+0.5/N,1.0/N),f_j3, color = plt.cm.YlOrRd(0.6), lw = 6., label = r"$\lambda = 96$")
	plt.plot(np.arange(0.5/N,1.0+0.5/N,1.0/N),f_j4, color = plt.cm.YlOrRd(0.8), lw = 6., label = r"$\lambda = 192$")
	plt.plot(np.arange(0.5/N,1.0+0.5/N,1.0/N),f_j5, color = plt.cm.YlOrRd(1.0), lw = 6., label = r"$\lambda = 384$")
	

elif gamma == 2. and alpha == -1.0 and beta == -1.0:
	plt.plot(np.arange(0.5/N,1.0+0.5/N,1.0/N),f_j1, color = plt.cm.YlOrRd(0.2), lw = 6., label = r"$\lambda = 4$")
	plt.plot(np.arange(0.5/N,1.0+0.5/N,1.0/N),f_j2, color = plt.cm.YlOrRd(0.4), lw = 6., label = r"$\lambda = 8$")
	plt.plot(np.arange(0.5/N,1.0+0.5/N,1.0/N),f_j3, color = plt.cm.YlOrRd(0.6), lw = 6., label = r"$\lambda = 24$")
	plt.plot(np.arange(0.5/N,1.0+0.5/N,1.0/N),f_j4, color = plt.cm.YlOrRd(0.8), lw = 6., label = r"$\lambda = 48$")
	plt.plot(np.arange(0.5/N,1.0+0.5/N,1.0/N),f_j5, color = plt.cm.YlOrRd(1.0), lw = 6., label = r"$\lambda = 96$")


elif gamma == 3.5 and alpha == -2.0 and beta == 1.0:
	plt.plot(np.arange(0.5/N,1.0+0.5/N,1.0/N),f_j1, color = plt.cm.YlOrRd(0.2), lw = 6., label = r"$\lambda = 12$")
	plt.plot(np.arange(0.5/N,1.0+0.5/N,1.0/N),f_j2, color = plt.cm.YlOrRd(0.4), lw = 6., label = r"$\lambda = 24$")
	plt.plot(np.arange(0.5/N,1.0+0.5/N,1.0/N),f_j3, color = plt.cm.YlOrRd(0.6), lw = 6., label = r"$\lambda = 48$")
	plt.plot(np.arange(0.5/N,1.0+0.5/N,1.0/N),f_j4, color = plt.cm.YlOrRd(0.8), lw = 6., label = r"$\lambda = 96$")
	plt.plot(np.arange(0.5/N,1.0+0.5/N,1.0/N),f_j5, color = plt.cm.YlOrRd(1.0), lw = 6., label = r"$\lambda = 192$")
	



elif gamma == 4. and alpha == -2. and beta == 1.:
	plt.plot(np.arange(0.5/N,1.0+0.5/N,1.0/N),f_j1, color = plt.cm.YlOrRd(0.2), lw = 6., label = r"$\lambda = 4$")
	plt.plot(np.arange(0.5/N,1.0+0.5/N,1.0/N),f_j2, color = plt.cm.YlOrRd(0.4), lw = 6., label = r"$\lambda = 8$")
	plt.plot(np.arange(0.5/N,1.0+0.5/N,1.0/N),f_j3, color = plt.cm.YlOrRd(0.6), lw = 6., label = r"$\lambda = 24$")
	plt.plot(np.arange(0.5/N,1.0+0.5/N,1.0/N),f_j4, color = plt.cm.YlOrRd(0.8), lw = 6., label = r"$\lambda = 48$")
	plt.plot(np.arange(0.5/N,1.0+0.5/N,1.0/N),f_j5, color = plt.cm.YlOrRd(1.0), lw = 6., label = r"$\lambda = 96$")





label_left_shift = 0.07

if gamma + 2.0 * alpha > 0:
	label_height = 17.5
else: 
	label_height = 10.5

if gamma + 2.0 * alpha < 0:
		plt.axvline(x = gamma / (-2. * alpha), lw = 5., ls = '--', color = 'k', alpha = 0.9)
		plt.axvline(x = (gamma + alpha) / (-alpha), lw = 5., ls = '--', color = 'k', alpha = 0.9)
		plt.annotate(r"$\overline{x}$", xy = (0.75 - label_left_shift,label_height) ,fontsize = 20.)
		plt.annotate(r"$x^*$", xy = (0.875 - label_left_shift,label_height) ,fontsize = 20.)
		

plt.axvline(x = -beta / alpha, lw = 5., ls = '--', color = 'k', alpha = 0.9)
plt.annotate(r"$x_{eq}$", xy = ((-beta/alpha) - label_left_shift,label_height), fontsize =20.)

if beta > 0:
	if gamma + 2. * alpha >= 0:
		plt.axis([0.,1.,0.,14.])
	else:
		plt.axis([0.,1.,0.,20.])
elif beta <= 0.:
	plt.axis([0.,1.,0.,14.])

plt.legend(loc = 'upper left', fontsize = 16.)

plt.xlabel(r"Fraction of Cooperators ($x$)", fontsize = 20., labelpad = 10.)
plt.ylabel(r"Probability Density ($f(x)$)", fontsize = 20.)

plt.tight_layout()


script_folder = os.getcwd()
pairwise_folder = os.path.dirname(script_folder)

if gamma == 1.5 and alpha == -1.0 and beta == -1.0:
	if group_type == "local normalization":
		plt.savefig(pairwise_folder + "/Figures/PDsteadyghostlocal.png")
	elif group_type == "Tullock":
		plt.savefig(pairwise_folder + "/Figures/PDsteadyghostTullock.png")
	elif group_type == "Fermi":
		plt.savefig(pairwise_folder + "/Figures/PDsteadyghostFermi.png")
elif gamma == 2. and alpha == -1.0 and beta == -1.0:
	if group_type == "Fermi":
		plt.savefig(pairwise_folder + "/Figures/PDsteadynoghostFermi.png")
elif gamma == 3.5 and alpha == -2. and beta == 1.:
	if group_type == "Fermi":
		plt.savefig(pairwise_folder + "/Figures/HDsteadyghostFermi.png")
elif gamma == 4. and alpha == -2. and beta == 1.:
	if group_type == "Fermi":
		plt.savefig(pairwise_folder + "/Figures/HDsteadynoghostFermi.png")
		




plt.show()




