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
	return  P + N * (0.5 * gamma * (2.0 * j + 1.0) * (N ** -2.0) + alpha * (1.0 / (3.0 * (N** 3.0))) * (3.0 * (j ** 2.0)  + 3.0 * j + 1.0) )
	
N = 200
time_step = 0.01
time_length = 9600
#time_length = 40000

group_type = "payoff"


"""
Picking the group-level victory probability used in the simulation.
"""

victory_prob = "Fermi"
#victory_prob = "pairwise locally normalized"
#victory_prob = "Tullock"


"""
Choosing the payoff parameters for the game. 
"""

gamma = 3.5
alpha = -2.
beta = 1.

#gamma = 1.5
#alpha = -1.
#beta = -1.

P = 1.


"""
Setting up other parameters for the numerical simulation.
"""

s = 1.
lamb = 5.
inv_a = 2.

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
		
	
def group_function(x,type,alpha,gamma,P):
	
	if group_type == "coop":
		return x
	elif group_type == "payoff":
		return G(x,gamma,alpha,P)
		


		
def group_switch_prob(x,u,s,inv_a,group_type,alpha,gamma,P):
	
	focal_group = group_function(x,group_type,alpha,gamma,P)
	role_group = group_function(u,group_type,alpha,gamma,P)
	
	if victory_prob == "Fermi":
		return 0.5 + 0.5 * np.tanh(s * (focal_group - role_group))
	elif victory_prob == "pairwise locally normalized":
		num = focal_group - role_group
		denom = np.abs(focal_group) + np.abs(role_group)
		if denom == 0.:
			return 0.5
		else:
			return 0.5 + 0.5 * (num / denom)
	elif victory_prob == "Tullock":
		if gamma >= 0:
			G_min = P

		if focal_group == G_min and role_group == G_min:
			return 0.5
		else:
			num = (focal_group - (G_min))**(inv_a)
			denom = (focal_group - (G_min))**(inv_a) + (role_group - (G_min))**(inv_a)
			return num / denom
	
	
"""
Calculating average group-level victory probability for z-punisher groups over u-punisher
groups for (z,u) \in [i/N,(i+1)/N] \times [j/N,(j+1)/N] using the trapezoidal rule and
our finite volume assumption that the density is a piecewise-constant function taking
constant values on each grid volume.
"""	
	
	
	
def group_switch_terms(j,k,N,s,inv_a,group_type,alpha,gamma,P):
	
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
	
def group_reproduction_rate(f,N,s,inv_a,group_type,group_matrix,alpha,gamma,P):
	group_matrix = group_switch_matrix(N,s,inv_a,group_type,alpha,gamma,P)
	return (1. / N) * np.dot(group_matrix,f) 
	
def average_group_payoff(f,N,gamma,alpha,P):
	return (1. / np.float(N)) * np.dot(f,Gj_vec(index_holder,N,gamma,alpha,P))
	
def mean_coop(f,N,index_vec):
	return (1. / np.float(N)) * np.dot(index_vec,f_j)
	
def group_reproduction_formula(lamb,alpha,beta,gamma,P,s,inv_a,N,group_type):
	if beta < 0:
		rho10 = group_switch_prob(1.,0.,s,inv_a,group_type,alpha,gamma,P)
	elif beta > 0 and alpha < 0:
		rho10 = group_switch_prob(1.,beta / (-alpha),s,inv_a,group_type,alpha,gamma,P)
	if lamb == 0.:
		return rho10
	else:
		avg_rho = 0.5 - 0.5 * (1. / lamb) * (beta + alpha)
		return min(rho10,avg_rho)
	
group_reproduction_formula_vec = np.vectorize(group_reproduction_formula)
	
	

peak_holder = [float(np.argmax(f_j))/N]


Z = [[0,0],[0,0]]
levels = np.arange(0.,time_step * time_length+ time_step,time_step)
CS3 = plt.contourf(Z, levels, cmap=cmap.get_cmap('viridis_r'))
plt.clf()


lamb_min = 0.
#lamb_max = 25.
lamb_max = 120.
#lamb_step = 0.25
lamb_step = 0.25

lamb_range = np.arange(lamb_min,lamb_max+lamb_step,lamb_step)

lamb_list = []
group_reproduction_holder = []
group_reproduction_analytical_list = []
avg_G_holder = []
final_peak_holder = []
mean_coop_holder = []

"""
Running the finite volume simulations for our model of multilevel selection with
pairwise group-level competition for a range of relative strengths \lambda of group-level
competition. We use these simulations to generate data for figures comparing the average
group-level success and the average fraction of cooperation achieved after 9,600 time-steps
as a function of \lambda. 
"""
	

for lamb in lamb_range:
	peak_holder = []
	
	f_j = np.ones(N)
	
	

	for time in range(time_length):

	
		between_group_effect = between_group_term(f_j,N,s,inv_a,group_type,group_matrix,alpha,gamma,P)
		within_group_effect = within_group(f_j,N,alpha,beta,index_holder)
		righthandside = lamb * between_group_effect + within_group_effect
		f_j = f_j + time_step * righthandside
		
		peak_holder.append(float(np.argmax(f_j))/N)
		
	print(lamb)
	print(group_reproduction_rate(f_j,N,s,inv_a,group_type,group_matrix,alpha,gamma,P)[-1])
	print(group_reproduction_formula(lamb,alpha,beta,gamma,P,s,inv_a,N,group_type))
	lamb_list.append(lamb)
	group_reproduction_holder.append(group_reproduction_rate(f_j,N,s,inv_a,group_type,group_matrix,alpha,gamma,P)[-1])
	group_reproduction_analytical_list.append(group_reproduction_formula(lamb,alpha,beta,gamma,P,s,inv_a,N,group_type))
	avg_G_holder.append(average_group_payoff(f_j,N,gamma,alpha,P))
	final_peak_holder.append(peak_holder[-1])
	mean_coop_holder.append(mean_coop(f_j,N,index_vec))
		

plt.xlabel(r"Fraction of Cooperators ($x$)", fontsize = 20.)
plt.ylabel(r"Probability Density ($f(t,x)$)", fontsize = 20.)

plt.colorbar(CS3) 


print(group_reproduction_rate(f_j,N,s,inv_a,group_type,group_matrix,alpha,gamma,P))

print(0.5 + 0.5 * np.tanh(s * (G(1.,gamma,alpha,P) - G(0.,gamma,alpha,P))))

plt.figure(3)

#group_reproduction_analytical = group_reproduction_formula_vec(lamb_range,alpha,beta,gamma,s,N,type)
plt.plot(lamb_range,group_reproduction_holder, lw =5.)
plt.plot(lamb_range,group_reproduction_formula_vec(lamb_range,alpha,beta,gamma,P,s,inv_a,N,group_type),lw = 5., ls = '--')

plt.figure(4)
plt.plot(lamb_range,avg_G_holder, lw =5.)


plt.figure(5)
plt.plot(lamb_range,final_peak_holder, lw =5.)

plt.figure(6)
plt.plot(lamb_range,mean_coop_holder, lw =5.)


"""
Saving results to generate figures comparing average payoff, fraction of cooperation, and
average group-level victory probability for long-time population as a function of the 
strength of group-level competition \lambda.
"""


script_folder = os.getcwd()
pairwise_folder = os.path.dirname(script_folder)
file_path = pairwise_folder


if gamma == 1.5 and alpha == -1. and beta == -1. and P == 0. and victory_prob == "Fermi":
	file1 = file_path + "/Simulation_Outputs/Rho1xPDgamma1p5P0.txt"
	file2 = file_path + "/Simulation_Outputs/AvgGPDgamma1p5P0.txt"
	file3 = file_path + "/Simulation_Outputs/PeakPDgamma1p5P0.txt"
	file4 = file_path + "/Simulation_Outputs/MeanCoopPDEgamma1p5P0.txt"
	
elif gamma == 1.5 and alpha == -1. and beta == -1. and P == 1. and victory_prob == "Fermi":
	file1 = file_path + "/Simulation_Outputs/Rho1xPDgamma1p5P1.txt"
	file2 = file_path + "/Simulation_Outputs/AvgGPDgamma1p5P1.txt"
	file3 = file_path + "/Simulation_Outputs/PeakPDgamma1p5P1.txt"
	file4 = file_path + "/Simulation_Outputs/MeanCoopPDEgamma1p5P1.txt"
	
elif gamma == 1.5 and alpha == -1. and beta == -1. and victory_prob == "pairwise locally normalized":
	file1 = file_path + "/Simulation_Outputs/Rho1xPDgamma1p5_Local.txt"
	file2 = file_path + "/Simulation_Outputs/AvgGPDgamma1p5_Local.txt"
	file3 = file_path + "/Simulation_Outputs/PeakPDgamma1p5_Local.txt"
	file4 = file_path + "/Simulation_Outputs/MeanCoopPDEgamma1p5_Local.txt"
	
elif gamma == 1.5 and alpha == -1. and beta == -1. and victory_prob == "Tullock":
	file1 = file_path + "/Simulation_Outputs/Rho1xPDgamma1p5_Tullock.txt"
	file2 = file_path + "/Simulation_Outputs/AvgGPDgamma1p5_Tullock.txt"
	file3 = file_path + "/Simulation_Outputs/PeakPDgamma1p5_Tullock.txt"
	file4 = file_path + "/Simulation_Outputs/MeanCoopPDEgamma1p5_Tullock.txt"
	
		
	
elif gamma == 3.5 and alpha == -2. and beta == 1. and P == 1. and victory_prob == "Fermi":
	file1 = file_path + "/Simulation_Outputs/Rho1xHDgamma3p5P1.txt"
	file2 = file_path + "/Simulation_Outputs/AvgGHDgamma3p5P1.txt"
	file3 = file_path + "/Simulation_Outputs/PeakHDgamma3p5P1.txt"
	file4 = file_path + "/Simulation_Outputs/MeanCoopHDgamma3p5P1.txt"
	

	
f1 = open(file1,'w+')
f1.write("between-group selection strength lambda")
f1.write('\n')
f1.write(str(lamb_list)[1:-1])
f1.write('\n')
f1.write("replication rate of all-cooperator group rho(1,x)")
f1.write('\n')
f1.write(str(group_reproduction_holder)[1:-1])
f1.write('\n')
f1.write("analytical formula for rho(1,x)")
f1.write('\n')
f1.write(str(group_reproduction_analytical_list)[1:-1])
f1.write('\n')
f1.close()
	
f2 = open(file2,'w+')
f2.write("between-group selection strength lambda")
f2.write('\n')
f2.write(str(lamb_list)[1:-1])
f2.write('\n')
f2.write("average payoff G(x) at steady state")
f2.write('\n')
f2.write(str(avg_G_holder)[1:-1])
f2.write('\n')
f2.close()


f3 = open(file3,'w+')
f3.write("between-group selection strength lambda")
f3.write('\n')
f3.write(str(lamb_list)[1:-1])
f3.write('\n')
f3.write("modal level of cooperation at steady state")
f3.write('\n')
f3.write(str(final_peak_holder)[1:-1])
f3.write('\n')
f3.close()


f4 = open(file4,'w+')
f4.write("between-group selection strength lambda")
f4.write('\n')
f4.write(str(lamb_list)[1:-1])
f4.write('\n')
f4.write("average level of cooperation at steady state")
f4.write('\n')
f4.write(str(mean_coop_holder)[1:-1])
f4.write('\n')
f4.close()


print(group_reproduction_analytical_list)
print(avg_G_holder)
plt.show()


