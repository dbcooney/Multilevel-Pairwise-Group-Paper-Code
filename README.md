# Multilevel-Pairwise-Group-Paper-Code

This repository accompanies the preprint "Steady-State and Dynamical Behavior of a PDE Model of Multilevel Selection with Pairwise Group-Level Competition", by Konstantinos Alexiou and Daniel B. Cooney. It will include all of the scripts used to generate the figures in the paper. It includes all of the scripts used to compute numerical finite volume simulations of the dimorphic and trimorphic multilevel selection models and to generate all of the figures in the paper.

The repository is organized into three folders: Scripts, Figures, and Simulation Outputs. All of the scripts can be run using Python 3.10.

For reference, below is a list of figures and the scripts that were used to generate each figure. For scripts that are used to generate more than one figure or figure panel, see that the caption in the figure to choose parameter values for the simulation. 

* Figure 6.1: Run fvpairwisegroup.py with PD payoff parameters
* Figure 6.2: Run steady_compare_fv_pairwise.py with PD payoff parameters and setting group_type == "Fermi"
* Figures 6.3 and 6.4: Run loopfvpairwise.py with PD payoff parameters and group_type == "Fermi" to generate simulation outputs, then run PD_lambda_plots_Fermi.py to produce the figure
* Figure 6.5: Run fvpairwisegroup.py with HD payoff parameters
* Figure 6.6: Run steady_compare_fv_pairwise.py with HD payoff parameters and setting group_type == "Fermi"
* Figures 6.7 and 6.8: Run loopfvpairwise.py with HD payoff parameters and group_type == "Fermi" generate simulation outputs, then run HD_lambda_plots.py to produce the figure
* Figure 6.9: Run fvpairwisegroup.py with HD payoff parameters
* Figure B.1(left): Run steady_compare_fv_pairwise.py with PD payoff parameters and setting group_type == "local normalization"
* Figure B.1(right): Run steady_compare_fv_pairwise.py with PD payoff parameters and setting group_type == "Tullock" 
* Figures B.2(left) and B.3(left): Run loopfvpairwise.py with PD payoff parameters and group_type == "pairwise locally normalized" to generate simulation outputs, then run PD_lambda_plots_Local.py to produce the figures
* Figures B.2(left) and B.3(right): Run loopfvpairwise.py with PD payoff parameters and group_type == "Tullock" to generate simulation outputs, then run PD_lambda_plots_Tullock.py to produce the figures
  
