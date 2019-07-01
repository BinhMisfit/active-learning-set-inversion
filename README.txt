READ ME (SET INVERSION)

*** INTRODUCTION ***
There are 3 distinct folders for 3 experiments
1. Ordinary Shapes (2D Circle, 3D Sphere, ...)
2. Lotka–Volterra (predator - preys)
3. SIR Model (The 17th century plague in Eyam)






*** STRUCTURE OF EACH FOLDER ***
===== Set_Inversion_Shapes =====
1. Folder 'Functions': containing all supporting functions
2. Train_Models.py: Script to train models
3. Plot_Results.py: Script to plot the results (after training models)

===== Set_Inversion_LV =====
1. Folder 'Functions': containing all supporting functions
2. Find_GT.py (Need to run first): Script to find the groundtruth of Lotka-Volterra problem --> Running this script will create Groundtruth figure + Folder 'Points_GT', which will contain groundtruth points (Coordinates of points and their true lables)
3. Train_Models: Script to train models
4. Plot_Results.py: Script to plot the results (after training models)

===== Set_Inversion_Shapes =====
(Same with Set_Inversion_LV)
This folder is organized as the same with Folder 'Set_Inversion_LV'
However, there is a file 'Data.csv'
1. Data.csv: The collected data about the 17th-century plague in Eyam





*** HOW TO USE ***
===== Set_Inversion_Shapes =====
[1] Set up parameters
- Support_Shape: List of shapes that OASIS supports (Do not change this)
- Shape_vec: List of shapes that you want OASIS to construct pre-image (Shape_vec must be a subset of Support_Shape)
- Random_Type: Type of sampling initial random points. OASIS supports Latin Hyper Cube, Uniform (Normal), and Sobol algorithm.
- K_samp: Number of initial random points
- Active_Points: Number of points for active learning
- tol_threshold: Threshold used in the optimization step when find a point on the decision boundary (Smaller is better, but difficult to solve --> 0.55 is optimal)
- method_numb_idx: List of index of classifers that you want to run (0: SVM, 1: KNN, 2: MLP, 3: RF) 
- Optimal_Run (boolean type): Whether run the optimal approach or not. Optimal Run is to seek the optimal hyperparameters in the given set of values that suit the best for classifiers
- [parameter]_Optimal_Run: Set of values for hyperparameters in which Optimal Run will search the best values. (only when Optimal_Run = True)
- [parameter]_Default: Default parameters for classifiers (only when Optimal_Run = False)

[2] DO NOT CHANGE THIS
List of variables that you do not want to mess up (including state space, set up for the shapes, ...)

[3] Run the script 'Train_Models.py' will create 5 files
- Model ... [sav file]: Trained Models
- Points ... [pkl file]: Coordinates of Points in state space and its predicted labels
- Result_Accuracy ... [csv file]: Summary of accuracy, processing time
- Result_Optimal ... [csv file]: Summary of optimal hyperparameters and the training accuracy when using those values
- Result_Optimal_Full ... [csv file]: Full information of hyperparameters when training, including their training accuracy at each iteration, and at each value of hyperparameter.

[4] Run the script 'Plot_Results.py' will create predicted result
[Note] Only plot the results after you have trained
- Shape_vec: List of shapes that you want to plot the results
- Other parameters: Make sure match with the parameters in Train_Models.py (Random_Type, K_samp, Active_Points)

===== Set_Inversion_LV =====
[1] Set up parameters
- Same with Set_Inversion_Shapes

[2] Run the script 'Find_GT.py' first to create the ground truth of the Lotka-Volterra
- We have already run it and saved the ground truth in Folder 'Points_GT'
- Set up parameters for the Lotka-Volterra
- N, X0, Time, M0, p1, p3: As described in the paper 
	+ N: Timestep from 0 to Time
	+ X0: Initial population [pop of preys, pop of predators]
	+ Time: The max period we want to simulate
	+ M0: Threshold of condition for population of prey (pop of prey needs to be always higher than M0)
	+ p1: birth rate of prey
	+ p3: death rate of predator
- If you change the above paras --> change in Train_Models.py as well

[3] Run the script 'Train_Models.py' to create models and predicted points
- Same with Set_Inversion_Shapes

[4] Run the script 'Plot_Results.py' to plot the predicted points
- Same with Set_Inversion_Shapes

===== Set_Inversion_SIR (Mostly the same with Set_Inversion_LV) =====
[1] Set up parameters
- Same with Set_Inversion_Shapes
- data: Read the Data.csv file

[2] Run the script 'Find_GT.py' first to create the ground truth of the SIR model
- We have already run it and saved the ground truth in Folder 'Points_GT'
- Set up parameters for the SIR
- N, X0, Time, alpha, R0: As described in the paper 
	+ N: Timestep from 0 to Time
	+ X0: Initial population [pop of susceptible, infected, removed] --> Initial pop of each compartment S, I, R
	+ Time: The max period we want to simulate
	+ alpha: removal rate
	+ R0: basic reproduction number
- If you change the above paras --> change in Train_Models.py as well

[3] Run the script 'Train_Models.py' to create models and predicted points
- Same with Set_Inversion_Shapes

[4] Run the script 'Plot_Results.py' to plot the predicted points
- Same with Set_Inversion_Shapes





*** QUICK TRY ***
- Currently, the default parameters are given as stated in the paper. However, it will take times to finish experiments. To make a quick attempt, please set the value of parameter Active_Points to be smaller (e.g. 10 ~ 20) 
- Remember to change in all related files (Train_Models.py, Plot_Results.py, ...)