This folder contains all of the past test I have done to test various architectural changes and design decisions in Graph Neural Networks. 
Each subfolder of the current folder has all of the relevant code to reproduce the results except for the nn_conv.py and utilities.py scripts. 
There should generally be an image or data in each subfolder to corroborate the results below.

Here is a list of things that I have tested and the results:
1. Optimal input parameters to the neural network in the CalTech paper: After training 4 networks with different input parameter for 15 epochs, 
the best combination for the data from the CalTech paper was x, y, a_{smooth}, and \nabla a_{smooth}.  