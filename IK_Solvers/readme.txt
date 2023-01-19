# What's included
main_results.py - the primary script discussed in the paper
core.py - same as main_results.py, but the one I actively edit
NLinkArm3d.py - the modified version of Python Robotics' forward and inverse kinematics solvers
a bunch of .csv and .npy files holding different sets of generated data

# Dependencies
pythonlibraries: numpy, matplotlib, chess, tensorflow (keras, if tensorflow doesn't autoinstall it)

# What will happen
running the main_results.py file will...
1. train the Single Network architecture on all the lift moves (if you want to generate your own data, 
    uncomment the # GENERATE DATA sections in main_together() or main_separate(). WARNING, this can take very long!

2. Plot the training and validation loss

3. Plot predictions for 4 random moves from the test set

4. Show an animation of the arm following the predicted joint angle trajectory while displaying the reference trajectory waypoints
(CREATING THIS ANIMATION WAS A SIGNIFICANT EFFORT NOT INCLUDED IN THE REPORT BECAUSE THERE WAS NO WAY TO SHOW IT)

5. Run the speed test described in the report

6. train the Nested Network architecture on all the lift moves

7. Plot the traininig and validation loss for both networks

8. Plot the path for one random move from the test set

9. Show an animation of the arm following the predicted joint angle trajectory while displaying the reference trajectory waypoints

10. Run the speed test described in the report

This whole process takes approximately 6 minutes of runtime, not including any time it is paused displaying a figure.