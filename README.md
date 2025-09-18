# GNSS_IMU
Python implementation of a loosely coupled GNSS/IMU sensor fusion based on matlab code and textbook by Paul Groves, "Principles of GNSS, Inertial, 
and Multisensor Integrated Navigation Systems" for demonstration and testing purposes.  

Includes two sample GNSS/IMU data sets, one collected from a vehicle, and the other collected while walking.

To process the default sample data set (from a vehicle), run the GNSS_IMU.py script.  To run with non-default configuration parameters, edit the drive_config.py file before running.  The default configuration parameters include several 15 second simulated GNSS outages to demonstrate performance under dead-reckoning conditions.  The first half of the data was taken on residential streets on a hill, and the second half was taken in a flat parking lot with tight maneuvers.

To process the walking sample data set, first modify the header of the GNSS_IMU.py script to select the walking data as indicated by the comments in the code, then run the script.  To run with non-default configuration parameters, edit the walk_config.py file before running.  The default configuration parameters include a 15 second simulated GNSS outage near the beginning of the data set.  The second half of the data includes a large section of GNSS float data.

This code was originally developed as a demonstration tool for a project to improve positioning for underground utility locators in challenging GNSS conditions.  Special thanks to Seescan Inc. for allowing the code to be shared as an open-source project.

Note that the GNSS input files and the GNSS/IMU output files are all in RTKLIB solution format, so can be plotted and explored with RTKLIB.  Also, there are a number of options that can be enabled or disabled in the configuration file including Kalman filter tuning parameters, simulated GNSS outages, initial yaw alignment, velocity-matching for near-real time solutions, zero-velocity updates, non-holonomic constraint updates, simulated IMU bias and scale factor errors. 

This is still a work in progress, so please comments on any issues or potential improvements you come across.  Pull requests to improve the code are welcome.

This software is provided “AS IS” without any warranties of any kind so please be careful, especially if using it in any kind of real-time application.



          
