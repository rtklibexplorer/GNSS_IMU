#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 Project:    Sensor Fusion for GNSS/IMU (Loosely Coupled Kalman Filter)
 File:       imu_plot.py
 Author:     Tim Everett
 Copyright:  (c) 2025 Tim Everett
 License:    BSD 3 License (see LICENSE file for details)

 Description:
     Plot functions for GNSS/IMU sensor fusion
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from imu_transforms import ECEF_to_NED, LLH_to_ECEF

def wrap(angle_deg):
    return (angle_deg + 180) % 360 - 180

def Plot_Results(in_gnss, in_imu, out_profile, title, run_dir):
    t_in = in_gnss[:,0] - in_gnss[0,0]
    t_out = out_profile[:,0] - in_gnss[0,0]
    coast_status = out_profile[:,10]

    ecef_pos_in = LLH_to_ECEF(in_gnss[:,1:4])
    ecef_origin = ecef_pos_in[run_dir]
    ned_pos_in, _ = ECEF_to_NED(ecef_pos_in, ecef_pos_in, ecef_origin)  # dummy vel
    ned_vel_in = in_gnss[:,14:17]
    
    ecef_pos_out = out_profile[:,1:4]
    ecef_vel_out = out_profile[:,4:7]
    ned_pos_out, ned_vel_out = ECEF_to_NED(ecef_pos_out, ecef_vel_out, ecef_origin)
    b_vel_out = out_profile[:,17:20]
    
    ned_pos_in_resamp = np.zeros_like(ned_pos_out)
    ned_vel_in_resamp = np.zeros_like(ned_vel_out)
    for i in range(3):
        ned_pos_in_resamp[:,i] = np.interp(t_out, t_in, ned_pos_in[:,i])
        ned_vel_in_resamp[:,i] = np.interp(t_out, t_in, ned_vel_in[:,i])
    
    rpy_out = np.rad2deg(np.unwrap(out_profile[:,7:10], axis=0))
    
    hspeed_in = np.sqrt(ned_vel_in[:,0]**2 + ned_vel_in[:,1]**2) / (t_in[1]-t_in[0]) 
    yaw_v_in = np.rad2deg(np.atan2(ned_vel_in[:,1], ned_vel_in[:,0]))
    
    # Calculate vel and pos differences between in and out
    vel_diff = ned_vel_out - ned_vel_in_resamp
    pos_diff = ned_pos_out - ned_pos_in_resamp
    vel_diff_h95 = np.percentile(abs(vel_diff), 95, axis=0)
    pos_diff_h95 = np.percentile(abs(pos_diff), 95, axis=0)

    # plot relative position in NED
    plt.figure()
    plt.title('%s: GNSS antenna position (NED)' % title)
    plt.plot(t_in, ned_pos_in[:,0], 'o-')
    plt.plot(t_in, ned_pos_in[:,1], 'o-')
    plt.plot(t_in, ned_pos_in[:,2], 'o-')
    for i in range(2):
        ix = np.where(out_profile[:,10]==i)[0]
        plt.plot(t_out[ix], ned_pos_out[ix,0], '.')
        plt.plot(t_out[ix], ned_pos_out[ix,1], '.')
        plt.plot(t_out[ix], ned_pos_out[ix,2], '.')
    plt.grid()
    plt.legend(['N_GNSS', 'E_GNSS', 'D_GNSS', 'N_OUT', 'E_OUT', 'D_OUT', 'N_OUT_coast', 'E_OUT_coast', 'D_OUT_coast'])
    plt.xlabel('secs')
    plt.ylabel('meters')
    plt.show()
    
    # plot velocity in NED
    plt.figure()
    plt.title('%s: GNSS antenna velocity (NED)' % title)
    plt.plot(t_in, ned_vel_in, 'o-')

    for i in range(2):
        ix = np.where(coast_status==i)[0]
        plt.plot(t_out[ix], ned_vel_out[ix], '.')
    plt.legend(['N_GNSS', 'E_GNSS', 'D_GNSS', 'N_OUT', 'E_OUT', 'D_OUT', 'N_OUT_coast', 'E_OUT_coast', 'D_OUT_coast'])
    plt.grid()
    plt.xlabel('secs')
    plt.ylabel('m/sec')
    plt.show()
    
    # plot velocity in body frame
    plt.figure()
    plt.title('%s: System origin velocity (body frame)' % title)
    plt.plot(t_out, b_vel_out, '.-')
    plt.legend(['X', 'Y','Z'])
    plt.grid()
    plt.xlabel('secs')
    plt.ylabel('m/sec')
    plt.show()
    
    # plot orientation in NED
    plt.figure()
    plt.title('%s: Orientation (NED)' % title)
    plt.plot(t_out, wrap(rpy_out), '.')
    ix = np.where(hspeed_in > 0.5)[0]
    plt.plot(t_in[ix], wrap(yaw_v_in[ix]),'.')
    plt.grid()
    plt.xlabel('secs')
    plt.ylabel('degrees')
    plt.legend(['R_OUT', 'P_OUT', 'Y_OUT', 'VEL_HDG'])
    plt.show()


    # plot differences between in and out
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('%s: Pos and Vel diffs' % title)
    axes[0].plot(t_out, pos_diff, '.-')
    axes[1].plot(t_out, vel_diff, '.-')
    axes[0].legend(['dpos_N: %.3f' % pos_diff_h95[0], 'dpos_E: %.3f' % pos_diff_h95[1], 'dpos_D: %.3f' % pos_diff_h95[2]])
    axes[1].legend(['dvel_N: %.3f' % vel_diff_h95[0], 'dvel_N: %.3f' % vel_diff_h95[1], 'dvel_N: %.3f' % vel_diff_h95[2]])
    axes[0].grid()
    axes[1].grid()
    axes[0].set_title('Position differences')
    axes[1].set_title('Velocity differences')
    axes[0].set_xlabel('secs')
    axes[1].set_xlabel('secs')
    axes[0].set_ylabel('meters')
    axes[1].set_ylabel('meters/sec')
    plt.show()
    
    # calculate imu misalignment
    yaw_v = np.interp(t_out, t_in, yaw_v_in)
    ix = np.where(b_vel_out[:,0] > 0.5)[0]
    err_rp = np.median(rpy_out[ix,0:2],axis=0)
    err_y = np.median(wrap(rpy_out[ix,2] - yaw_v[ix]))
    print('   IMU misalignment: R=%.3f P=%.3f, Y=%.3f (degrees)' % (err_rp[0], err_rp[1], err_y))
    
    
    return ned_pos_in, ned_pos_out

    
def Plot_Biases(bias_est, title, time0):
    t = bias_est[:,0] - time0
    
    # plot accel biases
    plt.figure()
    plt.title('%s: Accel biases' % title)
    plt.plot(t, bias_est[:,1:4], '.-')
    plt.legend(['X', 'Y', 'Z'])
    plt.grid()
    plt.xlabel('secs')
    plt.ylabel('m/sec^2')
    plt.show()
    
    # plot gyro biases
    plt.figure()
    plt.title('%s: Gyro biases' % title)
    plt.plot(t, np.rad2deg(bias_est[:,4:7]), '.-')
    plt.legend(['R', 'P', 'Y'])
    plt.grid()
    plt.xlabel('secs')
    plt.ylabel('deg/sec')
    plt.show()
        
    if bias_est.shape[1] == 13:
    # plot accel scale factors
        plt.figure()
        plt.title('%s: Accel scale factors' % title)
        plt.plot(t, 1 + bias_est[:,7:10], '.-')
        plt.legend(['X', 'Y', 'Z'])
        plt.grid()
        plt.xlabel('secs')
        plt.ylabel('Scale factor')
        plt.show()
        
        # plot gyro scale factors
        plt.figure()
        plt.title('%s: Gyro scale factors' % title)
        plt.plot(t, 1 +bias_est[:,10:13], '.-')
        plt.legend(['R', 'P', 'Y'])
        plt.grid()
        plt.xlabel('secs')
        plt.ylabel('Scale factor')
        plt.show()
        
def Plot_Uncertainties(KF_SD, title, time0):
    t = KF_SD[:,0] - time0
    
    # plot pos and vel stdevs
    plt.figure()
    plt.title('%s: Vel/Pos state stdevs (m/sec, m)' % title)
    plt.plot(t, KF_SD[:,4:10], '.-')
    plt.legend(['Xvel', 'Yvel', 'Zvel', 'Xpos', 'Ypos', 'Zpos'])
    plt.grid()
    plt.xlabel('secs')
    plt.show()
    
    # plot other stdevs
    plt.figure()
    plt.title('%s: Other state stdevs (deg, m/sec^2, deg/sec)' % title)
    plt.plot(t, np.rad2deg(KF_SD[:,1:4]), '.-')
    plt.plot(t, KF_SD[:,10:13], '.-')
    plt.plot(t, np.rad2deg(KF_SD[:,13:16]), '.-')
    plt.legend(['R', 'P', 'Y', 'Xacc_bias', 'Yacc_bias', 'Zacc_bias', 'R_bias', 'P_bias', 'Y_bias'])
    plt.grid()
    plt.xlabel('secs')
    plt.show()


        
def Plot_IMU(imu_data, gravity, title, time0):
    t = imu_data[:,0] - time0
    
    # plot raw accel measurements
    plt.figure()
    plt.title('%s: Raw Accel Measurements' % title)
    plt.plot(t, imu_data[:,1:4] * gravity, '.-')
    plt.legend(['X', 'Y', 'Z'])
    plt.grid()
    plt.xlabel('secs')
    plt.ylabel('m/sec^2')
    plt.show()
    
    # plot raw gyro measurements
    plt.figure()
    plt.title('%s: Raw Gyro Measurements' % title)
    plt.plot(t, np.rad2deg(imu_data[:,4:7]), '.-')
    plt.legend(['R', 'P', 'Y'])
    plt.grid()
    plt.xlabel('secs')
    plt.ylabel('deg/sec')
    plt.show()
