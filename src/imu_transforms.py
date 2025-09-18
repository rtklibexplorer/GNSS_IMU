#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 Project:    Sensor Fusion for GNSS/IMU (Loosely Coupled Kalman Filter)
 File:       imu_transforms.py
 Author:     Tim Everett
 Copyright:  (c) 2025 Tim Everett
 License:    BSD 3 License (see LICENSE file for details)

 Description:
     Coordinate transform functions for GNSS/IMU sensor fusion
===============================================================================
"""

import numpy as np
import datetime

def datetime_to_utc(timestamp_str):
    # Split to get milliseconds separately
    base, millis = timestamp_str.split('.')
    # Parse datetime part
    dt = datetime.datetime.strptime(base, '%Y/%m/%d %H:%M:%S')
    dt = dt.replace(tzinfo=datetime.timezone.utc)
    # Add milliseconds
    dt_with_millis = dt + datetime.timedelta(milliseconds=int(millis))
    # Convert to seconds since epoch
    seconds = dt_with_millis.timestamp()
    return seconds

def utc_secs_to_datetime(seconds):
    # Convert to datetime object with microsecond precision
    dt = datetime.datetime.utcfromtimestamp(seconds)
    # Extract milliseconds
    milliseconds = int(dt.microsecond / 1000)
    # Format as "YYYY/MM/DD HH:MM:SS.mmm"
    formatted = dt.strftime('%Y/%m/%d %H:%M:%S') + f'.{milliseconds:03d}'
    return formatted

def CTM_to_Euler(C):
    #CTM_to_Euler - Converts a coordinate transformation matrix to the
    #corresponding set of Euler angles
    #
    # Inputs:
    #   C       coordinate transformation matrix describing transformation from
    #           beta to alpha
    #
    # Outputs:
    #   eul     Euler angles describing rotation from beta to alpha in the 
    #           order roll, pitch, yaw(rad)

    eul = np.zeros(3)
    eul[0] = np.arctan2(C[1, 2], C[2, 2])  # roll
    if abs(C[0,2]) > 1:
        eul[1] = -np.pi/2 * np.sign(C[0,2])
    else:
        eul[1] = -np.arcsin(C[0, 2])          # pitch
    eul[2] = np.arctan2(C[0, 1], C[0, 0])  # yaw
    return eul

def Euler_to_CTM(eul):
    #Euler_to_CTM - Converts a set of Euler angles to the corresponding
    #coordinate transformation matrix
    # Inputs:
    #   eul     Euler angles describing rotation from beta to alpha in the 
    #           order roll, pitch, yaw(rad)
    #
    # Outputs:
    #   C       coordinate transformation matrix describing transformation from
    #           beta to alpha
    
    sphi, cphi = np.sin(eul[0]), np.cos(eul[0])
    stheta, ctheta = np.sin(eul[1]), np.cos(eul[1])
    spsi, cpsi = np.sin(eul[2]), np.cos(eul[2])
    
    return np.array([
        [ctheta*cpsi                  ,  ctheta*spsi                 , -stheta],
        [-cphi*spsi + sphi*stheta*cpsi,  cphi*cpsi + sphi*stheta*spsi, sphi*ctheta],
        [ sphi*spsi + cphi*stheta*cpsi, -sphi*cpsi + cphi*stheta*spsi, cphi*ctheta]
    ])

def compute_C_e_n(lat, lon):
    # Calculate ECEF to NED coordinate transformation matrix using (2.150) 
    sphi, cphi = np.sin(lat), np.cos(lat)
    slam, clam = np.sin(lon), np.cos(lon)
    return np.array([
        [-sphi*clam, -sphi*slam,  cphi],
        [   -slam,       clam,    0.0],
        [-cphi*clam, -cphi*slam, -sphi]
    ])

def pvc_ECEF_to_LLH(r_eb_e, v_eb_e, C_b_e):
    #ECEF_to_LLH - Converts Cartesian  to curvilinear position, velocity
    #resolving axes from ECEF to NED and attitude from ECEF- to NED-referenced
    # Inputs:
    #   r_eb_e        Cartesian position of body frame w.r.t. ECEF frame, resolved
    #                 along ECEF-frame axes (m)
    #   v_eb_e        velocity of body frame w.r.t. ECEF frame, resolved along
    #                 ECEF-frame axes (m/s)
    #   C_b_e         body-to-ECEF-frame coordinate transformation matrix
    #
    # Outputs:
    #   L_b           latitude (rad)
    #   lambda_b      longitude (rad)
    #   h_b           height (m)
    #   v_eb_n        velocity of body frame w.r.t. ECEF frame, resolved along
    #                 north, east, and down (m/s)
    #   C_b_n         body-to-NED coordinate transformation matrix

    # Parameters
    R_0 = 6378137
    e = 0.0818191908425

    lambda_b = np.arctan2(r_eb_e[1], r_eb_e[0])
    k1 = np.sqrt(1 - e**2) * np.abs(r_eb_e[2])
    k2 = e**2 * R_0
    beta = np.sqrt(r_eb_e[0]**2 + r_eb_e[1]**2)
    E = (k1 - k2) / beta
    F = (k1 + k2) / beta
    P = 4/3 * (E*F + 1)
    Q = 2 * (E**2 - F**2)
    D = P**3 + Q**2
    V = (np.sqrt(D) - Q)**(1/3) - (np.sqrt(D) + Q)**(1/3)
    G = 0.5 * (np.sqrt(E**2 + V) + E)
    T = np.sqrt(G**2 + (F - V * G) / (2 * G - E)) - G
    L_b = np.sign(r_eb_e[2]) * np.arctan((1 - T**2) / (2 * T * np.sqrt(1 - e**2)))
    h_b = (beta - R_0 * T) * np.cos(L_b) + \
          (r_eb_e[2] - np.sign(r_eb_e[2]) * R_0 * np.sqrt(1 - e**2)) * np.sin(L_b)

    C_e_n = compute_C_e_n(L_b, lambda_b)

    # Transform velocity using (2.73)
    v_eb_n = C_e_n @ v_eb_e
    # Transform attitude using (2.15)
    C_b_n = C_e_n @ C_b_e

    return L_b, lambda_b, h_b, v_eb_n, C_b_n

def pvc_LLH_to_ECEF(L_b, lambda_b, h_b, v_eb_n, C_b_n):
    #LLH_to_ECEF - Converts curvilinear to Cartesian position, velocity
    #resolving axes from NED to ECEF and attitude from NED- to ECEF-referenced
    # Inputs:
    #   L_b           latitude (rad)
    #   lambda_b      longitude (rad)
    #   h_b           height (m)
    #   v_eb_n        velocity of body frame w.r.t. ECEF frame, resolved along
    #                 north, east, and down (m/s)
    #   C_b_n         body-to-NED coordinate transformation matrix
    #
    # Outputs:
    #   r_eb_e        Cartesian position of body frame w.r.t. ECEF frame, resolved
    #                 along ECEF-frame axes (m)
    #   v_eb_e        velocity of body frame w.r.t. ECEF frame, resolved along
    #                 ECEF-frame axes (m/s)
    #   C_b_e         body-to-ECEF-frame coordinate transformation matrix
    
    # Parameters
    R_0 = 6378137
    e = 0.0818191908425
    
    # Calculate transverse radius of curvature using (2.105)
    R_E = R_0 / np.sqrt(1 - (e * np.sin(L_b))**2)
    
    # Convert position using (2.112)
    cos_lat = np.cos(L_b)
    sin_lat = np.sin(L_b)
    cos_long = np.cos(lambda_b)
    sin_long = np.sin(lambda_b)

    r_eb_e = np.array([
        (R_E + h_b) * cos_lat * cos_long,
        (R_E + h_b) * cos_lat * sin_long,
        ((1 - e**2) * R_E + h_b) * sin_lat
    ])

    # Calculate ECEF to NED coordinate transformation matrix using (2.150)
    C_e_n = compute_C_e_n(L_b, lambda_b)

    # Transform velocity using (2.73)
    v_eb_e = C_e_n.T @ v_eb_n
    
    # Transform attitude using (2.15)
    C_b_e = C_e_n.T @ C_b_n

    return r_eb_e, v_eb_e, C_b_e

def pv_ECEF_to_LLH(r_eb_e, v_eb_e):
    #pv_ECEF_to_LLH - Converts Cartesian to curvilinear position and velocity
    #resolving axes from ECEF to NED
    
    # Inputs:
    #   r_eb_e        Cartesian position of body frame w.r.t. ECEF frame, resolved
    #                 along ECEF-frame axes (m)
    #   v_eb_e        velocity of body frame w.r.t. ECEF frame, resolved along
    #                 ECEF-frame axes (m/s)
    #
    # Outputs:
    #   L_b           latitude (rad)
    #   lambda_b      longitude (rad)
    #   h_b           height (m)
    #   v_eb_n        velocity of body frame w.r.t. ECEF frame, resolved along
    #                 north, east, and down (m/s)

    # Parameters
    R_0 = 6378137
    e = 0.0818191908425

    # Convert position using Borkowski closed-form exact solution
    # From (2.113)
    lambda_b = np.arctan2(r_eb_e[1], r_eb_e[0])
    k1 = np.sqrt(1 - e**2) * np.abs(r_eb_e[2])
    k2 = e**2 * R_0
    beta = np.sqrt(r_eb_e[0]**2 + r_eb_e[1]**2)
    E = (k1 - k2) / beta
    F = (k1 + k2) / beta
    P = 4/3 * (E*F + 1)
    Q = 2 * (E**2 - F**2)
    D = P**3 + Q**2
    V = (np.sqrt(D) - Q)**(1/3) - (np.sqrt(D) + Q)**(1/3)
    G = 0.5 * (np.sqrt(E**2 + V) + E)
    T = np.sqrt(G**2 + (F - V * G) / (2 * G - E)) - G
    L_b = np.sign(r_eb_e[2]) * np.arctan((1 - T**2) / (2 * T * np.sqrt(1 - e**2)))
    h_b = (beta - R_0 * T) * np.cos(L_b) + \
          (r_eb_e[2] - np.sign(r_eb_e[2]) * R_0 * np.sqrt(1 - e**2)) * np.sin(L_b)

    # Calculate ECEF to NED coordinate transformation matrix using (2.150)
    C_e_n = compute_C_e_n(L_b, lambda_b)

    # Transform velocity using (2.73)
    v_eb_n = C_e_n @ v_eb_e
    return L_b, lambda_b, h_b, v_eb_n

def ECEF_to_NED(ecef_pos, ecef_vel, ecef_origin):
    """
    Convert an array of ECEF coordinates to NED using specified ecef origin.
    Parameters:
        ecef_coords (np.ndarray): Nx3 array of [x, y, z] ECEF coordinates.
    Returns:
        ned_coords (np.ndarray): Nx3 array of [north, east, down] coordinates.
    """
    
    # Rotation matrix from ECEF to NED
    lat0, lon0, _, _ = pv_ECEF_to_LLH(ecef_origin, [0,0,0])
    R = compute_C_e_n(lat0, lon0)
    
    # Vector from origin to each point
    delta = ecef_pos - ecef_origin

    # Apply rotation
    ned_pos = delta @ R.T
    ned_vel = ecef_vel @ R.T
    
    return ned_pos, ned_vel

def NED_to_ECEF(L_b, lambda_b, h_b, v_eb_n, C_b_n):
    #NED_to_ECEF - Converts curvilinear to Cartesian position, velocity
    #resolving axes from NED to ECEF and attitude from NED- to ECEF-referenced
    # Inputs:
    #   L_b           latitude (rad)
    #   lambda_b      longitude (rad)
    #   h_b           height (m)
    #   v_eb_n        velocity of body frame w.r.t. ECEF frame, resolved along
    #                 north, east, and down (m/s)
    #   C_b_n         body-to-NED coordinate transformation matrix
    #
    # Outputs:
    #   r_eb_e        Cartesian position of body frame w.r.t. ECEF frame, resolved
    #                 along ECEF-frame axes (m)
    #   v_eb_e        velocity of body frame w.r.t. ECEF frame, resolved along
    #                 ECEF-frame axes (m/s)
    #   C_b_e         body-to-ECEF-frame coordinate transformation matrix
    
    # Parameters
    R_0 = 6378137
    e = 0.0818191908425
    
    # Calculate transverse radius of curvature using (2.105)
    R_E = R_0 / np.sqrt(1 - (e * np.sin(L_b))**2)
    
    # Convert position using (2.112)
    cos_lat = np.cos(L_b)
    sin_lat = np.sin(L_b)
    cos_long = np.cos(lambda_b)
    sin_long = np.sin(lambda_b)

    r_eb_e = np.array([
        (R_E + h_b) * cos_lat * cos_long,
        (R_E + h_b) * cos_lat * sin_long,
        ((1 - e**2) * R_E + h_b) * sin_lat
    ])

    return r_eb_e #, v_eb_e, C_b_e

def LLH_to_ECEF(llh):
    
    # WGS84 constants
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = f * (2 - f)

    """
    Vectorized conversion from LLH to ECEF.
    llh: Nx3 array [lat (rad), lon (rad), h (m)]
    Returns Nx3 ECEF [x, y, z]
    """
    sin_lat = np.sin(llh[:, 0])
    cos_lat = np.cos(llh[:, 0])
    cos_lon = np.cos(llh[:, 1])
    sin_lon = np.sin(llh[:, 1])
    h = llh[:, 2]

    N = a / np.sqrt(1 - e2 * sin_lat**2)

    x = (N + h) * cos_lat * cos_lon
    y = (N + h) * cos_lat * sin_lon
    z = (N * (1 - e2) + h) * sin_lat

    return np.column_stack((x, y, z))

def NED_to_ECEF_vel(ned, llh):
    """
    Vectorized conversion from NED to ECEF velocity.
    ned: Nx3 array [vn, ve, vd]
    llh: Nx3 array [lat (deg), lon (deg), h (m)]
    Returns Nx3 ECEF velocity vectors
    """
    lat = np.radians(llh[:, 0])
    lon = np.radians(llh[:, 1])

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    # Precompute rotation matrices from NED to ECEF
    # Shape: Nx3x3
    R = np.empty((llh.shape[0], 3, 3))
    R[:, 0, 0] = -sin_lat * cos_lon
    R[:, 0, 1] = -sin_lon
    R[:, 0, 2] = -cos_lat * cos_lon
    R[:, 1, 0] = -sin_lat * sin_lon
    R[:, 1, 1] =  cos_lon
    R[:, 1, 2] = -cos_lat * sin_lon
    R[:, 2, 0] =  cos_lat
    R[:, 2, 1] =  0
    R[:, 2, 2] = -sin_lat

    # Apply matrix multiplication: ECEF_vel = R @ ned
    ecef_vel = np.einsum('nij,nj->ni', R, ned)
    return ecef_vel

def ECEF_to_NED_vel(vel_ecef, lat_deg, lon_deg):
    """
    Convert an array of ECEF velocity vectors to NED velocity vectors.
    
    Parameters:
    vel_ecef_array : ndarray
        An (N, 3) array of ECEF velocity vectors.
    lat_deg : float
        Latitude in degrees.
    lon_deg : float
        Longitude in degrees.
    
    Returns:
    vel_ned_array : ndarray
        An (N, 3) array of NED velocity vectors.
    """
    # Convert degrees to radians
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    # Create transformation matrix from ECEF to NED
    R = np.array([
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon),  np.cos(lat)],
        [-np.sin(lon),                np.cos(lon),               0],
        [-np.cos(lat) * np.cos(lon), -np.cos(lat) * np.sin(lon), -np.sin(lat)]
    ])

    # Convert to NED for each velocity vector
    vel_ecef = np.atleast_2d(vel_ecef)
    vel_ned = vel_ecef @ R.T  # Matrix multiplication
    
    return vel_ned

def LLH_NED_to_ECEF(llh, ned_vel):
    """
    Convert arrays of LLH positions and NED velocities to ECEF coordinates and velocities.
    Vectorized for large arrays.
    
    Parameters:
        llh_array: Nx3 array of [lat, lon, height] in degrees and meters.
        ned_vel_array: Nx3 array of [vn, ve, vd] in m/s.
    
    Returns:
        ecef_positions: Nx3 array of ECEF [x, y, z] in meters.
        ecef_velocities: Nx3 array of ECEF [vx, vy, vz] in m/s.
    """
    ecef_positions = LLH_to_ECEF(llh)
    ecef_velocities = NED_to_ECEF_vel(ned_vel, llh)
    return ecef_positions, ecef_velocities