# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 16:51:37 2024

@author: derek
"""

import odrive
from odrive.enums import *
import time
from datetime import datetime

import pygame
    
import pandas as pd
import numpy as np


filename = str(datetime.now())
filename = filename.replace(":","")

time_start = time.time()
vel_CMD = 0
velRange = 22
Kp = 1.5
Ki = 0

Im_CMD = 0
e_Im_past = 0
e_Im_integral = 0

state = 'good'
time_stuck = 0

def log_data_to_excel(index, time, velocity, other_variable, position, current, current_cmd):
    # Create a DataFrame with the data
    data = {
        'Time': time,
        'Velocity': velocity,
        'Velocity Command': other_variable,
        'Position': position,
        'Current': current,
        'Current Command': current_cmd
    }
    df = pd.DataFrame(data, index=[index])

    # Read existing data from the Excel file
    try:
        existing_df = pd.read_excel('logs/'+filename+'.xlsx', index_col=0)
    except FileNotFoundError:
        existing_df = pd.DataFrame()

    # Concatenate existing data with new data
    updated_df = pd.concat([existing_df, df])

    # Write the updated DataFrame to the Excel file
    updated_df.to_excel('logs/'+filename+'.xlsx')

    
def odrive_init():
    print('Initialize odrive')
    # Connect to ODrive
    odrv = odrive.find_any()
    #  serial_number=self.yaml['serial-number']
    
    # Check if ODrive is connected
    if odrv is not None:
        print("ODrive connected")
    else:
        print("ODrive not found")
        exit()
    
    # Access motor axis
    axis0 = odrv.axis0
    
    
    # axis0.config.startup_motor_calibration
    # axis0.config.startup_encoder_offset_calibration
    axis0.requested_state = AxisState.FULL_CALIBRATION_SEQUENCE
    axis0.requested_state = 8
    
    
    axis0.controller.config.control_mode = ControlMode.VELOCITY_CONTROL
    axis0.controller.config.vel_ramp_rate = 50.0
    axis0.controller.config.input_mode = InputMode.VEL_RAMP
    
    axis0.controller.config.vel_gain = 0.01
    axis0.controller.config.vel_integrator_gain = 0.01
    
    axis0.controller.config.vel_limit = 200  # Set velocity limit in counts/s
        
    bus_voltage = odrv.vbus_voltage
    
    # Print the bus voltage
    print("Bus Voltage:", bus_voltage, "V")
    print(axis0.current_state)
    
    return odrv, axis0
    

def joystick_init():
    print('Initialize joysticks')
    # Initialize Pygame
    pygame.init()
    pygame.joystick.init()
    
    # Check for connected joysticks
    num_joysticks = pygame.joystick.get_count()
    if num_joysticks < 1:
        print("No joystick detected.")
    else:
        print('joystick detected.')
    # Initialize the first joystick
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    return joystick


def get_drive_joystick(joystick):
    y = -joystick.get_axis(1)
    if abs(y) <= 0.1:
        y = 0
    return y


def get_deadman(kill):
    kill = joystick.get_button(0)
    return kill
    

def clamp_CMD(u, limit):
    if u >= limit:
        u = limit
    elif u <= -limit:
        u = -limit
    return u


def check_stuck(vel, Im, state, time_stuck):
    
    vel_stuck_bound = .125
    
    if abs(vel) <= vel_stuck_bound and state == 'good':
        time_stuck = time.time()
        state ='hold up'
        
    elif time.time() - time_stuck >= 5.0 and abs(vel) <= vel_stuck_bound and state =='hold up':
        state='stuck'
        time_stuck = time.time()
        
    elif time.time() - time_stuck >= 5.0 and state == 'stuck':
        state = 'good'
        
        
    return state, time_stuck


def get_motor_variables(axis0):
    vel = axis0.pos_vel_mapper.vel
    pos = axis0.pos_vel_mapper.pos_abs
    Im = axis0.motor.foc.Iq_measured
    return vel, pos, Im
    

def get_inputs(joytick):
    y = get_drive_joystick(joystick)
    kill = get_deadman(joystick)
    return y, kill
   

def telemetry_readout(Im, vel, vel_CMD, state, kill):
    print()
    print('Im:', round(Im,2), 'A')
    print('Vel_m:',round(vel,2), 't/s') 
    print('Vel_CMD:', round(vel_CMD, 2), 't/s')
    print('System State:', state)
    print('Deadman:', kill)   
    return 'done'


def current_filter(Im_hist):
    y = (1/len(Im_hist))*(sum(Im_hist))
    return y


Im_hist = np.array([0, 0, 0, 0, 0])
cutoff_freq = 1.0
sample_time = 0.001

odrv, axis0 = odrive_init()
joystick = joystick_init()

i = 0
running = True
while running:
    # Check for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
    pygame.time.wait(10)
    axis0.watchdog_feed()
 
    vel, pos, Im = get_motor_variables(axis0)
    y, kill = get_inputs(joystick);
    
    telemetry_readout(Im, vel, vel_CMD, state, kill)

    Im_hist[0] = Im_hist[1]
    Im_hist[1] = Im_hist[2]
    Im_hist[2] = Im_hist[3]
    Im_hist[3] = Im_hist[4]
    Im_hist[4] = Im
    
    Im = current_filter(Im_hist)

    if y == 0:
        if state == 'good' or state == 'hold up':
            Im_CMD = 0
            
            e_Im = Im_CMD - Im
            
            e_Im_integral += Ki * 0.01*(e_Im + e_Im_past) / 2
            e_Im_past = e_Im
            
            # Set velocity
            vel_CMD = Kp*e_Im + e_Im_integral
        
        elif state == 'stuck':
            vel_CMD = 5
        
        state, time_stuck = check_stuck(vel, Im, state, time_stuck)
        
    else:
        vel_CMD = y*velRange
        Im_CMD = 0
       
    if kill == 1:
        vel_CMD = y*velRange
        Im_CMD = 0
        
    vel_CMD = clamp_CMD(vel_CMD, velRange)
    axis0.controller.input_vel = vel_CMD
        
        
    log_data_to_excel(i, time.time() - time_start, vel, vel_CMD, pos, Im, Im_CMD)
    
    if axis0.current_state != 8:
        state='bad'
        odrv, axis0 = odrive_init()
        joystick = joystick_init()
        state='good'
        
    i+=1
    
# Clean up
pygame.quit()

    