import serial
import csv
import time
from datetime import datetime

import math 
import numpy as np
import matplotlib.pyplot as plt 
from sympy import *


"""
Use the timings in the C code to make assumptions about the time between each read, and just calc vel. based on that
"""

# how many encoder values?
num_values = 5
order =  [ 0,            1,            2,                   3,            4]
labels = ["Renishaw 1", "Renishaw 2", "Renishaw Adjusted", "Zettlex", "Netzer"]
ordered_lables = [labels[s] for s in order]
# Configure the serial port
serial_port = 'COM4'  # Replace with your serial port
baud_rate = 9600              # Set your baud rate
timeout = 1                   # Read timeout in seconds

# Open the serial port
ser = serial.Serial(serial_port, baud_rate, timeout=timeout)

# Open the CSV file for writing
timestamp = datetime.now()
timestamp = str(timestamp).replace(":","_")
timestamp = str(timestamp).replace(" ","_")
# prev_time = timestamp.time()
init_time = time.monotonic_ns() #* 1e-9
print('init time', init_time)
#prev_time = init_time
dt = 0.0
time_array = np.array([[np.double(0.0)]])



log_name = input("Enter file name for log:  ")
csv_file = 'logs/Encoder-Testbed-Log_' + log_name + '.csv'
with open(csv_file, mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    # Write the header
    csv_writer.writerow( ordered_lables)

    print(f"Listening on {serial_port} at {baud_rate} baud rate...")
    fuckin = True

    while fuckin:
        if not ser.isOpen():
            ser.open()
        try:
            data_array = np.array([[1,2,3,4,5]], dtype=np.double)
            while True:
                # Read a line from the serial port
                line = ser.readline().decode('utf-8').strip()
                # print(f"Serial Raw: {line}")
                # curr_time = datetime.now()
                try:
                    if line:
                        #dt = float(curr_time - prev_time)
                        #prev_time = curr_time
                        curr_time = [time.monotonic_ns() - init_time]
                        # Split the line into values
                        print('elapsed time', curr_time)
                        pos_values = line.split(',')
                        pos_values = [np.double(s) for s in pos_values]
                        # Write the values to the CSV file
                        # csv_writer.writerow(curr_time + pos_values)
                        print(f"Written to CSV: {pos_values}")
                        
                        #np.append(data_array, values, 0)
                        #data_array = np.append(data_array, np.atleast_2d(np.array(curr_time)), axis=0)
                        data_array = np.append(data_array, np.atleast_2d(np.array(pos_values)), axis=0)
                        time_array = np.append(time_array, np.atleast_2d(np.array(curr_time)), axis=0) #, dtype= np.float64))     
                except:
                    pass
                # time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("Program terminated by user")
            fuckin = False
            isShit = False
        except ValueError:
            print("Program terminated by being shit")    
            time.sleep(.25)
            isShit = True
        finally:
            ser.close()
            print("Serial port closed")

            # setting up arrays
            data_array = data_array[5:] #exclude first 5 encoder readings rows
            time_array = time_array[5:] #exclude first 5 encoder readings rows
            for i in range(len(data_array)):
                for j in range(num_values):
                    # data_array[i,j] = float(data_array[i,j])
                    if data_array[i,j] > 180.0:
                        data_array[i,j] -= 360
            
            
            plt.figure("pos-error")
            pos_err_array = np.empty_like(data_array)
            for i in [0,1,2,3,4]:
                pos_err_array[:,i] = data_array[:,i]-data_array[:,2]
                plt.plot(time_array, pos_err_array[:,i], label = labels[order[i]])
            plt.legend()
            plt.savefig(log_name+'_pos-error.png')

            plt.figure("raw data")
            for i in range(num_values):
                plt.plot(time_array,data_array[:,i], label=ordered_lables[i])
            plt.legend()
            plt.savefig(log_name+'_raw-data.png')

            def vel_calc1():
                for i in range(1, data_array.shape[0]): #iterate by row
                # dt = time_array[i] - time_array[i-1]
                # dt = 105072463.8
                    dt = np.double(0.1)
                    #dt = np.double(time_array[i]- time_array[i-1])
                    for j in range(num_values): #by columns (will be 5 times)
                        dy = data_array[i][j]-data_array[i-1][j]
                        dy = np.double(min(abs(dy),abs(dy+360),abs(dy-360)))
                        # print( 'dy', dy )
                        vel_array[i][j] = (dy/dt)/(360.0)
            
            
            def vel_calc2(group):
                cnt = 0
                dt = 0.1 * group * 2
                dy = np.zeros(5)
                avrg1 = np.zeros(5)
                avrg2 = np.zeros(5)
                for i in range(data_array.shape[0]): #iterate by row
                    cnt += 1.0
                    for j in range(num_values): #by columns (will be 5 times)
                        if cnt <=group:
                            avrg1[j] = np.double(avrg1[j]+data_array[i][j])
                            if cnt == group:
                                avrg1 /= cnt
                        elif cnt <= group*2:
                            avrg2[j] = np.double(avrg2[j]+data_array[i][j])
                            if cnt == group*2:
                                avrg2 /= group
                                dy = abs(avrg1 - avrg2)
                                dy = [np.double(min(abs(p),abs(p+360),abs(p-360))) for p in dy]
                                cnt = group
                                avrg1 = avrg2
                                avrg2 = np.zeros(5)
                    vel_array[i] = [np.double(p/dt)/360.0 for p in vel_array[i]]
           
            plt.figure('velocity')
            plt.ylabel('rev/s')
            vel_array = np.empty_like(data_array, dtype = np.float64)
            #vel_calc2(2)
            vel_calc1()
            for i in range(num_values):
                plt.plot(time_array, vel_array[:,i], label = labels[order[i]])
                plt.ylim((-10,10))
                print('vel of,',labels[order[i]],': \n', vel_array[:,i])
            plt.legend()
            plt.savefig(log_name+'_velocity.png')

            plt.figure("vel-error")
            vel_err_array = np.empty_like(data_array)
            for i in [0,1,2,3,4]:
                vel_err_array[:,i] = vel_array[:,i]-vel_array[:,2]
                plt.plot(time_array, vel_err_array[:,i], label = labels[order[i]])
            plt.legend()
            plt.savefig(log_name+'_vel-error.png')

            combined_array = np.hstack((time_array, data_array, pos_err_array, vel_array, vel_err_array))

            # Transpose the combined array to switch rows and columnsS
            transposed_array = combined_array.T

            # Iterate over rows in the transposed array and write to CSV
            for row in transposed_array:
                csv_writer.writerow(row)


            if not isShit:
                plt.show()
                
