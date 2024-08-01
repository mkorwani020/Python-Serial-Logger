import numpy as np
import csv
from scipy.signal import butter,filtfilt
import plotly.graph_objects as go
import matplotlib.pyplot as plt
# Filter requirements.

order_read = [0, 1, 2, 3, 4]
labels = ["Renishaw 1", "Renishaw 2", "Renishaw Adjusted", "Zettlex", "Netzer"]

# Create a dictionary by zipping the two lists
order_labels_dict = dict(zip(order_read, labels))



def butter_filter(data, cutoff, fs, order, nyq):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False, output = 'ba', fs=fs)
    y = filtfilt(b, a, data)
    return y

def get_shape(lst):
            shape = []
            while isinstance(lst, list):
                shape.append(len(lst))
                lst = lst[0] if lst else []
            return tuple(shape)

def readfile(log_name):
    csv_file = 'logs/Encoder-Testbed-Log_' + log_name + '.csv'
    # data_array= ([[]])
    with open(csv_file,  newline='') as file:
        csv_reader = csv.reader(file)
         # Skip the first 6 rows
        all_rows = list(csv_reader)[1:-1] #excludes first and last row that has encoder names in order
        
        # Convert rows to lists of integers
        nested_list = [list(map(np.double, row)) for row in all_rows]
        # reader = file.read().split('\n')
        # # time_array = reader[0]
        # for i in reader[6:]:
        #     # new = np.array(i.split(','), dtype = np.double)
        #     new = i.split(',')
        #     data_array.append([new])
        
        
        return nested_list

def limit_angle(arr):
    for i in range(len(arr)):
            for j in range(0, arr.shape[1]):
                # data_array[i,j] = float(data_array[i,j])
                if arr[i,j] > 180.0:
                    arr[i,j] -= 360
    return arr   

def moving_avrg(arr, window_size=3):
 
    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []
    
    # Loop through the array to consider
    # every window of size n
    while i < len(arr) - window_size + 1:
        dy = arr[i+(window_size//2)]-arr[i-window_size//2]
        dy = np.double(min(abs(dy),abs(dy+360),abs(dy-360))) #check for smallest angle

        # Store elements from i to i+window_size
        # in list to get the current window
        window = arr[i : i + window_size]
        if dy < 0.25:
            # Calculate the average of current window
            window_average = sum(window) / window_size
            # Store the average of current
            
            # window in moving average list
            moving_averages.append(window_average)
        else:
            moving_averages.append(arr[i])
        # Shift window to right by one position
        time.append(time_array[i])
        i += 1
    return moving_averages

def moving_avrg2(arr):
    i = 0
    window_size=1
    # Initialize an empty list to store moving averages
    moving_averages = [arr[0]]
    window = [arr[0]]
    # Loop through the array to consider
    # every window of size n
    while i < len(arr):
        dy = arr[i]-window[-1]
        dy = np.double(min(abs(dy),abs(dy+360),abs(dy-360))) #check for smallest angle

        # Store elements from i to i+window_size
        # in list to get the current window
        window = arr[i : i + window_size]
        if dy > 0.2:
            moving_averages.append(window_average)
            
            window_size = 1
        else:
            moving_averages.append(moving_averages[-1])
            window_average = sum(window) / window_size
            window_size += 1
        # Shift window to right by one position
        time.append(time_array[i])
        i += 1
        
    return moving_averages

def vel_calc1(num):
    dt = np.double(0.1)
    rows = shape[0]
    #dt = np.double(time_array[i]- time_array[i-1])
    for j in range(1, rows): #by columns (will be 5 times)
        dy = data_array[j][num]-data_array[j-1][num]
        dy = np.double(min(abs(dy),abs(dy+360),abs(dy-360)))
        vel_array[j][num] = np.double(dy/dt)/(360.0)

def plot1():
    fig = go.Figure()
    fig.update_layout(
        title= log_name + " Position",
        # xaxis_title="seconds",
        yaxis_title="degrees",
        legend_title="encoder type",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )
    fig2 = go.Figure()
    fig2.update_layout(
        title= log_name + " velocity",
        # xaxis_title="seconds",
        yaxis_title="rev/s",
        legend_title="encoder type",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )
    y1_v2 = moving_avrg(netz_pos,3)
    plot(netz_pos, 'netzer', y1_v2)

    y2_v2 = moving_avrg(resa1_pos,3)
    plot(resa1_pos, 'Reinishaw 1', y2_v2)

    y5_v2 = moving_avrg(zet_pos,3)
    plot(zet_pos, 'zettlex', y5_v2)
    fig.show()

    for l in range(0,5):
            fig2.add_trace(go.Scatter(
                y = vel_array[:,l],
                line =  dict(shape =  'spline' ),
                name = order_labels_dict[l] + ' signal with noise'))
    fig2.show()
    def plot(enc, enc_name, f):
        fig.add_trace(go.Scatter(
                    # x= time_array,
                    y = enc,
                    line =  dict(shape =  'spline' ),
                    name = enc_name + ' signal with noise'
                    ))
        fig.add_trace(go.Scatter(
                    # x= time_array,
                    y = f,
                    line =  dict(shape =  'spline' ),
                    name = 'filtered signal: moving avrg w/ win 5'#order =' + str(order) + '  cutoff ='+ str(cutoff) #
                    ))



log_name = input("Please enter the filename: ")
h = 0
h = int(input("is the data organized by rows? Enter 0 for false and 1 for true:  "))
data_array= readfile(log_name)

shape = (get_shape(data_array))
print(shape)

if (h==1): # transposes data in the case the Mamta was dumb during some of the tests and wrote data by row for each encoder
    data_array = np.array(data_array, dtype = np.double)
    data_array = data_array.T
    data_array = data_array[6:] #gets rid of the data before they are all syncyed up 
    print(data_array.shape)
else:
     data_array = data_array[6:] #gets rid of the data before they are all syncyed up 
     data_array = np.array(data_array, dtype = np.double)

time_array = data_array[:,0]
data_array= np.delete(data_array, [0], axis=1)
shape = data_array.shape

T = np.double(time_array[-1] * 1e-9)    # Sample Period
# T= 0.1
fs = 10.0       # sample rate, Hz

nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # sin wave can be approx represented as quadratic
n = float(shape[0]) # total number of samples
cutoff =   10 # desired cutoff frequency of the filter, Hz , slightly higher than actual 1.2 Hz

data_array = limit_angle(data_array)
resa1_pos = data_array[:,0]
resa2_pos = data_array[:,1]
resa_adj_pos = data_array[:,2]
zet_pos = data_array[:,3]
netz_pos = data_array[:,4]

vel_array = np.zeros((shape[0], 5), dtype = np.double)
if (shape[1] > 5):
    for p in range (5, 10) : vel_array[:,(p-5)] = data_array[:,p]
else:
     for p in range (0,5) : vel_calc1(p)
'''
# Filter the data, and plot both the original and filtered signals.
y1 = butter_filter(netz_pos, cutoff, fs, order, nyq)
plot(netz_pos, 'netzer', y1)

y2 = butter_filter(resa1_pos, cutoff, fs, order, nyq)
plot(netz_pos, 'Reinishaw 1', y2)
'''

plt.figure("pos-error")
pos_err_array = np.empty_like(data_array)
for i in [0,1,2,4]:
    pos_err_array[:,i] = data_array[:,i]-data_array[:,2]
    plt.plot(time_array, pos_err_array[:,i], label = order_labels_dict[i])
plt.legend()
plt.savefig(log_name+'_pos-error.png')

plt.figure("raw data")
for i in range(5):
    plt.plot(time_array, data_array[:,i], label=order_labels_dict[i])
plt.legend()

plt.figure("filtered position")
for i in range(5):
    time = []
    w = 2
    if(order_labels_dict[i] == "Zettlex"): w = 2
    plt.plot(time, moving_avrg(data_array[:,i],w), label=order_labels_dict[i])
plt.legend()


plt.figure('velocity')
plt.ylabel('rev/s')
for i in range(5):
    plt.plot(time_array, vel_array[:,i], label=order_labels_dict[i])
    plt.ylim((-10,10))
plt.legend()

plt.show()
