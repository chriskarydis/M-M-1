import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
from pylab import *

# Read I_values from a file
def read_I_values(filename):
    I_values = []
    with open(filename, 'r') as file:
        for line in file:
            I_values.append(float(line.strip()))  # Convert each line to float and append to list
    return I_values

# Function to clear the contents of output files
def clear_output_files():
    file_names = ['T_R.txt', 'TQ_R.txt', 'N_R.txt', 'NQ_R.txt', 'NQMAX_R.txt']
    for file_name in file_names:
        open(file_name, 'w').close()

# Initialize random number generator for exponential distribution
rng = np.random.default_rng()

# Generate exponentially distributed random numbers
def exponential():
	U = rng.random()
	return -np.log(U)

# Calculate time of next arrival event
def next_arrival(l, cur_t, t_arr):
	return cur_t + l * exponential()

# Calculate service completion time for a client
def service_till(m, cur_t, t_ser):
	return cur_t + m * exponential()

# Update current time based on next arrival or service completion 
def time_jump(t_arr, t_ser, cur_t):
	if t_arr <= t_ser:
		cur_time = t_arr
	elif cur_t <= t_ser:
		cur_t = t_ser
	return cur_t

# Add client to the queue upon arrival
def arrival(cur_t, queue, client, queue_data, client_data):
	queue.append(client)
	queue_data.append([cur_t, len(queue)])
	client_data.append([cur_t, 0, 0])

# Service a client from the queue
def service(cur_t, t_ser, queue, client_data, queue_data):
	client = queue.pop(0)
	queue_data.append([cur_t, len(queue)])
	client_data[client][1] = cur_t
	client_data[client][2] = t_ser

# Run simulations for different mean arrival intervals
def run_simulation(I_values):
	print("Starting simulation loop...\n")
	
	# Ensure the 'sim_data' directory exists
	if not os.path.exists('sim_data'):
		os.makedirs('sim_data')
    
	# Loop over different arrival intervals
	for idx, l in enumerate(I_values):
		with open(f'T_R.txt', 'a') as t_r_file, \
			 open(f'TQ_R.txt', 'a') as tq_r_file, \
			 open(f'N_R.txt', 'a') as n_r_file, \
			 open(f'NQ_R.txt', 'a') as nq_r_file, \
			 open(f'NQMAX_R.txt', 'a') as nqmax_r_file:
		
		
			print(f"Running simulation {idx + 1} with Ι = {l}...")
			queue = [] # Initialize queue
			queue_data = [] # Queue data for plotting
			client_data = [] # Client data for statistics
			
			client = 0  # Next client to come
			t_ser = 0   # Time when service will be IDLE
			t_arr = 0   # Time of next arrival
			cur_t = 0   # Current time
			t_max = 100000 # Maximum simulation time
			sim_num = idx + 1 # Simulation number
			
			# Start first arrival and service
			t_arr = next_arrival(l, cur_t, t_arr)
			cur_t = t_arr
			arrival(cur_t, queue, client, queue_data, client_data)
			client += 1
			t_arr = next_arrival(l, cur_t, t_arr)
			t_ser = service_till(1.0, cur_t, t_ser)
			service(cur_t, t_ser, queue, client_data, queue_data)
			
			total_customers = 0 # Total number of customers in the system
			last_event_time = 0 # Time of the last event

			# Simulation loop
			while cur_t < t_max:
				total_customers += len(queue) * (cur_t - last_event_time) # Keeps a running total of the "customer-time" for the entire simulation
				last_event_time = cur_t	
				cur_t = time_jump(t_arr, t_ser, cur_t)
				if cur_t == t_arr:
					arrival(cur_t, queue, client, queue_data, client_data)
					client += 1
					t_arr = next_arrival(l, cur_t, t_arr)
				elif cur_t >= t_ser and len(queue) > 0:
					t_ser = service_till(1.0, cur_t, t_ser)
					service(cur_t, t_ser, queue, client_data, queue_data)
				else:
					cur_t = t_arr
	
			# Calculate mean number of customers
			mean_customers = total_customers / cur_t 
			
			# Write simulation data to files
			with open(f'sim_data/{sim_num}_client_data.dat', 'w') as f:
				for i in client_data:
					f.write(f"{i[0]},{i[1]},{i[2]}\n")
			f.close()
			
			with open(f'sim_data/{sim_num}_queue_data.dat', 'w') as f:
				for i in queue_data:
					f.write(f"{i[0]},{i[1]}\n")
			f.close()
			
			# Calculate statistics
			T = 0
			Tq = 0
			Ts = 0
			s_clients = 0  # Number of serviced clients
			
			with open(f'sim_data/{sim_num}_client_data.dat', 'r') as f:
				lines = f.readlines()
				for line in lines:
					line = line.split(',')
					if float(line[2]) > 0:
						T += float(line[2]) - float(line[0]) 
						Tq += float(line[1]) - float(line[0]) 
						Ts += float(line[2]) - float(line[1])
						s_clients += 1
			f.close()
			
			T /= s_clients
			Tq /= s_clients
			Ts /= s_clients
			    
			# Additional code for Nq_max calculation
			with open(f'sim_data/{sim_num}_queue_data.dat', 'r') as f:
				lines = f.readlines()
			f.close()
			    
			Nq = []
			Nq_max = 0
			Nqt_max = 0
			for ln in lines:
				a = ln.split(',')
				Nq.append(int(a[1]))
				if int(a[1]) > Nq_max:
					Nq_max = int(a[1])
					Nqt_max = float(a[0])
			
			mean_Nq = sum(Nq) / len(lines)
			
			# Write statistics to file
			with open(f'sim_data/{sim_num}_sim_data.dat', 'w') as f:
				f.write(f"l = {l}, m = 1.0, t_max = {t_max}\n")
				f.write(f"λ = {1/l}, μ = 1.0\n")
				f.write("Queue Theory\n")
				f.write(f"T = {(1.0*l)/(l-1.0)}, Tq = {(1.0/l)/((l-1.0)/(1.0*l))}, Ts = 1.0, Mean Nq = {((1/l)*(1/l))/(1-(1/l))}, Mean N = {(1/l)/(1-(1/l))}\n")
				f.write("Simulation\n")
				f.write(f"T = {T}, Tq = {Tq}, Ts = {Ts}, Mean Nq = {mean_Nq}, Mean N = {mean_customers}, Nq_max = {Nq_max}, Nq_max time = {Nqt_max} \n")
				f.write(f"Number of serviced clients = {s_clients}\n")
			f.close()
			
			# Prefix to generate the 'ρ'
			prefix = "{:.2f},".format((idx + 1) * 0.05)

			# Write results to output files
			t_r_file.write(prefix + f"{(1.0*l)/(l-1.0)},{T}\n")
			tq_r_file.write(prefix + f"{(1.0/l)/((l-1.0)/(1.0*l))},{Tq}\n")
			n_r_file.write(prefix + f"{(1/l)/(1-(1/l))},{mean_customers}\n")
			nq_r_file.write(prefix + f"{((1/l)*(1/l))/(1-(1/l))},{mean_Nq}\n")
			nqmax_r_file.write(prefix + f"{Nq_max}\n")
		
		print(f"Simulation {idx + 1} completed.\n")
	print("Simulation loop finished.\n")

# Define the filename containing l_values
I_values_file = 'I_values.txt'

# Read I_values from the file
I_values = read_I_values(I_values_file)

# Clear the output files
clear_output_files()

# Run simulations for each value in the list
run_simulation(I_values)

# Define a function to plot data from a file
def plot_data(filename, xlabel, ylabel, title):
	data = np.loadtxt(filename, delimiter=',')
	x = data[:, 0]
	if data.shape[1] == 2:  # Only simulation data available
		y_simulation = data[:, 1]
		plt.plot(x, y_simulation, label='Simulation')
	else:  # Both theoretical and simulation data available
		y_theoretical = data[:, 1]
		y_simulation = data[:, 2]
		plt.plot(x, y_theoretical, label='Theoretical')
		plt.plot(x, y_simulation, label='Simulation')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.legend()
	plt.grid(True)
   
# Define a function to create custom figure labels
def set_figure_label(plot_count, total_plots):
	fig = pylab.gcf()
	fig.canvas.manager.set_window_title(f"plot {plot_count}/{total_plots}")   

# Define a function to save the plot
def save_plot(filename):
    plt.savefig(filename)
    print(f"Plot saved as '{filename}'")
   
# Define the total plots and the plot count
total_plots=5
plot_count=0   
   
# Plot N_R.txt
plot_count+=1
plot_data('N_R.txt', 'R', 'N', 'N - R')
set_figure_label(plot_count, total_plots)
save_plot('N_R_plot.png')
plt.show()

# Plot NQ_R.txt
plot_count+=1
plot_data('NQ_R.txt', 'R', 'NQ', 'NQ - R')
set_figure_label(plot_count, total_plots)
save_plot('NQ_R_plot.png')
plt.show()

# Plot NQMAX_R.txt
plot_count+=1
plot_data('NQMAX_R.txt', 'R', 'NQMAX', 'NQMAX - R')
set_figure_label(plot_count, total_plots)
save_plot('NQMAX_R_plot.png')
plt.show()

# Plot T_R.txt
plot_count+=1
plot_data('T_R.txt', 'R', 'T', 'T - R')
set_figure_label(plot_count, total_plots)
save_plot('T_R_plot.png')
plt.show()

# Plot TQ_R.txt
plot_count+=1
plot_data('TQ_R.txt', 'R', 'TQ', 'TQ - R')
set_figure_label(plot_count, total_plots)
save_plot('TQ_R_plot.png')
plt.show()
