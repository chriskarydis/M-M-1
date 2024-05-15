import numpy as np
import matplotlib.pyplot as plt
import time
import resource
from matplotlib import pylab
from pylab import *

rng = np.random.default_rng()

def exponential():
	U = rng.random()
	return -np.log(U)
	    
def next_arrival(l, cur_t, t_arr):
	return cur_t + l * exponential()
	    
def service_till(m, cur_t, t_ser):
	return cur_t + m * exponential()
	    
def time_jump(t_arr, t_ser, cur_t):
	if t_arr <= t_ser:
		cur_time = t_arr
	elif cur_t <= t_ser:
		cur_t = t_ser
	return cur_t
	
def arrival(cur_t, queue, client, queue_data, client_data):
	queue.append(client)
	queue_data.append([cur_t, len(queue)])
	client_data.append([cur_t, 0, 0])
	    
def service(cur_t, t_ser, queue, client_data, queue_data):
	client = queue.pop(0)
	queue_data.append([cur_t, len(queue)])
	client_data[client][1] = cur_t
	client_data[client][2] = t_ser

def run_simulation(l_values):
	print("Starting simulation loop...")
	for idx, l in enumerate(l_values):
		# Open the output files outside the simulation loop
		with 	open(f'T_R.txt', 'a') as t_r_file, \
			open(f'TQ_R.txt', 'a') as tq_r_file, \
			open(f'N_R.txt', 'a') as n_r_file, \
			open(f'NQ_R.txt', 'a') as nq_r_file, \
			open(f'NQMAX_R.txt', 'a') as nqmax_r_file:
		
		
			print(f"Running simulation {idx + 1} with l = {l}...")
			start_time = time.time()  # Record start time
			start_resources = resource.getrusage(resource.RUSAGE_SELF)  # Record start resources
			queue = []
			queue_data = []
			client_data = []
			
			client = 0  # next client to come
			t_ser = 0   # time when service will be IDLE
			t_arr = 0   # time of next arrival
			cur_t = 0   # current time
			
			t_max = 100000
			sim_num = idx + 1
			
			t_arr = next_arrival(l, cur_t, t_arr)
			cur_t = t_arr
			arrival(cur_t, queue, client, queue_data, client_data)
			client += 1
			t_arr = next_arrival(l, cur_t, t_arr)
			t_ser = service_till(1.0, cur_t, t_ser)
			service(cur_t, t_ser, queue, client_data, queue_data)
			
			total_customers = 0 # total number of customers in the system
			last_event_time = 0 # time of the last event

			while cur_t < t_max:
				total_customers += len(queue) * (cur_t - last_event_time) # keeps a running total of the "customer-time" for the entire simulation
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

			mean_customers = total_customers / cur_t 
			print(f"Mean number of customers in the system: {mean_customers}")
			
			# Write simulation data to files
			with open(f'sim_data/{sim_num}_client_data.dat', 'w') as f:
				for i in client_data:
					f.write(f"{i[0]},{i[1]},{i[2]}\n")
			f.close()
			
			with open(f'sim_data/{sim_num}_queue_data.dat', 'w') as f:
				for i in queue_data:
					f.write(f"{i[0]},{i[1]}\n")
			f.close()
			
			# Calculate and write statistics
			T = 0
			Tq = 0
			Ts = 0
			s_clients = 0  # number of serviced clients
			
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
    
			end_resources = resource.getrusage(resource.RUSAGE_SELF)  # Record end resources
			end_time = time.time()  # Record end time
			elapsed_time = end_time - start_time  # Calculate elapsed time
			
			# Calculate and write system resource usage
			utime = end_resources.ru_utime - start_resources.ru_utime  # User CPU time
			stime = end_resources.ru_stime - start_resources.ru_stime  # System CPU time
			maxrss = end_resources.ru_maxrss  # Maximum resident set size (in kilobytes)
			
			with open(f'sim_data/{sim_num}_resource_usage.txt', 'w') as f:
				f.write(f"Simulation {idx + 1}:\n")
				f.write(f"Elapsed time: {elapsed_time} seconds\n")
				f.write(f"User CPU time: {utime} seconds\n")
				f.write(f"System CPU time: {stime} seconds\n")
				f.write(f"Maximum resident set size: {maxrss} kilobytes\n")
			f.close()
			    
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
			
			with open(f'sim_data/{sim_num}_sim_data.dat', 'w') as f:
				f.write(f"l = {l}, m = 1.0, t_max = {t_max}\n")
				f.write(f"λ = {1/l}, μ = 1.0\n")
				f.write("Queue Theory\n")
				f.write(f"T = {(1.0*l)/(l-1.0)}, Tq = {(1.0/l)/((l-1.0)/(1.0*l))}, Ts = 1.0, Mean Nq = {((1/l)*(1/l))/(1-(1/l))}, Mean N = {(1/l)/(1-(1/l))}\n")
				f.write("Simulation\n")
				f.write(f"T = {T}, Tq = {Tq}, Ts = {Ts}, Mean Nq = {mean_Nq}, Mean N = {mean_customers}, Nq_max = {Nq_max}, Nq_max time = {Nqt_max}\n")
				f.write(f"Number of serviced clients = {s_clients}\n")
			f.close()
			
			prefix = "{:.2f},".format((idx + 1) * 0.05)

			#################################################################################
			t_r_file.write(prefix + f"{(1.0*l)/(l-1.0)},{T}\n")
			tq_r_file.write(prefix + f"{(1.0/l)/((l-1.0)/(1.0*l))},{Tq}\n")
			n_r_file.write(prefix + f"{(1/l)/(1-(1/l))},{mean_customers}\n")
			nq_r_file.write(prefix + f"{((1/l)*(1/l))/(1-(1/l))},{mean_Nq}\n")
			nqmax_r_file.write(prefix + f"{Nq_max}\n")
			#################################################################################
		print(f"Simulation {idx + 1} completed.")
	print("Simulation loop finished.")

# Define list of mean arriving time intervals
l_values = [20.00, 10.00, 6.66, 5.00, 4.00, 3.33, 2.86, 2.50, 2.22, 2.00, 1.82, 1.66, 1.54, 1.43, 1.33, 1.25, 1.18, 1.11, 1.05]  # Example list

# Clear the contents of the output files at the start of the program
with open('T_R.txt', 'w') as t_r_file, \
     open('TQ_R.txt', 'w') as tq_r_file, \
     open('N_R.txt', 'w') as n_r_file, \
     open('NQ_R.txt', 'w') as nq_r_file, \
     open('NQMAX_R.txt', 'w') as nqmax_r_file:
    pass  # Just opening the files to clear them

# Run simulations for each value in the list
run_simulation(l_values)

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
   
# Define the total plots and the plot count
total_plots=5
plot_count=0   
   
# Plot N_R.txt
plot_count+=1
plot_data('N_R.txt', 'R', 'N', 'N - R')
set_figure_label(plot_count, total_plots)
plt.show()

# Plot NQ_R.txt
plot_count+=1
plot_data('NQ_R.txt', 'R', 'NQ', 'NQ - R')
set_figure_label(plot_count, total_plots)
plt.show()

# Plot NQMAX_R.txt
plot_count+=1
plot_data('NQMAX_R.txt', 'R', 'NQMAX', 'NQMAX - R')
set_figure_label(plot_count, total_plots)
plt.show()

# Plot T_R.txt
plot_count+=1
plot_data('T_R.txt', 'R', 'T', 'T - R')
set_figure_label(plot_count, total_plots)
plt.show()

# Plot TQ_R.txt
plot_count+=1
plot_data('TQ_R.txt', 'R', 'TQ', 'TQ - R')
set_figure_label(plot_count, total_plots)
plt.show()

