

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
#import matplotlib.pyplot as plt
#import tables as tb

class gradientdescentsolver():
	
	def __init__(self,score_func,gradient_score_func,sensor_arena):
		self.score_func = score_func;
		self.gradient_func = gradient_score_func
		self.accel_coeff = .4
		self.posseries_data =[] #np.empty(np.shape(self.current_pos))
		self.centroidseries_data = [0]#np.empty(self.no_dim);
		self.spreadseries_data = [0];
		self.sensor_arena = sensor_arena
		self.score_data =[]
	def compute_gradient(self,point):
		self.grad = self.gradient_func(point)
	def descent(self,point):
		self.new_point = point - self.grad*self.accel_coeff
	def get_descented_point(self,point):
		self.point =point
		self.compute_gradient(point);
		self.descent(point)
		return self.new_point
	def report(self):
		print("The solver is currently at "+ str(self.point) +"\n")
		print(self.sensor_arena.get_score(self.point))
		print(self.score_func(self.point))
		print("The current score is " + str(self.sensor_arena.get_score(self.point))+ "\n")
		print("The computed gradient is " + str(self.grad) + "\n")
		print("The point moved to " +str(self.new_point)+"\n")
		print("The new score is" + str(self.score_func(self.new_point))+"\n")
		
	def valueIO(self):	
		self.current_pos_dummy = np.copy(sensor_arena.sensor_loc)
		self.current_pos_dummy = np.vstack((self.current_pos_dummy,sensor_arena.target_loc))
		self.current_pos_dummy = np.vstack((self.current_pos_dummy,self.new_point))
		self.posseries_data.append(self.current_pos_dummy)
		self.spreadseries_data.append(1)
		self.score_data.append(self.sensor_arena.get_score(self.point))
		#np.save('data_tracker.npy',[np.array(self.posseries_data
		np.save('data_tracker1',[np.array(self.posseries_data),np.array(self.centroidseries_data),np.array(self.spreadseries_data),np.array(self.score_data)])
	def plot_score(self):	
		a = np.load('data_tracker1.npy');
		plt.plot(a[3],'go')
		#plt.show()
		
		
	

class PSO():
	def __init__(self,localize_object,no_particles,no_dim,self_accel_coeff,global_accel_coeff,dt):
		"""no_particles, no_dim, self_accel_coeff, global_accel_coeff, dt , A total of 5 parameters
		no_particles = Number of swarm particles to create and engage
		no_dim = Dimension of the search space 	"""
		self.localize_object = localize_object;
		self.no_particles = no_particles;
		self.no_dim = no_dim;

		self.self_accel_coeff = self_accel_coeff;
		self.global_accel_coeff = global_accel_coeff;
		
		self.dt_velocity = 1;
		self.dt_pos = 1
		self.weight = .5
		self.maxvel = 10;
		self.minvel = 0;
		
		self.initialize_swarm();

		self.posseries_data =[] #np.empty(np.shape(self.current_pos))
		self.centroidseries_data = []#np.empty(self.no_dim);
		self.spreadseries_data = [];
		self.globalmin_data=[]
		self.globalminloc_data =[]
		self.centroid = np.empty(self.no_dim)
		self.spread = 0;
		
	def initialize_swarm(self):
		self.velocity = np.random.random((self.no_particles,self.no_dim));
		self.current_pos = np.random.random((self.no_particles,self.no_dim))*2+[4,4]
		self.paramspacebounds = [-10,10];
		self.selfminlocation = self.current_pos #np.random.random((self.no_particles,self.no_dim));
		self.selfminval = self.funcdef(self.selfminlocation,self.no_particles);
		self.globalmin = np.min(self.selfminval);
		self.globalminlocation = self.selfminlocation[np.argmin(self.selfminval)];
		self.curr_score = self.funcdef(self.current_pos,self.no_particles);


	def update_selfmin(self):
		update_req_array = self.curr_score<self.selfminval

		self.selfminval = np.multiply(update_req_array,self.curr_score)+np.multiply(np.invert(update_req_array),self.selfminval);

		update_req_array = np.transpose(update_req_array);

		update_req_array = np.reshape(update_req_array,(self.no_particles,1))
		print(np.shape(update_req_array));

		print(np.shape(self.current_pos));
		self.selfminlocation = (update_req_array*self.current_pos)+(np.invert(update_req_array)*self.selfminlocation);

	def update_globalmin(self):
		curr_globalmin = np.amin(self.curr_score);
		curr_globalmin_loc = self.current_pos[self.curr_score==curr_globalmin]
		curr_globalmin_loc = curr_globalmin_loc[0]		
		state = curr_globalmin<self.globalmin	
		state = np.squeeze(state)
		self.globalmin = np.multiply(state,curr_globalmin)+np.multiply(np.invert(state),self.globalmin);
		self.globalminlocation = (state*curr_globalmin_loc)+(np.invert(state)*self.globalminlocation);

	def funcdef(self,locs,no_particles):
		score_curr = np.empty(no_particles);
		
		for i in range(no_particles):
			score_curr[i] = self.localize_object.get_score(locs[i])
			#print(score_curr[i])
		return score_curr
		
		
	def update_currscores(self):
		self.curr_score = self.funcdef(self.current_pos,self.no_particles);	

	def update_velocities(self):
		self_accel = np.random.random()*(self.selfminlocation-self.current_pos);
		global_accel = np.random.random()*(self.globalminlocation-self.current_pos);
		accel = self.self_accel_coeff*self_accel+self.global_accel_coeff*global_accel;
		self.velocity = self.weight*self.velocity+accel*self.dt_velocity;
		self.velocity[self.velocity>self.maxvel] = self.maxvel;
		self.velocity[self.velocity<(-self.maxvel)] = -self.maxvel;
	def update_pos(self):
		self.current_pos = self.current_pos + self.velocity*self.dt_pos;
	def calc_swarm_props(self):
		centroid = np.mean(self.current_pos,axis=0);
		spread = 0;
		for i in range(self.no_particles):
			spread = np.linalg.norm(self.current_pos-centroid)+spread;
		self.centroid = centroid
		self.spread = spread
		self.best_score = np.min(self.curr_score);
		return(centroid,spread)

		
	def valueIO(self):
		self.current_pos_dummy = np.copy(self.current_pos)
		self.current_pos_dummy = np.vstack((self.current_pos_dummy,self.localize_object.sensor_loc))
		self.current_pos_dummy = np.vstack((self.current_pos_dummy,self.localize_object.target_loc))
		self.posseries_data.append(self.current_pos_dummy) #np.hstack((self.posseries_data,self.current_pos))
		self.spreadseries_data.append(self.spread)# = np.append(self.spreadseries_data,self.spread)
		self.centroidseries_data.append(self.centroid)# = np.vstack((self.centroidseries_data,self.centroid))
		self.globalmin_data.append(self.globalmin)
		self.globalminloc_data.append(self.globalminlocation)
		np.save('data_tracker2'
,[np.array(self.posseries_data),np.array(self.centroidseries_data),np.array(self.spreadseries_data),np.array(self.globalmin_data),np.array(self.globalminloc_data)])
		
	def report(self):
		print("The centroid is at " + str(self.centroid)+"\n");
		print("The spread is " + str(self.spread)+"\n");
		print("------")
		print("The current best score  is"+str(np.min(self.curr_score))+"\n");
		print("The global best is at "+str(self.globalminlocation)+"\n");
		print("The gloabl best score is" + str(self.globalmin) + "\n")
		#print("Ca")
		
		print("current score \t\t best self score\t\t current_position \t\t best self position \t\t current velocity \n")
		for i in range(self.no_particles):
			print(str(self.curr_score[i]) + '\t\t ' + str(self.selfminval[i])+'\t\t '+ str(self.current_pos[i]) + '\t\t  '+ str(self.selfminlocation[i])+'\t\t '+ str(self.velocity[i]))
			#print(self.current_pos[i])
			#print(self.velocity[i])
		#print("The current score is  "+str(self.curr_score)+"\n");
		#print("The positions are " + str(self.current_pos)+"\n");
		#print("-----");
		
		
		#print("The current velocities are "+ str(self.velocity)+"\n");
		print("#$#$#$#$#$#$ \n")
	def plot_score(self):	
		a = np.load('data_tracker2.npy');
		plt.plot(a[3],'ro')
		#plt.show()
		
	
		
def plot_surface(fitfunc):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	X1 = np.arange(0, 10, 0.25)
	Y2 = np.arange(0, 10, 0.25)
	X, Y = np.meshgrid(X1, Y2)
	Z = X1;
	for i in range(len(X)):
		Z[i] = fitfunc(np.array([X1[i],Y2[i]]))
	Z = np.array(Z)
	#R = np.sqrt(X**2 + Y**2)
	#Z = np.sin(R)
	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
	#ax.set_zlim(-1.01, 1.01)

	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	fig.colorbar(surf, shrink=0.5, aspect=5)

	plt.show()
	
	

class localize():
	
	def __init__(self,no_sensors):
		self.no_sensors = no_sensors
		self.create_sensors()
		self.create_target_object()
		self.get_original_ranges()
		self.get_noisy_ranges()
		
	def dist_from_ithsensor(self,i,point):
		if i>self.no_sensors:
			print "Exceeded"
			return
		else:
			return np.linalg.norm(point-self.sensor_loc[i])
	
	
	
	def gradient_score(self,point):
		point = np.array(point);
		dim = point.shape[0];
		gradi = np.empty(dim)
		dist_vector = [self.dist_from_ithsensor(i,point) for i in range(self.no_sensors)]
		dist_vector = np.array(dist_vector)
		common_factor_vector = [1-((self.noisy_ranges[i])/dist_vector[i]) for i in range(self.no_sensors)]
		common_factor_vector = np.array(common_factor_vector)
		dim_diff_vector = point-self.sensor_loc;
		dim_gradient_vector = np.transpose(common_factor_vector*np.transpose(dim_diff_vector));
		dim_gradient = np.sum(dim_gradient_vector,axis=0)
		return dim_gradient*(2./self.no_sensors)
		#grad_presum_vector = [np.dot(common_factor_vector[i],dim_diff_vector[i])
		
		#for i in range(	dim):		
			##gradi[dim] = 2*(self.noisy_ranges[i]/self.dist_from_ithsensor(i,)
			
		
	def create_sensors(self):
		#self.sensor_loc = np.random.random((self.no_sensors,2))*10
		#self.sensor_loc =np.array([[1,2],[3,4],[5,6]])
		self.sensor_loc = np.array([[0.969,.266],[.66,.41],[.52,.78]]) * 10
	def create_target_object(self):
		#self.target_loc = np.random.random((1,2))*10
		self.target_loc = [5,5]
		
	def get_original_ranges(self):
		self.orig_ranges = self.sensor_loc-self.target_loc;
		self.orig_ranges = np.linalg.norm(self.orig_ranges,axis=1)
	
	def get_noisy_ranges(self):
		sigma = .1;
		
		mean_vector = self.orig_ranges;
		path_loss_coeff = 0.01;
		variance_vector = (sigma)*(np.power(self.orig_ranges,path_loss_coeff));
		#print mean_vector
		#print variance_vector
		self.mean = mean_vector
		self.var = variance_vector
		nse = np.arange(self.no_sensors)
		for i in range(self.no_sensors):			
			nse[i] = np.random.normal(mean_vector[i],variance_vector[i])
		#nse = np.array(n)
		
		self.noisy_ranges = nse
	def get_score(self,particle_loc):
		score = 0;
		cartesian_distance = np.linalg.norm(particle_loc -self.sensor_loc, axis=1)
		#print cartesian_distance
		#cartesian_distance = np.power(cartesian_distance,.5)
		#print cartesian_distance
		score_vector = self.noisy_ranges-cartesian_distance;
		#print score_vector
		score = np.mean(np.power(score_vector,2))
		
		return score
	
		
		
		
	#def initalize_swarm_param(self):
		
	
	
	
		
		
loc_algo = localize(3)
pso = PSO(loc_algo,10,2,1,1.5,1);
indx = 0;
lst_score = float("inf");
pos_tracker = np.empty(np.shape(pso.current_pos));
spread = 0;
centroid = np.empty(pso.no_dim);


sensor_arena = loc_algo
gdsolver = gradientdescentsolver(sensor_arena.get_score,sensor_arena.gradient_score,sensor_arena)

point = np.random.random((1,2))
iterindx =1


#while np.abs(np.amin(pso.curr_score)-lst_score)>.001:
while indx<100:
	pso.update_pos();
	pso.update_currscores();
	pso.update_selfmin();
	pso.update_globalmin();
	pso.update_velocities();
	(centroid,spread) =pso.calc_swarm_props();
	print('\n \n \n \n \n')
	print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
	print("The running index is "+ str(indx)+"\n")
	print(str(pso.globalminlocation))
	indx = indx+1;
	if pso.no_particles<20:
		pso.report();
	pso.valueIO();
	
	print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n')
	curr_score = sensor_arena.get_score(point);
	point = gdsolver.get_descented_point(point);
	descented_score = sensor_arena.get_score(point);
	deltascore = descented_score-curr_score;
	curr_score = descented_score;
	gdsolver.report();
	gdsolver.valueIO();
	
##pso.plot_score()

#gdsolver.plot_score()
#plt.show()

plt.plot(np.load('data_tracker1.npy')[3],'r')
plt.show()
plt.plot(np.load('data_tracker2.npy')[3],'b')
plt.show()
