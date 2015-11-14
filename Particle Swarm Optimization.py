# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import matplotlib.pyplot as plt

class PSO(self,no_particles,no_dim,self_accel_coeff,global_accel_coeff,dt):
    self.no_particles = no_particles;
    self.no_dim = no_dim;
    
    
    self.self_accel_coeff = self.accel_coeff;
    self.global_accel_coeff = self.global_accel_coeff;
    
    self.velocity = np.random.random((self.no_particles,self.no_dim));
    
    self.current_pos = np.random.random((self.no_particles,self.no_dim))
    self.paramspacebounds = [-10,10];
    
    self.dt_velocity = .05;
    self.dt_pos = .1
    
    self.selfminlocation = np.random.random((self.no_particles,self.no_dim));
    self.selfminval = self.funcdef(self.selfminlocation,self.no_particles);
    
    self.globalminlocation = np.random.random(self.no_dim);
    self.globalmin = self.funcdef(self.globalminlocation,1);
    
    
    self.curr_score = np.empty((1,self.no_particles));
    
    def update_selfmin(self):
        update_req_array = self.curr_score<self.selfmin
        self.selfminval = np.multiply(update_req_array,self.curr_score)+np.multiply(np.invert(update_req_array),self.selfmin);
        
        update_req_array = np.transpose(update_req_array);
        self.selfminlocation = np.multiply(update_req_array,self.current_pos)+np.multiply(np.invert(update_req_array),self.selfminlocation);
        
    def update_globalmin(self):
        curr_globalmin = np.amin(self.curr_score);
        state = curr_globalmin<self.globalmin
        self.globalmin = np.multiply(state,curr_globalmin)+np.multiply(np.invert(state),self.globalmin);
        self.globalminlocation = np.multiply(state,curr_globalmin)+np.mulptiply(np.invert(state),self.globalminlocation);
        
    def funcdef(self,loc,no_particles):
        for i in range(no_particles):
            score(1,i) = np.linalg.norm(loc(i));
        return score
    
    def update_velocities(self):
        self_accel = self.selfminlocation-self.current_pos;
        global_accel = self.globalminlocation-self.current_pos;
        accel = self.self_accel_coeff*self_accel+self.global_accel_coeff*global_accel;
        self.velocity = self.velocity+accel*self.dt_velocity;
    def update_pos(self):
        self.current_pos = self.current_pos + self.velocity*self.dt_pos;
    def calc_swarm_props(self):
        centroid = np.mean(self.current_pos,axis=0);
        spread = 0;
        for i in range(self.no_particles):
            spread = np.linalg.norm(self.current_pos-centroid)+spread;
        return(centroid,spread)
    
    
    
    
        
    
    

# <codecell>

pso = PSO(20,2,2,2,.05)

# <codecell>

pso?

# <codecell>

pso

# <codecell>

print(pso.velocity)

# <codecell>


# <codecell>


