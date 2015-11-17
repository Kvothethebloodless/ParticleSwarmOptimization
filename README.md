<h2>A 2-Dimensional Particle Swarm Algorithm Implementation </h2>

This is an implementation of the Particle Swarm Algorithm for solving two-dimensional optimization problems.


<h3>Usage: </h3>

The class **PSO** is the base class to carry out all operations such as updating velocities , positions, global best, local best position values.

```python
class PSO():
    def __init__(self,no_particles,no_dim,self_accel_coeff,global_accel_coeff,dt):
```

The method ```PSO.funcdef()``` holds the required defitinion of the score. It is currently the __norm__ function.

```python
def funcdef(self,locs,no_particles):
    score_curr = np.empty(no_particles);
    for i in range(no_particles):
        score_curr[i] = np.linalg.norm(locs[i])
        print(score_curr[i])
    return score_curr
```

The animation doesn't occure in realtime. The ```PSO.valueIO() ``` method writes all states that the particles go through, to a __.npy__ file. The script __plotter.py__ reads the file and animates the states using scatter plots.


Score Function can be changed by changing the method```PSO.funcdef() ```


Feel free to fork and experiment!

Reference: http://www.softcomputing.net/aciis.pdf

Algorithm implemented from directions given in the above reference article.


