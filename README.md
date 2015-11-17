<h2> A 2-Dimensional Particle Swarm Algorithm Implementation </h2>

<p> This is an implementation of the Particle Swarm Algorithm for solving two-dimensional optimization problems. </p>

<h3> Usage: </h3>

<p> The code has a class **PSO** to which holds the required defitinion of the score function as *PSO.funcdef* . It is currently the *_norm_* function.</p>

<p> The animation doesn't occure realtime. The *PSO.valueIO* writes all states the particles go through to a *npy* file. *plotter.py* reads the file and animates the states. </p>

<p> Score Function can be changed by changing *PSO.funcdef*

<p> Feel free to fork and experiment! </p>



