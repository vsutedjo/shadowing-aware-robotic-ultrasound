# shadowing-aware-robotic-ultrasound

# Python Robot Path Optimization for Ultrasound Compounding
This code runs the optimization for ultrasound compounding. Here is a quick guide on how to use it.

# What does it do?
You can define a voxel grid which represents your (unknown?) volume of interest. The classical trajectory would scan the volume in a perpendicular manner. Instead, we try to optimize the trajectory modularily.

We run two optimization iterations:
- Volume coverage: try to cover the defined voxel grid as efficiently as possible.
- Occlusion prevention: try to re-scan voxels that have detected occlusions in the first iteration.

# Workflow
You can define all your parameters (dimensions of your voxel grid, the tilt angles of your scan, the width of the transducer and the focal distance relative to the plane number) in [main.py](https://gitlab.lrz.de/ga87viv/optimization_python/-/blob/master/python/main.py) (see next chapter).

Once you have defined your setup, you can run the code. It will optimize for volume coverage and output the workspace file for imfusion, which you can directly open using the ImFusionSuite GUI. Run the Hybrid Ultrasound Simulation and you will get the respective scan.

In the coming days, the scan will be executed automatically through Python and a confidence map will be calculated on that scan that is piped back into the second iteration of the algorithm. 

### Parameters in [main.py](https://gitlab.lrz.de/ga87viv/optimization_python/-/blob/master/python/main.py#L18)
- ***[nr_rows, nr_cols, nr_planes](https://gitlab.lrz.de/ga87viv/optimization_python/-/blob/master/python/main.py#L18)*: int**: defines how your voxel grid will look like. The transducer will always scan from the top surface.
- ***[angles:](https://gitlab.lrz.de/ga87viv/optimization_python/-/blob/master/python/main.py#L21)* list of int in range [-45,45]:** The tilt angles you want to consider. (Currently, only one degree of rotation is implemented. It's possible float works fine as well but it hasn't throughly been tested. Make sure to enter the angles in order from smallest to largest)
- ***[transducer_width:](https://gitlab.lrz.de/ga87viv/optimization_python/-/blob/master/python/main.py#L22)* float:** The width of the transducer relative to a voxel. So a width of 1 means one scan will be the width of one voxel, 1.5 would also cover half of the neighboring voxel. (Most throughouly tested with 1).
- ***[focal_dist:](https://gitlab.lrz.de/ga87viv/optimization_python/-/blob/master/python/main.py#L22)* float:** The focal distance of a transducer in voxel heights. So a focal_dist of 2.5 means that a perpendicular scan is the sharpest in the second voxel layer starting from the transducer apex (the surface of the voxel grid).
- ***[visualize_scans:](https://gitlab.lrz.de/ga87viv/optimization_python/-/blob/master/python/main.py#L24)* bool:** Whether or not you want to visualize the synthetic scans.
### Synthetic occlusions
For the second iteration - occlusion prevention - you can also define a synthetic occlusion, if necessary. For that, go to functions/helpers/occlusion.py. There, you can define your [occlusion_fun(x,y)](https://gitlab.lrz.de/ga87viv/optimization_python/-/blob/master/python/functions/helpers/occlusion.py#L7) function relative to the x,y coordinates. Right now, only surface occlusions parallel to the voxel grid surface are implemented. If you want to simply visualize your current occlusion, uncomment the [visualize_occlusion line in main.py](https://gitlab.lrz.de/ga87viv/optimization_python/-/blob/master/python/main.py#L233) (and comment out the optimizer()).
### Perpendicular scan
For comparative evaluation data, you can uncomment [perpendicular_scan() in main.py](https://gitlab.lrz.de/ga87viv/optimization_python/-/blob/master/python/main.py#L232) main function to get a perpendicular scan (either a synthetic one for synthetic results or simulated US one).
