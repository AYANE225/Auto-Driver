# One predicted motion hypothesis for a tracked object.
float32 confidence              # relative likelihood of this hypothesis in [0, 1]
string mode                     # motion model, e.g. "constant_velocity", "ctrv"
float32[] times                 # time offsets [s] of each waypoint from "now"
geometry_msgs/Point[] waypoints # forecast positions (x, y, z) at each time
