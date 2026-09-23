# A single confirmed track from the multi-object tracker.
int32 id                        # persistent track id
string label                    # object class, e.g. "car", "pedestrian"
float32 score                   # detection/track confidence in [0, 1]
geometry_msgs/Pose pose         # center pose (position + orientation from yaw)
geometry_msgs/Vector3 dimensions # bounding-box extent (l, w, h) in metres
geometry_msgs/Vector3 velocity  # (vx, vy, 0) in m/s, ground plane
float32 yaw_rate                # estimated turn rate in rad/s
uint8 age                       # frames since the track was created
uint8 hits                      # number of successful measurement updates
