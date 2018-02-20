from rdn import Pioneer
import rospy
from math import pi

robot = Pioneer(rospy)
[x, y, theta] = robot.get_position()
print("x = ", round(x, 3), "y = ", round(y, 3), "theta = ", int(360*theta/2/pi)), 
