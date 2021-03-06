import vrep
import math
import time

def to_rad(deg):
    return 2*math.pi*deg/360

def to_deg(rad):
    return rad*360/(2*math.pi)

class VrepPioneerSimulation:
    def __init__(self):

        self.ip = '127.0.0.1'
        self.port = 19997
        self.scene = './simu.ttt'
        self.gain = 2
        self.initial_position = [1, 1,to_rad(45)]

        self.r = 0.096 # wheel radius
        self.R = 0.267 # demi-distance entre les r

        self.activated_sensors = [1, 4, 5, 8, 9, 12, 13, 16]

        #print('New pioneer simulation started')
        vrep.simxFinish(-1)
        self.client_id = vrep.simxStart(self.ip, self.port, True, True, 5000, 5)

        if self.client_id!=-1:
            #print ('Connected to remote API server on %s:%s' % (self.ip, self.port))
            res = vrep.simxLoadScene(self.client_id, self.scene, 1, vrep.simx_opmode_oneshot_wait)
            res, self.pioneer = vrep.simxGetObjectHandle(self.client_id, 'Pioneer_p3dx', vrep.simx_opmode_oneshot_wait)
            res, self.left_motor = vrep.simxGetObjectHandle(self.client_id, 'Pioneer_p3dx_leftMotor', vrep.simx_opmode_oneshot_wait)
            res, self.right_motor = vrep.simxGetObjectHandle(self.client_id, 'Pioneer_p3dx_rightMotor', vrep.simx_opmode_oneshot_wait)
            self.proximity_sensors = [0 for i in range(len(self.activated_sensors))]

            for i in range(len(self.activated_sensors)):
                res, self.proximity_sensors[i] = vrep.simxGetObjectHandle(self.client_id, 'Pioneer_p3dx_ultrasonicSensor' + str(self.activated_sensors[i]), vrep.simx_opmode_oneshot_wait)

        else:
            print('Unable to connect to %s:%s' % (self.ip, self.port))
            exit()

        self.set_position2(self.initial_position)
        #self.set_position(self.initial_position)
        #vrep.simxStartSimulation(self.client_id, vrep.simx_opmode_oneshot_wait)
        firt_load_of_sensors = self.load_proximity_sensors()

    # This function seems to not work properly, use set_position2 instead
    def set_position(self, position):
        """Set the position (x,y,theta) of the robot

        Args:
            position (list): the position [x,y,theta]
        """

        vrep.simxSetObjectPosition(self.client_id, self.pioneer, -1, [position[0], position[1], 0.13879], vrep.simx_opmode_oneshot_wait)
        vrep.simxSetObjectOrientation(self.client_id, self.pioneer, -1, [0, 0, to_deg(position[2])], vrep.simx_opmode_oneshot_wait)

    def set_position2(self, position):
        """Set the position (x,y,theta) of the robot

        Args:
            position (list): the position [x,y,theta]
        """
        vrep.simxStopSimulation(self.client_id, vrep.simx_opmode_oneshot_wait)
        time.sleep(0.1) # Ensure that the simulation is properly stopped
        self.set_position(position)
        vrep.simxStartSimulation(self.client_id, vrep.simx_opmode_oneshot_wait)

    def get_position(self):
        """Get the position (x,y,theta) of the robot

        Return:
            position (list): the position [x,y,theta]
        """
        position = []
        res, tmp = vrep.simxGetObjectPosition(self.client_id, self.pioneer, -1, vrep.simx_opmode_oneshot_wait)
        position.append(tmp[0])
        position.append(tmp[1])

        res, tmp = vrep.simxGetObjectOrientation(self.client_id, self.pioneer, -1, vrep.simx_opmode_oneshot_wait)
        position.append(tmp[2]) # en radian

        return position

    def set_motor_velocity(self, control):
        """Set a target velocity on the pioneer motors, multiplied by the gain
        defined in self.gain

        Args:
            control(list): the control [left_motor, right_motor]
        """
        vrep.simxSetJointTargetVelocity(self.client_id, self.left_motor, self.gain*control[0], vrep.simx_opmode_oneshot_wait)
        vrep.simxSetJointTargetVelocity(self.client_id, self.right_motor, self.gain*control[1], vrep.simx_opmode_oneshot_wait)

    def load_proximity_sensors(self):
        """First call to initiate proximity sensors.

        active sensors : 0, 3, 4, 7 (1,4,5,8), in front of robot.
        """
        values = [0 for i in range(len(self.proximity_sensors))];
        for sensor in range(len(self.proximity_sensors)):
            values[sensor] = self.load_proximity_sensor(sensor)
        return values

    def load_proximity_sensor(self, sensor):
        """First call to initiate given proximity sensor.

        Args:
            sensor(int): the index of the sensor to load.
        """
        return vrep.simxReadProximitySensor(self.client_id, self.proximity_sensors[sensor], vrep.simx_opmode_streaming);

    def get_proximity_sensors(self):
        """Get values for all proximity sensors.

        """
        values = [0 for i in range(len(self.proximity_sensors))];
        for sensor in range(len(self.proximity_sensors)):
            values[sensor] = self.get_proximity_sensor(sensor)
        return values

    def get_proximity_sensor(self, sensor):
        """Get value of given proximity sensor.

        Args:
            sensor(int): the index of the sensor to get value of.
        """
        return vrep.simxReadProximitySensor(self.client_id, self.proximity_sensors[sensor], vrep.simx_opmode_buffer);

    def get_sensors_distances(self):
        """ Returns the distances of all the sensors in a list.

        Args:
            (list): sensors to get distance of.

        Return:
            (list): the sensors distances.
        """
        sensors_values = self.get_proximity_sensors()

        invert_distances = []
        for value in sensors_values :
            vector = value[2]
            distance = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
            if (value[1]) :
                if (distance == 0) :
                    invert_distances.append(100)
                else :
                    invert_distances.append(1 / distance)
            else :
                invert_distances.append(0)

        return invert_distances
