# online_trainer.py

# The OnlineTrainer class monitor a lesson until an ending criterion is met
# There are 3 ending criterion :
#     - Interruption by the run.py script (self.running set to False)
#     - Convergence of the robot : the relative error must stay under a given threshold for a given duration
#     - Maximum experiment duration reached
# These parameter can be set in the config.cfg file

# The trainer is run on a new thread to enable run.py to stop it
# It performs several actions :
#     - Calling the Neuron Network iterations
#     - Setting the robot velocity
#     - Computing the gradient
#     - Calling the backpropagation

import time
import math
import json
from math import cos, sin, atan2, exp, log, pi, sqrt, atan
import os
from movement_equations import get_grad

def create_folder_if_necessary(dirname):
    if "/" in dirname:
        l = os.listdir(dirname[:dirname.rfind("/")])
    else:
        l = os.listdir()
    if dirname[dirname.find("/")+1:] not in l:
        os.mkdir(dirname)
    elif not os.path.isdir(dirname):
        os.remove(dirname)
        os.mkdir(dirname)

def create_path_if_necessary(dirname):
    l = dirname.split("/")
    for i in range(len(l)):
        create_folder_if_necessary("/".join(l[:i+1]))

def create_file_path_if_necessary(filepath):
    if "/" in filepath:
        create_path_if_necessary(filepath[:filepath.rfind("/")])

def get_relative_error(position, target, size):
    """Returns the relative error between the current robot position
    and its target
    0 means that the target is reached
    1 means that the robot is at the furthest possible position from the target"""
    ex = (position[0] - target[0]) / size
    ey = (position[1] - target[1]) / size
    etheta = (position[2] - target[2]) / math.pi  #Faut trouver la bonne relation pour etheta
    if etheta<=-1:
       etheta+=2
    elif etheta>1:
       etheta-=2
    relative_error = math.sqrt(ex**2 / 3 + ey**2 / 3 + etheta**2 / 3)
    return relative_error

#Il faut ajouter devant les atan un paramètres pour atténuer les oscillations
def theta_s(x, y, ratio, experimental_theta):
    """Returns the theta shift of the robot
    The goal of this shift if to prevent the robot from being stuck
    when he reach a local minimum of the standard error"""
    if not experimental_theta:
        if x>0:
            return math.atan(ratio*y)
        if x<=0:
            return math.atan(-ratio*y)
    else:
        if x>0:
            t = math.atan(ratio*y)
        else:
            t = -math.atan(ratio*y)

        size = 4
        m = 0.4
        rate = 0.50
        X = -m/log((1 - rate))
        d = abs(x)
        c = exp(-d / X)
        c = 1 + c
        e = 1 - exp(-(sqrt(x**2+y**2) / 0.40))
        return t
        #return c*t * e

def deep_copy(l):
    if isinstance(l, list):
        return list(map(deep_copy, l))
    return l

class OnlineTrainer:
    def __init__(self, robot, NN):
        """
        Args:
            robot (Robot): a robot instance following the pattern of
                VrepPioneerSimulation
            target (list): the target position [x,y,theta]
        """
        self.robot = robot
        self.network = NN

        self.alpha = [1/4,1/4,1/((math.pi))]
        self.sensor_alpha = 1/4

        self.ready_to_exit = False

    def train(self, target, options={}, tick=0.050, prediction=False, experimental_theta=False, gain=1, restrict_theta_shift=True, restrict_propagation=False, invert_restriction=False, random_ratio=0, learning_step=0.05, theta_shift_ratio=10, size=4, maximum_duration=0, stop_criterion=None, verbose=False, log='', neuron_file_name='', description="No Description"):

        self.ready_to_exit = False

        self.alpha = [1/size,1/size,1/((math.pi))]

        # This coefficient can reduce the theta input to [-0.75, +0.75]
        self.alpha[2] *= 1

        position = self.robot.get_position() # This function takes 50 ms to compute
        sensors = self.robot.get_sensors_distances();

        self.stop_automatically = stop_criterion!=None
        if self.stop_automatically:
            self.percent_criterion = stop_criterion[0]
            self.duration_criterion = stop_criterion[1]
        self.stop_criterion_reached = False

        network_input = [0] * 11
        network_input[0] = (position[0]-target[0])*self.alpha[0]
        network_input[1] = (position[1]-target[1])*self.alpha[1]
        network_input[2] = (position[2]-target[2])*self.alpha[2]
        for i in range(len(sensors)) :
            network_input[i+3] = sensors[i] * 1

        t0 = time.time()

        if log or neuron_file_name:
            times = []
        if log:
            positions = []
            commands = []
            criterions = []
            gradients = []
            theta_shifts = []
        if neuron_file_name:
            neurons_input = []
            neurons_output = []

        criterion_reach_date = time.time()

        delta_t = tick

        relative_error = 1000

        # Loop until one of the ending criterions is met
        while self.running and not (self.stop_automatically and self.stop_criterion_reached):
            new_system = True

            t = time.time()

            # Get the robot new position
            position = self.robot.get_position() # this takes 50 ms to process
            sensors = self.robot.get_sensors_distances()

            new_error = get_relative_error(position, target, size)

            error_is_smaller = new_error < relative_error
            #12/12 essai en passant de .20 à .40
            enable_theta_shift = (error_is_smaller or new_error > 0.40)

            # Compute the new input values for the Neuron Network
            if (not restrict_theta_shift) or enable_theta_shift:
                network_input[2]=(((position[2]-target[2]-theta_s(position[0]-target[0], position[1]-target[1], theta_shift_ratio, experimental_theta)+math.pi)%(2*math.pi))-math.pi)*self.alpha[2]
            else:
                network_input[2]=(((position[2]-target[2]+math.pi)%(2*math.pi))-math.pi)*self.alpha[2]
            info = ", t_s=" + str(int(theta_s(position[0]-target[0], position[1]-target[1], theta_shift_ratio, experimental_theta)*360/2/math.pi))
            network_input[1] = (position[1]-target[1])*self.alpha[1]
            network_input[0] = (position[0]-target[0])*self.alpha[0]
            for i in range(len(sensors)) :
                network_input[i+3] = sensors[i] * self.sensor_alpha

            if log:
                theta_shifts.append(theta_s(position[0]-target[0], position[1]-target[1], theta_shift_ratio, experimental_theta))

            # Tell the Neuron Network to make an iteration (propagation)
            command = self.network.runNN(network_input)

            # Change the robot wheel velocity accordingly
            self.robot.set_motor_velocity(command) # this takes 100 ms to process

            if log or neuron_file_name:
                times.append(t - t0)
            if log:
                # Save the current state of the lesson
                positions.append(self.robot.get_position()) # This function takes 50 ms to compute
                commands.append(command[:])
                criterions.append(get_relative_error(position, target, size))
            if neuron_file_name:
                neurons_input.append(deep_copy(self.network.wi))
                neurons_output.append(deep_copy(self.network.wo))

            if self.training:

                if prediction:
                    sensors_influence_rd = 0
                    sensors_influence_rg = 0

                    r = self.robot.r
                    R = self.robot.R
                    x = position[0]
                    y = position[1]
                    theta = position[2]
                    x_target = target[0]
                    y_target = target[1]
                    if (not restrict_theta_shift) or enable_theta_shift or self.training:
                        theta_target = ((target[2] + theta_s(position[0] - target[0], position[1] - target[1], theta_shift_ratio, experimental_theta) + math.pi)%(2 * math.pi)) - math.pi # With theta_shift
                    else:
                        theta_target = ((target[2] + math.pi)%(2 * math.pi)) - math.pi # Without theta_shift

                    # ce dictionnaire contient les angles des capteurs (utile pour obtenir leur influence)
                    sensors_angle = {0: -math.pi / 6, 1: -math.pi/3, 2: -2*math.pi/3, 3: -5*math.pi/6, 4: 5*math.pi/6, 5: 2*math.pi/3, 6: math.pi/3, 7: math.pi/6}
                    # TODO: Modify grad for proper retro-propagation
                    for k in range(len(sensors)):
                        offset = network_input[3 + k] * (time.time() - t) * self.robot.r / self.robot.R
                        if k == 0 or k == 7:
                            vects = [1, -1]
                        elif k == 1 or k == 2:
                            vects = [1, 1]
                        elif k == 3 or k == 4:
                            vects = [-1, 1]
                        else:
                            vects = [-1, -1]
                        sensors_influence_rd += vects[0] * offset
                        sensors_influence_rg += vects[1] * offset

                    grad = get_grad(self, r, R, size, x, y, theta, x_target, y_target, theta_target, delta_t)
                    grad = [gain * (grad[0] - sensors_influence_rg), gain * (grad[1] - sensors_influence_rd)]

                else: # Old method
                    sensors_influence_rd = 0
                    sensors_influence_rg = 0
                    # ce dictionnaire contient les angles des capteurs (utile pour obtenir leur influence)
                    sensors_angle = {0: -math.pi / 6, 1: -math.pi/3, 2: -2*math.pi/3, 3: -5*math.pi/6, 4: 5*math.pi/6, 5: 2*math.pi/3, 6: math.pi/3, 7: math.pi/6}
                    # TODO: Modify grad for proper retro-propagation
                    for k in range(len(sensors)):
                        offset = network_input[3 + k] * delta_t * self.robot.r / self.robot.R
                        if k == 0 or k == 7:
                            vects = [1, -1]
                        elif k == 1 or k == 2:
                            vects = [1, 1]
                        elif k == 3 or k == 4:
                            vects = [-1, 1]
                        else:
                            vects = [-1, -1]
                        sensors_influence_rd += vects[0] * offset
                        sensors_influence_rg += vects[1] * offset

                    grad = [
                        ((-1)/(delta_t**2))*(network_input[0]*delta_t*self.robot.r*math.cos(position[2])
                        +network_input[1]*delta_t*self.robot.r*math.sin(position[2])
                        -network_input[2]*delta_t*self.robot.r/(2*self.robot.R))
                        -sensors_influence_rg,

                        ((-1)/(delta_t**2))*(network_input[0]*delta_t*self.robot.r*math.cos(position[2])
                        +network_input[1]*delta_t*self.robot.r*math.sin(position[2])
                        +network_input[2]*delta_t*self.robot.r/(2*self.robot.R))
                        -sensors_influence_rd
                    ]

                if log:
                    gradients.append(grad)

                if (error_is_smaller and not invert_restriction) or (invert_restriction and not error_is_smaller):
                    # The two args after grad are the gradient learning steps for t
                    # and t-1
                    self.network.backPropagate(grad, learning_step, 0)
                elif random_ratio > 0:
                    self.network.random_update(random_ratio)
                elif not restrict_propagation:
                    self.network.backPropagate(grad, learning_step, 0)

            # Compute the relative error
            relative_error = get_relative_error(position, target, size)

            # Display the current data
            s = "[" + str(round(t-t0, 1)) + " sec.] x=" + str(round(position[0], 3)) + ", y=" + str(round(position[1], 3)) + ", th=" + str(int(position[2]*360/2/math.pi)) + ", err=" + str(round(100*relative_error, 1)) + "%" + info
            if len(s) < 60:
                s += " "*(60-len(s))
            if self.running:
                if verbose:
                    print(s)
                else:
                    print(s, end="\r")

            if maximum_duration > 0 and t >= t0 + maximum_duration:
                # Maximum duration reached : the lesson is forced to stop
                self.running = False
            elif self.stop_automatically:
                # Check if the robot has converged
                if relative_error <= self.percent_criterion:
                    if t >= criterion_reach_date + self.duration_criterion:
                        self.stop_criterion_reached = True
                else:
                    criterion_reach_date = t

            # Wait to prevent underflow
            t_after_iteration = time.time()
            if tick > t_after_iteration - t:
                time.sleep(tick - (t_after_iteration - t))

        # End of the lesson
        self.robot.set_motor_velocity([0,0]) # This function takes 100 ms to compute
        self.running = False

        if log:
            # Save the logs in a file
            duration = time.time() - t0
            obj = {"options":dict(options), "target":target, "duration":duration, "times":times, "positions":positions, "commands":commands, "criterions":criterions, "gradients":gradients, "theta_shifts":theta_shifts}
            create_file_path_if_necessary("logs/"+log)
            with open("logs/"+log, 'w') as f:
                json.dump(obj, f)
                f.close()
            autosave_name = "logs/AUTOSAVE/"+str(int(time.time()))+".json"
            create_file_path_if_necessary(autosave_name)
            with open(autosave_name, 'w') as f:
                json.dump(obj, f)
                f.close()

        if neuron_file_name:
            obj = {"times":times, "input":neurons_input, "output":neurons_output}
            create_file_path_if_necessary("networks/"+log)
            with open("networks/"+log, 'w') as f:
                json.dump(obj, f)
                f.close()

        self.ready_to_exit = True
