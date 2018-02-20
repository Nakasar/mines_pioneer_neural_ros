# run.py

# This is the main script that must be run in order to perform a learning session
# It can work with a real pioneer robot, or with a vrep simulation

# Last update by Victor Greiner (greinervictor@gmail.com)

from BackProp_Python_v2 import NN
from vrep_pioneer_simulation import VrepPioneerSimulation
from rdn import Pioneer
import rospy
from online_trainer import OnlineTrainer
import json
import threading
import configparser
import time

def ask(question, answers=['y', 'n']):
    """Ask a question to the user until the input result belongs to the answers list
    question : str
    answers : str list
    returns the chosen answer"""
    choice = input(question)
    while choice not in answers:
        choice = input(question)
    return choice

def simulation(options):
    """Performs a learning session with the given option dictionnary"""

    # Extract data from the option dictionnary

    RealRobot = options["RealRobot"]

    Verbose = options["Verbose"]
    LogFileName = options["LogFileName"]
    WeightFileName = options["WeightFileName"]
    NeuronFileName = options["NeuronFileName"]
    Description = options["Description"]
    SingleSimulation = options["SingleSimulation"]=='1'
    MaximumDuration = float(options["MaximumDuration"])
    Tick = float(options["Tick"])
    Prediction = options["Prediction"]=='1'
    ExperimentalTheta = options["ExperimentalTheta"]=='1'

    Gain = float(options["Gain"])
    RestrictThetaShift = options["RestrictThetaShift"]=='1'
    RestrictPropagation = options["RestrictPropagation"]=='1'
    InvertRestriction = options["InvertRestriction"]=='1'
    NewSigmoid = options["NewSigmoid"]=='1'
    RandomRatio = float(options["RandomRatio"])

    Load = options["Load"]
    Learn = options["Learn"]
    HiddenNeuronNumber = int(options["HiddenNeuronNumber"])
    LearningStep = float(options["LearningStep"])
    ThetaShiftRatio = float(options["ThetaShiftRatio"])
    Size = float(options["Size"])
    
    FixedStartingPosition = options["FixedStartingPosition"]=='1'
    StartingPositionX = float(options["StartingPositionX"])
    StartingPositionY = float(options["StartingPositionY"])
    StartingPositionTheta = float(options["StartingPositionTheta"])

    FixedTargetPosition = options["FixedTargetPosition"]=='1'
    TargetPositionX = float(options["TargetPositionX"])
    TargetPositionY = float(options["TargetPositionY"])
    TargetPositionTheta = float(options["TargetPositionTheta"])

    StopAutomatically = options["StopAutomatically"]
    CriterionPercent = float(options["CriterionPercent"])
    CriterionDuration = float(options["CriterionDuration"])

    # Create the right robot model (pioneer API or vrep API)
    if RealRobot=='a':
        RealRobot = ask('Is it a real robot ? (y/n) --> ')

    if RealRobot=='y':
        robot = Pioneer(rospy)
    else:
        robot = VrepPioneerSimulation()

    # Create the Neuron Network model
    # and load its values from a save file if necessary
    if WeightFileName:
        if Load=='a':
            choice = ask('Do you want to load previous network ? (y/n) --> ')
        else:
            choice = Load
        if choice == 'y':
            try:
                with open(WeightFileName) as fp:
                    json_obj = json.load(fp)
            except:
                print("Warning : could not find or open \""+WeightFileName+"\", starting from a random network...")
                network = NN(3, HiddenNeuronNumber, 2)
            else:
                # Check the size of the saved network
                n = len(json_obj["output_weights"])
                if n!=HiddenNeuronNumber:
                    print("Warning : the config.cfg file wanted ", HiddenNeuronNumber, " hidden neurons, but the saved network has ", n, " hidden neurons. The saved network will still be used...", sep='')
                    HiddenNeuronNumber = n

                network = NN(3, HiddenNeuronNumber, 2)
                # Load the weight values

                for i in range(3):
                    for j in range(HiddenNeuronNumber):
                        network.wi[i][j] = json_obj["input_weights"][i][j]
                for i in range(HiddenNeuronNumber):
                    for j in range(2):
                        network.wo[i][j] = json_obj["output_weights"][i][j]
        else:
            network = NN(3, HiddenNeuronNumber, 2)
    else:
        network = NN(3, HiddenNeuronNumber, 2)

    if NewSigmoid:
        network.newSigmoid = True

    # The trainer will monitor the lesson until one of the stopping criterion is met
    # It works on a separate thread, started by its start() method
    # See online_trainer.py for more details
    trainer = OnlineTrainer(robot, network)

    if Learn=='a':
        choice = ask('Do you want to learn ? (y/n) --> ')
    else:
        choice = Learn

    if choice == 'y':
        trainer.training = True
    else: # choice = 'n'
        trainer.training = False

    # Move the robot to its starting location
    # This is only possible with a vrep simulation
    # since the real robot always starts at [0, 0, 0]
    if not (RealRobot=='y'):
        if FixedStartingPosition:
            location = [StartingPositionX, StartingPositionY, StartingPositionTheta]
            robot.set_position2(location)
        else:
            location = []
            while len(location) != 3:
                location = input("Enter the starting location : x y radian --> ")
                location = location.split()
            for i in range(len(location)):
                location[i] = float(location[i])
            robot.set_position2(location)
            
    if FixedTargetPosition:
        target = [TargetPositionX, TargetPositionY, TargetPositionTheta]
    else:
        target = []
        while len(target) != 3:
            target = input("Enter the first target : x y radian --> ")
            target = target.split()
        for i in range(len(target)):
            target[i] = float(target[i])
        print('New target : [%d, %d, %d]'%(target[0], target[1], target[2]))

    if StopAutomatically=='a':
        StopAutomatically = ask("Do you want to stop the simulation when the ending criterion is met ? --> ")
    StopAutomatically = StopAutomatically=='y'

    if Verbose=='a':
        Verbose = ask("Do you want to display data while the trainer is running ? (y/n) --> ")
    Verbose = Verbose=='y'

    continue_running = True
    # Each iteration correspond to one lesson
    # If the SingleSimulation parameter is not set to 1 in config.cfg,
    # Several successive lessons can be done
    while(continue_running):

        if StopAutomatically:
            stop_criterion = [CriterionPercent, CriterionDuration]
        else:
            stop_criterion = None

        thread = threading.Thread(target=trainer.train, args=(target, options, Tick, Prediction, ExperimentalTheta, Gain, RestrictThetaShift, RestrictPropagation, InvertRestriction, RandomRatio, LearningStep, ThetaShiftRatio, Size, MaximumDuration, stop_criterion, Verbose, LogFileName, NeuronFileName, Description))
        thread.daemon = True # Kill the current simulation when this script is killed
        trainer.running = True
        begin_date = time.time()
        thread.start() # Start the simulation on another thread

        if StopAutomatically or MaximumDuration>0:
            while trainer.running:
                time.sleep(0.1)
        else: # StopAutomatically=='n' and MaximumDuration==0:
            input("Press Enter to stop the current training\n")

        elapsed_time = time.time() - begin_date
        print(" "*60, end='\r')
        print("Finished in "+str(round(elapsed_time, 1))+" sec.")

        trainer.running = False

        if SingleSimulation:
            choice = 'n'
        else:
            choice = ask("Do you want to continue ? (y/n) --> ")

        if choice == 'y':
            if Learn=='a':
                choice_learning = ask('Do you want to learn ? (y/n) --> ')
            else:
                choice_learning = Learn
            if choice_learning =='y':
                trainer.training = True
            elif choice_learning == 'n':
                trainer.training = False
            if FixedStartingPosition:
                location = [StartingPositionX, StartingPositionY, StartingPositionTheta]
                robot.set_position2(location)
            else:
                choice_reset_position = ask("Do you want to change the robot location ? (y/n) --> ")
                if choice_reset_position=='y':
                    location = []
                    while len(location) != 3:
                        location = input("Enter the new location : x y radian --> ")
                        location = location.split()
                    for i in range(len(location)):
                        location[i] = float(location[i])
                    robot.set_position2(location)

            if FixedTargetPosition:
                target = [TargetPositionX, TargetPositionY, TargetPositionTheta]
            else:
                target = []
                while len(target) != 3:
                    target = input("Enter the new target : x y radian --> ")
                    target = target.split()
                for i in range(len(target)):
                    target[i] = float(target[i])
                print('New target : [%d, %d, %d]'%(target[0], target[1], target[2]))
        else: # choice = 'n'
            continue_running = False

    if WeightFileName:
        # Save the current Neuron Network
        with open(WeightFileName, 'w') as fp:
            json_obj = {"input_weights": network.wi, "output_weights": network.wo}
            json.dump(json_obj, fp)
            fp.close()

    while not trainer.ready_to_exit:
        time.sleep(0.1)

    return elapsed_time

if __name__=="__main__":
    # Extract data from the config file
    try:
        config = configparser.ConfigParser()
        config.read("config.cfg")
        file_options = config['Options']

    except:
        print("Error : Could not retrieve options from the config.cfg file")
        exit()
    # Run the lesson
    simulation(file_options)

