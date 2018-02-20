# script.py
# Author : Victor Greiner (2016)

import configparser
import time
import os
from math import pi
from run import simulation
import random

def ask(question, answers=['y', 'n']):
    """Ask a question to the user until the input result belongs to the answers list
    question : str
    answers : str list
    returns the chosen answer"""
    choice = input(question)
    while choice not in answers:
        choice = input(question)
    return choice

def get_standard_options():
    """Return a dictionary containing the default parameters"""
    initial_config = configparser.ConfigParser()
    initial_config.read("config.cfg")

    config = configparser.ConfigParser()
    config.read("config.cfg")
    options = config['Options']

    options['LogFileName'] = 'last.json'
    options['WeightFileName'] = 'weights.json'
    options['NeuronFileName'] = 'last.json'

    options['RealRobot'] = 'n'
    options['Description'] = 'No Description'
    options['HiddenNeuronNumber'] = '3'
    options['LearningStep'] = '0.05'
    options['ThetaShiftRatio'] = '10'
    options['Size'] = '4'
    options['Prediction'] = '1'
    options['ExperimentalTheta'] = '1'

    options["Gain"] = '1'
    options["RestrictThetaShift"] = '1'
    options["RestrictPropagation"] = '0'
    options["InvertRestriction"] = '0'
    options["NewSigmoid"] = '0'
    options["RandomRatio"] = '0'

    options['Verbose'] = 'n'

    options['SingleSimulation'] = '1'
    options['MaximumDuration'] = '150'
    options['Tick'] = '0.020'

    options['Learn'] = 'y'
    options['Load'] = 'n'

    options['FixedStartingPosition'] = '1'
    options['StartingPositionX'] = '1'
    options['StartingPositionY'] = '1'
    options['StartingPositionTheta'] = '0'

    options['FixedTargetPosition'] = '1'
    options['TargetPositionX'] = '0'
    options['TargetPositionY'] = '0'
    options['TargetPositionTheta'] = '0'

    options['StopAutomatically'] = 'y'
    options['CriterionPercent'] = '0.03'
    options['CriterionDuration'] = '10'

    return options

options = get_standard_options()

choices = ["first_lessons", "100_lessons", "100_lessons_old", "3_lessons", "9_lessons", "4_lessons", "4_lessons_tests", "18_lessons", "testxy"]
choice = ask("Which script do you want to launch ? "+str(choices)+" -> ", choices)

LogFileName = choice + "/logs"

if choice=="first_lessons": # 10 independant first lessons
    nmax = 10
    for i in range(nmax):
        options['LogFileName'] = LogFileName+str(i+1)+".json"
        print("Running simulation "+str(i+1)+"/"+str(nmax)+"...")
        simulation(options)
        time.sleep(1)

elif choice=="100_lessons": # 100 successive lessons
    nmax = 100
    options['StartingPositionX'] = '1'
    options['StartingPositionY'] = '1'
    options['StartingPositionTheta'] = '0'
    options['TargetPositionX'] = '0'
    options['TargetPositionY'] = '0'
    for i in range(nmax):
        options['LogFileName'] = LogFileName+str(i+1)+".json"
        print("Running simulation "+str(i+1)+"/"+str(nmax)+"...")
        simulation(options)
        time.sleep(1)
        options['Load'] = 'y'

elif choice=="100_lessons_old": # 100 successive lessons with the old gradient formula
    nmax = 100
    options['Prediction'] = '0'
    options['ExperimentalTheta'] = '1'
    for i in range(nmax):
        options['LogFileName'] = LogFileName+str(i+1)+".json"
        print("Running simulation "+str(i+1)+"/"+str(nmax)+"...")
        simulation(options)
        time.sleep(1)
        options['Load'] = 'y'

elif choice=="3_lessons": # 3 lessons

    options['ExperimentalTheta'] = '1'

    # First lesson :
    options['StartingPositionX'] = '2'
    options['StartingPositionY'] = '2'
    options['StartingPositionTheta'] = '0'
    options['LogFileName'] = "3_lessons/1.json"
    print("Running simulation 1/4...")
    simulation(options)
    time.sleep(1)

    # Second lesson :
    options['Load'] = 'y'
    options['LogFileName'] = "3_lessons/2.json"
    print("Running simulation 2/4...")
    simulation(options)
    time.sleep(1)

    # Third lesson :
    options['StartingPositionX'] = '2'
    options['StartingPositionY'] = '0'
    options['StartingPositionTheta'] = '3.14'
    options['LogFileName'] = "3_lessons/3.json"
    print("Running simulation 3/4...")
    simulation(options)
    time.sleep(1)

    # Test :
    options['Learn'] = 'n'
    options['StartingPositionX'] = '2'
    options['StartingPositionY'] = '0'
    options['StartingPositionTheta'] = '1.57'
    options['LogFileName'] = "3_lessons/4.json"
    print("Running simulation 4/4...")
    simulation(options)
    time.sleep(1)

elif choice=="4_lessons": # 4 lessons

    n = 0
    nmax = 5

    for x in [-1, 1]:
        for y in [-1, 1]:
            n += 1
            options['LogFileName'] = LogFileName+str(n)+".json"
            options['StartingPositionX'] = str(x)
            options['StartingPositionY'] = str(y)
            options['StartingPositionTheta'] = str(1.57)
            print("Running simulation "+str(n)+"/"+str(nmax)+"...")
            simulation(options)
            time.sleep(1)
            options['Load'] = 'y'

    # Test :
    options['LogFileName'] = LogFileName+str(5)+".json"
    options['Learn'] = 'n'
    options['StartingPositionX'] = '-1'
    options['StartingPositionY'] = '1'
    options['StartingPositionTheta'] = '1'
    print("Now testing")
    print("Running simulation "+str(nmax)+"/"+str(nmax)+"...")
    simulation(options)

elif choice=="4_lessons_tests": # 4 lessons avec 10 tests

    n = 0
    nmax = 4

    for x in [-1, 1]:
        for y in [-1, 1]:
            n += 1
            options['LogFileName'] = LogFileName+str(n)+".json"
            options['StartingPositionX'] = str(x)
            options['StartingPositionY'] = str(y)
            options['StartingPositionTheta'] = str(1.57)
            print("Running simulation "+str(n)+"/"+str(nmax)+"...")
            simulation(options)
            time.sleep(1)
            options['Load'] = 'y'

    # Test :
    for i in range (0, 10):
        x = random.randint(-1, 1)
        y = random.randint(-1, 1)
        n += 1
        options['LogFileName'] = LogFileName+str(n)+".json"
        options['Learn'] = 'n'
        options['StartingPositionX'] = str(x)
        options['StartingPositionY'] = str(y)
        options['StartingPositionTheta'] = '1'
        print("Running simulation "+str(nmax)+"/"+str(nmax)+"...")
        simulation(options)

elif choice=="9_lessons": # 9 lessons

    n = 0
    nmax = 10
    options['Size'] = '3'
    for x in [-2, 0, 2]:
        for y in [-2, 0, 2]:
            options['LogFileName'] = LogFileName+str(n)+".json"
            options['StartingPositionX'] = str(1)
            options['StartingPositionY'] = str(1)
            options['StartingPositionTheta'] = str(0)
            n += 1
            print("Running simulation "+str(n)+"/"+str(nmax)+"...")
            simulation(options)
            time.sleep(1)
            options['Load'] = 'y'

    # Test :
    options['LogFileName'] = LogFileName+str(5)+".json"
    options['Learn'] = 'n'
    options['StartingPositionX'] = '1'
    options['StartingPositionY'] = '1'
    options['StartingPositionTheta'] = '1'
    print("Running simulation "+str(nmax)+"/"+str(nmax)+"...")
    simulation(options)

elif choice=="18_lessons": # 18 lessons

    n = 0
    nmax = 19
    options['Size'] = '3'
    for i in range (0, 2) :
    	for x in [-2, 0, 2]:
        	for y in [-2, 0, 2]:
            		options['LogFileName'] = LogFileName+str(n)+".json"
            		options['StartingPositionX'] = str(x)
            		options['StartingPositionY'] = str(y)
            		options['StartingPositionTheta'] = str(0)
            		n += 1
            		print("Running simulation "+str(n)+"/"+str(nmax)+"...")
            		simulation(options)
            		time.sleep(1)
            		options['Load'] = 'y'

    # Test :
    options['LogFileName'] = LogFileName+str(5)+".json"
    options['Learn'] = 'n'
    options['StartingPositionX'] = '1'
    options['StartingPositionY'] = '1'
    options['StartingPositionTheta'] = '1'
    print("Running simulation "+str(nmax)+"/"+str(nmax)+"...")
    simulation(options)

elif choice=="testxy": # un run avec X = x et Y = y
    x = 1
    y = 1
    theta = 0
    options['LogFileName'] = LogFileName+str(1)+".json"
    options['TargetPositionX'] = str(x)
    options['TargetPositionY'] = str(y)
    options['TargetPositionTheta'] = str(theta)
    print("Running simulation ...")
    simulation(options)

else:
    print("Incorrect choice")
