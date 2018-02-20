# theta_s_2.py

# This quick script display the direction of the theta shift for several positions.

import matplotlib.pyplot as mp
import json
import os
from math import atan, cos, sin

class Graph():
    def __init__(self):
        pass

    def add_data_set(self, x, y):
        #mp.plot(x, y, color="blue")
        mp.plot(x, y)

    def show(self):
        mp.show()

    def draw(self, x, y):
        self.add_data_set(x, y)
        self.show()

def draw_from_set(x, y):
    graph = Graph()
    graph.draw(x, y)

def draw_from_file(name):
    with open(name) as f:
        data = json.load(f)
        graph = Graph()
        graph.add_data_set([p[0] for p in data["positions"]], [p[1] for p in data["positions"]])
        graph.draw()

def get_data_from_json(name):
    with open(name) as f:
        data = json.load(f)
        return data

def get_theta(x, y, coef):
    if x>=0:
        return atan(coef*y)
    else:
        return atan(-coef*y)

def forward(x1, y1):
    x = [x1]
    y = [y1]
    while abs(y1) > epsilon and abs(y1) <= 4 and x1<0:
        #print("1 :", y1)
        theta = get_theta(x1, y1, coef)
        x2 = x1 + cos(theta)*d
        y2 = y1 + sin(theta)*d
        x.append(x2)
        y.append(y2)
        x1 = x2
        y1 = y2
    g.add_data_set(x, y)

def backward(x1, y1):
    x = [x1]
    y = [y1]
    while abs(y1) > epsilon and abs(y1) <= 4 and x1>0:
        #print("1 :", y1)
        theta = get_theta(x1, y1, coef)
        x2 = x1 - cos(theta)*d
        y2 = y1 - sin(theta)*d
        x.append(x2)
        y.append(y2)
        #print(round(x2, 2), round(y2, 2))
        x1 = x2
        y1 = y2
    #print(x, y)
    g.add_data_set(x, y)

if __name__ == "__main__":
    mp.axis('equal')
    d = 0.05
    n = 50
    coef = 10
    epsilon = 0.01
    g = Graph()
    for i in range(0, n):
        x1 = 4*i/n
        y1 = 4
        backward(x1, y1)

        x1 = 4*i/n
        y1 = -4
        backward(x1, y1)

        x1 = -4*i/n
        y1 = 4
        forward(x1, y1)

        x1 = -4*i/n
        y1 = -4
        forward(x1, y1)


    mp.title("Lignes de directions de theta_strategique")
    g.show()

