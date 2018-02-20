# graph_displayer.py
# Author : Victor Greiner (2016)

import matplotlib.pyplot as mp
import json
import os
from ask import ask
from mpl_toolkits.mplot3d import Axes3D

class Graph():
    def __init__(self):
        pass

    def add_data_set(self, x, y):
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
        data["t"] = data["times"]
        data["x"]= [p[0] for p in data["positions"]]
        data["y"] = [p[1] for p in data["positions"]]
        data["theta"] = [p[2] for p in data["positions"]]
        data["command1"] = [c[0] for c in data["commands"]]
        data["command2"] = [c[1] for c in data["commands"]]
        try :
            data["grad1"] = [c[0] for c in data["gradients"]]
            data["grad2"] = [c[1] for c in data["gradients"]]
        except:
            pass
        return data

if __name__ == "__main__":

    os.chdir('logs')
    directories = [d for d in os.listdir() if os.path.isdir(d) and d!="AUTOSAVE"]
    main_json_files = [f for f in os.listdir() if (not os.path.isdir(f)) and f[-5:]==".json"]
    main_json_files_alt = [f[:-5] for f in os.listdir() if (not os.path.isdir(f)) and f[-5:]==".json"]
    json_files = [f for f in os.listdir("AUTOSAVE") if (not os.path.isdir(f)) and f[-5:]==".json"]
    json_files_alt = [f[:-5] for f in os.listdir("AUTOSAVE") if (not os.path.isdir(f)) and f[-5:]==".json"]
    l = [f[:-5] for f in os.listdir("AUTOSAVE") if (not os.path.isdir(f)) and f[-5:]==".json" and len(f)==15]
    l.sort(key=lambda a:int(a[:-5]))
    l = list(reversed(l))
    json_files_alt_2 = list(reversed([f[-4:] for f in l[:10]]))

    #print(json_files_alt_2)

    choice = 'l'
    while choice=='l':
        choice = ask("Which graph do you want to show : "+str([d for d in os.listdir() if (os.path.isdir(d) or d[-5:]==".json") and d!="AUTOSAVE"])+" --> ", directories + json_files + json_files_alt + json_files_alt_2 + ['l'] + main_json_files + main_json_files_alt)
        if choice=='l':
            print(json_files_alt_2)

    g = Graph()

    if choice in directories:
        os.chdir(choice)

        mp.title("Trajectories")
        for name in os.listdir():
            data = get_data_from_json(name)
            g.add_data_set(data["x"], data["y"])
        g.show()

        g = Graph()
        mp.title("Criterions")
        for name in os.listdir():
            data = get_data_from_json(name)
            g.add_data_set(data["times"], data["criterions"])
        g.show()

        g = Graph()
        mp.title("Durations")
        n = []
        durations = []
        l = os.listdir()
        try:
            l.sort(key=lambda a:int(a[4:-5]))
            for name in l:
                data = get_data_from_json(name)
                if data["duration"] < 59:
                    n.append(int(name[4:-5]))
                    durations.append(data["duration"]-10)
            g.add_data_set(n, durations)
            g.show()
        except:
            pass

        g = Graph()
        mp.title("theta")
        for name in os.listdir():
            data = get_data_from_json(name)
            g.add_data_set(data["t"], data["theta"])
        g.show()

        g = Graph()
        mp.title("commands")
        for name in os.listdir():
            data = get_data_from_json(name)
            g.add_data_set(data["t"], data["command1"])
        g.add_data_set(data["t"], data["command2"])
        g.show()

        #Courbes 3D (07/03/17)
        fig = mp.figure()
        mp.title("3D")
        ax = fig.add_subplot(111, projection='3d')
        for name in os.listdir():
            data = get_data_from_json(name)
            ax.plot(data["theta"],data["x"],data["y"])

        try:
            g = Graph()
            mp.title("gradient")
            for name in os.listdir():
                 data = get_data_from_json(name)
                 g.add_data_set(data["t"], data["grad1"])
                 g.add_data_set(data["t"], data["grad2"])
            g.show()
        except:
            pass

        g = Graph()
        mp.title("theta_shift")
        g.add_data_set(data["t"], data["theta_shifts"])
        g.show()

        g = Graph()
        mp.title("xt")
        for name in os.listdir():
            data = get_data_from_json(name)
            g.add_data_set(data["t"], data["x"])
        g.show()

        g = Graph()
        mp.title("yt")
        for name in os.listdir():
            data = get_data_from_json(name)
            g.add_data_set(data["t"], data["y"])
        g.show()

    elif choice in main_json_files:
        data = get_data_from_json(choice)
        g.add_data_set(data["x"], data["y"])
        if "description" in data["options"] and data["options"]["description"] and data["options"]["description"]!="No description":
            mp.title(data["options"]["description"].replace("[duration]", str(int(data["duration"])-10)))
        #mp.title("Temps de réponse : "+str(int(data["duration"])-10)+" sec.")
        g.show()

        g = Graph()
        mp.title("theta")
        g.add_data_set(data["t"], data["theta"])
        g.show()
    elif choice in main_json_files_alt:
        data = get_data_from_json(choice+".json")
        g.add_data_set(data["x"], data["y"])
        if "description" in data["options"] and data["options"]["description"] and data["options"]["description"]!="No description":
            mp.title(data["options"]["description"].replace("[duration]", str(int(data["duration"])-10)))
        #mp.title("Temps de réponse : "+str(int(data["duration"])-10)+" sec.")
        #print(data["options"])
        g.show()

        # Other graphs
        g = Graph()
        mp.title("criterion")
        g.add_data_set(data["t"], data["criterions"])
        g.show()

        g = Graph()
        mp.title("theta")
        g.add_data_set(data["t"], data["theta"])
        g.show()

        g = Graph()
        mp.title("commands")
        g.add_data_set(data["t"], data["command1"])
        g.add_data_set(data["t"], data["command2"])
        g.show()

        try:
            g = Graph()
            mp.title("gradient")
            g.add_data_set(data["t"], data["grad1"])
            g.add_data_set(data["t"], data["grad2"])
            g.show()
        except:
            pass

        g = Graph()
        mp.title("theta_shift")
        g.add_data_set(data["t"], data["theta_shifts"])
        g.show()





    elif choice in json_files:
        os.chdir("AUTOSAVE")
        data = get_data_from_json(choice)
        g.add_data_set(data["x"], data["y"])
        if "description" in data["options"] and data["options"]["description"] and data["options"]["description"]!="No description":
            mp.title(data["options"]["description"].replace("[duration]", str(int(data["duration"])-10)))
        #mp.title("Temps de réponse : "+str(int(data["duration"])-10)+" sec.")
        g.show()
    elif choice in json_files_alt:
        os.chdir("AUTOSAVE")
        data = get_data_from_json(choice+".json")
        g.add_data_set(data["x"], data["y"])
        if "description" in data["options"] and data["options"]["description"] and data["options"]["description"]!="No description":
            mp.title(data["options"]["description"].replace("[duration]", str(int(data["duration"])-10)))
        #mp.title("Temps de réponse : "+str(int(data["duration"])-10)+" sec.")
        g.show()
    elif choice in json_files_alt_2:
        os.chdir("AUTOSAVE")
        l = [f[:-5] for f in os.listdir() if (not os.path.isdir(f)) and f[-5:]==".json" and len(f)==15]
        l.sort(key=lambda a:int(a[:-5]))
        l = list(reversed(l))
        #print(l)
        c = ''
        for name in l:
            if (not c) and name[-4:]==choice:
                c = name
        if not c:
            print("Error : could not find the file or folder")
            exit(-1)
        choice = c
        #print(choice)
        data = get_data_from_json(choice+".json")
        g.add_data_set(data["x"], data["y"])
        if "description" in data["options"] and data["options"]["description"] and data["options"]["description"]!="No description":
            mp.title(data["options"]["description"].replace("[duration]", str(int(data["duration"])-10)))
        #mp.title("Temps de réponse : "+str(int(data["duration"])-10)+" sec.")
        g.show()
    else:
        print("Error : could not find the file or folder")
