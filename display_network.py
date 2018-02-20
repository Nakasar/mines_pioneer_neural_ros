from graph_displayer import Graph
import json
import os
from ask import ask

def display(name):
    with open(name) as f:
        data = json.load(f)
        _in = data["input"]
        _inputs = [[d[i][j] for d in _in] for i in range(len(_in[0])) for j in range(len(_in[0][0]))]
        #for iii in _inputs:
            #print(iii)
        x = data["times"]
        g = Graph()
        for i in _inputs:
            g.add_data_set(x, i)
        print("Displaying ",len(_inputs), " set of ", len(_inputs[0]), " values...", sep='')
        g.show()
        g = Graph()

        _in = data["output"]
        _inputs = [[d[i][j] for d in _in] for i in range(len(_in[0])) for j in range(len(_in[0][0]))]
        #for iii in _inputs:
            #print(iii)
        g = Graph()
        for i in _inputs:
            g.add_data_set(x, i)
        print("Displaying ",len(_inputs), " set of ", len(_inputs[0]), " values...", sep='')
        g.show()
        g = Graph()

if __name__ == "__main__":
    
    os.chdir('networks')
    l = [f[:-5] for f in os.listdir() if (not os.path.isdir(f)) and f[-5:]==".json"]

    choice = 'l'
    while choice=='l':
        choice = ask("Which graph do you want to show : "+str(l)+" --> ", l + ['l'])
        if choice=='l':
            print(l)

    name = choice + ".json"

    display(name)
