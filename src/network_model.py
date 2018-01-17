import networkx as nx
import math
import matplotlib.pyplot as plt
import random

def getModel(time, probabilityP, probabilityQ,probabilityD, per):
    probabilityR = 1.0-probabilityP-probabilityQ
    G = nx.DiGraph()
    G.add_node(0)
    per = time*(per/100.0)
    """
    At each time step
    with probability p, a new node is created attaching to a directed node,
    with probability q, a new node is created attached by a directed link 
    with probability r, a directed link is created between the old nodes.
    """
    for step in range(time):

        arrayNode = list(G.nodes)    
        newIn = []
        newOut = []
        D = []
        
        for i in arrayNode:
            if G.in_degree(i) > 0:
                newIn.append(i)
            if  G.out_degree(i) > per:
                newOut.append(i)
            if G.degree(i)<per:
                D.append(i)

        random_valueA = random.random()
        random_valueB = random.random()
        random_valueC = random.random()
        random_valueD = random.random()

        if random_valueA < probabilityP:
            if newIn == []:
                node_i = random.choice(arrayNode)
            else:
                node_i = random.choice(newIn)
            G.add_node(step)
            G.add_edge(node_i, step)

        if random_valueB < probabilityQ:
            if newOut == []:
                node_i = random.choice(arrayNode)
            else:
                node_i = random.choice(newOut)
                G.add_node(step)
                G.add_edge(step, node_i)

        if random_valueC < probabilityR:
                node1 = random.choice(arrayNode)
                node2 = random.choice(arrayNode)
                G.add_edge(node1, node2)
        
        if D != []:
            if random_valueD < probabilityD:
                node_i = random.choice(D)
                G.remove_node(node_i)

    #nx.draw(G, with_labels=True, node_color = "blue", node_shape='o', alpha = 0.9, linewidths = 6)
    #plt.show()
    return G

def getLambda(degrees):
    exp = 0
    for d in degrees:
        exp+=math.log10(d/degrees[0])
    exp = 1+ len(degrees)/exp
    return exp

def plotDegree(exp, networkDegrees, networkDegreesList):
    lengthDL = len(networkDegreesList)
    lengthD = len(networkDegrees)
    
    probability = [networkDegrees.count(networkDegreesList[i])/float(lengthD) for i in range(lengthDL)]
    ylambd = [j**-exp for j in networkDegreesList]
    
    plt.scatter(networkDegreesList,probability,c='b',marker='x', label = "Users activity")
    plt.plot(networkDegreesList, ylambd, label='Lambda = -'+str(exp), color = "orange")
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Degree (log)')
    plt.ylabel('Probability (log)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
def saveNetwork(path, G):
    import csv
    f = open(path,'w')
    f.write('\n') # We write the blank line at the top
    writerN = csv.writer(f,delimiter=',')
    writerN.writerow(['Source','Target'])
    writerN.writerows(G.edges())
    f.close()
    
 if __name__=="__main__":
    probabilityP = 0.067
    probabilityQ = 0.042 #q<p
    probabilityD = 0.002
    time = 10000
    per = 0.1
    network = getModel(time, probabilityP, probabilityQ, per)
    
    networkDegrees = [network.degree(x) for x in network.nodes]
    networkDegrees.sort()
    networkDegreesList = list(set(networkDegrees))
    networkDegreesList.sort()
    
    lmd = getLambda(networkDegreesList)
    plotDegree(lmd, networkDegrees, networkDegreesList)
    
    #path = ""
    #saveNetwork(path, network)
