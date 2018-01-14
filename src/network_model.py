import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

probabilityP = 0.1 #A
probabilityQ = 0.01 #q<p B
probabilityR = 1.0-probabilityP-probabilityQ #C
coeffL = 0.01
coeffV = 0.02
time = 1000

G = nx.DiGraph()
G.add_node(0)

""""
At each time step,
with probability p, a new node is created attaching to a directed node,
with probability q, a new node is created attached by a directed link 
with probability r, a directed link is created between the old nodes.
"""
for step in range(time):

    arrayNode = list(G.nodes)
    numOfNode = len(arrayNode)
    allIO = 0 #all incoming/outgoing links in the network

    newIn = []
    newOut = []
    for i in range(numOfNode):
        allIO+=G.in_degree(arrayNode[i])
        if G.in_degree(arrayNode[i]) > 0:
            newIn.append(arrayNode[i])
        if  G.out_degree(arrayNode[i]) > 1:
            newOut.append(arrayNode[i])

    random_valueA = random.random()
    random_valueB = random.random()
    random_valueC = random.random()

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
        if newOut+newIn == []:
            node1 = random.choice(arrayNode)
            node2 = random.choice(arrayNode)
            G.add_edge(node1, node2)
        else:
            node1 = random.choice(newIn+newOut)
            node2 = random.choice(newIn+newOut)
            G.add_edge(node1, node2)


nx.draw(G, with_labels=True, node_color = "blue", node_shape='o', alpha = 0.9, linewidths = 6)
plt.show()

lambdaIn = (1+coeffL*(probabilityR+probabilityQ))/(1.0-probabilityQ)+1
lambdaOut = (1+coeffV*(probabilityR+probabilityQ))/(1-probabilityP)+1
print "LambdaIn", lambdaIn
print "LambdaOut", lambdaOut
powerLaw = 0
min = 1
for i in G.nodes():
    powerLaw+= np.log(G.degree(i)/float(min))
print "Power law",1+numOfNode*(1.0/powerLaw)

ba_c = nx.degree_centrality(G)
# To convert normalized degrees to raw degrees
ba_c = {k:int(v*(len(G)-1)) for k,v in ba_c.iteritems()}
plt.xscale('log')
plt.yscale('log')
plt.scatter(ba_c.keys(),ba_c.values(),c='b',marker='x')
plt.xlabel('Connections')
plt.ylabel('Frequency')
plt.show()