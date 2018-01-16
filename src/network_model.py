import networkx as nx
import math
import matplotlib.pyplot as plt
import random

probabilityP = 0.06
probabilityQ = 0.05 #q<p
probabilityR = 1.0-probabilityP-probabilityQ
time = 10000

G = nx.DiGraph()
G.add_node(0)

"""
At each time step
with probability p, a new node is created attaching to a directed node,
with probability q, a new node is created attached by a directed link 
with probability r, a directed link is created between the old nodes.
"""
for step in range(time):

    arrayNode = list(G.nodes)
    numOfNode = len(arrayNode)
    arrayNodeDegree = []
    allIO = 0 #all incoming/outgoing links in the network

    newIn = []
    newOut = []
    for i in arrayNode:
        allIO+=G.in_degree(i)
        if G.in_degree(i) > 0:
            newIn.append(i)
        if  G.out_degree(i) > 3:
            newOut.append(i)
        arrayNodeDegree.append(G.degree(i))

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


#nx.draw(G, with_labels=True, node_color = "blue", node_shape='o', alpha = 0.9, linewidths = 6)
#plt.show()

ba_c = [G.degree(x) for x in G.nodes]
ba_c.sort()
y = list(set(ba_c))
y.sort()

exp = 0
for d in y:
    exp+=math.log10(d/y[0])
exp = 1+ len(y)/exp
print ("Lambda", exp)

plt.xscale('log')
plt.yscale('log')
x = y
yA = [ba_c.count(y[i])/float(len(ba_c)) for i in range(len(y))]
plt.scatter(x,yA,c='b',marker='x', label = "Users activity")
ylambd = [i**-exp for i in y]
plt.plot(y, ylambd, label='Lambda = -'+str(exp), color = "orange")
plt.xlabel('Probability (log)')
plt.ylabel('Users count (log)')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
