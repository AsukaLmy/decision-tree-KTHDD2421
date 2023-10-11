# author: Mengyu Liang

import monkdata as m
from dtree import *
import drawtree_qt5 as qt5
import random
import matplotlib.pyplot as pl
import numpy as np

# assignment 1: calculate the entropy
def calculate_entropy_assignment1():
    e1 = entropy(m.monk1)
    e2 = entropy(m.monk2)
    e3 = entropy(m.monk3)
    print("")
    print("entropys of monk1&2&3 are:", e1,e2,e3)
    print("")


# assignment 2: Explain entropy for a uniform distribution and a non-uniform distribution, present some example distributions with high and low entropy
# A: bulabulabula


# assignment 3: calculate the average gain
def calculate_average_gain_assignment3():
    # monk1
    print("average gain in monk1:")
    print("a1", averageGain(m.monk1, m.attributes[0]))
    print("a2", averageGain(m.monk1, m.attributes[1]))
    print("a3", averageGain(m.monk1, m.attributes[2]))
    print("a4", averageGain(m.monk1, m.attributes[3]))
    print("a5", averageGain(m.monk1, m.attributes[4]))
    print("a6", averageGain(m.monk1, m.attributes[5]))
    # monk2
    print("average gain in monk2:")
    print("a1", averageGain(m.monk2, m.attributes[0]))
    print("a2", averageGain(m.monk2, m.attributes[1]))
    print("a3", averageGain(m.monk2, m.attributes[2]))
    print("a4", averageGain(m.monk2, m.attributes[3]))
    print("a5", averageGain(m.monk2, m.attributes[4]))
    print("a6", averageGain(m.monk2, m.attributes[5]))
    # monk3
    print("average gain in monk3:")
    print("a1", averageGain(m.monk3, m.attributes[0]))
    print("a2", averageGain(m.monk3, m.attributes[1]))
    print("a3", averageGain(m.monk3, m.attributes[2]))
    print("a4", averageGain(m.monk3, m.attributes[3]))
    print("a5", averageGain(m.monk3, m.attributes[4]))
    print("a6", averageGain(m.monk3, m.attributes[5]))
    print("")
# Based on average gain, in monk1 gain of a5 significantly exceed others', which means a5 should be one of nodes.
# In monk2, no attribute exceed others much and they all have low gain. Maybe using single attribute as split is not a wise choice.
# In monk3, both a2 and a5 have gains that significantly exceed others'. Both of attribute 2&5 will be the node.

# assignment 4: bulabulabula.......

#################################################################################################################################################

# assignment 5: Build the full decision trees for all three Monk datasets using buildTree. 
# Then, use the function check to measure the performance of the decision tree on both the training and test datasets.

# # decide next node
# # monk1, first node is a5(which can be 1,2,3,4)
# # when a5 = 1 to 4 step 1
def calculate_next_node():
    for i in range(1,5,1):
        subset = select(m.monk1,m.attributes[4],i)  # get subset
        # information gain in next level
        print("average gain when a5 ==",i,":")
        print("a1", averageGain(subset, m.attributes[0]))
        print("a2", averageGain(subset, m.attributes[1]))
        print("a3", averageGain(subset, m.attributes[2]))
        print("a4", averageGain(subset, m.attributes[3]))
        print("a5", averageGain(subset, m.attributes[4]))
        print("a6", averageGain(subset, m.attributes[5]))
    print("")
# # next level node: none when a5==1; a4 when a5==2; a6 when a5==3; a1 when a5==4

# using builTtree
def calculate_error_assignment5():
    tree1 = buildTree(m.monk1,m.attributes)
    error_test1 = 1-check(tree1, m.monk1test)
    error1 = 1-check(tree1, m.monk1)

    tree2 = buildTree(m.monk2,m.attributes)
    error_test2 = 1-check(tree2, m.monk2test)
    error2 = 1-check(tree2, m.monk2)

    tree3 = buildTree(m.monk3,m.attributes)
    error_test3 = 1-check(tree3, m.monk3test)
    error3 = 1-check(tree3, m.monk3)

    print("monk1:")
    print("error_train ==",error1," ","error_test ==",error_test1)
    print("monk2:")
    print("error_train ==",error2," ","error_test ==",error_test2)
    print("monk3:")
    print("error_train ==",error3," ","error_test ==",error_test3)

#################################################################################################################################################

# Pruning
# assignment 7
def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]


def pruning_and_plot_assignment7():
    error_monk1 = [1,1,1,1,1,1]
    error_monk3 = [1,1,1,1,1,1]
    frac = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # monk1
    for i in range(0,6,1):
        monk1train, monk1val = partition(m.monk1, frac[i])
        newtree1 = buildTree(monk1train,m.attributes)
        alternativetree = allPruned(newtree1)
        minerror = 1
        ult_tree_id = 114514
        for j in range(0,len(alternativetree),1):
            temp_error = 1 - check(alternativetree[j],monk1val)
            if temp_error < minerror:
                minerror = temp_error
                ult_tree_id = j
        ult_tree = alternativetree[ult_tree_id]
        error_monk1[i] = 1-check(ult_tree,m.monk1test)
        
    print(error_monk1)    
    
    # monk3
    for i in range(0,6,1):
        monk3train, monk3val = partition(m.monk3, frac[i])
        newtree3 = buildTree(monk3train,m.attributes)
        alternativetree = allPruned(newtree3)
        minerror = 1
        ult_tree_id = 114514
        for j in range(0,len(alternativetree),1):
            temp_error = 1 - check(alternativetree[j],monk3val)
            if temp_error < minerror:
                minerror = temp_error
                ult_tree_id = j
        ult_tree = alternativetree[ult_tree_id]
        error_monk3[i] = 1-check(ult_tree,m.monk3test)
    print(error_monk3)  

    # plot
    pl.plot(frac, error_monk1, color='#000000', marker='o', label = "monk1")
    pl.plot(frac, error_monk3, color='#BB1216', marker='o', label = "monk3")
    pl.title("monk1&monk3")
    pl.xlabel("fractions")
    pl.ylabel("errors")
    pl.legend(loc='upper right', frameon=False)
    pl.show()

# repeat 100 times /mean and std 
def pruning_and_plot_assignment7_100times():
    error_monk1 = [0,0,0,0,0,0]
    error_monk3 = [0,0,0,0,0,0]
    temperr_monk1 = np.zeros((100,6))
    temperr_monk3 = np.zeros((100,6))
    std_monk1 = [0,0,0,0,0,0]
    std_monk3 = [0,0,0,0,0,0]

    frac = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # monk1
    for k in range(0,100,1):
        for i in range(0,6,1):
            monk1train, monk1val = partition(m.monk1, frac[i])
            newtree1 = buildTree(monk1train,m.attributes)
            alternativetree = allPruned(newtree1)
            minerror = 1
            ult_tree_id = 114514
            for j in range(0,len(alternativetree),1):
                temp_error = 1 - check(alternativetree[j],monk1val)
                if temp_error < minerror:
                    minerror = temp_error
                    ult_tree_id = j
            ult_tree = alternativetree[ult_tree_id]
            temperr_monk1[k,i] = 1 - check(ult_tree,m.monk1test)
            error_monk1[i] = error_monk1[i] + 1 - check(ult_tree,m.monk1test)
            
           
        
        # monk3
        for i in range(0,6,1):
            monk3train, monk3val = partition(m.monk3, frac[i])
            newtree3 = buildTree(monk3train,m.attributes)
            alternativetree = allPruned(newtree3)
            minerror = 1
            ult_tree_id = 114514
            for j in range(0,len(alternativetree),1):
                temp_error = 1 - check(alternativetree[j],monk3val)
                if temp_error < minerror:
                    minerror = temp_error
                    ult_tree_id = j
            ult_tree = alternativetree[ult_tree_id]
            temperr_monk3[k,i] = 1 - check(ult_tree,m.monk3test)
            error_monk3[i] = error_monk3[i] + 1 - check(ult_tree,m.monk3test)

    #error     
    for i in range(0,6,1):
        error_monk1[i] = error_monk1[i]/100
        error_monk3[i] = error_monk3[i]/100

    #std
    for i in range(0,6,1):
        totaldeviation_monk1 = 0
        totaldeviation_monk3 = 0
        for j in range(0,100,1):
            totaldeviation_monk1 = totaldeviation_monk1 + (temperr_monk1[j,i]-error_monk1[i])**2
            totaldeviation_monk3 = totaldeviation_monk3 + (temperr_monk3[j,i]-error_monk3[i])**2
        std_monk1[i] = totaldeviation_monk1/100
        std_monk3[i] = totaldeviation_monk3/100

    # plot
    pl.figure(1)
    pl.plot(frac, error_monk1, color='#000000', marker='o', label = "monk1")
    pl.plot(frac, error_monk3, color='#BB1216', marker='o', label = "monk3")
    pl.title("error of monk1&monk3")
    pl.xlabel("fractions")
    pl.ylabel("errors")
    pl.legend(loc='upper right', frameon=False)
  
    pl.figure(2)
    pl.plot(frac, std_monk1, color='#000000', marker='o', label = "monk1")
    pl.plot(frac, std_monk3, color='#BB1216', marker='o', label = "monk3")
    pl.title("std of monk1&monk3")
    pl.xlabel("fractions")
    pl.ylabel("std")
    pl.legend(loc='upper right', frameon=False)
    pl.show()

def main():
    calculate_entropy_assignment1()
    calculate_average_gain_assignment3()
    calculate_error_assignment5()
#   pruning_and_plot_assignment7()
    pruning_and_plot_assignment7_100times()


main()



