from __future__ import print_function
from IPython.display import Image

'''
Question 1
Given two strings s and t, determine whether some anagram of t is a substring of s. 
For example: if s = "udacity" and t = "ad", then the function returns True. Your function 
definition should look like: question1(s, t) and return a boolean True or False.
'''

def question1(s, t):
    
    if (type(s) != str) or (type(t) != str):
        return "NotStrError"
    
    if (len(s) < 1) or (len(t) < 1) :
        return "EmptyStrError"
    
    
    t_char_counts = {}
    for char in t:
        if char in t_char_counts:
            t_char_counts[char] += 1
        else:
            t_char_counts[char] = 1
    
    
    lenT = len(t)
    lenS = len(s)
    
    if lenS > lenT:
        rnge = lenS - lenT + 1
    elif lenS == lenT:
        rnge = 1
    else:
        return False    
    
    for i in range(rnge):
        s_subString = s[i:i+lenT]
        
        s_char_counts = {}
        for char in s_subString:
            if char in s_char_counts:
                s_char_counts[char] += 1
            else:
                s_char_counts[char] = 1
                
        stringMatched = True
        for key in t_char_counts:
            if key in s_char_counts.keys():
                if t_char_counts[key] != s_char_counts[key]:
                    stringMatched = False
            else:
                stringMatched = False        
    
        if stringMatched:
            return True
    return False
    
    
def test1():
    print ("Testing Question 1")
    print ( "Test case: check for value:", "Fail-Input value is blank" if "EmptyStrError" == question1("","") else "Pass")
    print ( "Test case: check for string:", "Fail-Input is not string" if "NotStrError" == question1(123, 1.23) else "Pass")
    print ( "Test case: udacity - (udacity, ad):", "Pass" if True == question1("udacity", "ad") else "Fail")
    print ( "Test case: search string is longer - (ad, udacity):", "Pass" if True == question1("ad","udacity") else "Fail-Input search string is longer")
    print ( "Test case: Same length strings (s equal to t):", "Pass" if True == question1("abcd", "abcd") else "Fail")
    print ( "Test case: Same length strings (s equal to t):", "Pass" if True == question1("silent", "listen") else "Fail")
    print ( "Test case: Same length strings (s equal to t):", "Pass" if True == question1("stackoverflow", "rove") else "Fail")
    print ( "Test case: non consecutive substring):", "Pass" if False == question1("stackoverflow", "slow") else "Fail")
    print ()
    print("----------------------------------------")

'''
Question 2
Given a string a, find the longest palindromic substring contained in a. Your function definition should 
look like question2(a), and return a string.
'''

def question2(a):    
    
    if (type(a) != str) :
        return "NotStrError"
    
    if (len(a) < 1) :
        return "EmptyStrError"
    
    if(len(a) == 1):
        return a
    
    lenA = len(a)
    plndrms = []
    maxPlnDrm = None
    maxLen = -1
    for i in range(lenA):    
        for k in range(1,lenA+1):
            if (i + k) <= lenA:
                st = a[i:i+k]
                if len(st) == 1:
                    plndrms.append(st)
                else:
                    lenSub = len(st)
                    if lenSub % 2 != 0:
                        plnDrm = True
                        midPt = int(lenSub / 2) 
                        for j in range(midPt + 1):
                            if st[midPt - j] != st[midPt + j]:
                                plnDrm = False
                        if plnDrm:
                            plndrms.append(st)
                    else:
                        plnDrm = True
                        midPt = int(lenSub / 2)
                        for j in range(midPt):
                            if st[j] != st[lenSub - (j + 1)]:
                                plnDrm = False
                        if plnDrm:
                            plndrms.append(st)
                        
                        for st in plndrms:
                            if len(st) > maxLen:
                                maxLen = len(st)
                                maxPlnDrm = st                            
    
    return maxPlnDrm


def test2():
    print ( "Testing Question 2")
    print ( "Test case: check for value:", "Fail-Input value is blank" if "EmptyStrError" == question1("","") else "Pass")
    print ( "Test case: check for string:", "Fail-Input is not string" if "NotStrError" == question1(123, 1.23) else "Pass")
    
    print( "Test case: Individual characters only - for test - t or e or s or t: ", "Pass" if ("t" or "e" or "s" or "t") == question2("test") else "Fail" )
    print( "Test case: whole string - for teset - teset: ", "Pass" if "teset" == question2("teset") else "Fail" )
    print( "Test case: whole string - only one character: ", "Pass" if "a" == question2("a") else "Fail" )
    print( "Test case: 2 character string - same characters: ", "Pass" if "aa" == question2("aa") else "Fail" )
    print( "Test case: 2 character string - different characters:", "Pass" if ("a" or "b") == question2("ab") else "Fail" )
    print( "Test case: whole string - even number of characters string :", "Pass" if "abba" == question2("abba") else "Fail" )
    print( "Test case: whole string - odd number of characters string :", "Pass" if "ababa" == question2("ababa") else "Fail" )
    
    print( "Test case: long string - forgeeksskeegfor :", "Pass" if "geeksskeeg" == question2("forgeeksskeegfor") else "Fail" )
    print("----------------------------------------")

'''
Question 3
Given an undirected graph G, find the minimum spanning tree within G. A minimum spanning tree connects all vertices in a graph with the smallest possible total weight of edges. Your function should take in and return an adjacency list structured like this:

{'A': [('B', 2)],
 'B': [('A', 2), ('C', 5)], 
 'C': [('B', 5)]}
Vertices are represented as unique strings. The function definition should be question3(G)

'''

def question3(G):
    edges=[]
    uniqEdges=[]
    
    vertices = G.keys()
    totVertices = len(vertices)
    
    if totVertices == 0:
        return "empDictError"
    
    vertices = [set(i) for i in vertices]
    
    for key in G.keys():
        for eg in G[key]:
            edge = (eg[1], key, eg[0])
            edgeRev = (eg[1], eg[0], key)
            if edgeRev not in edges:
                uniqEdges.append(edge)
    
    uniqEdges = sorted(uniqEdges)
    
    finalEdges = set()    

    for i in range(len(uniqEdges)):
        wt = uniqEdges[i][0]
        
        vrt1 = uniqEdges[i][1]
        vrt2 = uniqEdges[i][2]
        
        for j in range(len(vertices)):
            if vrt1 in vertices[j]:
                vrt_1_set_index = j
            if vrt2 in vertices[j]:
                vrt_2_set_index = j
        
        if vrt_1_set_index < vrt_2_set_index:
            vertices[vrt_1_set_index] = set.union(vertices[vrt_1_set_index], vertices[vrt_2_set_index])
            vertices.pop(vrt_2_set_index)
            finalEdges.add(uniqEdges[i])
        if vrt_1_set_index > vrt_2_set_index:
            vertices[vrt_2_set_index] = set.union(vertices[vrt_1_set_index], vertices[vrt_2_set_index])
            vertices.pop(vrt_1_set_index)
            finalEdges.add(uniqEdges[i])  

        finalGraph = {}        
     
        for edge in finalEdges:            
            if edge[1]  in finalGraph:
                finalGraph[edge[1]].append((edge[2],edge[0]))
            else:
                finalGraph[edge[1]] = [(edge[2],edge[0])]
                
            if edge[2]  in finalGraph:
                finalGraph[edge[2]].append((edge[1],edge[0]))
            else:
                finalGraph[edge[2]] = [(edge[1],edge[0])]                
            
        
    return(finalGraph) 

def test3_sub(expected_output_G,output_G ):
    result = "Pass"
    if len(expected_output_G.keys()) != len(output_G.keys()):
        result = "Fail"
    else:
        for key in expected_output_G.keys():
            if key in output_G.keys():
                expected_values = expected_output_G[key]
                actual_values = output_G[key]
                if len(expected_values) != len(actual_values):
                    result = "Fail"
                else:
                    for val in expected_output_G[key]:
                        if val not in output_G[key]:
                            result = "Fail"               
                
            else:
                result = "Fail" 
    return result

def test3():
    
    print()
    
    print( "Testing Question 3")
    
    input_G = {'A': [('B', 2)],
         'B': [('A', 2), ('C', 5)],
         'C': [('B', 5)]}
    output_G = question3(input_G)
    
    expected_output_G = input_G
    
   
       
    result = test3_sub(expected_output_G,output_G )
    print("Test Case - udacity example : ", result ) 
    print()
    print("Example from website : http://www.geeksforgeeks.org/greedy-algorithms-set-2-kruskals-minimum-spanning-tree-mst")
    
    input_G = {'0': [('1', 4), ('7', 8)],
    '1': [('0', 4), ('2', 8),('7', 11)],
     '2': [('1', 8), ('3', 7), ('5', 4), ('8', 2)],
     '3': [('2', 7), ('4', 9), ('5',14)],
     '4': [('3', 9), ('5', 10)],
     '5': [('2',4),('3', 14), ('4', 10), ('6', 2)],
     '6': [('5', 2), ('7', 1), ('8', 6)],
     '7': [('0', 8), ('1', 11),('6',1),('8',7)],
     '8': [('2', 2), ('6', 6),('7',7)]
    } 
    
    expected_output_G = {'2': [('5', 4), ('3', 7), ('8', 2)], '4': [('3', 9)], '1': [('0', 4)], '3': [('4', 9), ('2', 7)], '5': [('2', 4), ('6', 2)], '8': [('2', 2)], '0': [('7', 8), ('1', 4)], '7': [('6', 1), ('0', 8)], '6': [('7', 1), ('5', 2)]}
    output_G = question3(input_G)
    
   
    result = test3_sub(expected_output_G,output_G )

    print("Test Case - geeksforgeeks example : ", result )     
    
    
    print()   
    input_G = {}    
    output_G = question3(input_G)

    print("Test Case - Edge Case - Empty Dictionary : ", "Fail" if output_G == "empDictError" else "Pass" )
    print("----------------------------------------")
    
    
'''
Question 4

Find the least common ancestor between two nodes on a binary search tree. The least common ancestor
 is the farthest node from the root that is an ancestor of both nodes. For example, the root is a 
 common ancestor of all nodes on the tree, but if both nodes are descendents of the root's left 
 child, then that left child might be the lowest common ancestor. You can assume that both nodes 
 are in the tree, and the tree itself adheres to all BST properties. The function definition should 
 look like question4(T, r, n1, n2), where T is the tree represented as a matrix, where the index of
 the list is equal to the integer stored in that node and a 1 represents a child node, r is a
 non-negative integer representing the root, and n1 and n2 are non-negative integers representing
 the two nodes in no particular order. For example, one test case might be

question4([[0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 0]], 3, 1, 4) 
and the answer would be 3.

'''

def question4(T, r, n1, n2):
    
    parentFound = False
    newParent = r
    
    while (not parentFound) & (newParent is not None):
        prod = (n1 - r) * (n2 - r)
        
        if prod < 0:
            parentFound = True
            return r
        else:
            parentFound = False
            prevParent = r
            
            chd = T[r]
            
            newParent = None
            
            for i in range(len(chd)):
                if chd[i] == 1:
                    if n1 > r:
                        if i > r:
                            newParent = i
                            break
                    else:
                        if i < r:
                            newParent = i
                            break
            r = newParent     
           
    return prevParent  

'''
Test Data: Tree used in testing below for non-udacity examples
                               4
        2                                          10       
   1          3                         7                      14
                                   6         8           12           15


'''
def test4():
    
    print()
    print ( "Testing Question 4")
    
    #print("Test case: Udacity example")
    
   
    T = [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 0]]
    print("Test case: Udacity example : ", "Pass" if 3 == question4(T, 3, 1, 4)  else "Fail")
    
    T = [        
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
         [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
         [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0], 
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1], 
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]            
        ]
   
    print("Test case: question4(T, 4, 3, 7) - ", "Pass" if 4 == question4(T, 4, 3, 7)  else "Fail")
    print("Test case: question4(T, 4, 6, 8) - ", "Pass" if 7 == question4(T, 4, 6, 8)  else "Fail")
    print("Test case: question4(T, 4, 1, 3) - ", "Pass" if 2 == question4(T, 4, 1, 3)   else "Fail")
    print("Test case: question4(T, 4, 12, 15) - ", "Pass" if 14 == question4(T, 4, 12, 15)  else "Fail")
    print("Test case: question4(T, 4, 12, 8) - ", "Pass" if 10 == question4(T, 4, 12, 8)  else "Fail")
    print("----------------------------------------")
    
'''
Question 5
Find the element in a singly linked list that's m elements from the end. For example, if a linked list 
has 5 elements, the 3rd element from the end is the 3rd element. The function definition should look 
like question5(ll, m), where ll is the first node of a linked list and m is the "mth number from the 
end". You should copy/paste the Node class below to use as a representation of a node in the linked 
list. Return the value of the node at that position.

class Node(object):
  def __init__(self, data):
    self.data = data
    self.next = None

'''    
class Node(object):
    def __init__(self, value):
        self.value = value
        self.next = None
        
class LinkedList(object):
    def __init__(self, head=None):
        self.head = head
        
    def append(self, new_element):
        current = self.head
        if self.head:
            while current.next:
                current = current.next
            current.next = new_element
        else:
            self.head = new_element
            
    def get_position(self, position):
        """Get an element from a particular position.
        Assume the first position is "1".
        Return "None" if position is not in the list."""
        posValue = None
        curPos = 0
        if self.head:
            current = self.head
            curPos = 1
            if curPos == position:
                return current
            else:            
                while current.next:
                    current = current.next
                    curPos += 1
                    if curPos == position:
                        return current
        else:
            return None

        return posValue
    def insert(self, new_element, position):
        curPos = 0
        if self.head:
            current = self.head
            curPos = 1

            if curPos == position:
                new_element.next = current            
                current = new_element
                self.head = new_element
            else:            
                while current.next:
                    curPos += 1
                    prev = current
                    current = current.next
                    if curPos == position:
                        new_element.next = current
                        prev.next = new_element
                        current = new_element
                        return

                        
    
    def delete(self, value):
        """Delete the first node with a given value."""
        
        if self.head:
            current = self.head
            while current:
                if current.value == value:
                    if current == self.head:
                        self.head = current.next
                    else:
                        prev.next = current.next
                    return
                else:
                    prev = current
                    current = current.next
                    
    def question5(self, ll, m):
        ll_size = 0
        lstValues = []
        retVal = None
        if ll:
            ll_size += 1
            lstValues.append(ll.value)
            while ll.next:
                ll_size += 1
                ll = ll.next
                lstValues.append(ll.value)
            if (m > ll_size) or (m == 0):
                #print(m, " from end, value = None" )
                return "outRange"
            else:
                retVal = lstValues[ll_size - m]
                #print(m, " from end, value = ",retVal )
            return retVal
        else:
            return retVal

def test5():
    n1 = Node(1)
   
    ll = LinkedList(n1)
    
    n2 = Node(2)
    n3 = Node(3)
    n4 = Node(4)
    n5 = Node(5)
    n6 = Node(6)
    n7 = Node(7)
    n8 = Node(8)
    n9 = Node(9)    
    
    ll.append(n2)
    ll.append(n3)
    ll.append(n4)
    ll.append(n5)
    
    print()
    print ( "Testing Question 5")
    print ( "Test case: LinkedList 1->2->3->4->5 : 0 from end :", "Fail-Outside the range" if "outRange" == ll.question5(n1, 0) else "Pass" )
    print ( "Test case: LinkedList 1->2->3->4->5 : 6 from end :", "Fail-Outside the range" if "outRange" == ll.question5(n1, 6) else "Pass" )
    print ( "Test case: LinkedList 1->2->3->4->5 : 1 from end :", "Pass" if 5 == ll.question5(n1, 1) else "Fail" )
    print ( "Test case: LinkedList 1->2->3->4->5 : 2 from end :", "Pass" if 4 == ll.question5(n1, 2) else "Fail" )
    print ( "Test case: LinkedList 1->2->3->4->5 : 3 from end :", "Pass" if 3 == ll.question5(n1, 3) else "Fail" )
    print ( "Test case: LinkedList 1->2->3->4->5 : 4 from end :", "Pass" if 2 == ll.question5(n1, 4) else "Fail" )
    print ( "Test case: LinkedList 1->2->3->4->5 : 5 from end :", "Pass" if 1 == ll.question5(n1, 5) else "Fail" )
    
    ll.delete(4)
    print ( "Test case After deleting 4: LinkedList 1->2->3->5 : 2 from end :", "Pass" if 3 == ll.question5(n1, 2) else "Fail" )

    ll.insert(n4,4)
    print ( "Test case After inserting 4 at 4: LinkedList 1->2->3->4->5 : 2 from end :", "Pass" if 4 == ll.question5(n1, 2) else "Fail" )
    print("----------------------------------------")
    
test1()

test2()   

test3() 

test4()

test5()

Image(filename='Fig-0-300x139.jpg')