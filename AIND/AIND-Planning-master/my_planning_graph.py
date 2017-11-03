from aimacode.planning import Action
from aimacode.search import Problem
from aimacode.utils import expr
from lp_utils import decode_state


class PgNode():
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    ''' Base class for planning graph nodes.
=======
    """Base class for planning graph nodes.
>>>>>>> dc9e870... Base Code
=======
    ''' Base class for planning graph nodes.
>>>>>>> 8d1ef1b... Submission_01
=======
    ''' Base class for planning graph nodes.
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0

    includes instance sets common to both types of nodes used in a planning graph
    parents: the set of nodes in the previous level
    children: the set of nodes in the subsequent level
    mutex: the set of sibling nodes that are mutually exclusive with this node
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    '''
=======
    """
>>>>>>> dc9e870... Base Code
=======
    '''
>>>>>>> 8d1ef1b... Submission_01
=======
    '''
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0

    def __init__(self):
        self.parents = set()
        self.children = set()
        self.mutex = set()

    def is_mutex(self, other) -> bool:
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        ''' Boolean test for mutual exclusion
=======
        """Boolean test for mutual exclusion
>>>>>>> dc9e870... Base Code
=======
        ''' Boolean test for mutual exclusion
>>>>>>> 8d1ef1b... Submission_01
=======
        ''' Boolean test for mutual exclusion
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0

        :param other: PgNode
            the other node to compare with
        :return: bool
            True if this node and the other are marked mutually exclusive (mutex)
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        '''
=======
        """
>>>>>>> dc9e870... Base Code
=======
        '''
>>>>>>> 8d1ef1b... Submission_01
=======
        '''
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        if other in self.mutex:
            return True
        return False

    def show(self):
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        ''' helper print for debugging shows counts of parents, children, siblings

        :return:
            print only
        '''
<<<<<<< HEAD
=======
        """helper print for debugging shows counts of parents, children, siblings

        :return:
            print only
        """
>>>>>>> dc9e870... Base Code
=======
        ''' helper print for debugging shows counts of parents, children, siblings

        :return:
            print only
        '''
>>>>>>> 8d1ef1b... Submission_01
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        print("{} parents".format(len(self.parents)))
        print("{} children".format(len(self.children)))
        print("{} mutex".format(len(self.mutex)))


class PgNode_s(PgNode):
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    '''
    A planning graph node representing a state (literal fluent) from a planning
    problem.
=======
    """A planning graph node representing a state (literal fluent) from a
    planning problem.
>>>>>>> dc9e870... Base Code
=======
    '''
    A planning graph node representing a state (literal fluent) from a planning
    problem.
>>>>>>> 8d1ef1b... Submission_01
=======
    '''
    A planning graph node representing a state (literal fluent) from a planning
    problem.
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0

    Args:
    ----------
    symbol : str
        A string representing a literal expression from a planning problem
        domain.

    is_pos : bool
        Boolean flag indicating whether the literal expression is positive or
        negative.
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
    '''

    def __init__(self, symbol: str, is_pos: bool):
        ''' S-level Planning Graph node constructor
<<<<<<< HEAD
=======
    """

    def __init__(self, symbol: str, is_pos: bool):
        """S-level Planning Graph node constructor
>>>>>>> dc9e870... Base Code
=======
    '''

    def __init__(self, symbol: str, is_pos: bool):
        ''' S-level Planning Graph node constructor
>>>>>>> 8d1ef1b... Submission_01
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0

        :param symbol: expr
        :param is_pos: bool
        Instance variables calculated:
            literal: expr
                    fluent in its literal form including negative operator if applicable
        Instance variables inherited from PgNode:
            parents: set of nodes connected to this node in previous A level; initially empty
            children: set of nodes connected to this node in next A level; initially empty
            mutex: set of sibling S-nodes that this node has mutual exclusion with; initially empty
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        '''
        PgNode.__init__(self)
        self.symbol = symbol
        self.is_pos = is_pos
        self.literal = expr(self.symbol)
        if not self.is_pos:
            self.literal = expr('~{}'.format(self.symbol))

    def show(self):
        '''helper print for debugging shows literal plus counts of parents, children, siblings

        :return:
            print only
        '''
        print("\n*** {}".format(self.literal))
        PgNode.show(self)

    def __eq__(self, other):
        '''equality test for nodes - compares only the literal for equality

        :param other: PgNode_s
        :return: bool
        '''
        if isinstance(other, self.__class__):
            return (self.symbol == other.symbol) \
                   and (self.is_pos == other.is_pos)

    def __hash__(self):
        return hash(self.symbol) ^ hash(self.is_pos)


class PgNode_a(PgNode):
    '''A-type (action) Planning Graph node - inherited from PgNode
    '''

    def __init__(self, action: Action):
        '''A-level Planning Graph node constructor
<<<<<<< HEAD
=======
        """
=======
        '''
>>>>>>> 8d1ef1b... Submission_01
        PgNode.__init__(self)
        self.symbol = symbol
        self.is_pos = is_pos
        self.literal = expr(self.symbol)
        if not self.is_pos:
            self.literal = expr('~{}'.format(self.symbol))

    def show(self):
        '''helper print for debugging shows literal plus counts of parents, children, siblings

        :return:
            print only
        '''
        print("\n*** {}".format(self.literal))
        PgNode.show(self)

    def __eq__(self, other):
        '''equality test for nodes - compares only the literal for equality

        :param other: PgNode_s
        :return: bool
        '''
        if isinstance(other, self.__class__):
            return (self.symbol == other.symbol) \
                   and (self.is_pos == other.is_pos)

    def __hash__(self):
        return hash(self.symbol) ^ hash(self.is_pos)


class PgNode_a(PgNode):
    '''A-type (action) Planning Graph node - inherited from PgNode
    '''

    def __init__(self, action: Action):
<<<<<<< HEAD
        """A-level Planning Graph node constructor
>>>>>>> dc9e870... Base Code
=======
        '''A-level Planning Graph node constructor
>>>>>>> 8d1ef1b... Submission_01
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0

        :param action: Action
            a ground action, i.e. this action cannot contain any variables
        Instance variables calculated:
            An A-level will always have an S-level as its parent and an S-level as its child.
            The preconditions and effects will become the parents and children of the A-level node
            However, when this node is created, it is not yet connected to the graph
            prenodes: set of *possible* parent S-nodes
            effnodes: set of *possible* child S-nodes
            is_persistent: bool   True if this is a persistence action, i.e. a no-op action
        Instance variables inherited from PgNode:
            parents: set of nodes connected to this node in previous S level; initially empty
            children: set of nodes connected to this node in next S level; initially empty
            mutex: set of sibling A-nodes that this node has mutual exclusion with; initially empty
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
       '''
=======
        """
>>>>>>> dc9e870... Base Code
=======
       '''
>>>>>>> 8d1ef1b... Submission_01
=======
       '''
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        PgNode.__init__(self)
        self.action = action
        self.prenodes = self.precond_s_nodes()
        self.effnodes = self.effect_s_nodes()
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        self.is_persistent = False
        if self.prenodes == self.effnodes:
            self.is_persistent = True

    def show(self):
        '''helper print for debugging shows action plus counts of parents, children, siblings

        :return:
            print only
        '''
        print("\n*** {}{}".format(self.action.name, self.action.args))
        PgNode.show(self)

    def precond_s_nodes(self):
        '''precondition literals as S-nodes (represents possible parents for this node).
<<<<<<< HEAD
=======
        self.is_persistent = self.prenodes == self.effnodes
        self.__hash = None
=======
        self.is_persistent = False
        if self.prenodes == self.effnodes:
            self.is_persistent = True
>>>>>>> 8d1ef1b... Submission_01

    def show(self):
        '''helper print for debugging shows action plus counts of parents, children, siblings

        :return:
            print only
        '''
        print("\n*** {}{}".format(self.action.name, self.action.args))
        PgNode.show(self)

    def precond_s_nodes(self):
<<<<<<< HEAD
        """precondition literals as S-nodes (represents possible parents for this node).
>>>>>>> dc9e870... Base Code
=======
        '''precondition literals as S-nodes (represents possible parents for this node).
>>>>>>> 8d1ef1b... Submission_01
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        It is computationally expensive to call this function; it is only called by the
        class constructor to populate the `prenodes` attribute.

        :return: set of PgNode_s
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        '''
        nodes = set()
        for p in self.action.precond_pos:
            n = PgNode_s(p, True)
            nodes.add(n)
        for p in self.action.precond_neg:
            n = PgNode_s(p, False)
            nodes.add(n)
        return nodes

    def effect_s_nodes(self):
        '''effect literals as S-nodes (represents possible children for this node).
=======
        """
=======
        '''
>>>>>>> 8d1ef1b... Submission_01
=======
        '''
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        nodes = set()
        for p in self.action.precond_pos:
            n = PgNode_s(p, True)
            nodes.add(n)
        for p in self.action.precond_neg:
            n = PgNode_s(p, False)
            nodes.add(n)
        return nodes

    def effect_s_nodes(self):
<<<<<<< HEAD
<<<<<<< HEAD
        """effect literals as S-nodes (represents possible children for this node).
>>>>>>> dc9e870... Base Code
=======
        '''effect literals as S-nodes (represents possible children for this node).
>>>>>>> 8d1ef1b... Submission_01
=======
        '''effect literals as S-nodes (represents possible children for this node).
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        It is computationally expensive to call this function; it is only called by the
        class constructor to populate the `effnodes` attribute.

        :return: set of PgNode_s
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        '''
        nodes = set()
        for e in self.action.effect_add:
            n = PgNode_s(e, True)
            nodes.add(n)
        for e in self.action.effect_rem:
            n = PgNode_s(e, False)
            nodes.add(n)
        return nodes

    def __eq__(self, other):
        '''equality test for nodes - compares only the action name for equality

        :param other: PgNode_a
        :return: bool
        '''
        if isinstance(other, self.__class__):
            return (self.action.name == other.action.name) \
                   and (self.action.args == other.action.args)

    def __hash__(self):
        return hash(self.action.name) ^ hash(self.action.args)


def mutexify(node1: PgNode, node2: PgNode):
    ''' adds sibling nodes to each other's mutual exclusion (mutex) set. These should be sibling nodes!
=======
        """
=======
        '''
>>>>>>> 8d1ef1b... Submission_01
=======
        '''
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        nodes = set()
        for e in self.action.effect_add:
            n = PgNode_s(e, True)
            nodes.add(n)
        for e in self.action.effect_rem:
            n = PgNode_s(e, False)
            nodes.add(n)
        return nodes

    def __eq__(self, other):
        '''equality test for nodes - compares only the action name for equality

        :param other: PgNode_a
        :return: bool
        '''
        if isinstance(other, self.__class__):
            return (self.action.name == other.action.name) \
                   and (self.action.args == other.action.args)

    def __hash__(self):
        return hash(self.action.name) ^ hash(self.action.args)


def mutexify(node1: PgNode, node2: PgNode):
<<<<<<< HEAD
<<<<<<< HEAD
    """ adds sibling nodes to each other's mutual exclusion (mutex) set. These should be sibling nodes!
>>>>>>> dc9e870... Base Code
=======
    ''' adds sibling nodes to each other's mutual exclusion (mutex) set. These should be sibling nodes!
>>>>>>> 8d1ef1b... Submission_01
=======
    ''' adds sibling nodes to each other's mutual exclusion (mutex) set. These should be sibling nodes!
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0

    :param node1: PgNode (or inherited PgNode_a, PgNode_s types)
    :param node2: PgNode (or inherited PgNode_a, PgNode_s types)
    :return:
        node mutex sets modified
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    '''
=======
    """
>>>>>>> dc9e870... Base Code
=======
    '''
>>>>>>> 8d1ef1b... Submission_01
=======
    '''
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
    if type(node1) != type(node2):
        raise TypeError('Attempted to mutex two nodes of different types')
    node1.mutex.add(node2)
    node2.mutex.add(node1)


class PlanningGraph():
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
    '''
    A planning graph as described in chapter 10 of the AIMA text. The planning
    graph can be used to reason about 
    '''

    def __init__(self, problem: Problem, state: str, serial_planning=True):
        '''
<<<<<<< HEAD
=======
    """
=======
    '''
>>>>>>> 8d1ef1b... Submission_01
    A planning graph as described in chapter 10 of the AIMA text. The planning
    graph can be used to reason about 
    '''

    def __init__(self, problem: Problem, state: str, serial_planning=True):
<<<<<<< HEAD
        """
>>>>>>> dc9e870... Base Code
=======
        '''
>>>>>>> 8d1ef1b... Submission_01
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        :param problem: PlanningProblem (or subclass such as AirCargoProblem or HaveCakeProblem)
        :param state: str (will be in form TFTTFF... representing fluent states)
        :param serial_planning: bool (whether or not to assume that only one action can occur at a time)
        Instance variable calculated:
            fs: FluentState
                the state represented as positive and negative fluent literal lists
            all_actions: list of the PlanningProblem valid ground actions combined with calculated no-op actions
            s_levels: list of sets of PgNode_s, where each set in the list represents an S-level in the planning graph
            a_levels: list of sets of PgNode_a, where each set in the list represents an A-level in the planning graph
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        '''
        #print("problem.initials = " , problem.initial)
=======
        """
>>>>>>> dc9e870... Base Code
=======
        '''
        #print("problem.initials = " , problem.initial)
>>>>>>> 8d1ef1b... Submission_01
=======
        '''
        #print("problem.initials = " , problem.initial)
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        self.problem = problem
        self.fs = decode_state(state, problem.state_map)
        self.serial = serial_planning
        self.all_actions = self.problem.actions_list + self.noop_actions(self.problem.state_map)
        self.s_levels = []
        self.a_levels = []
        self.create_graph()

    def noop_actions(self, literal_list):
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        '''create persistent action for each possible fluent
=======
        """create persistent action for each possible fluent
>>>>>>> dc9e870... Base Code
=======
        '''create persistent action for each possible fluent
>>>>>>> 8d1ef1b... Submission_01
=======
        '''create persistent action for each possible fluent
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0

        "No-Op" actions are virtual actions (i.e., actions that only exist in
        the planning graph, not in the planning problem domain) that operate
        on each fluent (literal expression) from the problem domain. No op
        actions "pass through" the literal expressions from one level of the
        planning graph to the next.

        The no-op action list requires both a positive and a negative action
        for each literal expression. Positive no-op actions require the literal
        as a positive precondition and add the literal expression as an effect
        in the output, and negative no-op actions require the literal as a
        negative precondition and remove the literal expression as an effect in
        the output.

        This function should only be called by the class constructor.

        :param literal_list:
        :return: list of Action
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        '''
=======
        """
>>>>>>> dc9e870... Base Code
=======
        '''
>>>>>>> 8d1ef1b... Submission_01
=======
        '''
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        action_list = []
        for fluent in literal_list:
            act1 = Action(expr("Noop_pos({})".format(fluent)), ([fluent], []), ([fluent], []))
            action_list.append(act1)
            act2 = Action(expr("Noop_neg({})".format(fluent)), ([], [fluent]), ([], [fluent]))
            action_list.append(act2)
        return action_list

    def create_graph(self):
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        ''' build a Planning Graph as described in Russell-Norvig 3rd Ed 10.3 or 2nd Ed 11.4
=======
        """ build a Planning Graph as described in Russell-Norvig 3rd Ed 10.3 or 2nd Ed 11.4
>>>>>>> dc9e870... Base Code
=======
        ''' build a Planning Graph as described in Russell-Norvig 3rd Ed 10.3 or 2nd Ed 11.4
>>>>>>> 8d1ef1b... Submission_01
=======
        ''' build a Planning Graph as described in Russell-Norvig 3rd Ed 10.3 or 2nd Ed 11.4
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0

        The S0 initial level has been implemented for you.  It has no parents and includes all of
        the literal fluents that are part of the initial state passed to the constructor.  At the start
        of a problem planning search, this will be the same as the initial state of the problem.  However,
        the planning graph can be built from any state in the Planning Problem

        This function should only be called by the class constructor.

        :return:
            builds the graph by filling s_levels[] and a_levels[] lists with node sets for each level
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        '''
=======
        """
>>>>>>> dc9e870... Base Code
=======
        '''
>>>>>>> 8d1ef1b... Submission_01
=======
        '''
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        # the graph should only be built during class construction
        if (len(self.s_levels) != 0) or (len(self.a_levels) != 0):
            raise Exception(
                'Planning Graph already created; construct a new planning graph for each new state in the planning sequence')

        # initialize S0 to literals in initial state provided.
        leveled = False
        level = 0
        self.s_levels.append(set())  # S0 set of s_nodes - empty to start
        # for each fluent in the initial state, add the correct literal PgNode_s
        for literal in self.fs.pos:
            self.s_levels[level].add(PgNode_s(literal, True))
        for literal in self.fs.neg:
            self.s_levels[level].add(PgNode_s(literal, False))
        # no mutexes at the first level

        # continue to build the graph alternating A, S levels until last two S levels contain the same literals,
        # i.e. until it is "leveled"
        while not leveled:
            self.add_action_level(level)
            self.update_a_mutex(self.a_levels[level])

            level += 1
            self.add_literal_level(level)
            self.update_s_mutex(self.s_levels[level])

            if self.s_levels[level] == self.s_levels[level - 1]:
                leveled = True
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    def add_action_level(self, level):
        ''' add an A (action) level to the Planning Graph
=======

    def add_action_level(self, level):
        """ add an A (action) level to the Planning Graph
>>>>>>> dc9e870... Base Code
=======
    def add_action_level(self, level):
        ''' add an A (action) level to the Planning Graph
>>>>>>> 8d1ef1b... Submission_01
=======
    def add_action_level(self, level):
        ''' add an A (action) level to the Planning Graph
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0

        :param level: int
            the level number alternates S0, A0, S1, A1, S2, .... etc the level number is also used as the
            index for the node set lists self.a_levels[] and self.s_levels[]
        :return:
            adds A nodes to the current level in self.a_levels[level]
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        '''
        # Add action A level to the planning graph as described in the Russell-Norvig text
=======
        """
        # TODO add action A level to the planning graph as described in the Russell-Norvig text
>>>>>>> dc9e870... Base Code
=======
        '''
        # Add action A level to the planning graph as described in the Russell-Norvig text
>>>>>>> 8d1ef1b... Submission_01
=======
        '''
        # Add action A level to the planning graph as described in the Russell-Norvig text
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        # 1. determine what actions to add and create those PgNode_a objects
        # 2. connect the nodes to the previous S literal level
        # for example, the A0 level will iterate through all possible actions for the problem and add a PgNode_a to a_levels[0]
        #   set iff all prerequisite literals for the action hold in S0.  This can be accomplished by testing
        #   to see if a proposed PgNode_a has prenodes that are a subset of the previous S level.  Once an
        #   action node is added, it MUST be connected to the S node instances in the appropriate s_level set.

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 8d1ef1b... Submission_01
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        a_nodes = []
        for action in self.all_actions:
            node_a = PgNode_a(action)
            if node_a.prenodes.issubset(self.s_levels[level]):
                a_nodes.append(node_a)
                for node_s in self.s_levels[level]:
                    node_s.children.add(node_a)
                    node_a.parents.add(node_s)
        # Build the action level from the reachable action nodes
        self.a_levels.append(a_nodes)
 
<<<<<<< HEAD
<<<<<<< HEAD
    def add_literal_level(self, level):
        ''' add an S (literal) level to the Planning Graph
=======
    def add_literal_level(self, level):
        """ add an S (literal) level to the Planning Graph
>>>>>>> dc9e870... Base Code
=======
    def add_literal_level(self, level):
        ''' add an S (literal) level to the Planning Graph
>>>>>>> 8d1ef1b... Submission_01
=======
    def add_literal_level(self, level):
        ''' add an S (literal) level to the Planning Graph
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0

        :param level: int
            the level number alternates S0, A0, S1, A1, S2, .... etc the level number is also used as the
            index for the node set lists self.a_levels[] and self.s_levels[]
        :return:
            adds S nodes to the current level in self.s_levels[level]
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        '''
        #  add literal S level to the planning graph as described in the Russell-Norvig text
=======
        """
        # TODO add literal S level to the planning graph as described in the Russell-Norvig text
>>>>>>> dc9e870... Base Code
=======
        '''
        #  add literal S level to the planning graph as described in the Russell-Norvig text
>>>>>>> 8d1ef1b... Submission_01
=======
        '''
        #  add literal S level to the planning graph as described in the Russell-Norvig text
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        # 1. determine what literals to add
        # 2. connect the nodes
        # for example, every A node in the previous level has a list of S nodes in effnodes that represent the effect
        #   produced by the action.  These literals will all be part of the new S level.  Since we are working with sets, they
        #   may be "added" to the set without fear of duplication.  However, it is important to then correctly create and connect
        #   all of the new S nodes as children of all the A nodes that could produce them, and likewise add the A nodes to the
        #   parent sets of the S nodes
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 8d1ef1b... Submission_01
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        actionNodes = self.a_levels[level - 1]
        
        nodeSS = set()
        for pgNodeA in actionNodes:
            nodesS = pgNodeA.effect_s_nodes()
            #self.s_levels.append(nodesS)
            for node in nodesS:
                nodeSS.add(node)
                pgNodeA.children.add(node)
                node.parents.add(pgNodeA) 
        self.s_levels.append(nodeSS)
    
<<<<<<< HEAD
<<<<<<< HEAD

    def update_a_mutex(self, nodeset):
        ''' Determine and update sibling mutual exclusion for A-level nodes
=======

    def update_a_mutex(self, nodeset):
        """ Determine and update sibling mutual exclusion for A-level nodes
>>>>>>> dc9e870... Base Code
=======

    def update_a_mutex(self, nodeset):
        ''' Determine and update sibling mutual exclusion for A-level nodes
>>>>>>> 8d1ef1b... Submission_01
=======

    def update_a_mutex(self, nodeset):
        ''' Determine and update sibling mutual exclusion for A-level nodes
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0

        Mutex action tests section from 3rd Ed. 10.3 or 2nd Ed. 11.4
        A mutex relation holds between two actions a given level
        if the planning graph is a serial planning graph and the pair are nonpersistence actions
        or if any of the three conditions hold between the pair:
           Inconsistent Effects
           Interference
           Competing needs

        :param nodeset: set of PgNode_a (siblings in the same level)
        :return:
            mutex set in each PgNode_a in the set is appropriately updated
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        '''
=======
        """
>>>>>>> dc9e870... Base Code
=======
        '''
>>>>>>> 8d1ef1b... Submission_01
=======
        '''
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        nodelist = list(nodeset)
        for i, n1 in enumerate(nodelist[:-1]):
            for n2 in nodelist[i + 1:]:
                if (self.serialize_actions(n1, n2) or
                        self.inconsistent_effects_mutex(n1, n2) or
                        self.interference_mutex(n1, n2) or
                        self.competing_needs_mutex(n1, n2)):
                    mutexify(n1, n2)

    def serialize_actions(self, node_a1: PgNode_a, node_a2: PgNode_a) -> bool:
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        '''
=======
        """
>>>>>>> dc9e870... Base Code
=======
        '''
>>>>>>> 8d1ef1b... Submission_01
=======
        '''
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        Test a pair of actions for mutual exclusion, returning True if the
        planning graph is serial, and if either action is persistent; otherwise
        return False.  Two serial actions are mutually exclusive if they are
        both non-persistent.

        :param node_a1: PgNode_a
        :param node_a2: PgNode_a
        :return: bool
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        '''
=======
        """
>>>>>>> dc9e870... Base Code
=======
        '''
>>>>>>> 8d1ef1b... Submission_01
=======
        '''
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        #
        if not self.serial:
            return False
        if node_a1.is_persistent or node_a2.is_persistent:
            return False
        return True

    def inconsistent_effects_mutex(self, node_a1: PgNode_a, node_a2: PgNode_a) -> bool:
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        '''
=======
        """
>>>>>>> dc9e870... Base Code
=======
        '''
>>>>>>> 8d1ef1b... Submission_01
=======
        '''
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        Test a pair of actions for inconsistent effects, returning True if
        one action negates an effect of the other, and False otherwise.

        HINT: The Action instance associated with an action node is accessible
        through the PgNode_a.action attribute. See the Action class
        documentation for details on accessing the effects and preconditions of
        an action.

        :param node_a1: PgNode_a
        :param node_a2: PgNode_a
        :return: bool
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        '''
        # TODO test for Inconsistent Effects between nodes

        if (node_a1.action.effect_add == node_a2.action.effect_rem) or ( node_a1.action.effect_rem == node_a2.action.effect_add):
            return True
         
        return False    
                                   
        

    def interference_mutex(self, node_a1: PgNode_a, node_a2: PgNode_a) -> bool:
        '''
<<<<<<< HEAD
=======
        """
=======
        '''
>>>>>>> 8d1ef1b... Submission_01
        # TODO test for Inconsistent Effects between nodes

        if (node_a1.action.effect_add == node_a2.action.effect_rem) or ( node_a1.action.effect_rem == node_a2.action.effect_add):
            return True
         
        return False    
                                   
        

    def interference_mutex(self, node_a1: PgNode_a, node_a2: PgNode_a) -> bool:
<<<<<<< HEAD
        """
>>>>>>> dc9e870... Base Code
=======
        '''
>>>>>>> 8d1ef1b... Submission_01
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        Test a pair of actions for mutual exclusion, returning True if the 
        effect of one action is the negation of a precondition of the other.

        HINT: The Action instance associated with an action node is accessible
        through the PgNode_a.action attribute. See the Action class
        documentation for details on accessing the effects and preconditions of
        an action.

        :param node_a1: PgNode_a
        :param node_a2: PgNode_a
        :return: bool
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        '''
        # TODO test for Interference between nodes

        prenodes1 = node_a1.prenodes
        effnodes1 = node_a1.effnodes
        
        prenodes2 = node_a1.prenodes
        effnodes2 = node_a2.effnodes
        
        for nodeS1 in prenodes1:
            for nodeS2 in effnodes2:
                    if (nodeS1.symbol == nodeS2.symbol) and (nodeS1.is_pos != nodeS2.is_pos):
                        return True      

        for nodeS1 in effnodes1:
            for nodeS2 in prenodes2:
               if (nodeS1.symbol == nodeS2.symbol) and (nodeS1.is_pos != nodeS2.is_pos):
                    return True
                      
        
        return False

    def competing_needs_mutex(self, node_a1: PgNode_a, node_a2: PgNode_a) -> bool:
        '''
<<<<<<< HEAD
=======
        """
=======
        '''
>>>>>>> 8d1ef1b... Submission_01
        # TODO test for Interference between nodes

        prenodes1 = node_a1.prenodes
        effnodes1 = node_a1.effnodes
        
        prenodes2 = node_a1.prenodes
        effnodes2 = node_a2.effnodes
        
        for nodeS1 in prenodes1:
            for nodeS2 in effnodes2:
                    if (nodeS1.symbol == nodeS2.symbol) and (nodeS1.is_pos != nodeS2.is_pos):
                        return True      

        for nodeS1 in effnodes1:
            for nodeS2 in prenodes2:
               if (nodeS1.symbol == nodeS2.symbol) and (nodeS1.is_pos != nodeS2.is_pos):
                    return True
                      
        
        return False

    def competing_needs_mutex(self, node_a1: PgNode_a, node_a2: PgNode_a) -> bool:
<<<<<<< HEAD
        """
>>>>>>> dc9e870... Base Code
=======
        '''
>>>>>>> 8d1ef1b... Submission_01
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        Test a pair of actions for mutual exclusion, returning True if one of
        the precondition of one action is mutex with a precondition of the
        other action.

        :param node_a1: PgNode_a
        :param node_a2: PgNode_a
        :return: bool
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        '''

        # TODO test for Competing Needs between nodes
        #prenodes1 = node_a1.prenodes
        #prenodes2 = node_a2.prenodes
        
        prenodes1 = node_a1.parents
        prenodes2 = node_a2.parents

        for nodeS1 in prenodes1:
            for nodeS2 in prenodes2:
               #if (nodeS1.symbol == nodeS2.symbol) and (nodeS1.is_pos != nodeS2.is_pos):
                   if nodeS1.is_mutex(nodeS2):
                       return True
        


        return False

    def update_s_mutex(self, nodeset: set):
        ''' Determine and update sibling mutual exclusion for S-level nodes
<<<<<<< HEAD
=======
        """
=======
        '''
>>>>>>> 8d1ef1b... Submission_01

        # TODO test for Competing Needs between nodes
        #prenodes1 = node_a1.prenodes
        #prenodes2 = node_a2.prenodes
        
        prenodes1 = node_a1.parents
        prenodes2 = node_a2.parents

        for nodeS1 in prenodes1:
            for nodeS2 in prenodes2:
               #if (nodeS1.symbol == nodeS2.symbol) and (nodeS1.is_pos != nodeS2.is_pos):
                   if nodeS1.is_mutex(nodeS2):
                       return True
        


        return False

    def update_s_mutex(self, nodeset: set):
<<<<<<< HEAD
        """ Determine and update sibling mutual exclusion for S-level nodes
>>>>>>> dc9e870... Base Code
=======
        ''' Determine and update sibling mutual exclusion for S-level nodes
>>>>>>> 8d1ef1b... Submission_01
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0

        Mutex action tests section from 3rd Ed. 10.3 or 2nd Ed. 11.4
        A mutex relation holds between literals at a given level
        if either of the two conditions hold between the pair:
           Negation
           Inconsistent support

        :param nodeset: set of PgNode_a (siblings in the same level)
        :return:
            mutex set in each PgNode_a in the set is appropriately updated
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        '''
=======
        """
>>>>>>> dc9e870... Base Code
=======
        '''
>>>>>>> 8d1ef1b... Submission_01
=======
        '''
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        nodelist = list(nodeset)
        for i, n1 in enumerate(nodelist[:-1]):
            for n2 in nodelist[i + 1:]:
                if self.negation_mutex(n1, n2) or self.inconsistent_support_mutex(n1, n2):
                    mutexify(n1, n2)

    def negation_mutex(self, node_s1: PgNode_s, node_s2: PgNode_s) -> bool:
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        '''
=======
        """
>>>>>>> dc9e870... Base Code
=======
        '''
>>>>>>> 8d1ef1b... Submission_01
=======
        '''
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        Test a pair of state literals for mutual exclusion, returning True if
        one node is the negation of the other, and False otherwise.

        HINT: Look at the PgNode_s.__eq__ defines the notion of equivalence for
        literal expression nodes, and the class tracks whether the literal is
        positive or negative.

        :param node_s1: PgNode_s
        :param node_s2: PgNode_s
        :return: bool
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        '''
        # TODO test for negation between nodes

        if node_s1.is_pos != node_s2.is_pos:
            return True
            
        return False

    def inconsistent_support_mutex(self, node_s1: PgNode_s, node_s2: PgNode_s):
        '''
=======
        """
=======
        '''
>>>>>>> 8d1ef1b... Submission_01
=======
        '''
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        # TODO test for negation between nodes

        if node_s1.is_pos != node_s2.is_pos:
            return True
            
        return False

    def inconsistent_support_mutex(self, node_s1: PgNode_s, node_s2: PgNode_s):
<<<<<<< HEAD
<<<<<<< HEAD
        """
>>>>>>> dc9e870... Base Code
=======
        '''
>>>>>>> 8d1ef1b... Submission_01
=======
        '''
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        Test a pair of state literals for mutual exclusion, returning True if
        there are no actions that could achieve the two literals at the same
        time, and False otherwise.  In other words, the two literal nodes are
        mutex if all of the actions that could achieve the first literal node
        are pairwise mutually exclusive with all of the actions that could
        achieve the second literal node.

        HINT: The PgNode.is_mutex method can be used to test whether two nodes
        are mutually exclusive.

        :param node_s1: PgNode_s
        :param node_s2: PgNode_s
        :return: bool
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        '''
        # TODO test for Inconsistent Support between nodes
        parentsS1 = node_s1.parents
        parentsS2 = node_s2.parents
        
        for act1 in parentsS1:
            for act2 in parentsS2:
                if not act1.is_mutex(act2):
                   return False 
        
   
        return True
    
    def h_levelsum(self) -> int:
        '''The sum of the level costs of the individual goals (admissible if goals independent)

        :return: int
        '''
        level_sum = 0
        # TODO implement
        # for each goal in the problem, determine the level cost, then add them together
        levels = len(self.s_levels)
        #print ( "levels = " , levels)

        goal_found_list=[]

        #print(" my_planning_graph # of goals = " , len(self.problem.goal))
        #print("levels = " , levels)
        
        for goal in self.problem.goal:
            if goal not in goal_found_list:
                for i in range(levels):
                    #print( " i = " , i)
                    for nodeS in self.s_levels[i]:
                        if goal == nodeS.literal:
                            #print(" goal - " , goal, " found. i = ", i)
                            level_sum += i
                            goal_found_list.append(goal)
                            break
                    if goal in goal_found_list:
                        break
            
        #print ( "level_sum = " , level_sum)
=======
        """
=======
        '''
>>>>>>> 8d1ef1b... Submission_01
=======
        '''
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        # TODO test for Inconsistent Support between nodes
        parentsS1 = node_s1.parents
        parentsS2 = node_s2.parents
        
        for act1 in parentsS1:
            for act2 in parentsS2:
                if not act1.is_mutex(act2):
                   return False 
        
   
        return True
    
    def h_levelsum(self) -> int:
        '''The sum of the level costs of the individual goals (admissible if goals independent)

        :return: int
        '''
        level_sum = 0
        # TODO implement
        # for each goal in the problem, determine the level cost, then add them together
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> dc9e870... Base Code
=======
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        levels = len(self.s_levels)
        #print ( "levels = " , levels)

        goal_found_list=[]

        #print(" my_planning_graph # of goals = " , len(self.problem.goal))
        #print("levels = " , levels)
        
        for goal in self.problem.goal:
            if goal not in goal_found_list:
                for i in range(levels):
                    #print( " i = " , i)
                    for nodeS in self.s_levels[i]:
                        if goal == nodeS.literal:
                            #print(" goal - " , goal, " found. i = ", i)
                            level_sum += i
                            goal_found_list.append(goal)
                            break
                    if goal in goal_found_list:
                        break
            
        #print ( "level_sum = " , level_sum)
<<<<<<< HEAD
>>>>>>> 8d1ef1b... Submission_01
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        return level_sum
