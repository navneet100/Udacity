from aimacode.logic import PropKB
from aimacode.planning import Action
from aimacode.search import (
    Node, Problem,
)
from aimacode.utils import expr
from lp_utils import (
    FluentState, encode_state, decode_state,
)
from my_planning_graph import PlanningGraph

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
from functools import lru_cache

>>>>>>> dc9e870... Base Code
=======
>>>>>>> 8d1ef1b... Submission_01
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0

class AirCargoProblem(Problem):
    def __init__(self, cargos, planes, airports, initial: FluentState, goal: list):
        """

        :param cargos: list of str
            cargos in the problem
        :param planes: list of str
            planes in the problem
        :param airports: list of str
            airports in the problem
        :param initial: FluentState object
            positive and negative literal fluents (as expr) describing initial state
        :param goal: list of expr
            literal fluents required for goal test
        """
        self.state_map = initial.pos + initial.neg
        self.initial_state_TF = encode_state(initial, self.state_map)
        Problem.__init__(self, self.initial_state_TF, goal=goal)
        self.cargos = cargos
        self.planes = planes
        self.airports = airports
        self.actions_list = self.get_actions()

    def get_actions(self):
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
        This method creates concrete actions (no variables) for all actions in the problem
        domain action schema and turns them into complete Action objects as defined in the
        aimacode.planning module. It is computationally expensive to call this method directly;
        however, it is called in the constructor and the results cached in the `actions_list` property.

        Returns:
        ----------
        list<Action>
            list of Action objects
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

        # TODO create concrete Action objects based on the domain action schema for: Load, Unload, and Fly
        # concrete actions definition: specific literal action that does not include variables as with the schema
        # for example, the action schema 'Load(c, p, a)' can represent the concrete actions 'Load(C1, P1, SFO)'
        # or 'Load(C2, P2, JFK)'.  The actions for the planning problem must be concrete because the problems in
        # forward search and Planning Graphs must use Propositional Logic

        def load_actions():
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
            '''Create all concrete Load actions and return a list

            :return: list of Action objects
            '''
            loads = []
            # TODO create all load ground actions from the domain Load action
            
            for a in self.airports:
                for p in self.planes:
                    for c in self.cargos:
                        precond_pos = [expr("At({}, {})".format(c, a)),expr("At({}, {})".format(p, a))]
                        precond_neg = [expr("In({}, {})".format(c, p))]
                        effect_add = [expr("In({}, {})".format(c, p))]
                        effect_rem = [expr("At({}, {})".format(c, a))]
                        load = Action(expr("Load({}, {}, {})".format(c, p, a)),
                                     [precond_pos, precond_neg],
                                     [effect_add, effect_rem])
                        loads.append(load)            
            return loads

        def unload_actions():
            '''Create all concrete Unload actions and return a list

            :return: list of Action objects
            '''
            unloads = []
            # TODO create all Unload ground actions from the domain Unload action
            #for p in self.planes:
            for a in self.airports:
                for p in self.planes:
                    for c in self.cargos:
                        precond_pos = [expr("In({}, {})".format(c, p)),expr("At({}, {})".format(p, a)),
                                       ]
                        precond_neg = []
                        effect_add = [expr("At({}, {})".format(c, a)),]
                        effect_rem = [expr("In({}, {})".format(c, p))]
                        unload = Action(expr("Unload({}, {}, {})".format(c, p, a)),
                                     [precond_pos, precond_neg],
                                     [effect_add, effect_rem])
                        unloads.append(unload)            
            return unloads

        def fly_actions():
            '''Create all concrete Fly actions and return a list

            :return: list of Action objects
            '''
=======
            """Create all concrete Load actions and return a list
=======
            '''Create all concrete Load actions and return a list
>>>>>>> 8d1ef1b... Submission_01
=======
            '''Create all concrete Load actions and return a list
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0

            :return: list of Action objects
            '''
            loads = []
            # TODO create all load ground actions from the domain Load action
            
            for a in self.airports:
                for p in self.planes:
                    for c in self.cargos:
                        precond_pos = [expr("At({}, {})".format(c, a)),expr("At({}, {})".format(p, a))]
                        precond_neg = [expr("In({}, {})".format(c, p))]
                        effect_add = [expr("In({}, {})".format(c, p))]
                        effect_rem = [expr("At({}, {})".format(c, a))]
                        load = Action(expr("Load({}, {}, {})".format(c, p, a)),
                                     [precond_pos, precond_neg],
                                     [effect_add, effect_rem])
                        loads.append(load)            
            return loads

        def unload_actions():
            '''Create all concrete Unload actions and return a list

            :return: list of Action objects
            '''
            unloads = []
            # TODO create all Unload ground actions from the domain Unload action
            #for p in self.planes:
            for a in self.airports:
                for p in self.planes:
                    for c in self.cargos:
                        precond_pos = [expr("In({}, {})".format(c, p)),expr("At({}, {})".format(p, a)),
                                       ]
                        precond_neg = []
                        effect_add = [expr("At({}, {})".format(c, a)),]
                        effect_rem = [expr("In({}, {})".format(c, p))]
                        unload = Action(expr("Unload({}, {}, {})".format(c, p, a)),
                                     [precond_pos, precond_neg],
                                     [effect_add, effect_rem])
                        unloads.append(unload)            
            return unloads

        def fly_actions():
            '''Create all concrete Fly actions and return a list

            :return: list of Action objects
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
            flys = []
            for fr in self.airports:
                for to in self.airports:
                    if fr != to:
                        for p in self.planes:
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
                        #for c in self.cargos:                               
                            precond_pos = [expr("At({}, {})".format(p, fr)),
                                           ]
                            precond_neg = [expr("At({}, {})".format(p, to))]
<<<<<<< HEAD
=======
                            precond_pos = [expr("At({}, {})".format(p, fr)),
                                           ]
                            precond_neg = []
>>>>>>> dc9e870... Base Code
=======
                        #for c in self.cargos:                               
                            precond_pos = [expr("At({}, {})".format(p, fr)),
                                           ]
                            precond_neg = [expr("At({}, {})".format(p, to))]
>>>>>>> 8d1ef1b... Submission_01
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
                            effect_add = [expr("At({}, {})".format(p, to))]
                            effect_rem = [expr("At({}, {})".format(p, fr))]
                            fly = Action(expr("Fly({}, {}, {})".format(p, fr, to)),
                                         [precond_pos, precond_neg],
                                         [effect_add, effect_rem])
                            flys.append(fly)
            return flys

        return load_actions() + unload_actions() + fly_actions()

    def actions(self, state: str) -> list:
        """ Return the actions that can be executed in the given state.

        :param state: str
            state represented as T/F string of mapped fluents (state variables)
            e.g. 'FTTTFF'
        :return: list of Action objects
        """
        # TODO implement
        possible_actions = []
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 8d1ef1b... Submission_01
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        
        
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        
        #print(" kb.clauses = " , kb.clauses)
        
        for action in self.actions_list:
            is_possible = True
            for clause in action.precond_pos:
                #print("precond_pos clause = ", clause)
                if clause not in kb.clauses:
                    is_possible = False
            for clause in action.precond_neg:
                #print("precond_neg clause = ", clause)
                if clause in kb.clauses:
                    is_possible = False
            if is_possible:
                #print("possible action.name = ", action.name, " action.args = " , action.args)
                possible_actions.append(action)
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0

        
        return possible_actions        
        
        
<<<<<<< HEAD
=======
        return possible_actions

>>>>>>> dc9e870... Base Code
=======

        
        return possible_actions        
        
        
>>>>>>> 8d1ef1b... Submission_01
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
    def result(self, state: str, action: Action):
        """ Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).

        :param state: state entering node
        :param action: Action applied
        :return: resulting state after action
        """
        # TODO implement
        new_state = FluentState([], [])
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 8d1ef1b... Submission_01
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        
        old_state = decode_state(state, self.state_map)
        
        for fluent in old_state.pos:
            if fluent not in action.effect_rem:
                new_state.pos.append(fluent)
        for fluent in action.effect_add:
            if fluent not in new_state.pos:
                new_state.pos.append(fluent)
        for fluent in old_state.neg:
            if fluent not in action.effect_add:
                new_state.neg.append(fluent)
        for fluent in action.effect_rem:
            if fluent not in new_state.neg:
                new_state.neg.append(fluent)        
        
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> dc9e870... Base Code
=======
>>>>>>> 8d1ef1b... Submission_01
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        return encode_state(new_state, self.state_map)

    def goal_test(self, state: str) -> bool:
        """ Test the state to see if goal is reached

        :param state: str representing state
        :return: bool
        """
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        for clause in self.goal:
            if clause not in kb.clauses:
                return False
        return True

    def h_1(self, node: Node):
        # note that this is not a true heuristic
        h_const = 1
        return h_const

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    def h_pg_levelsum(self, node: Node):
        '''
        This heuristic uses a planning graph representation of the problem
        state space to estimate the sum of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
        '''
=======
    @lru_cache(maxsize=8192)
=======
>>>>>>> 8d1ef1b... Submission_01
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
    def h_pg_levelsum(self, node: Node):
        '''
        This heuristic uses a planning graph representation of the problem
        state space to estimate the sum of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
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
        # requires implemented PlanningGraph class
        pg = PlanningGraph(self, node.state)
        pg_levelsum = pg.h_levelsum()
        return pg_levelsum

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
    def h_ignore_preconditions(self, node: Node):
        '''
        This heuristic estimates the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions by ignoring the preconditions required for an action to be
        executed.
        '''
        # TODO implement (see Russell-Norvig Ed-3 10.2.3  or Russell-Norvig Ed-2 11.2)
        count = 0
        #print("node.literal = " , node.literal)
        nodeState = decode_state(node.state, self.state_map)
        fsPos = nodeState.pos
        #print("node.state = " , nodeState.pos)
        #print("self.goal = " , self.goal)
        
        goalsPresent = 0
        for goal in self.goal:
             if goal in fsPos:
                 goalsPresent += 1
        
        count = len(self.goal) - goalsPresent
        
<<<<<<< HEAD
=======
    @lru_cache(maxsize=8192)
=======
>>>>>>> 8d1ef1b... Submission_01
    def h_ignore_preconditions(self, node: Node):
        '''
        This heuristic estimates the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions by ignoring the preconditions required for an action to be
        executed.
        '''
        # TODO implement (see Russell-Norvig Ed-3 10.2.3  or Russell-Norvig Ed-2 11.2)
        count = 0
<<<<<<< HEAD
>>>>>>> dc9e870... Base Code
=======
        #print("node.literal = " , node.literal)
        nodeState = decode_state(node.state, self.state_map)
        fsPos = nodeState.pos
        #print("node.state = " , nodeState.pos)
        #print("self.goal = " , self.goal)
        
        goalsPresent = 0
        for goal in self.goal:
             if goal in fsPos:
                 goalsPresent += 1
        
        count = len(self.goal) - goalsPresent
        
>>>>>>> 8d1ef1b... Submission_01
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
        return count


def air_cargo_p1() -> AirCargoProblem:
    cargos = ['C1', 'C2']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]
    neg = [expr('At(C2, SFO)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('At(C1, JFK)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('At(P1, JFK)'),
           expr('At(P2, SFO)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p2() -> AirCargoProblem:
    # TODO implement Problem 2 definition
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 8d1ef1b... Submission_01
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
    
    cargos = ['C1', 'C2','C3']
    planes = ['P1', 'P2', 'P3']
    airports = ['JFK', 'SFO','ATL']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(C3, ATL)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           expr('At(P3, ATL)'),
           ]
    neg = [
           expr('At(C1, JFK)'),
           expr('At(C1, ATL)'),

           expr('At(C2, SFO)'),
           expr('At(C2, ATL)'),
<<<<<<< HEAD
<<<<<<< HEAD

           expr('At(C3, JFK)'),
           expr('At(C3, SFO)'),

           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('In(C1, P3)'),

           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('In(C2, P3)'),

           expr('In(C3, P1)'),
           expr('In(C3, P2)'),
           expr('In(C3, P3)'),


           expr('At(P1, JFK)'),
           expr('At(P1, ATL)'),

           expr('At(P2, SFO)'),
           expr('At(P2, ATL)'),

           expr('At(P3, JFK)'),
           expr('At(P3, SFO)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C3, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)
    

def air_cargo_p3() -> AirCargoProblem:
    # TODO implement Problem 3 definition
    
    cargos = ['C1', 'C2','C3','C4']
    planes = ['P1', 'P2', 'P3', 'P4']
    airports = ['JFK', 'SFO','ATL', 'ORD']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(C3, ATL)'),
           expr('At(C4, ORD)'),

           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),

           ]
    neg = [
           expr('At(C1, JFK)'),
           expr('At(C1, ATL)'),
           expr('At(C1, ORD)'),

           expr('At(C2, SFO)'),
           expr('At(C2, ATL)'),
           expr('At(C2, ORD)'),

           expr('At(C3, JFK)'),
           expr('At(C3, SFO)'),
           expr('At(C3, ORD)'),

           expr('At(C4, JFK)'),
           expr('At(C4, SFO)'),
           expr('At(C4, ATL)'),

           expr('In(C1, P1)'),
           expr('In(C1, P2)'),


           expr('In(C2, P1)'),
           expr('In(C2, P2)'),


           expr('In(C3, P1)'),
           expr('In(C3, P2)'),


           expr('In(C4, P1)'),
           expr('In(C4, P2)'),


           expr('At(P1, JFK)'),
           expr('At(P1, ATL)'),
           expr('At(P1, ORD)'),

           expr('At(P2, SFO)'),
           expr('At(P2, ATL)'),
           expr('At(P2, ORD)'),



           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C3, JFK)'),
            expr('At(C4, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)
    
=======
    pass
=======
>>>>>>> 8d1ef1b... Submission_01
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0

           expr('At(C3, JFK)'),
           expr('At(C3, SFO)'),

           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('In(C1, P3)'),

           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('In(C2, P3)'),

           expr('In(C3, P1)'),
           expr('In(C3, P2)'),
           expr('In(C3, P3)'),


           expr('At(P1, JFK)'),
           expr('At(P1, ATL)'),

           expr('At(P2, SFO)'),
           expr('At(P2, ATL)'),

           expr('At(P3, JFK)'),
           expr('At(P3, SFO)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C3, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)
    

def air_cargo_p3() -> AirCargoProblem:
    # TODO implement Problem 3 definition
<<<<<<< HEAD
<<<<<<< HEAD
    pass
>>>>>>> dc9e870... Base Code
=======
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
    
    cargos = ['C1', 'C2','C3','C4']
    planes = ['P1', 'P2', 'P3', 'P4']
    airports = ['JFK', 'SFO','ATL', 'ORD']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(C3, ATL)'),
           expr('At(C4, ORD)'),

           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),

           ]
    neg = [
           expr('At(C1, JFK)'),
           expr('At(C1, ATL)'),
           expr('At(C1, ORD)'),

           expr('At(C2, SFO)'),
           expr('At(C2, ATL)'),
           expr('At(C2, ORD)'),

           expr('At(C3, JFK)'),
           expr('At(C3, SFO)'),
           expr('At(C3, ORD)'),

           expr('At(C4, JFK)'),
           expr('At(C4, SFO)'),
           expr('At(C4, ATL)'),

           expr('In(C1, P1)'),
           expr('In(C1, P2)'),


           expr('In(C2, P1)'),
           expr('In(C2, P2)'),


           expr('In(C3, P1)'),
           expr('In(C3, P2)'),


           expr('In(C4, P1)'),
           expr('In(C4, P2)'),


           expr('At(P1, JFK)'),
           expr('At(P1, ATL)'),
           expr('At(P1, ORD)'),

           expr('At(P2, SFO)'),
           expr('At(P2, ATL)'),
           expr('At(P2, ORD)'),



           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C3, JFK)'),
            expr('At(C4, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)
    
<<<<<<< HEAD
>>>>>>> 8d1ef1b... Submission_01
=======
>>>>>>> a7568d9f54a58f4956b458fc6a3732a6565cdda0
