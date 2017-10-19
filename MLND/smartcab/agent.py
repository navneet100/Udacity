import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha 
        
        # Learning factor

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        self.gamma = 0.5
        self.trial_local = 0
        
        self.N = 100.0
        
        self.epsilon_differential = epsilon/self.N
        self.alpha_differential = alpha/self.N
        
        #self.enforce_deadline = True #navneet
        #self.learning = True
        #self.env.enforce_deadline = True      
        
        #self.next_waypoint = self.planner.next_waypoint()


    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        
        ########### 
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        
        # new code added start        
        if testing:
            self.epsilon = 0
            self.alpha = 0
        else:
            a = 0.0001
            self.trial_local = self.trial_local + 1
            print "trial# " , self.trial_local
            self.epsilon = self.epsilon  * math.exp(-1 * a * self.trial_local)
            self.alpha = self.alpha  * math.exp(-1 * a * self.trial_local)
            
            #self.Q = dict()
            
            #self.alpha = 0.5

            #self.epsilon = self.epsilon  * math.exp(-self.alpha*self.trial_local)
            
            #self.epsilon = self.epsilon - 1.0 * self.epsilon_differential 
            #self.alpha = self.alpha - self.alpha_differential 
            '''
            #self.alpha = self.alpha  * math.exp(1 * 20 *a )
            #self.alpha = self.alpha - self.trial_local * 0.0005
            #self.alpha = self.epsilon - 0.0005
            #self.epsilon =  math.exp(-1 * a * self.env.t)
            #self.epsilon = 0.9 * self.epsilon
            '''
        # new code added end        

        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ########### 
        ## TO DO ##
        ###########
        # Set 'state' as a tuple of relevant data for the agent
        # When learning, check if the state is in the Q-table
        #   If it is not, create a dictionary in the Q-table for the current 'state'
        #   For each action, set the Q-value for the state-action pair to 0
        
        #state = None
        #state = (waypoint,inputs['light'],inputs['left'],inputs['right'],inputs['oncoming'], deadline)
        #state = (waypoint,inputs['light'],inputs['left'],inputs['right'],inputs['oncoming'])
        
        
        #state = (waypoint,inputs['light'],inputs['left'],inputs['right'])
        #state = (waypoint,inputs['light'],inputs['left'], deadline)
           
        #state = (str({'waypoint':waypoint}),str({'light':inputs['light']}),str({'left':inputs['left']}))
        #state = (str({'waypoint':waypoint}),str({'light':inputs['light']}),str({'left':inputs['left']}),str({'right':inputs['right']}),str({'deadline':deadline}))
        #state = (str({'waypoint':waypoint}),str({'light':inputs['light']}),str({'left':inputs['left']}),str({'right':inputs['right']}))
        
        #state = (str({'waypoint':waypoint}),str({'inputs':{'light':inputs['light'],'left':inputs['left'],'oncoming':inputs['oncoming']}}), str({'deadline':deadline}))
        #state = (str({'waypoint':waypoint}),str({'inputs':{'light':inputs['light'],'left':inputs['left']}}), str({'deadline':deadline}))
        
        #A+,A+
        #
        state = (str({'waypoint':waypoint}),str({'inputs':{'light':inputs['light'],'left':inputs['left'],'oncoming':inputs['oncoming']}}))
        #state = (str({'waypoint':waypoint}),str({'inputs':{'light':inputs['light'],'left':inputs['left']}}))
      
        

        
        

        
        if self.learning:
               self.createQ(state)
        return state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state

        maxQ = None
        print "action count: - " ,len(self.Q[state])
        lst = []
        for action, reward in self.Q[state].iteritems():
           print "action, reward - ", action, reward
           if maxQ is None:
               maxQ = (action, reward)
           else:
               if  reward > maxQ[1] :
                   maxQ = [action, reward]  
        print "Max action, reward 1 - ", maxQ
        
        for action, reward in self.Q[state].iteritems():
            if reward == maxQ[1]:
                lst.append(action)
        act = random.choice(lst)
        maxQ = [act,self.Q[state][act]]
        print "Max action, reward 2 - ", maxQ
        return maxQ 


    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0
        
        if self.learning:
            #print self.Q.keys()
            if not state in self.Q.keys():
                #print "################################################"
                #print "state is not in Q table"
                #print "################################################"
                self.Q[state] = {}
                for action in self.env.valid_actions:                 
                    #self.Q[state][action] = 0
                    #self.Q.update({state:{action:10.0}})
                    
                    self.Q[state][action] = 0.0
                    #print "action=", action, " self.Q[state][action] = " , self.Q[state][action ]
            #print "self.env.valid_actions - " , len(self.env.valid_actions)
            #print "self.Q[state] - " , self.Q[state], "len(self.Q[state]) = " , len(self.Q[state])
            
            #for action, reward_1 in self.Q[state].iteritems():
            #     print action, reward_1
        return


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        #action = None

        ########### 
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        #   Otherwise, choose an action with the highest Q-value for the current state
 
        if self.learning:
            if self.epsilon < random.random():
                action = self.get_maxQ(state)[0]
            else:
                action = random.choice(self.valid_actions)
        else:
            #action = random.choice(Environment.valid_actions[1:])
            action = random.choice(self.valid_actions)
        return action


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives an award. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        print " state = ", state
        #print " self.Q.keys() - " , self.Q.keys()
        if self.learning:            
            '''
            if state not in self.Q.keys():                
                for action in self.env.valid_actions:                 
                    #self.Q[state][action] = 0
                    self.Q.update({state:{action:0.0}})            
            '''
            #for action, reward_1 in self.Q[state].iteritems():
                #print "before reward - " , self.Q[state][action], " reward = " , reward_1, "input reward = ", reward    
                #reward_1 =  reward_1 + self.alpha * ( reward + self.gamma * self.get_maxQ(state)[1] - reward_1)               
            self.Q[state][action] = self.Q[state][action] + self.alpha * ( reward + self.gamma * self.get_maxQ(state)[1] - self.Q[state][action])
                #print "after reward - " , reward_1
            #self.Q[self.state][action] = (1-self.alpha) * self.Q[self.state][action] + self.alpha*reward
            #print " Q Value = ", self.Q[self.state][action], len(self.Q)
            #print " Q len = ",len(self.Q)
            
            #check with following function
            #self.Q[self.state][action] = (1-self.alpha)self.Q[self.state][action] + self.alpha * (reward + gamma*self.Q[next_state][maxQ])
                   
        return


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return
        

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment(verbose = True)
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent, learning = True, alpha = 0.8,epsilon = 0.9)
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent, enforce_deadline = True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env, update_delay = 0.0001, log_metrics = True, optimized = True)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    #sim.run(tolerance = 0.05,n_test = 20) #A+, A+ with  a = 0.00001
    #sim.run(tolerance = 0.01,n_test = 20)  #A+, A with  a = 0.0001 with a removing one zero, need lower tolerance to reach high rating
    sim.run(tolerance = 0.01,n_test = 10)

if __name__ == '__main__':
    run()
