from cmath import sqrt
import numpy as np
import helper
import random

#   This class has all the functions and variables necessary to implement snake game
#   We will be using Q learning to do this

class SnakeAgent:

    #   This is the constructor for the SnakeAgent class
    #   It initializes the actions that can be made,
    #   Ne which is a parameter helpful to perform exploration before deciding next action,
    #   LPC which ia parameter helpful in calculating learning rate (lr) 
    #   gamma which is another parameter helpful in calculating next move, in other words  
    #            gamma is used to blalance immediate and future reward
    #   Q is the q-table used in Q-learning
    #   N is the next state used to explore possible moves and decide the best one before updating
    #           the q-table
    def __init__(self, actions, Ne, LPC, gamma):
        self.actions = actions
        self.Ne = Ne
        self.LPC = LPC
        self.gamma = gamma
        self.reset()

        # Create the Q and N Table to work with
        self.Q = helper.initialize_q_as_zeros()
        self.N = helper.initialize_q_as_zeros()


    #   This function sets if the program is in training mode or testing mode.
    def set_train(self):
        self._train = True

     #   This function sets if the program is in training mode or testing mode.       
    def set_eval(self):
        self._train = False

    #   Calls the helper function to save the q-table after training
    def save_model(self):
        helper.save(self.Q)

    #   Calls the helper function to load the q-table when testing
    def load_model(self):
        self.Q = helper.load()

    #   resets the game state
    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    #   This is a function you should write. 
    #   Function Helper:IT gets the current state, and based on the 
    #   current snake head location, body and food location,
    #   determines which move(s) it can make by also using the 
    #   board variables to see if its near a wall or if  the
    #   moves it can make lead it into the snake body and so on. 
    #   This can return a list of variables that help you keep track of
    #   conditions mentioned above.
    def helper_func(self, state):
        #print("IN helper_func")
        q_list_indexes = [0, 0, 0, 0, 0, 0, 0, 0]
        q_list_indexes[0] = 1 if state[0] == 40 else 2 if state[0] == 480 else 0
        q_list_indexes[1] = 1 if state[1] == 40 else 2 if state[1] == 480 else 0
        q_list_indexes[2] = 1 if state[3] < state[0] else 2 if state[3] > state[0] else 0
        q_list_indexes[3] = 1 if state[4] < state[1] else 2 if state[4] > state[1] else 0
        q_list_indexes[4] = 1 if (state[0] - 40, state[1]) in state[2] else 0
        q_list_indexes[5] = 1 if (state[0] + 40, state[1]) in state[2] else 0
        q_list_indexes[6] = 1 if (state[0], state[1] - 40) in state[2] else 0
        q_list_indexes[7] = 1 if (state[0], state[1] + 40) in state[2] else 0
        no_no_moves = set()
        yes_moves = set()
        if q_list_indexes[4] == 1:
            no_no_moves.add(2)
        if q_list_indexes[1] == 1:
            no_no_moves.add(0)
        if q_list_indexes[5] == 1:
            no_no_moves.add(3)
        if q_list_indexes[1] == 2:
            no_no_moves.add(1)
        if q_list_indexes[6] == 1:
            no_no_moves.add(0)
        if q_list_indexes[0] == 1:
            no_no_moves.add(2)
        if q_list_indexes[7] == 1:
            no_no_moves.add(1)
        if q_list_indexes[0] == 2:
            no_no_moves.add(3)
        if q_list_indexes[2] == 1 and q_list_indexes[3] == 0:
            yes_moves.add(2)
        if q_list_indexes[2] == 2 and q_list_indexes[3] == 0:
            yes_moves.add(3)
        if q_list_indexes[2] == 0 and q_list_indexes[3] == 1:
            yes_moves.add(0)
        if q_list_indexes[2] == 0 and q_list_indexes[3] == 2:
            yes_moves.add(1)
        return q_list_indexes, no_no_moves, yes_moves



    # Computing the reward, need not be changed.
    def compute_reward(self, points, dead):
        if dead:
            return -1
        elif points > self.points:
            return 1
        else:
            return -0.1

    #   This is the code you need to write. 
    #   This is the reinforcement learning agent
    #   use the helper_func you need to write above to
    #   decide which move is the best move that the snake needs to make 
    #   using the compute reward function defined above. 
    #   This function also keeps track of the fact that we are in 
    #   training state or testing state so that it can decide if it needs
    #   to update the Q variable. It can use the N variable to test outcomes
    #   of possible moves it can make. 
    #   the LPC variable can be used to determine the learning rate (lr), but if 
    #   you're stuck on how to do this, just use a learning rate of 0.7 first,
    #   get your code to work then work on this.
    #   gamma is another useful parameter to determine the learning rate.
    #   based on the lr, reward, and gamma values you can update the q-table.
    #   If you're not in training mode, use the q-table loaded (already done)
    #   to make moves based on that.
    #   the only thing this function should return is the best action to take
    #   ie. (0 or 1 or 2 or 3) respectively. 
    #   The parameters defined should be enough. If you want to describe more elaborate
    #   states as mentioned in helper_func, use the state variable to contain all that.
    def agent_action(self, state, points, dead):
        #print("IN AGENT_ACTION")
        if self._train == False:
            help, no_no_moves, yes_moves = self.helper_func(state)
            actions = self.Q[tuple(help)]
            action = max(enumerate(actions), key=lambda key: (key[1], random.random()))[0]
            return action
        else:
            #exploration
            help, no_no_moves, yes_moves = self.helper_func(state)
            state_help = tuple(help)
            actions = self.Q[state_help]
            for i in range(4):
                if i in no_no_moves:
                    actions[i] = 0.7 * -5.3 + (1-0.7) * actions[i]
                    #actions[i] = -4
                if i in yes_moves:
                    #actions[i] = 3
                    actions[i] = 0.7 * 4.3 + (1-0.7) * actions[i]
            action = max(enumerate(actions), key=lambda key: (key[1], random.random()))[0]
            actions[action] = 0.7 * self.compute_reward(points, dead) + (1-0.7) * actions[action]
            self.points = points
            return action
        #UNCOMMENT THIS TO RETURN THE REQUIRED ACTION.
        #return action