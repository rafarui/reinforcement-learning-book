# -*- coding: utf-8 -*-
"""
Created on Tue May  7 08:27:53 2019

@author: Pichau
"""

import numpy as np
import itertools


class tic_tac_toe(object):
    
    def __init__(self):
        
        self.reset()
        
        
    def reset(self):
        
        # define all possible states
        # each state is a 1x9 vector, 
        # where the elements 0 to 2 correspond to the top row of the board,
        # elements 3 to 5 correspond to the middle row of the board,
        # and elements 6 to 8 correspond to the bottom row of the board
        # and 1=X, 2=O and 0=blank
        self.states = np.array(list(itertools.product(range(3), repeat=9)))
        
        # generate initial value functions for both players
        # with all values equal to 0.5
        self.value_fun_x = np.ones(len(self.states)) * 0.5
        self.value_fun_o = np.ones(len(self.states)) * 0.5

        # construct a mask with all winning situations for X
        x_wins = np.all(self.states[:, 0:3]==1, axis=1) | \
                np.all(self.states[:, 3:6]==1, axis=1) | \
                np.all(self.states[:, 6:9]==1, axis=1) | \
                np.all(self.states[:, [0, 3, 6]]==1, axis=1) | \
                np.all(self.states[:, [1, 4, 7]]==1, axis=1) | \
                np.all(self.states[:, [2, 5, 8]]==1, axis=1) | \
                np.all(self.states[:, [0, 4, 8]]==1, axis=1) | \
                np.all(self.states[:, [2, 4, 6]]==1, axis=1)

        # construct a mask with all winning situations for O
        o_wins = np.all(self.states[:, 0:3]==2, axis=1) | \
                np.all(self.states[:, 3:6]==2, axis=1) | \
                np.all(self.states[:, 6:9]==2, axis=1) | \
                np.all(self.states[:, [0, 3, 6]]==2, axis=1) | \
                np.all(self.states[:, [1, 4, 7]]==2, axis=1) | \
                np.all(self.states[:, [2, 5, 8]]==2, axis=1) | \
                np.all(self.states[:, [0, 4, 8]]==2, axis=1) | \
                np.all(self.states[:, [2, 4, 6]]==2, axis=1)

        # put a 1 in all winning positions and a 0 in all losing positions
        self.value_fun_x[x_wins & ~o_wins] = 1
        self.value_fun_o[x_wins & ~o_wins] = 0 

        self.value_fun_x[o_wins & ~x_wins] = 0
        self.value_fun_o[o_wins & ~x_wins] = 1
        
        
    def play(self, num_games, eps=0.1, alpha=0.1, reset=False):
        
        if reset:
            self.reset()
        
        # keep track of results
        wins_x = 0
        wins_o = 0
        draws = 0
        
        # for each game...
        for game in range(num_games):
            
            # initialise game variables
            board = np.zeros((3, 3))
            game_over = False
            if game % 2 == 0:
                turn = 'x'
            else:
                turn = 'o' 
                
            # until game over...
            while not game_over:

                # get the current state of the board
                current_state = board.flatten()
                i_current = np.where(np.all(current_state == self.states, axis=1))[0][0]

                # get positions on board which are already taken
                taken = current_state != 0

                # find all possible states with the same taken positions
                i_possible = np.where(np.all(current_state[taken] == self.states[:, taken], axis=1))[0]

                # of these, find states with one less taken position than current state
                i_possible = i_possible[(self.states[i_possible] == 0).sum(axis=1) == (current_state == 0).sum() - 1]

                if turn == 'x':
                    
                    # of these, find states with one more X than current states
                    i_possible = i_possible[(self.states[i_possible] == 1
                                            ).sum(axis=1) == (current_state == 1).sum() + 1]

                    # get the state with the highest probability (break ties randomly)
                    max_value = self.value_fun_x[i_possible].max()
                    i_best = i_possible[self.value_fun_x[i_possible] == max_value]
                    if ~np.isscalar(i_best):
                        i_best = i_best[np.random.choice(len(i_best), 1)][0]

                    # with probability 1-eps choose the best next state
                    if np.isscalar(i_best) | (np.random.rand(1)[0] > eps):
                        i_next = i_best

                        # backup value
                        self.value_fun_x[i_current] = self.value_fun_x[i_current] + \
                                                alpha * (self.value_fun_x[i_next] - self.value_fun_x[i_current])

                    else:
                        # choose one of the other states randomly
                        i_possible = i_possible[i_possible != i_best]
                        i_next = i_possible[np.random.choice(len(i_possible), 1)][0]

                    # change turn
                    turn = 'o'  

                else:
                    
                    # of these, find states with one more O than current states
                    i_possible = i_possible[(self.states[i_possible] == 2
                                            ).sum(axis=1) == (current_state == 2).sum() + 1] 

                    # get the state with the highest probability (break ties randomly)
                    max_value = self.value_fun_o[i_possible].max()
                    i_best = i_possible[self.value_fun_o[i_possible] == max_value]
                    if ~np.isscalar(i_best):
                        i_best = i_best[np.random.choice(len(i_best), 1)][0]

                    # with probability 1-eps choose the best next state
                    if np.isscalar(i_best) | (np.random.rand(1)[0] > eps):
                        i_next = i_best

                        # backup value
                        self.value_fun_o[i_current] = self.value_fun_o[i_current] + \
                                                alpha * (self.value_fun_o[i_next] - self.value_fun_o[i_current])

                    else:
                        # choose one of the other states randomly
                        i_possible = i_possible[i_possible != i_best]
                        i_next = i_possible[np.random.choice(len(i_possible), 1)] 

                    # change turn
                    turn = 'x' 

                # add new move to board
                next_state = self.states[i_next].copy()
                board = np.reshape(next_state, (3, 3))

                # check if game over
                if ~np.any(board == 0):
                    draws += 1
                    game_over = True

                elif np.any(np.all(board == 1, axis=1)) | \
                    np.any(np.all(board == 1, axis=0)) | \
                    np.all(np.diag(board) == 1) | \
                    np.all(np.diag(np.fliplr(board)) == 1):
                    wins_x += 1
                    game_over = True

                elif np.any(np.all(board == 2, axis=1)) | \
                    np.any(np.all(board == 2, axis=0)) | \
                    np.all(np.diag(board) == 2) | \
                    np.all(np.diag(np.fliplr(board)) == 2):
                    wins_o += 1
                    game_over = True

        return {"wins_x": wins_x, "wins_o": wins_o, "draws": draws}
        
        