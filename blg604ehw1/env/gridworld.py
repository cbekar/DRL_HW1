from gym.envs.toy_text import discrete
from itertools import product
import numpy as np

from .render import Renderer

"""
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
"""
# You can change the map to experiment with agents.
MAP0 =["WWWWWW",
       "WEFEGW",
       "WEWEHW",
       "WSEEEW",
       "WWWWWW",
       ]
MAP = ["WWWWWWWWWWWWWWWWWWW",
       "WEEEEEEEEEWEEEEEEGW",
       "WEEEEEEEEEWEEEEEEEW",
       "WEEEEEEEEEEEEEEEEEW",
       "WEEEEEEEEEWEEEEEEEW",
       "WEEESSEEEEWEEEEEEEW",
       "WWWWWWWWWWWWWWWWWWW",
       ]
MAP3 =["WWWWWWWWWWWWWWWW",
       "WEEEEFEEEFEEEFGW",
       "WEEFEFEFEFEFEFEW",
       "WEEFEFEFEFEFEFEW",
       "WEEFEFEFEFEFEFEW",
       "WSEFEEEFEEEFEEEW",
       "WWWWWWWWWWWWWWWW",
       ]
MAP2 =["WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",
       "WEEEEEEEEEEWEEEEEWWEEEEEEEEEWEEEEEGW",
       "WEEWWWWWWWEWEEEEEWWEEEEEEEEEWEEEEEEW",
       "WEEEEEEEEWEWEEEEEEEEEEEEEEEEEEEEEEEW",
       "WWWWWWWWEWEWEEEEEWWEEEEEEEEEWEEEEEEW",
       "WSEEEEEEEWEEEEEEEWWEEEEEEEEEWEEEEEEW",
       "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",
       ]
MAP1 =["WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",
       "WEEEEEEEEEEWEEEEEWWEEEEEEEEEWEEEEEGW",
       "WEEWWWWWWWEWEEEEEWWEEEEEEEEEWEEEEEEW",
       "WEEEEEEEEWEWEEEEEEEEEEEEEEEEEEEEEEEW",
       "WWWWWWWWEWEWEEEEEWWEEEEEEEEEWEEEEEEW",
       "WSEEEEEEEWEEEEEEEWWEEFEEEEEEWEEEEEEW",
       "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",
       ]
REWARD_MAP = {
    b"W" : 0,       # Wall
    b"S" : 0,       # Start
    b"E" : 0,       # Empty
    b"F" : -0.1,    # Fire
    b"H" : -1.0,    # Hole
    b"G" : 1.0,     # Goal
}

TERMINATION_MAP = {
    b"W" : 0,
    b"S" : 0,
    b"E" : 0,
    b"F" : 0,
    b"H" : 1,
    b"G" : 1,
}
class GridWorldEnv(discrete.DiscreteEnv):
    r"""
        Grid environment with different tile types. Both the initial state
        distribution and transition map is defined so that the environment
        can be modeled as a MDP. This environment has the same basic methods
        as any other gym environment.
        
        Args:
            gridmap (list): List of strings defining the map of the grid.
            randomness (float): Probability of acting differently then expected.
    """

    metadata = {'render.modes': ["human", "ansi", "visual", "notebook"]}

    def __init__(self, gridmap=MAP, randomness=0.05):
        self.grid = np.asarray(gridmap, dtype='c')
        self.heigth, self.width = self.grid.shape

        nactions = 4
        nstates = self.heigth*self.width

        # Initial starting states and the corresponding distribution.
        initial_states = np.argwhere(self.grid == b"S")
        initial_state_dist = [(1.0/len(initial_states), state) for state in initial_states]

        # All the states that an agent may visit. (Non wall states)
        all_states = np.argwhere(self.grid != b"W")


        # This is the main table for the MDP. You can construct any MDP by modifying this.
        #   transition_map(P) := P(state, action) -> [(probability, next state, reward, termination), ...]
        # From any state action pair an agent can travel to its neighbouring tiles with the corresponding 
        # probability, reward value and termination.
        # Transition map is a dictionary of a dictionaries of lists
        # You supposed to fill the table in order to construct the MDP.
        transition_map = {tuple(state): {act: [] for act in range(nactions)} for state in all_states}
        
        ### YOUR CODE HERE ###
        def inc(row, col, a): # helper function
            if a==0: # left
                col = max(col-1,0)
            elif a==1: # down
                row = min(row+1,self.heigth-1)
            elif a==2: # right
                col = min(col+1,self.width-1)
            elif a==3: # up
                row = max(row-1,0)
            return (row, col)
        
        for row, col in all_states:
            s = (row, col)
            letter = self.grid[s]
            for a in range(4):
                li = transition_map[s][a]
                neighbours = [(True,-1,s,letter)] # option to fall onto self
                if self.grid[s] in b'GH':
                    li.append((1.0, s, 0, 1)) # termination
                else:
                    for b in range(4):
                        newstate = inc(row, col, b)
                        newletter = self.grid[newstate]
                        neighbours.append([True if newletter != b'W' else False,b,newstate,newletter])
                    validN = 0
                    validA = False
                    for cell in neighbours:
                        if cell[0]:
                            validN += 1
                            if cell[1] == a:
                                validA = True
                    for cell in neighbours:
                        done = TERMINATION_MAP[cell[3]]
                        rew = REWARD_MAP[cell[3]]
                        if validA:
                            if cell[0]:
                                if a == cell[1]:
                                    li.append((1-randomness, cell[2], rew, done))
                                else:
                                    li.append((randomness/(validN-1), cell[2], rew, done))
                        else:
                            if cell[0]:
                                li.append((1/validN, cell[2], rew, done))
                #print(s,a,transition_map[s][a],sep=" -> ")
            ### END ###

        super().__init__(nstates, nactions, transition_map, initial_state_dist)
        self.renderer = Renderer(self.grid)

    def reset(self):
        indx = discrete.categorical_sample([prob for prob, state in self.isd], self.np_random)
        self.lastaction=None
        _, state = self.isd[indx]
        self.s = tuple(state)
        return self.s

    def render(self, mode, **kwargs):
        if mode=="visual":
            self.renderer.visaul_render(self.s, **kwargs)
        elif mode == "notebook":
            self.renderer.buffer_render(self.s, **kwargs)
        elif mode in ("ansi", "stdout"):
            self.renderer.string_render(self.s, **kwargs)
