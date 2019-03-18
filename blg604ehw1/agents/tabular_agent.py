from collections import defaultdict
from collections import namedtuple
import random
import math

class TabularAgent:
    r""" Base class for the tabular agents. This class
    provides policies for the tabular agents.
    """

    def __init__(self, nact):
        self.values = defaultdict(lambda: [0.0]*nact)
        self.nact = nact
        random.seed()
    
    def greedy_policy(self, state):
        # Returns the best action.
        return max(range(self.nact), key = lambda a: self.values[state][a])

    def e_greedy_policy(self, state, epsilon):
        # Returns the best action with the probability (1 - e) and 
        # action with probability e
        return self.greedy_policy(state) if random() > epsilon else random.randrange(self.nact)
    
    def soft_policy(self, state):
        # Probabilistic policy where the probability of returning an
        # action is proportional to the value of the action.
        tv = [0.0]*self.nact
        for a in range(self.nact):
            tv[a] = self.values[state][a] # todo: learn lambda
        return random.choices(list(range(self.nact)), weights=tv)[0] if max(tv) != 0 \
    else random.randrange(self.nact)

    def update(self):
        raise NotImplementedError # have to pass alpha, not neccessary

class QAgent(TabularAgent):
    r""" Q learning agent where the update is done
    accordig to the off-policy Q update.        
    """

    def __init__(self, nact):
        super().__init__(nact)
    
    def update(self, trans, alpha, gamma):
        """ update (QTransition: trans, float: alpha, float: gamma) -> float: td_error
        QTransition: (state, action, reward, next_state, terminal)
        """
        
        td_error = trans.reward + \
        gamma * self.values[trans.next_state][self.greedy_policy(trans.next_state)] - \
        self.values[trans.state][trans.action]
        self.values[trans.state][trans.action] += alpha*td_error
        return td_error
    
class SarsaAgent(TabularAgent):
    r""" Sarsa agent where the update is done
    accordig to the on-policy Sarsa update. 
    """
                                 
    def __init__(self, nact):
        super().__init__(nact)

    def update(self, trans, alpha, gamma):
        """ update (SarsaTransition: trans, float: alpha, float: gamma) -> float: td_error
        SarsaTransition: (state, action, reward, next_state, next_action, terminal)
        """
        
        td_error = trans.reward + gamma * self.values[trans.next_state][trans.next_action] - self.values[trans.state][trans.action]
        self.values[trans.state][trans.action] += alpha*td_error
        return td_error