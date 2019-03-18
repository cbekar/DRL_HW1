import numpy as np
from collections import namedtuple
import random

QTransition = namedtuple("QTransition", "state action reward next_state terminal")
SarsaTransition = namedtuple("QTransition", "state action reward next_state next_action terminal")

class AbstractApproximateAgent():
    r""" Base class for the approximate methods. This class
    provides policies.
    """

    def __init__(self, nobs, nact):
        self.nact = nact
        self.nobs = nobs
        self.weights = np.array(np.random.uniform(-0.1, 0.1, size=(nobs, nact)), dtype=np.float64)
        random.seed()

    def q_values(self, state):
        return np.dot(state, self.weights)

    def greedy_policy(self, state):
        # Returns the best possible action according to the values.
        return np.argmax(self.q_values(state))

    def e_greedy_policy(self, state, epsilon=0.4):
        # Returns the best action with the probability (1 - e) and 
        # action with probability e
        return self.greedy_policy(state) if random.random() > epsilon else random.randrange(self.nact)

    def soft_policy(self, state):
        # Probabilistic policy where the probability of returning an
        # action is proportional to the value of the action.
        tv = [0.0]*self.nact
        for a in range(self.nact):
            tv[a] = self.q_values[state][a] # todo: learn lambda
        return random.choices(list(range(self.nact)), weights=tv)[0] if max(tv) != 0 else random.randrange(self.nact)
                                        
    def update(self, *arg, **kwargs):
        raise NotImplementedError

class ApproximateQAgent(AbstractApproximateAgent):
    r""" Approximate Q learning agent where the learning is done
    via minimizing the mean squared value error with **semi**-gradient decent.
    This is an off-policy algorithm.
    """

    def __init__(self, nobs, nact):
        super().__init__(nobs, nact)
    
    def update(self, tran, alpha, gamma):
        """ Updates the parameters that parameterized the value function.
            update(QTransition: tran, float: alpha, float: gamma) -> mean_square_td_error
            QTransition: (state, action, reward, next_state, terminal)
        """

        mean_squared_td_error = np.float64(.0)
        grad = np.array(np.gradient(self.weights)[0],dtype=np.float64)
        mean_squared_td_error = (tran.reward + gamma * self.q_values(tran.next_state)[tran.action]-\
                                 self.q_values(tran.state)[tran.action]) * grad
        self.weights += alpha * mean_squared_td_error
        return mean_squared_td_error

class ApproximateSarsaAgent(AbstractApproximateAgent):
    #import ipdb; ipdb.set_trace()
    def __init__(self, nobs, nact):
        super().__init__(nobs, nact)

    def update(self, tran, alpha, gamma):
        """ Updates the parameters that parameterized the value function.
            update (SarsaTransition: trans, float: alpha, float: gamma) -> float: mean_square_td_error
            SarsaTransition: (state, action, reward, next_state, next_action, terminal)
        """
        
        grad = np.array(np.gradient(self.weights)[0],dtype=np.float64)
        mean_squared_td_error = np.float64(.0)
        if tran.terminal:
            mean_squared_td_error = (tran.reward - self.q_values(tran.state)[tran.action]) * grad
        else:
            mean_squared_td_error = (tran.reward + \
                                     gamma * self.q_values(tran.next_state)[tran.next_action] - \
                                     self.q_values(tran.state)[tran.action]) * grad
        self.weights += alpha * mean_squared_td_error
        return mean_squared_td_error

class LSTDQ(AbstractApproximateAgent):
    r""" Least Square Temporal Difference Q learning algorithm.
    Unlike the tabular counterpart of the LSTD, this method uses
    samples transitions from the environment and updates the parameters
    that parameterized the value function at one step. Note that
    in this implementation RBFs(Radial Basis Functions) are used
    as features and value function is defined as the linear combination
    of these functions.
    """

    def __init__(self, nobs, nact, features=60):
        #import ipdb; ipdb.set_trace()
        super().__init__(nobs, nact)
        self.weights = np.random.uniform(-0.1, 0.1, size=(features))
        self.features = features
        # You can modify RBFs centers
        self.rbf_centers = np.random.normal(loc=0, scale=0.5, size=(features, nobs+nact))
        self.rbf_centers[:, -2:] = 0
        self.rbf_centers[:features//2, -2] = 1.0
        self.rbf_centers[features//2:, -1] = 1.0
        self.rbf_centers[:,0] *= 3 # pos
        self.rbf_centers[:,1] *= 3   # posdot !guess
        self.rbf_centers[:,2] *= 3  # deg
        self.rbf_centers[:,3] *= 3   # degdot !guess
        
    def get_value(self, state, action):
        return np.dot(self.phi(state, action), self.weights)

    def phi(self, state, action):
        # Features of the (state, action) pair. This method returns feature vector 
        # using RBFs for the given pair.
        action1hot = np.empty((0))
        action1hot = np.append(np.append(action1hot,1),0) if action == 0 else np.append(np.append(action1hot,0),1)
        features = np.exp(-np.true_divide(np.linalg.norm(np.subtract(np.concatenate([state, action1hot]), self.rbf_centers), axis=1),1))
        return features

    def greedy_policy(self, state):
        # Override the base greedy policy to make it compatibale with the RBFs.
        return 0 if self.get_value(state,0) > self.get_value(state,1) else 1 
        
    def soft_policy(self, state):
        # Override the base soft policy to make it compatibale with the RBFs.
        tv = [0.0]*self.nact
        for a in range(self.nact):
            tv[a] = self.get_value(state, a) # todo: learn lambda
        return random.choices(list(range(self.nact)), weights=tv)[0] if max(tv) != 0 else random.randrange(self.nact)

    def optimize(self, trans, gamma=0.99, epsilon=0.01):
        """ Optimize the parameters using the transitions sampled from the
        recent policy. Transitions  argument consists of QTransitions. 
        optimize(List: trans, float: gamma) -> None
        """
        
        L2 = len(trans)
        Fi, PFi, R = [0]*L2, [0]*L2, [0]*L2
        for iteration in range(L2):
            Fi[iteration] = np.transpose(self.phi(trans[iteration].state,trans[iteration].action))
            PFi[iteration] = \
            np.transpose(self.phi(trans[iteration].next_state,trans[iteration].next_action))
            R[iteration] = trans[iteration].reward
        A = 1/L2 * np.dot(np.transpose(Fi), (Fi-np.dot(gamma,PFi)))
        b = 1/L2 * np.dot(np.transpose(Fi), R)
        self.weights = np.dot(np.linalg.inv(A), b)