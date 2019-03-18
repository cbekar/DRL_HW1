from decimal import *
class DPAgent():
    r""" Base Dynamic Programming class. DP methods
    requires the transition map in order to optimize
    policies. This class provides the policy and
    one step policy evaluation as well as policy
    improvement.
    """

    def __init__(self, nact, nobs, transitions_map, init_value=0.0):
        self.nact = nact
        self.nobs = nobs
        self.transitions_map = transitions_map
        self.values = {s: Decimal(init_value) for s in self.transitions_map.keys()}
        self.policy_dist = {s: [1.0/nact]*nact for s in self.transitions_map.keys()}

    def policy(self, state):
        # Policy pi that returns the action with the highest Q value.
        return max(range(self.nact), key = lambda a: self.policy_dist[state][a])

    def one_step_policy_eval(self, gamma=1.0):
        # One step policy evaluation.
        ### YOUR CODE HERE ###
        delta = .0
        for s in self.transitions_map.keys():
            v = self.values[s]
            tv = Decimal(.0)
            for a in range(self.nact):
                li = self.transitions_map[s][a]
                t = Decimal(.0)
                for branch in li:
                    t += Decimal(branch[0])*\
                    (Decimal(branch[2])+Decimal(gamma)*self.values[branch[1]])
                tv += Decimal(self.policy_dist[s][a])*t
            self.values[s] = tv
            delta = max(delta, abs(v-self.values[s]))
        ### END ###
        return delta
            
    def policy_improvement(self, gamma=1.0):
        # Policy impovement updates the policy according to the
        # most recent values
        ### YOUR CODE HERE ###
        is_policy_stable = True
        for s in self.transitions_map.keys():
            old = self.policy(s)
            tv = [Decimal(.0)]*self.nact
            for a in range(self.nact):
                li = self.transitions_map[s][a]
                t = Decimal(.0)
                for branch in li:
                    t += Decimal(branch[0])*\
                    (Decimal(branch[2])+Decimal(gamma)*self.values[branch[1]])
                tv[a] += t
            maxv = max(tv)
            mc = tv.count(maxv)
            for a in range(self.nact):
                self.policy_dist[s][a] = 1/mc if tv[a] == maxv else 0
            if self.policy(s) != old:
                is_policy_stable = False 
        ### END ###
        return is_policy_stable
    
    def printPolicy(self):
        for row in range(1,6):
            for col in range(1,35):
                try:
                    print(self.policy((row, col)),end=" ")
                except:
                    print("x",end=" ")
            print()
        print()
    def printValues(self):
        for row in range(1,6):
            for col in range(1,35):
                try:
                    print("%f" % self.values[(row, col)],end=" ")
                except:
                    print("xxx",end=" ")
            print()
        print()
    def printPolicyDist(self):
        for row in range(1,6):
            for col in range(1,35):
                for a in range(4):
                    try:
                        print(self.policy_dist[(row, col)][a],end=" ")
                    except:
                        print("x",end=" ")
                print("-")
            print("-")
        print()
class PolicyIteration(DPAgent):
    r""" Policy Iteration algorithm that first evaluates the
    values until they converge within epsilon range, then
    updates the policy and repeats the process until the
    policy no longer changes.
    """

    def __init__(self, nact, nobs, transitions_map):
        super().__init__(nact, nobs, transitions_map)

    def optimize(self, gamma, epsilon, max_iterations=100):
        # optimizer (float: gamma, float: epsilon) -> None
        ### YOUR CODE HERE ###
        for a in range(max_iterations):
            while self.one_step_policy_eval(gamma) > epsilon:
                continue
            if self.policy_improvement(gamma):
                break
        #self.printPolicy()
        #self.printValues()
        ### END ###

class ValueIteration(DPAgent):
    r""" Value Iteration algorithm iteratively evaluates
    the values and updates the policy until the values
    converges within epsilon range.
    """

    def __init__(self, nact, nobs, transitions_map):
        super().__init__(nact, nobs, transitions_map)
    
    def optimize(self, gamma, epsilon):
        #import ipdb; ipdb.set_trace()
        # optimize(float: gamma, float: epsilon) -> None
        ### YOUR CODE HERE ###
        while True:
            delta = .0
            for s in self.transitions_map.keys():
                v = self.values[s]
                tv = [Decimal(.0)]*self.nact
                for a in range(self.nact):
                    li = self.transitions_map[s][a]
                    t = Decimal(.0)
                    for branch in li:
                        t += Decimal(branch[0])*(Decimal(branch[2])+Decimal(gamma)*self.values[branch[1]])
                    tv[a] += t
                maxv = max(tv)
                mc = tv.count(maxv)
                self.values[s] = maxv
                for a in range(self.nact):
                    self.policy_dist[s][a] = 1/mc if tv[a] == maxv else 0
                delta = max(delta, abs(v-self.values[s]))
            if delta < epsilon:
                break
        #self.printPolicy()
        #self.printPolicyDist()
        #self.printValues()
        ### END ###