
from typing import Optional, List
import heapq


class State:
    
    # the limit on the number of different treatments can be achieved here
    def getval(self):
        pass

    # if the state is valid (the limit on the number of different treatments can be achieved here)
    def isValid(self) -> bool:
        pass

    # if the state is the end state
    def isEnd(self) -> bool:
        pass


class Treatment:
    def getval(self):
        pass

class Convertor:
    # Convert the state and treatment to the next state and return reward
    # the reward must be non-negative
    # return None if the next state is not valid
    def predict(self, s: State, t: Treatment) -> Optional[tuple[State, float]]:
        pass


class RestEstimator:

    # return the estimated of rest reward of the state
    # the estimation must bigger than the actual value
    # and when the state is terminal, the estimation must be 0
    def estimate(self, s: State) -> float:
        pass


class AStarSearcher:

    # treatments: the treatment list(means edges of a point (state))
    def __init__(self, start_state: State,
                 treatments: List[Treatment],
                 convertor: Convertor,
                 rest_estimator: RestEstimator):
        self.start_state = start_state
        self.treatments = treatments
        self.convertor = convertor
        self.rest_estimator = rest_estimator

    class Node:

        # fval is the total estimated reward of the path
        # gval is the actual reward from start_node to now
        def __init__(self, s: State, fval: float, gval: float, treatmens: List[Treatment], all_gvals: List[float]) -> None:
            self.s = s
            self.fval = fval
            self.gval = gval
            self.treatments = treatmens
            self.all_gvals = all_gvals

        # because the heap created by heapq is a min heap
        # but we want a max heap
        # so we need to reverse the order of the fval
        def __lt__(self, other):
            return self.fval > other.fval


    # return the max reward and end state and treatment path
    # return None if no solution found
    def search(self) -> Optional[tuple[float, State, List[Treatment]]]:
        max_heap=[]
        heapq.heappush(max_heap,
                       AStarSearcher.Node(self.start_state,
                                     self.rest_estimator.estimate(self.start_state),
                                     0.0,
                                     [], []))
        count = 0
        while max_heap:
            count = count+1
            node : AStarSearcher.Node =heapq.heappop(max_heap)
            if node.s.isEnd():
                print(count)
                return node.gval, node.s, node.treatments, node.all_gvals

            for treat in self.treatments:
                result = self.convertor.predict(node.s, treat)
                if result is None:
                    continue
                new_state, reward = result
                new_gval = node.gval + reward
                new_fval = new_gval + self.rest_estimator.estimate(new_state)
                new_treatmens = node.treatments + [treat]
                new_all_gvals = node.all_gvals + [new_gval]
                new_node = AStarSearcher.Node(new_state, new_fval, new_gval, new_treatmens, new_all_gvals)
                heapq.heappush(max_heap, new_node)

        return None
