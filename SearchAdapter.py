# warnings:
#   1. this module is not thread safe
#   2. the reward must be non-negative


from AStarSearch import State, Treatment, Convertor, RestEstimator, AStarSearcher
import numpy as np
import random

from typing import List, Optional
import torch

max_timestep_limit = 10
max_changes_limit = 5

difference_logger = []


class CRNState(State):


    # rnn_state: (h, c) used to run rnn
    # V: input to rnn, look at the crn paper's picture for details
    # output: output to be inputed, look at the crn paper's picture for details
    def __init__(self, rnn_state, V, output, timestep: int, changes:int):
        self.state = (rnn_state, V, output)
        self.timestep = timestep
        self.changes = changes
    
    # return rnn_state, V, output, timestep, changes
    def getval(self):
        return self.state+(self.timestep,self.changes)
    
    def isValid(self) -> bool:
        assert self.timestep<=max_timestep_limit
        return self.changes<=max_changes_limit
    
    def isEnd(self) -> bool:
        assert self.isValid()
        return self.timestep==max_timestep_limit
    
    def difference(self, other: 'CRNState') -> float:
        global difference_logger
        s_r, sv, so = self.state
        o_r, ov, oo = other.state
        rd = torch.sum(torch.square(s_r[1]-o_r[1])+torch.square(s_r[0]-o_r[0])).item()
        vd = np.sum(np.square(sv-ov))
        od = np.sum(np.square(so-oo))
        ans = (rd+vd+od)**0.5
        difference_logger.append(ans)
        return ans
    

class CRNTreatment(Treatment):

    def __init__(self, treat):
        self.treat = treat
    
    def getval(self):
        return self.treat
    
    def isClose(self, other: 'CRNTreatment') -> bool:
        assert type(self.getval()) == int
        return self.getval() == other.getval()

class CRNConvertor(Convertor):

    # func: (rnn_state, V, output, treatment) -> (rnn_state, V, output, reward)
    # reward must be a scalar larger than 0
    def __init__(self, func, initial_treatments:List[CRNTreatment]):
        self.func = func
        self.initial_treatments = initial_treatments

    def predict(self, s: CRNState, t: CRNTreatment) -> Optional[tuple[State, float]]:
        in_rnn_state, in_V, in_output, in_timestep, in_changes = s.getval()
        out_rnn_state, out_V, out_output, reward = self.func(in_rnn_state, in_V, in_output, t)
        out_timestep = in_timestep + 1
        if t.isClose(self.initial_treatments[in_timestep]):  # no change
            out_changes = in_changes
        else:
            out_changes = in_changes + 1
        
        out_state = CRNState(out_rnn_state, out_V, out_output, out_timestep, out_changes)

        if out_state.isValid():
            return out_state, reward
        
        return None

class CRNRestEstimator(RestEstimator):


    # self.anchorset: List[CRNState], the timestep and changes in CRNState here is not used
    # self.anchor_rewards: three dim numpy array, id of state, timestep, level
    # LVals: np.ndarray
    # initial_treatments: List[Treatment]
    # all_treatments:List[Treatment]
    # init_state: CRNState
    # convertor: CRNConvertor
    # M: number of paths to sample
    def __init__(self, LVals: np.ndarray, initial_treatments: List[Treatment],
                 all_treatments:List[Treatment],
                 init_state: CRNState, convertor: CRNConvertor,
                 M: int) -> None:
        
        self.LVals = LVals
        self.initial_treatments = initial_treatments
        self.all_treatments = all_treatments
        self.init_state = init_state
        self.convertor = convertor

        self.getAnchorSet(M)
        self.calcAnchorRewards()


    # return: treatment path
    def sampleATreatmentPath(self)->List[Treatment]:
        k = random.randint(1, max_changes_limit)
        ktreats: List[Treatment] = random.choices(self.all_treatments, k=k)

        assert np.sum(self.LVals)>1e-8

        assert np.min(self.LVals)>=0.0

        places = np.random.choice(np.arange(max_timestep_limit),
                         size=k, replace=False,
                         p=self.LVals / np.sum(self.LVals)).astype(np.int32)
        ans = self.initial_treatments.copy()

        for i in range(k):
            ans[places[i]]=ktreats[i]

        return ans


    # M: how many path to sample
    def getAnchorSet(self, M: int)->None:
        anchorset=[]
        anchorset.append(self.init_state)
        for i in range(M):
            path = self.sampleATreatmentPath()
            tmp: CRNState = self.init_state
            for t in path:
                s, _ = self.convertor.predict(tmp, t)
                assert s.isValid()
                anchorset.append(s)
        
        self.anchorset=anchorset
    
    def estimate(self, s: CRNState) -> float:

        if s.isEnd():
            return 0.0

        timestep = s.timestep
        changes = s.changes

        ans = None

        assert self.LVals[timestep]>=0.0

        for i in range(len(self.anchorset)):
            diff = s.difference(self.anchorset[i])
            assert diff >=0.0
            tmp = self.anchor_rewards[timestep][changes][i] + diff * self.LVals[timestep]

            if ans is None:
                ans = tmp
            # 找到最接近的估计
            elif ans > tmp:
                ans = tmp

        assert ans is not None

        return ans
        


    def calcAnchorRewards(self)->None:
        self.anchor_rewards = np.zeros((max_timestep_limit+1,
                                        max_changes_limit+1,
                                        len(self.anchorset)),
                                        dtype=np.float32)
        for i in range(max_timestep_limit-1,-1, -1):
            for j in range(max_changes_limit+1):
                for k in range(len(self.anchorset)):
                    treat = self.initial_treatments[i]
                    now_state = CRNState(*(self.anchorset[k].getval()))
                    now_state.timestep = i
                    now_state.changes = j
                    result = self.convertor.predict(now_state, treat)
                    if result is not None:
                        next_state, reward = result
                        assert next_state.isValid()
                        assert next_state.timestep == i +1
                        assert next_state.changes == j

                        self.anchor_rewards[i][j][k] = max(self.anchor_rewards[i][j][k],
                                                           reward + self.estimate(next_state))
                    if j < max_changes_limit:
                        for t in self.all_treatments:
                            if t.isClose(self.initial_treatments[i]):
                                continue
                            result = self.convertor.predict(now_state, t)
                            if result is not None:
                                next_state, reward = result
                                assert next_state.isValid()
                                assert next_state.timestep == i +1
                                assert next_state.changes == j+1

                                self.anchor_rewards[i][j][k] = max(self.anchor_rewards[i][j][k],
                                                                reward + self.estimate(next_state))                        

            


# return the max reward and end state and treatment path
# return None if no solution found
# init_rnn_state, init_V, init_output are used to initialize the init state for search, see the paper's picture for details
# LVals, see the paper's search part for details
# func: (rnn_state, V, output, treatment) -> (rnn_state, V, output, reward), one step of decoder, see the paper's picture for details
# initial_treatments: the actual treatments for this patient
# all_treatments: all of the possible treatments, it must be discrete
# MForEstimator: used to specify how many random paths the estimator need
# max_timestep_limit
# max_change_limit
def getResult(
    init_rnn_state,
    init_V,
    init_output,
    LVals: np.ndarray,
    func,
    initial_treatments: List[CRNTreatment],
    all_treatments: List[CRNTreatment],
    MForEstimator:int = 10,
    timestep_limit = 10,
    changes_limit = 5
              ) -> Optional[tuple[float, State, List[Treatment]]]:
    

    global max_timestep_limit, max_changes_limit

    max_timestep_limit = timestep_limit
    max_changes_limit = changes_limit
    init_state = CRNState(init_rnn_state, init_V, init_output, 0, 0)
    convertor = CRNConvertor(func, initial_treatments)
    estimator = CRNRestEstimator(LVals, initial_treatments, all_treatments, init_state, convertor, MForEstimator)
    searcher = AStarSearcher(init_state, all_treatments, convertor, estimator)
    ans = searcher.search()
    print(len(difference_logger))
    return ans