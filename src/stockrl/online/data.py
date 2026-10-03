"""Legacy learner metrics and imports retained for stored Experience records."""
from __future__ import annotations
from ..experience import (Experience, IncrementalMarketCSV, MarketObservation,
    REWARD_VERSION, REWARD_DEFINITION, parse_horizon)



PAPER_EXPLORATION_EPSILON=0.05


ONLINE_TRAINABLE_BLOCKS=4


class TrainingMetrics:
    """Keep each learner's measurements in the same runtime status object."""
    def __init__(self,store,learner):
        self.store=store; self.learner=learner
    def key(self,key):
        if self.learner=="candidate": return key
        if key.startswith("candidate_"): return "champion_"+key[len("candidate_"):]
        if key.startswith("last_candidate_"): return "last_champion_"+key[len("last_candidate_"):]
        if key in ("weight_delta_l1","paper_examples_trained","teacher_examples_trained"):
            return "champion_"+key
        return key
    def __getitem__(self,key): return self.store[self.key(key)]
    def __setitem__(self,key,value): self.store[self.key(key)]=value
    def get(self,key,default=None): return self.store.get(self.key(key),default)
    def setdefault(self,key,default=None): return self.store.setdefault(self.key(key),default)
