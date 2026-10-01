"""Live inference precedes queued learning; each priority retains FIFO order."""
from collections import deque
from contextlib import contextmanager
import threading
import time


class FairGpuScheduler:
    def __init__(self):
        self.condition=threading.Condition()
        self.queue=deque()
        self.active=None
        self.last={}

    @staticmethod
    def priority(role):
        if role in ("champion_live","candidate_live"):
            return 0
        if role=="candidate_publish":
            return 1
        if role.startswith("validation_"):
            return 2
        return 3

    def next_request(self):
        return min(self.queue,key=lambda item:(self.priority(item[1]),item[2]))

    @contextmanager
    def work(self,role):
        token=object();requested=time.perf_counter()
        with self.condition:
            self.queue.append((token,role,requested))
            self.condition.notify_all()
            self.condition.wait_for(lambda:self.active is None and self.next_request()[0] is token)
            self.queue.remove((token,role,requested))
            started=time.perf_counter()
            self.active=(token,role,started)
        try:
            yield
        finally:
            finished=time.perf_counter()
            with self.condition:
                self.last[role]={"wait_seconds":started-requested,
                                 "work_seconds":finished-started}
                self.active=None
                self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            now=time.perf_counter()
            return {"policy":"live_inference_first_then_publish_validation_learning",
                "active":self.active[1] if self.active else None,
                "active_seconds":now-self.active[2] if self.active else 0,
                "waiting":[{"role":role,"wait_seconds":now-requested}
                           for _,role,requested in sorted(self.queue,key=lambda item:(self.priority(item[1]),item[2]))],
                "last":{role:dict(data) for role,data in self.last.items()}}
