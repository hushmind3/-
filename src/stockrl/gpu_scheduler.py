"""Live inference precedes queued learning; each priority retains FIFO order."""
from collections import deque
from contextlib import contextmanager
import threading
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


class FairGpuScheduler:
    def __init__(self,preopen_learning=False,clock=None):
        self.condition=threading.Condition()
        self.queue=deque()
        self.active=None
        self.last={}
        self.preopen_learning=preopen_learning
        self.clock=clock or (lambda:datetime.now(timezone.utc))

    def learning_first(self):
        now=self.clock()
        korea=now.astimezone(ZoneInfo("Asia/Seoul"))
        new_york=now.astimezone(ZoneInfo("America/New_York"))
        return self.preopen_learning and korea.hour>=20 and (new_york.hour,new_york.minute)<(9,30)

    def priority(self,role):
        if self.learning_first():
            if "_learning_" in role:
                return 0
            if role=="candidate_publish":
                return 1
            return 3 if role.startswith("validation_") else 2
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
            now_utc=self.clock()
            market_open=now_utc.astimezone(ZoneInfo("America/New_York")).replace(hour=9,minute=30,second=0,microsecond=0)
            return {"policy":("preopen_replay_learning_first" if self.learning_first()
                    else "live_inference_first_then_publish_validation_learning"),
                "learning_priority_until_utc":market_open.astimezone(timezone.utc).isoformat() if self.learning_first() else None,
                "active":self.active[1] if self.active else None,
                "active_seconds":now-self.active[2] if self.active else 0,
                "waiting":[{"role":role,"wait_seconds":now-requested}
                           for _,role,requested in sorted(self.queue,key=lambda item:(self.priority(item[1]),item[2]))],
                "last":{role:dict(data) for role,data in self.last.items()}}
