"""Explicit broker boundary. The bundled adapter never sends a real order."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from queue import Queue, Empty
from threading import Event, Thread
from typing import Any


@dataclass
class OrderRequest:
    symbol: str
    side: str
    quantity: float
    timestamp: str
    limit_price: float | None = None


class BrokerAdapter(ABC):
    """Implement this interface in a separate optional live-broker plugin."""
    is_live: bool = False

    def supports_symbol(self, symbol: str) -> bool:
        """Live adapters must explicitly allow symbols before order routing."""
        return False

    def size_order(self, symbol: str, side: str) -> float:
        """Return a risk-limited quantity; the default refuses to size orders."""
        return 0.0

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def place_order(self, order: OrderRequest) -> dict[str, Any]: ...

    @abstractmethod
    def emergency_stop(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...




class BrokerWorker:
    """Queue based broker executor, isolated from the UI event loop."""
    def __init__(self, adapter: BrokerAdapter, result_queue: Queue):
        self.adapter=adapter; self.inbox: Queue=Queue(); self.results=result_queue
        self.stop_event=Event(); self.thread=Thread(target=self._run,name="broker-order-worker",daemon=True)

    def start(self):
        self.thread.start()

    def submit(self, request: OrderRequest):
        if self.stop_event.is_set():
            return False
        self.inbox.put(request)
        return True

    def _discard_pending(self):
        while True:
            try: self.inbox.get_nowait()
            except Empty: return

    def emergency_stop(self):
        self.stop_event.set()
        # Attempt the adapter's kill switch out-of-band so a blocked order or
        # network call cannot freeze the GUI's emergency-stop action.
        Thread(target=self._emergency_adapter,name="broker-kill-switch",daemon=True).start()

    def _emergency_adapter(self):
        try: self.adapter.emergency_stop()
        except Exception as exc: self.results.put({"ok":False,"stage":"emergency_stop","error":str(exc)})

    def close(self):
        self.stop_event.set()
        if self.thread.is_alive(): self.thread.join(timeout=5)

    def _run(self):
        try:
            self.adapter.connect()
            self.results.put({"ok":True,"stage":"connected"})
            while not self.stop_event.is_set():
                try: request=self.inbox.get(timeout=.25)
                except Empty: continue
                try: self.results.put({"ok":True,"stage":"order","result":self.adapter.place_order(request)})
                except Exception as exc:
                    # Fail closed on the first order error. Do not process
                    # requests already queued from the same decision batch.
                    self.stop_event.set()
                    self._discard_pending()
                    try: self.adapter.emergency_stop()
                    except Exception as stop_exc:
                        self.results.put({"ok":False,"stage":"emergency_stop","error":str(stop_exc)})
                    self.results.put({"ok":False,"stage":"order","error":f"{type(exc).__name__}: {exc}"})
                    break
        except Exception as exc:
            self.stop_event.set()
            self._discard_pending()
            self.results.put({"ok":False,"stage":"connect","error":f"{type(exc).__name__}: {exc}"})
        finally:
            try: self.adapter.close()
            except Exception as exc: self.results.put({"ok":False,"stage":"close","error":str(exc)})
