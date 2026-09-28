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


class MockBroker(BrokerAdapter):
    """Safe test broker that records simulated orders and has no network client."""
    is_live = False

    def __init__(self):
        self.connected = False
        self.orders: list[dict[str, Any]] = []
        self.stopped = False

    def connect(self) -> None:
        self.connected = True; self.stopped = False

    def place_order(self, order: OrderRequest) -> dict[str, Any]:
        if not self.connected or self.stopped:
            raise RuntimeError("mock broker is disconnected or emergency-stopped")
        result = {"status": "paper_filled", "symbol": order.symbol, "side": order.side.upper(),
                  "quantity": order.quantity, "timestamp": order.timestamp, "live_order": False}
        self.orders.append(result)
        return result

    def emergency_stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.connected = False


class BrokerWorker:
    """Queue based broker executor, isolated from the UI event loop."""
    def __init__(self, adapter: BrokerAdapter, result_queue: Queue):
        self.adapter=adapter; self.inbox: Queue=Queue(); self.results=result_queue
        self.stop_event=Event(); self.thread=Thread(target=self._run,name="broker-order-worker",daemon=True)

    def start(self):
        self.thread.start()

    def submit(self, request: OrderRequest): self.inbox.put(request)

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
                except Exception as exc: self.results.put({"ok":False,"stage":"order","error":f"{type(exc).__name__}: {exc}"})
        except Exception as exc:
            self.results.put({"ok":False,"stage":"connect","error":f"{type(exc).__name__}: {exc}"})
        finally:
            try: self.adapter.close()
            except Exception as exc: self.results.put({"ok":False,"stage":"close","error":str(exc)})
