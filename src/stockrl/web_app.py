"""Public TradingMoE web entrypoint; Python serves the built React UI."""
from .web.runtime import Supervisor
from .web.server import serve
