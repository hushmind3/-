"""Web entrypoint. Implementations: stockrl.web; UI assets: web/assets."""
from __future__ import annotations
import csv
from collections import deque
from contextlib import closing
import json
import os
import sqlite3
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime, time as day_time, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from .paths import default_runtime_dir, ensure_project_path, validate_model_dir
from .paper_account import KR_SELL_TAX_ASSUMPTION, SEED_CASH
from .account_diagnostics import summarize_account, valid_bid_ask_count, input_availability
from .operating_rules import operating_rules, daily_boundary
from .state_io import atomic_json
from .web.resources import ROOT, PAGE
from .web.runtime import Supervisor
from .web.server import serve
from .web.health import _agent_progress_health, _candidate_progress_health, _json, _market_group, _market_overview, _session, _utc_datetime
