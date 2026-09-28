"""PySide6 dashboard and process supervisor for the persistent global agent."""
from __future__ import annotations

import csv
import importlib
import json
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue, Empty

from PySide6.QtCore import QProcess, QTimer, Qt
from PySide6.QtGui import QCloseEvent, QColor, QFontDatabase
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
    QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QInputDialog)

from .broker import BrokerAdapter, BrokerWorker, OrderRequest


ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _load_broker_plugin() -> BrokerAdapter | None:
    spec = os.environ.get("STOCKRL_LIVE_BROKER_ADAPTER", "").strip()
    if not spec or ":" not in spec:
        return None
    module_name, class_name = spec.split(":", 1)
    adapter_type = getattr(importlib.import_module(module_name), class_name)
    adapter = adapter_type()
    if not isinstance(adapter, BrokerAdapter) or not getattr(adapter, "is_live", False):
        raise TypeError("Live broker plugin must subclass BrokerAdapter and declare is_live=True")
    return adapter


class StatCard(QFrame):
    def __init__(self, title: str, value: str = "-"):
        super().__init__(); self.setObjectName("statCard")
        layout=QVBoxLayout(self); layout.setContentsMargins(12,9,12,9)
        self.title=QLabel(title); self.title.setObjectName("statTitle")
        self.value=QLabel(value); self.value.setObjectName("statValue"); self.value.setWordWrap(True)
        layout.addWidget(self.title); layout.addWidget(self.value)


class GlobalAgentWindow(QMainWindow):
    def __init__(self, runtime: str | Path = "runtime-global-desktop", device: str = "auto"):
        super().__init__()
        if os.name=="nt" and not QFontDatabase.families():
            # Qt's offscreen plugin may not enumerate Windows fonts; loading
            # the installed system face is local-only and does not bundle it.
            for font_file in (Path(os.environ.get("WINDIR",r"C:\Windows"))/"Fonts"/"segoeui.ttf",
                              Path(os.environ.get("WINDIR",r"C:\Windows"))/"Fonts"/"segoeuib.ttf"):
                if font_file.is_file(): QFontDatabase.addApplicationFont(str(font_file))
        self.runtime = (ROOT / runtime).resolve() if not Path(runtime).is_absolute() else Path(runtime).resolve()
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.settings_path = self.runtime / "desktop_settings.json"
        self.settings = _read_json(self.settings_path)
        self.mode = str(self.settings.get("mode", "live"))
        self.bars = int(self.settings.get("mock_bars", 24))
        self.speed = float(self.settings.get("mock_interval_seconds", .5))
        self.candidate_every = int(self.settings.get("candidate_every", 256))
        self.fee = float(self.settings.get("fee", .001))
        self.source_data = Path(self.settings.get("mock_source", "data/global_market_daily.csv"))
        self.run_requested = False
        self._stopping = False
        self._was_stopped = False
        self._close_when_stopped = False
        self.live_armed = False
        self.broker_worker: BrokerWorker | None = None
        self.broker_results: Queue = Queue()
        self.order_adapter: BrokerAdapter | None = None
        self.order_cursor = 0
        self._build_ui()
        self._restore_settings()
        self.refresh_timer=QTimer(self); self.refresh_timer.timeout.connect(self.refresh_status); self.refresh_timer.start(1000)
        self.refresh_status()

    def _build_ui(self):
        self.setWindowTitle("Global Market Agent - Paper Trading")
        self.resize(1420, 900)
        root=QWidget(); outer=QVBoxLayout(root); outer.setContentsMargins(16,14,16,14); outer.setSpacing(10)
        header=QHBoxLayout(); title=QLabel("GLOBAL MARKET AGENT"); title.setObjectName("appTitle")
        self.overall=QLabel("Idle"); self.overall.setObjectName("statusPill")
        header.addWidget(title); header.addStretch(); header.addWidget(self.overall)
        outer.addLayout(header)
        stats=QGridLayout(); stats.setHorizontalSpacing(8); stats.setVerticalSpacing(8)
        self.cards={}
        items=[("feed","Market data"),("model","Model status"),("champion","Champion version"),
               ("replay","Replay experiences"),("reward","Paper net reward"),("last","Last observation"),
               ("learn","Last training"),("candidate","Candidate status"),
               ("gpu","GPU / VRAM"),("portfolio","Paper positions"),
               ("validation","Promotions / rejections"),("order","Live orders")]
        for index,(key,label) in enumerate(items):
            card=StatCard(label); self.cards[key]=card
            stats.addWidget(card,index//4,index%4)
        outer.addLayout(stats)

        controls=QHBoxLayout()
        self.mode_combo=QComboBox(); self.mode_combo.addItem("Historical real data (mock feed)", "mock")
        self.mode_combo.addItem("Public live API feed", "live")
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        self.bars_spin=QSpinBox(); self.bars_spin.setRange(4,2000); self.bars_spin.setValue(self.bars); self.bars_spin.setSuffix(" bars")
        self.speed_spin=QDoubleSpinBox(); self.speed_spin.setRange(.05,60); self.speed_spin.setSingleStep(.1)
        self.speed_spin.setDecimals(2); self.speed_spin.setValue(self.speed); self.speed_spin.setSuffix(" s/bar")
        self.fee_spin=QDoubleSpinBox(); self.fee_spin.setRange(0,.05); self.fee_spin.setSingleStep(.0001)
        self.fee_spin.setDecimals(4); self.fee_spin.setValue(self.fee); self.fee_spin.setPrefix("fee ")
        self.candidate_spin=QSpinBox(); self.candidate_spin.setRange(8,100000); self.candidate_spin.setSingleStep(64)
        self.candidate_spin.setValue(self.candidate_every); self.candidate_spin.setPrefix("candidate / ")
        self.start_button=QPushButton("Start system"); self.start_button.setObjectName("primaryButton"); self.start_button.clicked.connect(self.start_system)
        self.stop_button=QPushButton("Stop"); self.stop_button.clicked.connect(self.stop_system); self.stop_button.setEnabled(False)
        self.emergency_button=QPushButton("Emergency stop"); self.emergency_button.setObjectName("dangerButton"); self.emergency_button.clicked.connect(self.emergency_stop)
        controls.addWidget(QLabel("Feed")); controls.addWidget(self.mode_combo); controls.addWidget(self.bars_spin)
        controls.addWidget(self.speed_spin); controls.addWidget(self.fee_spin); controls.addWidget(self.candidate_spin); controls.addStretch()
        controls.addWidget(self.start_button); controls.addWidget(self.stop_button); controls.addWidget(self.emergency_button)
        outer.addLayout(controls)

        lower=QHBoxLayout(); lower.setSpacing(12)
        self.table=QTableWidget(0,7); self.table.setHorizontalHeaderLabels(["Time (UTC)","Symbol","Action","Confidence","SELL / HOLD / BUY","Value", "Model"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False); self.table.setAlternatingRowColors(True)
        lower.addWidget(self.table, 3)
        right=QVBoxLayout()
        training=QFrame(); training.setObjectName("panel"); form=QFormLayout(training)
        self.training_status=QLabel("Waiting for experience"); self.promotion_status=QLabel("No candidate evaluation yet")
        self.config_status=QLabel("Saved state will be restored"); self.config_status.setWordWrap(True)
        form.addRow("Learner",self.training_status); form.addRow("Candidate gate",self.promotion_status)
        form.addRow("Restart state",self.config_status)
        right.addWidget(training)
        self.live_checkbox=QCheckBox("Live orders OFF")
        self.live_checkbox.setObjectName("liveSwitch")
        self.live_checkbox.setChecked(False); self.live_checkbox.stateChanged.connect(self._live_toggle)
        right.addWidget(self.live_checkbox)
        note=QLabel("Orders default to paper only. Live mode needs your broker adapter and two explicit confirmations.")
        note.setWordWrap(True); note.setObjectName("smallNote"); right.addWidget(note)
        self.log_label=QLabel("Logs: logs/desktop-global-online.log"); self.log_label.setWordWrap(True); self.log_label.setObjectName("smallNote")
        right.addWidget(self.log_label); right.addStretch()
        lower.addLayout(right,1); outer.addLayout(lower,1)
        self.setCentralWidget(root)
        self.setStyleSheet("""
          QMainWindow,QWidget { background:#111820; color:#e7edf3; font-family:'Segoe UI'; font-size:10pt; }
          #appTitle { font-size:18pt; font-weight:700; letter-spacing:1px; }
          #statusPill { background:#20352c; color:#9ce2b5; border-radius:11px; padding:8px 14px; font-weight:600; }
          #statCard,#panel { background:#1a242f; border:1px solid #2d3b48; border-radius:9px; }
          #statTitle { color:#92a5b7; font-size:9pt; }
          #statValue { font-size:12pt; font-weight:650; color:#f2f6f9; }
          QPushButton,QComboBox,QSpinBox,QDoubleSpinBox { background:#263543; border:1px solid #415568; border-radius:6px; padding:7px 10px; }
          QPushButton:hover { background:#34495c; }
          #primaryButton { background:#287b61; font-weight:700; }
          #dangerButton { background:#853a43; font-weight:700; }
          #liveSwitch { background:#46282b; padding:10px; border-radius:7px; font-weight:700; }
          QTableWidget { background:#17212b; alternate-background-color:#1d2935; gridline-color:#2d3b48; border:1px solid #2d3b48; }
          QHeaderView::section { background:#263543; padding:8px; border:0; font-weight:600; }
          #smallNote { color:#9bacba; font-size:9pt; }
        """)

    def _restore_settings(self):
        index=self.mode_combo.findData(self.mode)
        if index>=0: self.mode_combo.setCurrentIndex(index)
        self.bars_spin.setValue(self.bars); self.speed_spin.setValue(self.speed); self.fee_spin.setValue(self.fee)
        self.candidate_spin.setValue(self.candidate_every)

    def _mode_changed(self, _index):
        self.mode=str(self.mode_combo.currentData()); self._save_settings()

    def _profile(self) -> tuple[Path,Path]:
        mode="mock" if self.mode=="mock" else "live"
        profile=self.runtime/mode; profile.mkdir(parents=True,exist_ok=True)
        data=profile/"market.csv"
        return data,profile/"agent"

    def _save_settings(self):
        self.mode=str(self.mode_combo.currentData()) if hasattr(self,"mode_combo") else self.mode
        self.bars=self.bars_spin.value() if hasattr(self,"bars_spin") else self.bars
        self.speed=self.speed_spin.value() if hasattr(self,"speed_spin") else self.speed
        self.fee=self.fee_spin.value() if hasattr(self,"fee_spin") else self.fee
        self.candidate_every=self.candidate_spin.value() if hasattr(self,"candidate_spin") else self.candidate_every
        self.settings={"mode":self.mode,"mock_bars":self.bars,"mock_interval_seconds":self.speed,"fee":self.fee,
                       "candidate_every":self.candidate_every,
                       "mock_source":str(self.source_data)}
        self.runtime.mkdir(parents=True,exist_ok=True)
        self.settings_path.write_text(json.dumps(self.settings,indent=2),encoding="utf-8")

    def _new_process(self,name:str,args:list[str]):
        proc=QProcess(self); proc.setWorkingDirectory(str(ROOT)); proc.setProgram(sys.executable); proc.setArguments(args)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(lambda p=proc,n=name:self._process_output(p,n))
        proc.finished.connect(lambda code,status,n=name:self._child_finished(n,code))
        proc.start(); return proc

    def _start_children(self):
        data,state=self._profile(); data.parent.mkdir(parents=True,exist_ok=True)
        (state/"stop.request").unlink(missing_ok=True); (data.parent/"feed.stop").unlink(missing_ok=True)
        if self.mode=="mock":
            args=["-u","-m","stockrl","mock-feed","--source",str((ROOT/self.source_data).resolve()),
                  "--output",str(data),"--bars",str(self.bars),"--interval-seconds",str(self.speed),
                  "--stop-file",str(data.parent/"feed.stop")]
            self.feed_proc=self._new_process("mock-feed",args)
        else:
            args=["-u","-m","stockrl","live-feed","--config",str(ROOT/"configs/live_symbols.json"),
                  "--output",str(data),"--poll-seconds","5","--stop-file",str(data.parent/"feed.stop")]
            self.feed_proc=self._new_process("live-feed",args)
        agent_args=["-u","-m","stockrl","global-online","--data",str(data),"--state-dir",str(state),
            "--follow","--poll-seconds","1","--initial-lookback-bars","128","--candidate-every",str(self.candidate_every),
            "--fee",str(self.fee),"--device",self.device]
        seed=ROOT/"runtime-global-verified/champion.pt"
        if seed.is_file() and not (state/"champion.pt").exists(): agent_args.extend(["--initial-champion",str(seed)])
        self.agent_proc=self._new_process("global-online",agent_args)
        self.config_status.setText(f"State: {state} / cursor, replay and champion saved automatically")

    def start_system(self):
        if self.run_requested: return
        self._save_settings(); self.run_requested=True; self._stopping=False; self._was_stopped=False
        self._start_children(); self.start_button.setEnabled(False); self.stop_button.setEnabled(True)
        for control in (self.mode_combo,self.bars_spin,self.speed_spin,self.fee_spin,self.candidate_spin): control.setEnabled(False)
        self.overall.setText("Observing market - paper mode")

    def _child_finished(self,name:str,code:int):
        if not self.run_requested: return
        if name=="mock-feed" and code==0:
            self.log_label.setText("Historical mock feed completed; the agent continues to process delayed outcomes and learn.")
            return
        self.log_label.setText(f"{name} exited ({code}); restarting from saved cursor and replay.")
        QTimer.singleShot(1200,lambda:self._restart_child(name))

    def _restart_child(self,name):
        if not self.run_requested: return
        data,state=self._profile()
        if name in ("mock-feed","live-feed"):
            if self.mode=="mock":
                args=["-u","-m","stockrl","mock-feed","--source",str((ROOT/self.source_data).resolve()),
                  "--output",str(data),"--bars",str(self.bars),"--interval-seconds",str(self.speed),
                  "--stop-file",str(data.parent/"feed.stop")]
                # Existing CSV dedup means a completed history restarts safely.
                self.feed_proc=self._new_process("mock-feed",args)
            else:
                args=["-u","-m","stockrl","live-feed","--config",str(ROOT/"configs/live_symbols.json"),
                      "--output",str(data),"--poll-seconds","5","--stop-file",str(data.parent/"feed.stop")]
                self.feed_proc=self._new_process("live-feed",args)
        else:
            args=["-u","-m","stockrl","global-online","--data",str(data),"--state-dir",str(state),
                  "--follow","--poll-seconds","1","--initial-lookback-bars","128","--candidate-every",str(self.candidate_every),
                  "--fee",str(self.fee),"--device",self.device]
            seed=ROOT/"runtime-global-verified/champion.pt"
            if seed.is_file() and not (state/"champion.pt").exists(): args.extend(["--initial-champion",str(seed)])
            self.agent_proc=self._new_process("global-online",args)

    def _process_output(self,proc,name):
        chunk=bytes(proc.readAllStandardOutput()).decode("utf-8",errors="replace").strip()
        if chunk:
            logdir=ROOT/"logs"; logdir.mkdir(exist_ok=True)
            with (logdir/f"desktop-{name}.log").open("a",encoding="utf-8") as f:
                f.write(chunk+"\n")

    def stop_system(self):
        if not self.run_requested and not getattr(self,"_stopping",False):
            self._save_settings(); return
        self.run_requested=False; self._stopping=True
        data,state=self._profile()
        state.mkdir(parents=True,exist_ok=True); data.parent.mkdir(parents=True,exist_ok=True)
        (state/"stop.request").write_text("stop",encoding="utf-8")
        (data.parent/"feed.stop").write_text("stop",encoding="utf-8")
        self.start_button.setEnabled(False); self.stop_button.setEnabled(False); self.overall.setText("Stopping safely - saving state")
        QTimer.singleShot(20000,self._force_shutdown)
        self._finish_shutdown_if_ready()
        self._save_settings()

    def _finish_shutdown_if_ready(self):
        if not self._stopping: return
        processes=[getattr(self,"agent_proc",None),getattr(self,"feed_proc",None)]
        if all(proc is None or proc.state()==QProcess.ProcessState.NotRunning for proc in processes):
            self._stopping=False; self.start_button.setEnabled(True); self.stop_button.setEnabled(False); self._was_stopped=True
            for control in (self.mode_combo,self.bars_spin,self.speed_spin,self.fee_spin,self.candidate_spin): control.setEnabled(True)
            self.overall.setText("Stopped - state saved")
            if self._close_when_stopped:
                self._close_when_stopped=False
                QTimer.singleShot(0,self.close)

    def _force_shutdown(self):
        if not self._stopping: return
        for proc in (getattr(self,"agent_proc",None),getattr(self,"feed_proc",None)):
            if proc and proc.state()!=QProcess.ProcessState.NotRunning: proc.kill()
        self._finish_shutdown_if_ready()

    def emergency_stop(self):
        self._disable_live_orders()
        self.stop_system()
        self.overall.setText("Emergency stop - live orders disabled")
        QMessageBox.warning(self,"Emergency stop","Live orders are OFF and market/model processes are stopping. Saved state is retained.")

    def _disable_live_orders(self):
        self.live_armed=False
        if self.broker_worker:
            self.broker_worker.emergency_stop(); self.broker_worker.close(); self.broker_worker=None
        self.order_adapter=None
        self.live_checkbox.blockSignals(True); self.live_checkbox.setChecked(False); self.live_checkbox.setText("Live orders OFF")
        self.live_checkbox.blockSignals(False)
        self.cards["order"].value.setText("OFF - paper only")

    def _live_toggle(self,state):
        if state!=Qt.CheckState.Checked.value:
            self._disable_live_orders(); return
        try:
            adapter=_load_broker_plugin()
        except Exception as exc:
            adapter=None; QMessageBox.critical(self,"Broker adapter error",str(exc))
        if adapter is None:
            QMessageBox.information(self,"Live broker is not connected",
                "No live broker API is bundled. Configure STOCKRL_LIVE_BROKER_ADAPTER=module:Class "
                "with a BrokerAdapter before enabling live orders.")
            self._disable_live_orders(); return
        first=QMessageBox.warning(self,"Live order confirmation 1/2",
            "This switch routes model BUY/SELL signals to your connected broker adapter. Paper results may differ from real fills. Continue?",
            QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,QMessageBox.StandardButton.No)
        if first!=QMessageBox.StandardButton.Yes:
            self._disable_live_orders(); return
        phrase,ok=QInputDialog.getText(self,"Live order confirmation 2/2","Type ENABLE LIVE ORDERS to continue:")
        if not ok or phrase.strip()!="ENABLE LIVE ORDERS":
            self._disable_live_orders(); return
        try:
            self.broker_worker=BrokerWorker(adapter,self.broker_results); self.broker_worker.start()
        except Exception as exc:
            try: adapter.close()
            except Exception: pass
            QMessageBox.critical(self,"Broker connection failed",str(exc)); self._disable_live_orders(); return
        self.order_adapter=adapter; self.live_armed=True
        _data,state=self._profile()
        decisions=state/"decisions.csv"; self.order_cursor=decisions.stat().st_size if decisions.exists() else 0
        self.live_checkbox.setText("Live orders ON - confirmed")

    def _dispatch_live_decisions(self):
        if not self.live_armed or not self.broker_worker: return
        data,state=self._profile(); path=state/"decisions.csv"
        if not path.exists(): return
        with path.open("rb") as f:
            f.seek(self.order_cursor); payload=f.read(); self.order_cursor=f.tell()
        for line in payload.decode("utf-8",errors="replace").splitlines():
            try: row=next(csv.DictReader(["date,symbol,action,value,p_sell,p_hold,p_buy",line]))
            except (StopIteration,ValueError): continue
            action=row.get("action","").upper()
            if action not in ("BUY","SELL"): continue
            supports=getattr(self.order_adapter,"supports_symbol",None)
            if not callable(supports) or not supports(row["symbol"]): continue
            size=getattr(self.order_adapter,"size_order",lambda _symbol,_side:0.0)(row["symbol"],action)
            if not isinstance(size,(int,float)) or size<=0: continue
            self.broker_worker.submit(OrderRequest(row["symbol"],action,float(size),row["date"]))
        try:
            while True:
                result=self.broker_results.get_nowait()
                if result.get("ok") and result.get("stage")=="connected":
                    self.cards["order"].value.setText("ON - broker connected")
                elif not result.get("ok"):
                    self.log_label.setText("Broker "+str(result.get("stage","operation"))+" error: "+result.get("error","unknown"))
                    self._disable_live_orders(); break
        except Empty: pass

    def _append_decision_rows(self, state:Path):
        path=state/"decisions.csv"
        if not path.exists(): return
        position=path.stat().st_size
        offset=getattr(self,"decision_offset",0)
        if position<offset: offset=0
        if not hasattr(self,"decision_seen"):
            self.decision_seen=deque(maxlen=80)
            with path.open("r",encoding="utf-8",errors="replace") as f:
                rows=deque(csv.DictReader(f),maxlen=80)
            for row in rows: self._insert_decision(row)
            self.decision_offset=position; return
        with path.open("rb") as f:
            f.seek(offset); new=f.read(); self.decision_offset=f.tell()
        lines=new.decode("utf-8",errors="replace").splitlines()
        for line in lines:
            try:
                values=next(csv.reader([line]))
                if len(values)!=7 or values[0]=="date": continue
                self._insert_decision(dict(zip(["date","symbol","action","value","p_sell","p_hold","p_buy"],values)))
            except (StopIteration,ValueError): continue
        self._dispatch_live_decisions()

    def _insert_decision(self,row):
        if not hasattr(self,"decision_seen"): self.decision_seen=deque(maxlen=80)
        self.decision_seen.append(row)
        rows=list(self.decision_seen); self.table.setRowCount(len(rows))
        names=["date","symbol","action","confidence","probabilities","value","version"]
        for i,item in enumerate(reversed(rows)):
            try:
                ps=float(item["p_sell"]); ph=float(item["p_hold"]); pb=float(item["p_buy"])
                prob=f"{ps:.2f} / {ph:.2f} / {pb:.2f}"; confidence=max(ps,ph,pb)
                vals=[item["date"],item["symbol"],item["action"],f"{confidence:.2f}",prob,
                      f"{float(item['value']):+.4f}","champion"]
            except (KeyError,ValueError): continue
            for j,val in enumerate(vals):
                cell=QTableWidgetItem(str(val)); cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if j==2:
                    color={"BUY":"#44c68a","SELL":"#ee7278","HOLD":"#c7b86b"}.get(val,"#e7edf3")
                    cell.setForeground(QColor(color))
                self.table.setItem(i,j,cell)

    def refresh_status(self):
        data,state=self._profile()
        feed_path=data.with_name("live_feed_metrics.json") if self.mode=="live" else data.with_name("mock_feed_metrics.json")
        feed=_read_json(feed_path); metrics=_read_json(state/"metrics.json")
        running=self.run_requested
        self.overall.setText("Observing market - paper mode" if running else ("Stopped - state saved" if self._was_stopped else "Idle"))
        feed_state=("Connected" if getattr(self,"feed_proc",None) and self.feed_proc.state()!=QProcess.ProcessState.NotRunning
                    else ("Playback complete" if self.mode=="mock" and feed else "Waiting / disconnected"))
        self.cards["feed"].value.setText(feed_state+f"\n{feed.get('appended_unique_rows',0):,} rows appended")
        agent_running=bool(getattr(self,"agent_proc",None) and self.agent_proc.state()!=QProcess.ProcessState.NotRunning)
        model_status="Running" if agent_running and metrics.get("observations") else ("Starting" if agent_running else "Stopped")
        self.cards["model"].value.setText(model_status+f"\n{metrics.get('parameters',0):,} params")
        champion=state/"champion.pt"
        champion_version=datetime.fromtimestamp(champion.stat().st_mtime,timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC") if champion.exists() else "Waiting for initialization"
        self.cards["champion"].value.setText(f"Promotions {metrics.get('promotions',0)}\n{champion_version}")
        self.cards["replay"].value.setText(f"{metrics.get('replay_count',0):,}")
        self.cards["reward"].value.setText(f"{100*float(metrics.get('paper_net_reward',0)):+.3f}%")
        self.cards["last"].value.setText(str(metrics.get("last_market_timestamp") or "-"))
        self.cards["learn"].value.setText(str(metrics.get("last_update_utc") or "Waiting for update"))
        self.cards["candidate"].value.setText("Training" if metrics.get("candidate_training") else
            f"Idle - {metrics.get('updates',0)} updates")
        used=metrics.get("cuda_memory_allocated_bytes",0); total=metrics.get("cuda_total_memory_bytes",0)
        gpu=metrics.get("cuda_device",metrics.get("device","-"))
        if total:
            gpu_text=f"{gpu}\n{used/1e9:.2f} / {total/1e9:.2f} GB" if agent_running else f"{gpu}\nIdle - peak {metrics.get('cuda_peak_allocated_bytes',0)/1e9:.2f} GB"
        else: gpu_text=str(gpu)
        self.cards["gpu"].value.setText(gpu_text)
        positions=_read_json(state/"live_positions.json")
        held=sum(1 for v in positions.values() if v)
        self.cards["portfolio"].value.setText(f"{held} active / {len(positions)} tracked")
        self.cards["validation"].value.setText(f"Promote {metrics.get('promotions',0)} / reject {metrics.get('rejections',0)}")
        self.cards["order"].value.setText("ON - broker connected" if self.live_armed else "OFF - paper only")
        self._finish_shutdown_if_ready()
        self.training_status.setText(("Candidate training" if metrics.get("candidate_training") else "Accumulating")+
                                     f" ? {metrics.get('replay_count',0)} replay")
        self.promotion_status.setText(f"{metrics.get('last_candidate_promoted','-')} - {metrics.get('last_update_utc') or 'No update yet'}")
        self._append_decision_rows(state)

    def closeEvent(self,event:QCloseEvent):
        if self.run_requested or self._stopping:
            event.ignore(); self._close_when_stopped=True
            self._disable_live_orders(); self.stop_system(); return
        self._disable_live_orders(); self._save_settings(); event.accept()


def run_desktop_app(runtime="runtime-global-desktop",device="auto"):
    app=QApplication.instance() or QApplication(sys.argv)
    window=GlobalAgentWindow(runtime,device); window.show()
    return app.exec()
