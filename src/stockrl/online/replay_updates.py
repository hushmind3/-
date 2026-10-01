"""Install replay maintenance and DB counters at a saved learner boundary."""
import ast
import time
from pathlib import Path
from .. import replay_store


def install_replay_updates(replay):
    path = Path(replay_store.__file__)
    stamp = path.stat().st_mtime_ns
    if getattr(replay, "_performance_code_stamp", None) == stamp:
        return
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GlobalReplayBuffer")
    names = {"_ensure_performance_indexes", "acknowledge_training", "compact", "_metadata", "stats"}
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(vars(replay_store))
    namespace["time"] = time
    # Compile as a class so staticmethod descriptors are retained.
    staged_class = ast.ClassDef(name="ReplayUpdates", bases=[], keywords=[], body=methods, decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[staged_class], type_ignores=[])), str(path), "exec"), namespace)
    with replay.lock:
        for name in names:
            setattr(replay_store.GlobalReplayBuffer, name, vars(namespace["ReplayUpdates"])[name])
        if replay.journal_path:
            from contextlib import closing
            with closing(replay._connect()) as db, db:
                replay._ensure_performance_indexes(db)
        replay._performance_code_stamp = stamp
