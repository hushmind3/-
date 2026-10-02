"""Native runtime state extraction, using the original MacroHFT reset code."""
import ast
from types import SimpleNamespace
import numpy as np
from torch import nn


class MacroHFTInputAdapter(nn.Module):
    def __init__(self,single_features,trend_features,reset_source):
        super().__init__()
        if len(single_features)!=36 or len(trend_features)!=9:raise ValueError("native MacroHFT feature schema differs")
        self.single_features,self.trend_features=single_features,trend_features
        namespace={"back_time_length":1}
        exec(compile(ast.parse(reset_source),"<original MacroHFT state extraction>","exec"),namespace)
        self.native_reset=namespace["reset"]

    def forward(self,frame,index,previous_action):
        if not set(self.single_features+self.trend_features+["timestamp","close"]).issubset(frame.columns):
            raise ValueError("feed lacks native MacroHFT 36+9 fields; use native feature dataset")
        if previous_action not in (0,1):raise ValueError("native prior action must be flat=0 or long=1")
        env=SimpleNamespace(df=frame.iloc[index:index+1],tech_indicator_list=self.single_features,
            tech_indicator_list_trend=self.trend_features,initial_action=previous_action,max_holding_number=.01,stack_length=1)
        single,trend,info=self.native_reset(env)
        if not np.isfinite(single.astype(float)).all() or not np.isfinite(trend.astype(float)).all():raise ValueError("nonfinite native state")
        return {"symbols":["ETHUSDT"],"as_of":str(frame.iloc[index].timestamp),"single_state":single.tolist(),
            "trend_state":trend.tolist(),"previous_action":[info["previous_action"]],"horizon":1,"sampling_seconds":60,
            "native_features_verified":True,"feature_schema":"MacroHFT_36+9","input_authenticity":"real_official_MacroHFT_ETHUSDT_features"}


def macro_adapter_metadata(root):
    from pathlib import Path
    path=Path(root)/"sources/MacroHFT"
    tree=ast.parse((path/"env/low_level_env.py").read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="Testing_Env")
    reset=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="reset")
    return {"single_features":np.load(path/"data/feature_list/single_features.npy",allow_pickle=True).tolist(),
        "trend_features":np.load(path/"data/feature_list/trend_features.npy",allow_pickle=True).tolist(),"reset_source":ast.unparse(reset)}
