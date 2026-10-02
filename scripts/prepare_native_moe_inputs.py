"""Download a small real ITCH prefix and retain official native ETH features."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from stockrl.expert_backends import source_module


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True);args=p.parse_args()
    root=args.root;path=root/"native_data/MarketGPT";path.mkdir(parents=True,exist_ok=True)
    target=path/"AAPL-20191230-native.json"
    if target.exists(): print(str(target));return
    url="https://huggingface.co/datasets/aaronwheeler/MarketGPT-datasets/resolve/main/12302019.NASDAQ_ITCH50_AAPL_message_proc.npy"
    rows=1512178;count=8;offset=128
    def column(j):
        start=offset+j*rows*8;end=start+count*8-1
        response=requests.get(url+f"?column={j}",headers={"Range":f"bytes={start}-{end}"},timeout=60)
        response.raise_for_status()
        if response.status_code!=206 or not response.headers.get("Content-Range","").startswith(f"bytes {start}-{end}/"):
            raise ValueError("server did not return the requested native column range")
        return np.frombuffer(response.content,dtype="<i8").copy()
    with ThreadPoolExecutor(max_workers=6) as pool:messages=np.column_stack(list(pool.map(column,range(18))))
    native=source_module("native_itch_encoding",root/"sources/MarketGPT/equities/data_processing/itch_encoding.py")
    tokens=native.encode_msgs(messages,native.Vocab().ENCODING).reshape(-1)
    if tokens.min()<0 or tokens.max()>=12160:raise ValueError("native vocabulary mismatch")
    np.save(path/"AAPL-20191230-real-messages.npy",messages)
    data={"symbols":["AAPL"],"as_of":"2019-12-30T"+str(np.timedelta64(int(messages[-1,10]),"s")).split(' ')[0],
        "data_date":"2019-12-30","itch_tokens":[tokens[-32:].tolist()],"token_schema":"MarketGPT_ITCH_Vocab_v3",
        "native_features_verified":True,"input_authenticity":"real_official_MarketGPT_preprocessed_ITCH",
        "source_url":url,"messages_sha256":hashlib.sha256(messages.tobytes()).hexdigest(),"horizon":1,"sampling_seconds":1}
    import pandas as pd
    data["as_of"]=str(pd.Timestamp("2019-12-30")+pd.Timedelta(seconds=int(messages[-1,10]),nanoseconds=int(messages[-1,11])))
    target.write_text(json.dumps(data),encoding="utf-8");print(json.dumps(data))


if __name__=="__main__":main()
