import json
from collections import defaultdict

import pandas as pd
from tqdm import tqdm


# jsonl file v.s. datalist
def load_jsonl(path):
    datalist = list()
    with open(path, 'rt') as f:
        for line in tqdm(f.readlines()):
            line_data = json.loads(line.strip())
            datalist.append(line_data)
    return datalist


def to_jsonl(datalist, path, mode='wt'):
    with open(path, mode) as f:
        for line_data in datalist:
            f.write(json.dumps(line_data) + '\n')


# dataframe v.s. datalist
def dataframe_to_datalist(dataframe):
    datalist = list()
    for _, row in dataframe.iterrows():
        datalist.append(row.to_dict())
    return datalist


def datalist_to_dataframe(datalist, colnames=None):
    frame_data = defaultdict(list)
    for line_data in datalist:
        if colnames:
            for k in colnames:
                frame_data[k].append(line_data.get(k, float('nan')))
        else:
            for k in line_data:
                frame_data[k].append(line_data[k])
    try:
        return pd.DataFrame(data=frame_data)
    except:
        return datalist_to_dataframe(datalist, colnames=list(frame_data.keys()))


def csv_to_datalist(csv_path):
    df = pd.read_csv(csv_path)
    datalist = dataframe_to_datalist(df)
    return datalist

def datalist_to_csv(datalist, csv_path):
    df = datalist_to_dataframe(datalist)
    df.to_csv(csv_path, index=False)

def datalist_to_csv(datalist, csv_path):
    df = datalist_to_dataframe(datalist)
    df.to_csv(csv_path)


def csv_to_jsonl(csv_path, jsonl_path):
    datalist = csv_to_datalist(csv_path)
    to_jsonl(datalist, jsonl_path)
