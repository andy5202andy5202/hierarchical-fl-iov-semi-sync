import pickle
import numpy as np
import os

OUTPUT_PATH = "./cifar_noniid_9groups_split"   # 輸出資料夾

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

print("檢查每個輸出檔案的 labels 分佈：")

for i in range(9):  # g0 ~ g8
    filename = os.path.join(OUTPUT_PATH, f"g{i}_train.pkl")
    content = load_pickle(filename)
    labels = np.array(content['labels'], dtype=np.int64)  
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"g{i}_train.pkl → labels: {unique_labels}, counts: {counts}")
