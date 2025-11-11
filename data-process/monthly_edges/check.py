import os
import pandas as pd

folder = "C:/Users/Quan/GitHub/NetWalk-test/data-process/monthly_edges/"
for f in sorted(os.listdir(folder)):
    if f.endswith(".csv"):
        df = pd.read_csv(os.path.join(folder, f))
        print(f, df.shape, df.columns.tolist())
