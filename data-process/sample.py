import pandas as pd

# 读取原始数据
df = pd.read_excel("shipping_sample.xlsx")

# 随机抽取 1% 的样本
sample_df = df.sample(frac=0.1, random_state=42)

# 保存为新文件
sample_df.to_excel("shipping_sample_0.001.xlsx", index=False)

print(f"原始数据 {len(df)} 行，抽样后 {len(sample_df)} 行。")
