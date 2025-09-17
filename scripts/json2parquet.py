import pandas as pd

# 读取 JSON 文件
df = pd.read_json("/mnt/bn/wxd-video-understanding/wangxd/data/Kwai-Klear/KlearReasoner-MathSub-30K/train_math_30K.json", lines=True)  # 如果你的 JSON 是一行一个对象，就加上 lines=True

# 转换并保存为 Parquet 文件
df.to_parquet("/mnt/bn/wxd-video-understanding/wangxd/data/Kwai-Klear/KlearReasoner-MathSub-30K/train_math_30K.parquet", engine="pyarrow", index=False)
