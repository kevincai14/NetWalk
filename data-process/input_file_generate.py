import pandas as pd
import os

# ========== 用户配置部分 ==========
INPUT_FILE = "shipping_sample_0.001.xlsx"   # 或者 "voyages.csv"
OUTPUT_DIR = "monthly_edges"  # 输出文件夹
TIME_COLUMN = "summary_time"  # 表示时间的列名
START_PORT = "leg_start_port_code"  # 起始港列
END_PORT = "leg_end_port_code"      # 到达港列
WEIGHT_COLUMN = "dwt"               # 权重列，可换成 "stay_dwt"、"teu"等
# =================================


def main():
    # === 1. 读取数据 ===
    print("📘 正在读取数据...")
    if INPUT_FILE.endswith(".xlsx"):
        df = pd.read_excel(INPUT_FILE)
    else:
        df = pd.read_csv(INPUT_FILE, sep="\t", encoding="utf-8")

    print(f"读取到 {len(df):,} 条记录。")

    # === 2. 检查关键字段是否存在 ===
    required_cols = [START_PORT, END_PORT, WEIGHT_COLUMN, TIME_COLUMN]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ 缺少必要字段: {col}")

    # === 3. 生成月份字段 ===
    print("📅 生成月份字段...")
    df["month"] = pd.to_datetime(df[TIME_COLUMN], errors="coerce").dt.to_period("M").astype(str)
    df = df.dropna(subset=["month"])

    # === 4. 只保留必要字段 ===
    df_edges = df[[START_PORT, END_PORT, WEIGHT_COLUMN, "month"]].copy()
    df_edges.columns = ["u", "v", "weight", "month"]

    # === 5. 按月聚合：同月同航线合并 ===
    print("⚙️ 按月聚合航线权重...")
    df_edges = (
        df_edges.groupby(["u", "v", "month"], as_index=False)
        .agg({"weight": "sum"})
    )

    # === 6. 输出每月文件 ===
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    months = sorted(df_edges["month"].unique())
    print(f"🗓 共检测到 {len(months)} 个月数据：{months}")

    for month in months:
        df_month = df_edges[df_edges["month"] == month][["u", "v", "weight"]]
        output_path = os.path.join(OUTPUT_DIR, f"edges_{month}.csv")
        df_month.to_csv(output_path, index=False)
        print(f"✅ 已输出: {output_path} ({len(df_month)} 条边)")

    # === 7. 输出总表 ===
    all_path = os.path.join(OUTPUT_DIR, "edges_all.csv")
    df_edges.to_csv(all_path, index=False)
    print(f"📂 已输出总表: {all_path}")

    print("\n🎯 数据处理完成！可直接输入 NetWalk_update 使用。")
    print(f"示例: NetWalk_update('{OUTPUT_DIR}/', walk_per_node=5, walk_len=3, init_months=1)")


if __name__ == "__main__":
    main()
