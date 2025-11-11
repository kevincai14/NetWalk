"""
NetWalk.py - 主程序
支持有向图 + 权重 + 月度快照输入
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import matplotlib.pyplot as plt

from framework.imports import *
import framework.Model as MD

from datetime import datetime
import tensorflow as tf
import numpy as np
import warnings

from framework.netwalk_update import NetWalk_update  # 这里用你改好的版本

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def print_time():
    return datetime.now().strftime('[INFO %Y-%m-%d %H:%M:%S]')


# =====================================================
# 主流程：加载数据、训练初始 embedding、按月更新
# =====================================================
def static_process(args):

    # ---------- STEP 0: 参数 ----------
    hidden_size = args.representation_size
    activation = tf.nn.sigmoid

    rho = 0.5           # 稀疏率
    lamb = 0.0017       # 权重衰减
    beta = 1            # 稀疏惩罚项
    gama = 340          # 自编码器权重
    walk_len = args.walk_length
    epoch = 400
    batch_size = 20
    learning_rate = 0.1
    optimizer = "rmsprop"
    corrupt_prob = [0]

    # ---------- STEP 1: 数据准备 ----------
    data_path = args.input
    netwalk = NetWalk_update(
        data_path,
        walk_per_node=args.number_walks,
        walk_len=args.walk_length,
        init_percent=args.init_percent,
        snap=args.snap,
        seed=args.seed
    )

    n = len(netwalk.vertices)
    print(f"{print_time()} 总节点数: {n}")
    print(f"{print_time()} 每个节点随机游走次数: {args.number_walks}")
    print(f"{print_time()} 游走长度: {args.walk_length}")

    dimension = [n, hidden_size]

    embModel = MD.Model(
        activation, dimension, walk_len, n, gama, lamb, beta, rho,
        epoch, batch_size, learning_rate, optimizer, corrupt_prob
    )

    # ---------- STEP 2: 初始训练 ----------
    print(f"{print_time()} === 初始化阶段 ===")
    init_walks = netwalk.getInitWalk()
    embedding_code(embModel, init_walks, n, args)
    print(f"{print_time()} 初始 embedding 计算完成")

    # ---------- STEP 3: 动态快照更新 ----------
    snapshotNum = 0
    while netwalk.hasNext():
        print(f"{print_time()} === 更新第 {snapshotNum + 1} 月快照 ===")
        snapshot_data = netwalk.nextOnehotWalks()
        embedding_code(embModel, snapshot_data, n, args)
        snapshotNum += 1
        print(f"{print_time()} 已处理至第 {snapshotNum} 月")

    print(f"{print_time()} 所有快照处理完毕 ✅")


# =====================================================
# 生成并保存 embedding
# =====================================================
def embedding_code(embModel, walks, n, args):
    # 训练模型
    embModel.fit(walks)

    # 获取 embedding
    embeddings = embModel.feedforward_autoencoder(walks)

    # 保存 embedding
    import os
    output_path = args.output
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.savetxt(output_path, embeddings, fmt="%g")
    print(f"Embeddings saved to {output_path}")




# =====================================================
# 主入口
# =====================================================
def main():
    parser = ArgumentParser("NETWALK", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

    parser.add_argument('--format', default='csv', help='输入文件格式')
    parser.add_argument('--snap', default=12, type=int, help='快照数量（默认12个月）')
    parser.add_argument('--init_percent', default=0.1, type=float, help='初始快照比例（例如0.1代表第1个月）')
    parser.add_argument('--input', required=True, help='输入数据文件路径（csv，含month列）')
    parser.add_argument('--output', default='./embedding.txt', help='输出embedding保存路径')

    parser.add_argument('--number_walks', default=5, type=int, help='每节点游走次数')
    parser.add_argument('--representation-size', default=16, type=int, help='节点嵌入维度')
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    parser.add_argument('--walk-length', default=5, type=int, help='随机游走长度')

    args = parser.parse_args()

    logging.basicConfig(format=LOGFORMAT)
    logger.setLevel(logging.INFO)

    static_process(args)


if __name__ == "__main__":
    import sys
    sys.argv = [
        "NetWalk.py",
        "--input", "C:/Users/Quan/GitHub/NetWalk-test/data-process/monthly_edges/",
        "--output", "results/embedding.txt",
        "--number_walks", "5",
        "--walk-length", "3",
        "--representation-size", "2",
        "--snap", "12"
    ]
    main()
