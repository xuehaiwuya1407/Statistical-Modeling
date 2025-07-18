import matplotlib.pyplot as plt
import networkx as nx
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
# 创建一个有向图
G = nx.DiGraph()

# 添加节点
nodes = [
    "原始数据集", "数据清洗", "特征选择", "数据归一化", "数据划分", "序列构造",
    "LSTM 模块", "GRU 模块", "注意力机制模块", "特征融合模块",
    "损失计算模块", "优化器模块", "训练循环模块", "训练数据", "训练输出"
]

for node in nodes:
    G.add_node(node)

# 添加边（流程）
edges = [
    ("原始数据集", "数据清洗"),
    ("数据清洗", "特征选择"),
    ("特征选择", "数据归一化"),
    ("数据归一化", "数据划分"),
    ("数据划分", "序列构造"),
    ("序列构造", "LSTM 模块"),
    ("序列构造", "GRU 模块"),
    ("LSTM 模块", "注意力机制模块"),
    ("GRU 模块", "注意力机制模块"),
    ("注意力机制模块", "特征融合模块"),
    ("特征融合模块", "训练输出"),
    ("训练数据", "LSTM 模块"),
    ("训练数据", "GRU 模块"),
    ("训练输出", "损失计算模块"),
    ("损失计算模块", "优化器模块"),
    ("优化器模块", "训练循环模块"),
    ("训练循环模块", "LSTM 模块"),
    ("训练循环模块", "GRU 模块")
]

for edge in edges:
    G.add_edge(edge[0], edge[1])

# 绘制图形
pos = nx.spring_layout(G)  # 使用 spring 布局
plt.figure(figsize=(20, 10))
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=3000)
nx.draw_networkx_edges(G, pos, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
plt.title('LSTM-GRU混合注意力模型流程示意图', fontsize=16)
plt.axis('off')
plt.show()