# 跨分片重放的注意力训练算子

同一训练步的两份分区资料来自合成窗口检测模型。 数据为本地合成。

- `data/capture.json` 保存采集原始内容。records 是交换记录，tensors 字段分开描述名称、类型、形状和值。整数索引从零开始，未指定闭合的区间均为左闭右开。
- `data/capacity.json` 保存制造端的容量、规模或分片条件。大尺寸是接口需考虑的使用规模。工作区计入新分配的解包、转换、归约、版本暂存及交接缓冲。调用方已有只读输入和已消费的输出不计入工作区，额外物化副本计入。

## 交换约定

`q` 轴为查询身份、查询头、特征，`k,v` 轴为键值身份、键值头、特征。`dy` 与输出同形。`head_map` 按查询头给键值头。查询和键的 `pos` 为会话内排序位置，逻辑 ID 与数组行按交接表对应。
可见键满足同会话且位置在 [q_pos-left, q_pos]，并落在该查询的允许键 ID 集。分数除以特征维平方根，softmax 与归并使用不低于 float64 的参考精度比较，输出及各梯度最大绝对误差 1e-8。
随机身份用于兼容训练方的种子格式。把训练步、样本名、查询头号、查询 ID 和键 ID 分别写成十进制或原 UTF-8 字符串，以字符 `|` 连接，用 SHA-256 的开头 8 个字节大端解出无符号字，按 [0, 2^64) 阈值比较。`keep` 的分子分母提供保留概率。

## 读取资料

使用 Python 3.11。首次创建本地环境时安装锁定依赖。只读查看显示原始字段和张量尺寸。

```sh
python -m pip install -r requirements.txt
python caseio.py data/capture.json
python -m unittest discover -s tests -v
```

现有检查核对采集格式和所附数据的结构。

## 训练算子 `window_attention.py`

`window_attention.py` 是可直接在训练项目内复用的纯 Python/numpy 缩放点积
注意力前后向算子，全部以逻辑 token 编号驱动，换键值分片布局、分片边界或到达
顺序都不改变同一步的掩码、dropout 与数值结果。

### 接口

- `Layout.from_records(records)`：从采集记录构建逻辑布局（会话、位置、显式
  允许集、`head_map`、窗口 `left`、训练步/样本、`keep`、前后向分片）。
  `layout.with_groups(...)` 在不改任何逻辑契约的前提下换分片布局。
- `AttentionEngine(layout, q, k, v, dy, max_workspace_bytes,
  max_receipt_bytes)`：`q/dy` 形状 `[nq, nqh, d]`，`k/v` 形状
  `[nk, nkh, d]`，float64。
- `forward()`：按键值分片流式 online-softmax，返回按逻辑查询行组织的输出；
  持久凭据只保存每行的 softmax 行最大值 `m`、dropout 前归一化分母 `l`、
  全屏蔽标记，以及反向阶段每个 (查询行, 查询头) 一个标量 `g_sum`。完整
  分数、概率、掩码从不落地。
- `backward_shard(group, version_step=...)`：反向第一遍，幂等地累积标量
  `g_sum` 并登记到达；重送已处理分片不会重复累积。版本号与训练步不符即判
  过期，保留受影响查询归属、登记原因并抛出 `StaleShardError`。
- `finalize_backward()`：全部合法分片到达后的第二遍，用全局 `g_sum` 计算
  正确的 softmax 梯度，返回 `GradientDelivery`。
- `revoke(query_ids, reason)`：结算前显式撤销查询归属。
- `delivery()` / `status()`：核对训练步、可提交的键/查询范围、缺席键与
  撤销原因。

### 交付与拒绝规则

- 全屏蔽查询（含 `keep=0` 全 dropout）输出为零且无梯度。
- 多个查询头经 `head_map` 共享同一逻辑 K/V 时，对同一键值头的贡献只交一次
  总和；任一必要键贡献缺席、存在过期分片或显式撤销时
  `GradientDelivery.committed=False`，`committed_*_ids` 为空，绝不用部分和
  冒充完整总和。
- 数据指纹（q/k/v/dy 哈希）、逻辑身份指纹（会话/位置/允许集/头映射/窗口/
  keep）与随机身份（SHA-256 逻辑五元组）在反向每个分片消费前核对，不属于
  同一步即 `ContractError`。分片布局刻意不进指纹。

### 预算计账

工作区只计入引擎内新分配的临时解包、转换、归约与版本暂存；调用方只读输入、
已消费输出以及随交付移交的单份梯度缓冲不计入。持久凭据计入
`max_receipt_bytes`。容量资料的极值规模（1536×2048、8/2 头、宽 32）下，
自动查询分块把单块峰值压在 3 MiB 工作区内，凭据约 300 KiB，低于 512 KiB。

```sh
python -m unittest discover -s tests -v
```

`tests/reference.py` 是仅供测试核对的密集参考实现（物化完整分数/概率），
`tests/test_window_attention.py` 覆盖与参考实现的前后向一致性、换布局不变性、
dropout 身份确定性、重送幂等、过期撤销、缺席不提交、契约不匹配与双预算强制。
