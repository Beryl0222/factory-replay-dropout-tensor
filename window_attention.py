"""跨分片可重放的缩放点积注意力训练算子（纯 Python/numpy 实现）。

设计目标
========
查询、键、值与上游梯度均以模型导出的 *逻辑 token 编号* 标识；前向按键值
分片流式计算，只保存预算内的训练凭据（每行 softmax 行最大值、dropout 前
归一化分母与全屏蔽标记），不物化每个查询到键的完整分数、概率或掩码。反向
按分片消费凭据，校验同一步的数据、逻辑身份、掩码契约与随机身份；支持重放
幂等、过期撤销与缺席不提交。

公开入口
========
- :class:`Layout`               逻辑布局（会话、位置、允许集、头映射、分片）
- :class:`AttentionEngine`      前向 / 反向 / 撤销 / 状态查询
- :class:`GradientDelivery`     反向交付结果（dQ/dK/dV、可提交范围、撤销原因）

数值契约：softmax 与分片归并使用不低于 float64 的精度，输出及各梯度最大绝对
误差 1e-8。dropout 作用于归一化概率，保留项按 ``den/num`` 反比例缩放，随机
身份为 SHA-256(``step|sample|head|qid|kid``) 前 8 字节大端无符号字，按
``[0, 2^64)`` 阈值比较。

工作区计账（遵循交接资料）：调用方只读输入与已消费输出不计入，引擎内新分配
的解包、转换、归约、版本暂存及交接缓冲计入 ``max_workspace_bytes``；持久训练
凭据（m、l、全屏蔽标记与指纹）计入 ``max_receipt_bytes``。作为反向交付物的
梯度缓冲只保留一份并随交付移交，不重复物化。
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "Layout",
    "AttentionEngine",
    "GradientDelivery",
    "WorkspaceLimitError",
    "ReceiptLimitError",
    "ContractError",
    "StaleShardError",
]

_U64 = 1 << 64


# --------------------------------------------------------------------------- #
# 预算与契约异常
# --------------------------------------------------------------------------- #
class WorkspaceLimitError(MemoryError):
    """临时工作区超过制造端 max_workspace_bytes。"""


class ReceiptLimitError(MemoryError):
    """持久训练凭据超过制造端 max_receipt_bytes。"""


class ContractError(ValueError):
    """数据、逻辑身份、掩码契约或随机身份与同一步凭据不一致。"""


class StaleShardError(ContractError):
    """反向分片来自过期版本，受影响查询归属已保留并撤销其梯度交付。"""


# --------------------------------------------------------------------------- #
# 逻辑布局
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Layout:
    """逻辑 token 布局。所有索引均为逻辑行号（与数组行一一对应）。

    参数
    ----
    query_ids, key_ids:
        逻辑 token ID 序列，顺序与 q/k/v/dy 数组行一致。
    query_session, key_session:
        每个查询/键的会话 ID（不同会话互不可见）。
    query_pos, key_pos:
        会话内排序位置。
    allow:
        显式允许范围，``{query_id: key_id 序列}``；未列出的查询表示除窗口
        与因果约束外不再额外限制。
    head_map:
        按查询头给出键值头的映射（多个查询头可共享同一逻辑 K/V 头）。
    left:
        窗口左界宽度，可见键位置满足 ``q_pos - left <= k_pos <= q_pos``
        （因果上界即键位置不晚于查询）。
    step, sample:
        训练步号与样本名，参与随机身份绑定。
    keep:
        ``(num, den)`` 保留概率，dropout 阈值为 ``2^64 * num / den``。
    forward_groups, backward_groups:
        前向 / 反向键值分片，每片是 key_id 的序列，组内顺序即到达顺序。
        前向各组的并集必须恰好覆盖全部键；反向允许重送（重复组）。
    """

    query_ids: tuple[str, ...]
    key_ids: tuple[str, ...]
    query_session: tuple[str, ...]
    key_session: tuple[str, ...]
    query_pos: tuple[int, ...]
    key_pos: tuple[int, ...]
    allow: dict[str, tuple[str, ...]]
    head_map: tuple[int, ...]
    left: int
    step: int
    sample: str
    keep: tuple[int, int]
    forward_groups: tuple[tuple[str, ...], ...]
    backward_groups: tuple[tuple[str, ...], ...]

    q_index: dict[str, int] = field(init=False, compare=False)
    k_index: dict[str, int] = field(init=False, compare=False)
    nq: int = field(init=False, compare=False)
    nk: int = field(init=False, compare=False)
    nqh: int = field(init=False, compare=False)
    nkh: int = field(init=False, compare=False)

    def __post_init__(self) -> None:
        q_index = {tid: i for i, tid in enumerate(self.query_ids)}
        k_index = {tid: i for i, tid in enumerate(self.key_ids)}
        if len(q_index) != len(self.query_ids):
            raise ValueError("query 逻辑 ID 重复")
        if len(k_index) != len(self.key_ids):
            raise ValueError("key 逻辑 ID 重复")
        num, den = self.keep
        if not (0 <= num <= den and den > 0):
            raise ValueError("keep 必须满足 0 <= num <= den 且 den > 0")
        if self.left < 0:
            raise ValueError("left 必须非负")
        seen: set[str] = set()
        for grp in self.forward_groups:
            for kid in grp:
                if kid not in k_index:
                    raise ValueError(f"前向分片含未知 key: {kid}")
                seen.add(kid)
        if seen != set(self.key_ids):
            raise ValueError("前向分片的并集必须恰好覆盖全部逻辑键")
        for grp in self.backward_groups:
            for kid in grp:
                if kid not in k_index:
                    raise ValueError(f"反向分片含未知 key: {kid}")
        for qid, kids in self.allow.items():
            if qid not in q_index:
                raise ValueError(f"allow 含未知 query: {qid}")
            for kid in kids:
                if kid not in k_index:
                    raise ValueError(f"allow 含未知 key: {kid}")
        object.__setattr__(self, "q_index", q_index)
        object.__setattr__(self, "k_index", k_index)
        object.__setattr__(self, "nq", len(self.query_ids))
        object.__setattr__(self, "nk", len(self.key_ids))
        object.__setattr__(self, "nqh", len(self.head_map))
        object.__setattr__(self, "nkh", int(max(self.head_map)) + 1)

    @classmethod
    def from_records(cls, records: dict) -> "Layout":
        """从采集资料的 records 字段构建布局。"""
        queries = [(qid, sess, int(pos)) for qid, sess, pos in records["queries"]]
        keys = [(kid, sess, int(pos)) for kid, sess, pos in records["keys"]]
        return cls(
            query_ids=tuple(q[0] for q in queries),
            key_ids=tuple(k[0] for k in keys),
            query_session=tuple(q[1] for q in queries),
            key_session=tuple(k[1] for k in keys),
            query_pos=tuple(q[2] for q in queries),
            key_pos=tuple(k[2] for k in keys),
            allow={qid: tuple(kids) for qid, kids in records["allow"].items()},
            head_map=tuple(int(h) for h in records["head_map"]),
            left=int(records["left"]),
            step=int(records["step"]),
            sample=str(records["sample"]),
            keep=(int(records["keep"][0]), int(records["keep"][1])),
            forward_groups=tuple(tuple(g) for g in records["forward_groups"]),
            backward_groups=tuple(tuple(g) for g in records["backward_groups"]),
        )

    def with_groups(
        self,
        forward_groups: Iterable[Iterable[str]],
        backward_groups: Iterable[Iterable[str]],
    ) -> "Layout":
        """换分片布局 / 到达顺序得到新布局，逻辑掩码契约保持不变。"""
        return Layout(
            query_ids=self.query_ids,
            key_ids=self.key_ids,
            query_session=self.query_session,
            key_session=self.key_session,
            query_pos=self.query_pos,
            key_pos=self.key_pos,
            allow=dict(self.allow),
            head_map=self.head_map,
            left=self.left,
            step=self.step,
            sample=self.sample,
            keep=self.keep,
            forward_groups=tuple(tuple(g) for g in forward_groups),
            backward_groups=tuple(tuple(g) for g in backward_groups),
        )

    # -- 掩码与随机身份 -----------------------------------------------------
    def legal_pairs(self, q_rows: Sequence[int], k_rows: Sequence[int]) -> np.ndarray:
        """返回 ``[len(q_rows), len(k_rows)]`` bool 合法配对矩阵。

        合法 = 同会话 且 位置在 ``[q_pos-left, q_pos]`` 且落在显式允许集。
        全部按逻辑身份计算，与分片布局、到达顺序无关。
        """
        qsess = np.asarray([self.query_session[i] for i in q_rows])
        ksess = np.asarray([self.key_session[j] for j in k_rows])
        qpos = np.asarray([self.query_pos[i] for i in q_rows], dtype=np.int64)
        kpos = np.asarray([self.key_pos[j] for j in k_rows], dtype=np.int64)
        same_session = qsess[:, None] == ksess[None, :]
        window = (kpos[None, :] >= qpos[:, None] - self.left) & (
            kpos[None, :] <= qpos[:, None]
        )
        allowed = np.ones((len(q_rows), len(k_rows)), dtype=bool)
        for li, qi in enumerate(q_rows):
            kids = self.allow.get(self.query_ids[qi])
            if kids is not None:
                permit = set(kids)
                allowed[li] = [self.key_ids[j] in permit for j in k_rows]
        return same_session & window & allowed

    def dropout_words(
        self, heads: Sequence[int], q_rows: Sequence[int], k_rows: Sequence[int]
    ) -> np.ndarray:
        """返回 ``[len(heads), len(q_rows), len(k_rows)]`` uint64 随机身份字。

        身份 = SHA-256(``step|sample|head|qid|kid``) 前 8 字节大端无符号字。
        纯逻辑身份决定，重放 / 换布局 / 换到达顺序结果不变。
        """
        words = np.empty((len(heads), len(q_rows), len(k_rows)), dtype=np.uint64)
        for hi, head in enumerate(heads):
            for li, qi in enumerate(q_rows):
                qid = self.query_ids[qi]
                for lj, kj in enumerate(k_rows):
                    payload = "|".join(
                        (str(self.step), self.sample, str(head), qid, self.key_ids[kj])
                    )
                    digest = hashlib.sha256(payload.encode("utf-8")).digest()
                    words[hi, li, lj] = int.from_bytes(digest[:8], "big")
        return words

    def keep_mask(self, words: np.ndarray) -> np.ndarray:
        """按 ``[0, 2^64)`` 阈值把随机字转成保留布尔掩码（无浮点参与）。"""
        num, den = self.keep
        if num == 0:
            return np.zeros(words.shape, dtype=bool)
        if num == den:
            return np.ones(words.shape, dtype=bool)
        threshold = (_U64 * num) // den
        return words < np.uint64(threshold)


# --------------------------------------------------------------------------- #
# 持久训练凭据（受 max_receipt_bytes 约束，不含完整分数/概率/掩码）
# --------------------------------------------------------------------------- #
@dataclass
class _Receipt:
    step: int
    sample: str
    layout_fingerprint: tuple
    tensor_fingerprint: tuple
    head_width: int
    scale: float
    row_max: np.ndarray            # [nq, nqh] float64，全屏蔽行存 0
    row_denom: np.ndarray          # [nq, nqh] float64，dropout 前分母 l
    empty_rows: np.ndarray         # [nq, nqh] bool，全屏蔽查询
    consumed_keys: set[str]
    # 反向阶段：每个 (查询行, 查询头) 一个标量 g_sum = Σ_键 p·(dy·V)，
    # 随分片在线累积；只是每查询行的归约标量，不是逐键分数/概率/掩码。
    g_sum: np.ndarray | None = None  # [nq, nqh] float64，finalize 前可逐片增长
    g_sum_seen: set[str] = field(default_factory=set)
    backward_seen: set[str] = field(default_factory=set)
    finalized: bool = False
    revoked_queries: set[str] = field(default_factory=set)
    revoke_reasons: list[str] = field(default_factory=list)

    def nbytes(self) -> int:
        total = (
            self.row_max.nbytes
            + self.row_denom.nbytes
            + self.empty_rows.nbytes
        )
        if self.g_sum is not None:
            total += self.g_sum.nbytes
        return total


# --------------------------------------------------------------------------- #
# 反向交付
# --------------------------------------------------------------------------- #
@dataclass
class GradientDelivery:
    """反向一次会话的梯度交付，按逻辑行组织。"""

    step: int
    sample: str
    dq: np.ndarray                  # [nq, nqh, d]，按逻辑查询行
    dk: np.ndarray                  # [nk, nkh, d]，按逻辑键行
    dv: np.ndarray                  # [nk, nkh, d]
    committed: bool
    """是否为可提交的完整总和。缺席 / 撤销 / 过期时为 ``False``。"""
    committed_key_ids: tuple[str, ...]
    """已并入且有效的逻辑键范围；未提交时仅供排查，不得提交训练器。"""
    committed_query_ids: tuple[str, ...]
    """可提交覆盖的逻辑查询行；未提交时为空元组。"""
    revoked_query_ids: tuple[str, ...]
    revoke_reasons: tuple[str, ...]
    missing_key_ids: tuple[str, ...]
    """完成全部反向组后仍缺席的逻辑键。"""

    def as_logical_rows(self) -> dict[str, np.ndarray]:
        return {"dq": self.dq, "dk": self.dk, "dv": self.dv}


# --------------------------------------------------------------------------- #
# 引擎
# --------------------------------------------------------------------------- #
class AttentionEngine:
    """流式前向 + 可校验反向的注意力算子。

    Parameters
    ----------
    layout:
        逻辑布局（含分片定义与随机身份绑定要素）。
    q, k, v:
        ``q [nq, nqh, d]``、``k/v [nk, nkh, d]``，按逻辑行排列，float64。
    dy:
        上游梯度，与输出同形。
    max_workspace_bytes, max_receipt_bytes:
        制造端工作区 / 凭据预算。
    q_block:
        查询分块大小；``None`` 时按工作区预算自动取最大可行块。
    """

    def __init__(
        self,
        layout: Layout,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        dy: np.ndarray,
        max_workspace_bytes: int,
        max_receipt_bytes: int,
        q_block: int | None = None,
    ) -> None:
        self.layout = layout
        # ascontiguousarray 在已是 float64 C 连续时不产生副本（只读输入不计账）
        self.q = np.ascontiguousarray(q, dtype=np.float64)
        self.k = np.ascontiguousarray(k, dtype=np.float64)
        self.v = np.ascontiguousarray(v, dtype=np.float64)
        self.dy = np.ascontiguousarray(dy, dtype=np.float64)
        self._check_shapes()
        self.max_workspace_bytes = int(max_workspace_bytes)
        self.max_receipt_bytes = int(max_receipt_bytes)
        self.d = self.q.shape[2]
        self.scale = 1.0 / math.sqrt(self.d)
        self.q_block = self._choose_q_block(q_block)
        self._receipt: _Receipt | None = None
        self._dk: np.ndarray | None = None
        self._dv: np.ndarray | None = None
        self._dq: np.ndarray | None = None

    def _check_shapes(self) -> None:
        lay = self.layout
        d = self.q.shape[2]
        if self.q.shape != (lay.nq, lay.nqh, d):
            raise ValueError("q 形状与布局不一致")
        if self.k.shape != (lay.nk, lay.nkh, d):
            raise ValueError("k 形状与布局不一致")
        if self.v.shape != self.k.shape:
            raise ValueError("v 与 k 形状不一致")
        if self.dy.shape != self.q.shape:
            raise ValueError("dy 必须与 q/输出同形")
        if any(h < 0 or h >= lay.nkh for h in lay.head_map):
            raise ValueError("head_map 越界")

    def _choose_q_block(self, q_block: int | None) -> int:
        """单块峰值约 6 个 f64 [b,g] 临时量加 4 个 [b/g,d] 归约缓冲。"""
        g = self.layout.nk
        per_row = 6 * g * 8 + 4 * self.d * 8
        budget = max(self.max_workspace_bytes, per_row)
        auto = max(1, budget // per_row)
        if q_block is None:
            return min(auto, self.layout.nq)
        if q_block <= 0:
            raise ValueError("q_block 必须为正")
        return q_block

    def _check_workspace(self, nbytes: int) -> None:
        if nbytes > self.max_workspace_bytes:
            raise WorkspaceLimitError(
                f"临时工作区 {nbytes} 字节超过预算 {self.max_workspace_bytes}"
            )

    # ------------------------------------------------------------------ #
    # 指纹：数据本身、逻辑身份、掩码契约、随机身份绑定到同一步。
    # 分片布局刻意不进指纹——同一步换布局必须仍然匹配。
    # ------------------------------------------------------------------ #
    def _layout_fingerprint(self) -> tuple:
        lay = self.layout
        return (
            lay.step,
            lay.sample,
            lay.query_ids,
            lay.key_ids,
            lay.query_session,
            lay.key_session,
            lay.query_pos,
            lay.key_pos,
            tuple(sorted((q, ks) for q, ks in lay.allow.items())),
            lay.head_map,
            lay.left,
            lay.keep,
        )

    def _tensor_fingerprint(self) -> tuple:
        return (
            self.q.shape,
            self.k.shape,
            hashlib.sha256(self.q.tobytes()).hexdigest(),
            hashlib.sha256(self.k.tobytes()).hexdigest(),
            hashlib.sha256(self.v.tobytes()).hexdigest(),
            hashlib.sha256(self.dy.tobytes()).hexdigest(),
        )

    def _verify_contract(self, receipt: _Receipt) -> None:
        if receipt.step != self.layout.step or receipt.sample != self.layout.sample:
            raise ContractError("训练步/样本与凭据不匹配")
        if receipt.layout_fingerprint != self._layout_fingerprint():
            raise ContractError("逻辑身份或掩码契约与凭据不属于同一步")
        if receipt.tensor_fingerprint != self._tensor_fingerprint():
            raise ContractError("q/k/v/dy 数据本身与凭据不属于同一步")

    # ------------------------------------------------------------------ #
    # 前向：在线 softmax 流式合并，凭据只留 m、l、空行标记
    # ------------------------------------------------------------------ #
    def forward(self) -> np.ndarray:
        """流式前向，返回按逻辑查询行组织的注意力输出（调用方持有）。"""
        lay = self.layout
        nq, nqh, d = self.q.shape
        row_max = np.full((nq, nqh), -np.inf, dtype=np.float64)
        row_denom = np.zeros((nq, nqh), dtype=np.float64)
        output = np.zeros((nq, nqh, d), dtype=np.float64)
        consumed: set[str] = set()
        r_drop = lay.keep[1] / lay.keep[0] if lay.keep[0] > 0 else 0.0

        for group in lay.forward_groups:
            k_rows = [lay.k_index[kid] for kid in group]
            kk = self.k[k_rows]
            vv = self.v[k_rows]
            g = len(k_rows)
            for start in range(0, nq, self.q_block):
                q_rows = list(range(start, min(start + self.q_block, nq)))
                b = len(q_rows)
                self._check_workspace(6 * b * g * 8 + 4 * max(b, g) * d * 8)
                legal = lay.legal_pairs(q_rows, k_rows)
                qq = self.q[q_rows]
                for h in range(nqh):
                    kh = lay.head_map[h]
                    scores = np.matmul(qq[:, h, :], kk[:, kh, :].T) * self.scale
                    words = lay.dropout_words([h], q_rows, k_rows)[0]
                    kept = lay.keep_mask(words)
                    effective = legal & kept

                    legal_scores = np.where(legal, scores, -np.inf)
                    block_max = legal_scores.max(axis=1)          # 合法项最大值
                    has_legal = np.isfinite(block_max)
                    prev_m = row_max[q_rows, h]
                    has_prev = np.isfinite(prev_m)
                    new_m = np.maximum(
                        np.where(has_prev, prev_m, -np.inf), block_max
                    )
                    has_any = np.isfinite(new_m)
                    safe_new = np.where(has_any, new_m, 0.0)
                    f_prev = np.where(
                        has_prev, np.exp(prev_m - safe_new), 0.0
                    )
                    # dropout 前归一化分母：只数合法项，统一以新全局最大值 new_m
                    # 为基准（块自身的 block_max 可能低于 new_m，不能直接用）
                    shifted = np.where(legal, scores - safe_new[:, None], 0.0)
                    block_l = np.where(legal, np.exp(shifted), 0.0).sum(axis=1)
                    new_l = (
                        np.where(has_prev, row_denom[q_rows, h], 0.0) * f_prev
                        + block_l
                    )
                    # 输出累积：保留项按 den/num 反比例缩放，同样以 new_m 为基准
                    block_w = np.where(
                        effective, np.exp(shifted) * r_drop, 0.0
                    )
                    block_out = np.matmul(block_w, vv[:, kh, :])
                    output[q_rows, h, :] = (
                        output[q_rows, h, :] * f_prev[:, None]
                        + block_out
                    )
                    row_max[q_rows, h] = new_m
                    row_denom[q_rows, h] = new_l
            consumed.update(group)

        finite = np.isfinite(row_max)
        empty_rows = ~finite
        safe_l = np.where(row_denom > 0, row_denom, 1.0)
        output = output / safe_l[:, :, None]
        output = np.where(empty_rows[:, :, None], 0.0, output)
        row_max = np.where(finite, row_max, 0.0)

        receipt = _Receipt(
            step=lay.step,
            sample=lay.sample,
            layout_fingerprint=self._layout_fingerprint(),
            tensor_fingerprint=self._tensor_fingerprint(),
            head_width=d,
            scale=self.scale,
            row_max=row_max,
            row_denom=row_denom,
            empty_rows=empty_rows,
            consumed_keys=consumed,
        )
        if receipt.nbytes() > self.max_receipt_bytes:
            raise ReceiptLimitError(
                f"训练凭据 {receipt.nbytes()} 字节超过预算 {self.max_receipt_bytes}"
            )
        self._receipt = receipt
        return output

    # ------------------------------------------------------------------ #
    # 反向
    # ------------------------------------------------------------------ #
    def _init_backward(self, receipt: _Receipt) -> None:
        # 梯度缓冲是反向交付物本身，只此一份并随 delivery 移交；此处仅做
        # 容量自检（与单个输出张量同量级的暂存约定）。
        nq, nqh, d = self.q.shape
        nk, nkh = self.layout.nk, self.layout.nkh
        staging = max(nq * nqh, nk * nkh) * d * 8
        self._check_workspace(staging)
        self._dq = np.zeros((nq, nqh, d), dtype=np.float64)
        self._dk = np.zeros((nk, nkh, d), dtype=np.float64)
        self._dv = np.zeros((nk, nkh, d), dtype=np.float64)
        receipt.g_sum = np.zeros((nq, nqh), dtype=np.float64)
        if receipt.nbytes() > self.max_receipt_bytes:
            raise ReceiptLimitError(
                f"训练凭据 {receipt.nbytes()} 字节超过预算 {self.max_receipt_bytes}"
            )

    def _affected_queries(self, k_rows: Sequence[int]) -> tuple[str, ...]:
        """与该片任一键存在合法配对的逻辑查询（撤销时保留归属）。"""
        lay = self.layout
        affected: list[str] = []
        for start in range(0, lay.nq, self.q_block):
            q_rows = list(range(start, min(start + self.q_block, lay.nq)))
            legal = lay.legal_pairs(q_rows, k_rows)
            for li, qi in enumerate(q_rows):
                if legal[li].any():
                    affected.append(lay.query_ids[qi])
        return tuple(sorted(affected))

    def _shard_probabilities(
        self,
        receipt: _Receipt,
        h: int,
        kh: int,
        q_rows: Sequence[int],
        k_rows: Sequence[int],
        legal: np.ndarray,
        r_drop: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """重建本片的 dropout 前 softmax ``a`` 与 dropout 后概率 ``p``。

        只用凭据里的行最大值 m、行分母 l 与逻辑随机身份，逐片即用即弃。
        """
        lay = self.layout
        qq = self.q[q_rows]
        kk = self.k[k_rows]
        scores = np.matmul(qq[:, h, :], kk[:, kh, :].T) * self.scale
        words = lay.dropout_words([h], q_rows, k_rows)[0]
        kept = lay.keep_mask(words)
        m = receipt.row_max[q_rows, h]
        l = receipt.row_denom[q_rows, h]
        empty = receipt.empty_rows[q_rows, h]
        safe_l = np.where(l > 0, l, 1.0)
        shifted = np.where(
            legal & ~empty[:, None], scores - m[:, None], 0.0
        )
        a = np.where(
            legal & ~empty[:, None],
            np.exp(shifted) / safe_l[:, None],
            0.0,
        )
        p = np.where(
            legal & kept & ~empty[:, None], a * r_drop, 0.0
        )
        return a, p

    def backward_shard(
        self,
        group: Sequence[str],
        *,
        version_step: int | None = None,
    ) -> None:
        """消费一个反向键值分片（两遍式反向的第一遍）。

        本遍只做两件事，且都幂等：
        1. 在线累积每个 (查询行, 查询头) 的标量 ``g_sum = Σ_键 p·(dy·V)``；
        2. 登记该片已到达。不写 dQ/dK/dV，不保存任何逐键量。

        - 重送同一分片（或其已处理子集）不会重复累积。
        - ``version_step`` 与凭据训练步不一致即判过期：保留受影响查询归属、
          登记撤销原因并抛出 :class:`StaleShardError`，该片不写入任何状态。
        全部合法分片到达后调用 :meth:`finalize_backward` 做第二遍梯度计算。
        """
        receipt = self._require_receipt()
        self._verify_contract(receipt)
        if receipt.finalized:
            raise ContractError("反向已 finalize，不能再消费新分片")
        lay = self.layout
        group = tuple(group)

        if version_step is not None and version_step != receipt.step:
            k_rows = [lay.k_index[kid] for kid in group]
            affected = self._affected_queries(k_rows)
            receipt.revoked_queries.update(affected)
            reason = (
                f"过期分片 {list(group)}：分片版本 step={version_step} 与凭据 "
                f"step={receipt.step} 不符，受影响查询 {list(affected)} 的"
                "梯度交付撤销"
            )
            receipt.revoke_reasons.append(reason)
            raise StaleShardError(reason)

        fresh = [kid for kid in group if kid not in receipt.g_sum_seen]
        if not fresh:
            return  # 整组重送：幂等，无重复累积
        if receipt.g_sum is None:
            self._init_backward(receipt)

        k_rows = [lay.k_index[kid] for kid in fresh]
        vv = self.v[k_rows]
        g = len(k_rows)
        nq, nqh, d = self.q.shape
        r_drop = lay.keep[1] / lay.keep[0] if lay.keep[0] > 0 else 0.0

        for start in range(0, nq, self.q_block):
            q_rows = list(range(start, min(start + self.q_block, nq)))
            b = len(q_rows)
            self._check_workspace(4 * b * g * 8 + 2 * max(b, g) * d * 8)
            legal = lay.legal_pairs(q_rows, k_rows)
            dyq = self.dy[q_rows]
            for h in range(nqh):
                kh = lay.head_map[h]
                _, p = self._shard_probabilities(
                    receipt, h, kh, q_rows, k_rows, legal, r_drop
                )
                g_logit = np.matmul(dyq[:, h, :], vv[:, kh, :].T)
                receipt.g_sum[q_rows, h] += (p * g_logit).sum(axis=1)
        receipt.g_sum_seen.update(fresh)
        receipt.backward_seen.update(fresh)

    def finalize_backward(self) -> GradientDelivery:
        """第二遍：用完整的全局标量 ``g_sum`` 计算 dQ/dK/dV。

        重跑所有已到达分片（纯函数式重建 p，不产生新的随机状态），按
        ``ds = p·g − a·g_sum`` 计算正确的 softmax 梯度。共享逻辑 K/V 的
        多个查询头对同一键值头的贡献在同一份 dK/dV 上只做一次总和。
        任一必要分片缺席或存在撤销时，``committed=False``。
        """
        receipt = self._require_receipt()
        self._verify_contract(receipt)
        if receipt.finalized:
            raise ContractError("反向已 finalize，请勿重复结算")
        lay = self.layout
        if self._dk is None:
            self._init_backward(receipt)
        nq, nqh, d = self.q.shape
        r_drop = lay.keep[1] / lay.keep[0] if lay.keep[0] > 0 else 0.0
        # 以去重后的已到达键组成结算分片（重送不影响结果）
        seen_order = [kid for kid in lay.key_ids if kid in receipt.backward_seen]

        for cstart in range(0, len(seen_order), self.q_block):
            chunk = seen_order[cstart : cstart + self.q_block]
            k_rows = [lay.k_index[kid] for kid in chunk]
            kk = self.k[k_rows]
            vv = self.v[k_rows]
            g = len(k_rows)
            for start in range(0, nq, self.q_block):
                q_rows = list(range(start, min(start + self.q_block, nq)))
                b = len(q_rows)
                self._check_workspace(5 * b * g * 8 + 3 * max(b, g) * d * 8)
                legal = lay.legal_pairs(q_rows, k_rows)
                qq = self.q[q_rows]
                dyq = self.dy[q_rows]
                live = np.asarray(
                    [
                        lay.query_ids[qi] not in receipt.revoked_queries
                        for qi in q_rows
                    ]
                )
                for h in range(nqh):
                    kh = lay.head_map[h]
                    a, p = self._shard_probabilities(
                        receipt, h, kh, q_rows, k_rows, legal, r_drop
                    )
                    g_logit = np.matmul(dyq[:, h, :], vv[:, kh, :].T)
                    # 全局 softmax 梯度：减去的是跨全部键的 g_sum 标量
                    ds = p * g_logit - a * receipt.g_sum[q_rows, h][:, None]
                    ds = np.where(live[:, None], ds, 0.0)
                    self._dv[k_rows, kh, :] += np.matmul(p.T, dyq[:, h, :])
                    self._dk[k_rows, kh, :] += np.matmul(ds.T, qq[:, h, :])
                    self._dq[q_rows, h, :] += np.matmul(ds, kk[:, kh, :])
        receipt.finalized = True
        return self.delivery()

    def backward_all(self) -> GradientDelivery:
        """消费全部反向分组（重送自动幂等）后结算并返回交付判定。"""
        for group in self.layout.backward_groups:
            self.backward_shard(group)
        return self.finalize_backward()

    def revoke(self, query_ids: Iterable[str], reason: str) -> None:
        """显式撤销指定逻辑查询的梯度归属（归属信息保留在凭据中）。

        必须在 :meth:`finalize_backward` 之前调用；结算后撤销会破坏已算出
        的 dQ，因此直接拒绝，避免交出无法解释的梯度。
        """
        receipt = self._require_receipt()
        if receipt.finalized:
            raise ContractError("反向已 finalize，撤销请求必须在结算前提出")
        qids = tuple(query_ids)
        for qid in qids:
            if qid not in self.layout.q_index:
                raise ContractError(f"未知查询: {qid}")
        receipt.revoked_queries.update(qids)
        receipt.revoke_reasons.append(f"显式撤销 {list(qids)}：{reason}")
        if self._dq is not None:
            for qid in qids:
                self._dq[self.layout.q_index[qid], :, :] = 0.0

    def delivery(self) -> GradientDelivery:
        """汇总当前反向状态并做出可提交判定（不触发任何计算）。"""
        receipt = self._require_receipt()
        lay = self.layout
        if self._dk is None:
            # 反向尚未开始：没有任何可提交内容
            dq = np.zeros_like(self.q)
            dk = np.zeros_like(self.k)
            dv = np.zeros_like(self.v)
        else:
            dq, dk, dv = self._dq, self._dk, self._dv
        missing = tuple(
            kid for kid in lay.key_ids if kid not in receipt.backward_seen
        )
        committed_keys = tuple(
            kid for kid in lay.key_ids if kid in receipt.backward_seen
        )
        revoked = tuple(sorted(receipt.revoked_queries))
        live_queries = tuple(
            qid for qid in lay.query_ids if qid not in receipt.revoked_queries
        )
        # 任一必要贡献缺席、任一撤销/过期、或未完成结算：不得提交总和
        committed = (
            receipt.finalized
            and len(missing) == 0
            and len(revoked) == 0
            and not receipt.revoke_reasons
        )
        dq = dq.copy()
        for qid in revoked:
            dq[lay.q_index[qid], :, :] = 0.0
        return GradientDelivery(
            step=lay.step,
            sample=lay.sample,
            dq=dq,
            dk=dk.copy(),
            dv=dv.copy(),
            committed=committed,
            committed_key_ids=committed_keys if committed else (),
            committed_query_ids=live_queries if committed else (),
            revoked_query_ids=revoked,
            revoke_reasons=tuple(receipt.revoke_reasons),
            missing_key_ids=missing,
        )

    def status(self) -> dict:
        """核对训练步、可提交梯度范围与撤销请求原因。"""
        receipt = self._require_receipt()
        lay = self.layout
        return {
            "step": lay.step,
            "sample": lay.sample,
            "forward_complete": receipt.consumed_keys == set(lay.key_ids),
            "backward_seen_keys": tuple(sorted(receipt.backward_seen)),
            "committed_key_range": tuple(
                kid for kid in lay.key_ids if kid in receipt.backward_seen
            ),
            "revoked_queries": tuple(sorted(receipt.revoked_queries)),
            "revoke_reasons": tuple(receipt.revoke_reasons),
            "receipt_bytes": receipt.nbytes(),
            "workspace_budget_bytes": self.max_workspace_bytes,
            "receipt_budget_bytes": self.max_receipt_bytes,
        }

    def _require_receipt(self) -> _Receipt:
        if self._receipt is None:
            raise ContractError("前向凭据不存在，必须先完成 forward()")
        return self._receipt
