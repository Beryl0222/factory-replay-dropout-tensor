"""测试用密集参考实现：物化完整分数/概率/掩码，仅用于与流式算子核对。"""
from __future__ import annotations

import numpy as np

from window_attention import Layout


def reference_attention(
    layout: Layout,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    dy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 (output, dq, dk, dv)，float64 参考精度。"""
    nq, nqh, d = q.shape
    nk, nkh = k.shape[0], k.shape[1]
    scale = 1.0 / np.sqrt(d)
    out = np.zeros_like(q)
    dq = np.zeros_like(q)
    dk = np.zeros_like(k)
    dv = np.zeros_like(v)
    num, den = layout.keep
    r_drop = den / num if num > 0 else 0.0
    all_k = list(range(nk))

    for qi in range(nq):
        legal = layout.legal_pairs([qi], all_k)[0]
        for h in range(nqh):
            kh = layout.head_map[h]
            scores = q[qi, h, :] @ k[:, kh, :].T * scale
            if not legal.any():
                # 全屏蔽查询：零输出、无梯度
                continue
            safe = np.where(legal, scores, -np.inf)
            m = safe.max()
            shifted = np.where(legal, scores - m, 0.0)  # 避免非法位置 exp(inf)
            a = np.where(legal, np.exp(shifted), 0.0)
            a = a / a.sum()
            words = layout.dropout_words([h], [qi], all_k)[0, 0]
            kept = layout.keep_mask(words)
            p = np.where(legal & kept, a * r_drop, 0.0)
            out[qi, h, :] = p @ v[:, kh, :]

            g_logit = dy[qi, h, :] @ v[:, kh, :].T
            pg = p * g_logit
            ds = pg - a * pg.sum()
            dv[:, kh, :] += p[:, None] * dy[qi, h, :][None, :]
            dk[:, kh, :] += ds[:, None] * q[qi, h, :][None, :]
            dq[qi, h, :] += ds @ k[:, kh, :]
    return out, dq, dk, dv
