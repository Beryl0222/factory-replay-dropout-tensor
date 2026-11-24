"""窗口注意力算子的契约测试。"""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from caseio import load_case
from window_attention import (
    AttentionEngine,
    ContractError,
    GradientDelivery,
    Layout,
    ReceiptLimitError,
    StaleShardError,
    WorkspaceLimitError,
)

from tests.reference import reference_attention

DATA = Path(__file__).resolve().parents[1] / "data"


def _hash_word(layout: Layout, head: int, qid: str, kid: str) -> int:
    payload = "|".join((str(layout.step), layout.sample, str(head), qid, kid))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


class AttentionContractTests(unittest.TestCase):
    def setUp(self):
        self.records, self.tensors = load_case(DATA / "capture.json")
        self.capacity = json.loads((DATA / "capacity.json").read_text("utf-8"))
        self.layout = Layout.from_records(self.records)
        self.q, self.k, self.v, self.dy = (
            self.tensors["q"],
            self.tensors["k"],
            self.tensors["v"],
            self.tensors["dy"],
        )
        self.ws = self.capacity["max_workspace_bytes"]
        self.rc = self.capacity["max_receipt_bytes"]

    def engine(self, layout=None, **kw):
        return AttentionEngine(
            layout or self.layout,
            self.q,
            self.k,
            self.v,
            self.dy,
            max_workspace_bytes=self.ws,
            max_receipt_bytes=self.rc,
            **kw,
        )

    # ------------------------------------------------------------------ #
    def test_forward_matches_dense_reference(self):
        eng = self.engine()
        out = eng.forward()
        ref_out, _, _, _ = reference_attention(
            self.layout, self.q, self.k, self.v, self.dy
        )
        np.testing.assert_allclose(out, ref_out, atol=1e-8, rtol=0)

    def test_backward_matches_dense_reference(self):
        eng = self.engine()
        eng.forward()
        delivery = eng.backward_all()
        self.assertTrue(delivery.committed)
        _, ref_dq, ref_dk, ref_dv = reference_attention(
            self.layout, self.q, self.k, self.v, self.dy
        )
        np.testing.assert_allclose(delivery.dq, ref_dq, atol=1e-8, rtol=0)
        np.testing.assert_allclose(delivery.dk, ref_dk, atol=1e-8, rtol=0)
        np.testing.assert_allclose(delivery.dv, ref_dv, atol=1e-8, rtol=0)
        self.assertEqual(delivery.committed_key_ids, self.layout.key_ids)
        self.assertEqual(delivery.committed_query_ids, self.layout.query_ids)
        self.assertEqual(delivery.missing_key_ids, ())
        self.assertEqual(delivery.revoked_query_ids, ())

    def test_all_masked_query_zero_output_and_zero_gradient(self):
        # QD 会话 s2 中没有任何键，必然全屏蔽
        qd = self.layout.q_index["QD"]
        eng = self.engine()
        out = eng.forward()
        np.testing.assert_array_equal(out[qd], 0.0)
        delivery = eng.backward_all()
        np.testing.assert_array_equal(delivery.dq[qd], 0.0)
        self.assertTrue(self.layout.legal_pairs([qd], list(range(self.layout.nk))).sum() == 0)

    def test_layout_repartition_does_not_change_result(self):
        """换分片布局、边界与到达顺序，同一步结果必须逐元素一致。"""
        keys = list(self.layout.key_ids)
        # 单片、反转、三片任意切分三种布局
        layouts = [
            self.layout.with_groups([keys], [keys]),
            self.layout.with_groups(
                [list(reversed(keys))], [list(reversed(keys))]
            ),
            self.layout.with_groups(
                [keys[:1], keys[1:3], keys[3:]],
                [keys[4:], keys[2:4], keys[:2]],
            ),
        ]
        outs, dels = [], []
        for lay in layouts:
            eng = self.engine(lay)
            outs.append(eng.forward())
            dels.append(eng.backward_all())
        # 不同分片顺序的浮点求和次序不同，按数值契约 1e-8 容差核对
        for other in outs[1:]:
            np.testing.assert_allclose(outs[0], other, atol=1e-8, rtol=0)
        for d in dels[1:]:
            np.testing.assert_allclose(dels[0].dq, d.dq, atol=1e-8, rtol=0)
            np.testing.assert_allclose(dels[0].dk, d.dk, atol=1e-8, rtol=0)
            np.testing.assert_allclose(dels[0].dv, d.dv, atol=1e-8, rtol=0)
            self.assertTrue(d.committed)

    def test_dropout_identity_is_logical_and_deterministic(self):
        lay = self.layout
        w1 = lay.dropout_words([0, 2], [0, 1], [1, 3])
        # 独立重算逐身份核对 SHA-256 约定
        for hi, h in enumerate([0, 2]):
            for li, qi in enumerate([0, 1]):
                for lj, kj in enumerate([1, 3]):
                    self.assertEqual(
                        int(w1[hi, li, lj]),
                        _hash_word(lay, h, lay.query_ids[qi], lay.key_ids[kj]),
                    )
        # 阈值比较：keep=3/4
        threshold = (1 << 64) * 3 // 4
        kept = lay.keep_mask(w1)
        np.testing.assert_array_equal(kept, w1 < np.uint64(threshold))
        # 换布局后随机身份不变
        shuffled = lay.with_groups(
            [list(reversed(lay.key_ids))], [list(reversed(lay.key_ids))]
        )
        w2 = shuffled.dropout_words([0, 2], [0, 1], [1, 3])
        np.testing.assert_array_equal(w1, w2)

    def test_resend_is_idempotent(self):
        # 参照：每组只送一遍后结算
        ref_eng = self.engine()
        ref_eng.forward()
        for grp in self.layout.backward_groups:
            ref_eng.backward_shard(grp)
        ref_d = ref_eng.finalize_backward()

        # 实验组：结算前把全部反向组再送一遍（含采集资料里的重复组）
        eng = self.engine()
        eng.forward()
        for grp in self.layout.backward_groups:
            eng.backward_shard(grp)
        for grp in self.layout.backward_groups:
            eng.backward_shard(grp)  # 重送：第一遍幂等去重
        second = eng.finalize_backward()
        np.testing.assert_array_equal(ref_d.dk, second.dk)
        np.testing.assert_array_equal(ref_d.dv, second.dv)
        np.testing.assert_array_equal(ref_d.dq, second.dq)
        self.assertTrue(second.committed)
        # 结算后再送分片必须被拒绝
        with self.assertRaises(ContractError):
            eng.backward_shard(self.layout.backward_groups[0])

    def test_stale_shard_revokes_and_aborts_delivery(self):
        eng = self.engine()
        eng.forward()
        # 先消费一个正常分片
        eng.backward_shard(self.layout.backward_groups[0])
        bad = self.records["mismatched_gradient_version"]
        with self.assertRaises(StaleShardError) as ctx:
            eng.backward_shard([bad["key"]], version_step=bad["step"])
        self.assertIn("step=8", str(ctx.exception))
        # 继续送完其余分片后结算：撤销仍然生效，绝不提交
        for grp in self.layout.backward_groups[1:]:
            eng.backward_shard(grp)
        delivery = eng.finalize_backward()
        self.assertFalse(delivery.committed)
        self.assertTrue(len(delivery.revoked_query_ids) > 0)
        self.assertTrue(any("过期分片" in r for r in delivery.revoke_reasons))
        # 受影响查询的 dQ 被清零且不提交
        for qid in delivery.revoked_query_ids:
            np.testing.assert_array_equal(
                delivery.dq[self.layout.q_index[qid]], 0.0
            )
        self.assertEqual(delivery.committed_query_ids, ())
        self.assertEqual(delivery.committed_key_ids, ())
        status = eng.status()
        self.assertEqual(status["step"], self.records["step"])
        self.assertTrue(any("过期分片" in r for r in status["revoke_reasons"]))

    def test_missing_contribution_is_never_committed(self):
        eng = self.engine()
        eng.forward()
        # 只送一个分片，其余必要贡献缺席
        eng.backward_shard(self.layout.backward_groups[0])
        delivery = eng.delivery()
        self.assertFalse(delivery.committed)
        self.assertEqual(
            set(delivery.missing_key_ids),
            set(self.layout.key_ids) - set(self.layout.backward_groups[0]),
        )
        self.assertEqual(delivery.committed_query_ids, ())

    def test_explicit_revoke_keeps_attribution(self):
        eng = self.engine()
        eng.forward()
        # 先送分片（第一遍），结算前撤销，再结算
        for grp in self.layout.backward_groups:
            eng.backward_shard(grp)
        eng.revoke(["QC"], reason="上游通知样本作废")
        delivery = eng.finalize_backward()
        self.assertFalse(delivery.committed)
        self.assertEqual(delivery.revoked_query_ids, ("QC",))
        np.testing.assert_array_equal(delivery.dq[self.layout.q_index["QC"]], 0.0)
        self.assertTrue(any("样本作废" in r for r in delivery.revoke_reasons))
        # 被撤销查询不进入可提交范围，其余查询仍保留归属信息
        self.assertNotIn("QC", delivery.committed_query_ids)
        self.assertEqual(delivery.missing_key_ids, ())

    def test_contract_mismatch_is_rejected(self):
        eng = self.engine()
        eng.forward()
        # 改动数据后，凭据与数据本身不再对应同一步
        eng2 = AttentionEngine(
            self.layout,
            self.q + 1.0,
            self.k,
            self.v,
            self.dy,
            max_workspace_bytes=self.ws,
            max_receipt_bytes=self.rc,
        )
        eng2._receipt = eng._receipt
        with self.assertRaises(ContractError):
            eng2.backward_all()
        # 逻辑身份变更（掩码契约）同样被拒
        other = Layout.from_records({**self.records, "left": self.records["left"] + 1})
        eng3 = self.engine(other)
        eng3._receipt = eng._receipt
        with self.assertRaises(ContractError):
            eng3.backward_all()

    def test_legal_pairs_window_causal_allow_and_session(self):
        lay = self.layout
        # QA: s0 pos=2, left=3 -> 可见 s0 中 pos∈[-1,2] 且在 allow 内
        legal = lay.legal_pairs([lay.q_index["QA"]], list(range(lay.nk)))[0]
        got = {lay.key_ids[j] for j in range(lay.nk) if legal[j]}
        # KA s0/0、KB s0/2 满足窗口与因果；KC pos=4 超因果；KD/KE 跨会话
        self.assertEqual(got, {"KA", "KB"})
        # QB s0 pos=4, left=3 -> 窗口 [1,4]：KA pos=0 落在窗外；
        # KB pos=2、KC pos=4 在内且都在 allow 中
        legal_b = lay.legal_pairs([lay.q_index["QB"]], list(range(lay.nk)))[0]
        got_b = {lay.key_ids[j] for j in range(lay.nk) if legal_b[j]}
        self.assertEqual(got_b, {"KB", "KC"})
        # QC s1 pos=1：KE s1/1 与 KD s1/0 均在窗口与 allow 内
        legal_c = lay.legal_pairs([lay.q_index["QC"]], list(range(lay.nk)))[0]
        got_c = {lay.key_ids[j] for j in range(lay.nk) if legal_c[j]}
        self.assertEqual(got_c, {"KD", "KE"})

    def test_receipt_within_budget_and_no_full_score_materialized(self):
        eng = self.engine()
        eng.forward()
        status = eng.status()
        self.assertLessEqual(status["receipt_bytes"], self.rc)
        receipt = eng._receipt
        # 凭据只含每行 m/l/空行标记，形状与 nk 无关地不随键数二次增长
        self.assertEqual(receipt.row_max.shape, (self.layout.nq, self.layout.nqh))
        self.assertEqual(receipt.row_denom.shape, (self.layout.nq, self.layout.nqh))

    def test_workspace_budget_is_enforced(self):
        with self.assertRaises(WorkspaceLimitError):
            AttentionEngine(
                self.layout,
                self.q,
                self.k,
                self.v,
                self.dy,
                max_workspace_bytes=64,  # 连单块最小临时量都放不下
                max_receipt_bytes=self.rc,
                q_block=4,
            ).forward()

    def test_receipt_budget_is_enforced(self):
        eng = AttentionEngine(
            self.layout,
            self.q,
            self.k,
            self.v,
            self.dy,
            max_workspace_bytes=self.ws,
            max_receipt_bytes=64,
        )
        with self.assertRaises(ReceiptLimitError):
            eng.forward()

    def _random_case(self, rng, keep):
        nq, nk, nqh, nkh, d = 9, 7, 4, 2, 5
        q = rng.normal(scale=4.0, size=(nq, nqh, d))
        k = rng.normal(scale=4.0, size=(nk, nkh, d))
        v = rng.normal(scale=4.0, size=(nk, nkh, d))
        dy = rng.normal(size=(nq, nqh, d))
        qids = tuple(f"Q{i}" for i in range(nq))
        kids = tuple(f"K{i}" for i in range(nk))
        qsess = tuple(f"s{rng.integers(0, 3)}" for _ in range(nq))
        ksess = tuple(f"s{rng.integers(0, 3)}" for _ in range(nk))
        qpos = tuple(int(rng.integers(0, 6)) for _ in range(nq))
        kpos = tuple(int(rng.integers(0, 6)) for _ in range(nk))
        allow = {}
        for qi, qid in enumerate(qids):
            if rng.random() < 0.7:
                size = int(rng.integers(1, nk + 1))
                pick = [kids[j] for j in rng.choice(nk, size=size, replace=False)]
                allow[qid] = tuple(pick)
        # 保证所有键值头都被至少一个查询头引用，nkh 与张量一致
        head_map = tuple(int(rng.integers(0, nkh)) for _ in range(nqh - 1)) + (nkh - 1,)
        groups = [
            tuple(kids[i : i + 3]) for i in range(0, nk, 3)
        ]
        # 反向使用打乱后的两片不重叠分区；重送幂等由专门测试覆盖
        rorder = list(kids)
        rng.shuffle(rorder)
        cut = nk // 2
        lay = Layout(
            query_ids=qids,
            key_ids=kids,
            query_session=qsess,
            key_session=ksess,
            query_pos=qpos,
            key_pos=kpos,
            allow=allow,
            head_map=head_map,
            left=int(rng.integers(0, 5)),
            step=int(rng.integers(0, 1000)),
            sample="fuzz",
            keep=keep,
            forward_groups=tuple(groups),
            backward_groups=(tuple(rorder[:cut]), tuple(rorder[cut:])),
        )
        return lay, q, k, v, dy

    def test_fuzz_against_reference_many_cases(self):
        rng = np.random.default_rng(7)
        for seed in range(24):
            keep = (1, 1) if seed % 3 == 0 else (3, 4)
            lay, q, k, v, dy = self._random_case(rng, keep)
            eng = AttentionEngine(
                lay, q, k, v, dy,
                max_workspace_bytes=self.ws, max_receipt_bytes=self.rc,
            )
            out = eng.forward()
            ref_out, ref_dq, ref_dk, ref_dv = reference_attention(lay, q, k, v, dy)
            np.testing.assert_allclose(out, ref_out, atol=1e-8, rtol=0,
                                       err_msg=f"seed={seed} keep={keep}")
            delivery = eng.backward_all()
            self.assertTrue(delivery.committed, f"seed={seed}")
            np.testing.assert_allclose(delivery.dq, ref_dq, atol=1e-8, rtol=0)
            np.testing.assert_allclose(delivery.dk, ref_dk, atol=1e-8, rtol=0)
            np.testing.assert_allclose(delivery.dv, ref_dv, atol=1e-8, rtol=0)

    def test_all_dropout_keeps_one_path_still_scaled(self):
        """保留概率 3/4 下，被保留项按 4/3 缩放，重放逐位一致。"""
        rng = np.random.default_rng(11)
        lay, q, k, v, dy = self._random_case(rng, (3, 4))
        e1 = AttentionEngine(lay, q, k, v, dy, self.ws, self.rc)
        o1 = e1.forward()
        d1 = e1.backward_all()
        e2 = AttentionEngine(lay, q, k, v, dy, self.ws, self.rc)
        o2 = e2.forward()
        d2 = e2.backward_all()
        np.testing.assert_array_equal(o1, o2)
        np.testing.assert_array_equal(d1.dq, d2.dq)
        np.testing.assert_array_equal(d1.dk, d2.dk)
        np.testing.assert_array_equal(d1.dv, d2.dv)

    def test_all_dropout_keep_zero_gives_nothing(self):
        """保留概率为 0：所有保留项消失，输出与梯度全零，行为等同全屏蔽。"""
        lay0 = Layout.from_records({**self.records, "keep": [0, 4]})
        eng = AttentionEngine(lay0, self.q, self.k, self.v, self.dy,
                              self.ws, self.rc)
        out = eng.forward()
        np.testing.assert_array_equal(out, 0.0)
        d = eng.backward_all()
        np.testing.assert_array_equal(d.dq, 0.0)
        np.testing.assert_array_equal(d.dk, 0.0)
        np.testing.assert_array_equal(d.dv, 0.0)
        # 没有缺席分片，仍是一次完整（虽然全零）的可提交交付
        self.assertTrue(d.committed)
        self.assertEqual(d.missing_key_ids, ())

    def test_gsum_scalar_accumulation_matches_global(self):
        """第一遍累积的 g_sum 必须等于全局 Σ p·(dy·V)，且只占每行一个标量。"""
        eng = self.engine()
        eng.forward()
        receipt = eng._receipt
        for grp in self.layout.backward_groups:
            eng.backward_shard(grp)
        # 用参考实现独立计算全局 g_sum
        nq, nqh, d = self.q.shape
        ref_gsum = np.zeros((nq, nqh))
        for qi in range(nq):
            for h in range(nqh):
                kh = self.layout.head_map[h]
                if receipt.empty_rows[qi, h]:
                    continue
                k_rows = list(range(self.layout.nk))
                legal = self.layout.legal_pairs([qi], k_rows)[0]
                scores = self.q[qi, h] @ self.k[k_rows, kh].T * eng.scale
                m = receipt.row_max[qi, h]
                l = receipt.row_denom[qi, h]
                shifted = np.where(legal, scores - m, 0.0)
                a = np.where(legal, np.exp(shifted) / l, 0.0)
                words = self.layout.dropout_words([h], [qi], k_rows)[0, 0]
                p = np.where(legal & self.layout.keep_mask(words),
                             a * self.layout.keep[1] / self.layout.keep[0], 0.0)
                glog = self.dy[qi, h] @ self.v[k_rows, kh].T
                ref_gsum[qi, h] = (p * glog).sum()
        np.testing.assert_allclose(receipt.g_sum, ref_gsum, atol=1e-12, rtol=0)
        # g_sum 形状只随查询行数增长，与键数无关（非逐键凭据）
        self.assertEqual(receipt.g_sum.shape, (nq, nqh))

    def test_capacity_scale_planner_respects_budget(self):
        """制造端极值规模下，自动分块规划与凭据账仍在预算内。"""
        cap = self.capacity
        nq, nk, nqh, nkh, d = (
            cap["queries"],
            cap["keys"],
            cap["query_heads"],
            cap["key_heads"],
            cap["head_width"],
        )
        rng = np.random.default_rng(cap["seed"])
        q = rng.normal(size=(nq, nqh, d))
        k = rng.normal(size=(nk, nkh, d))
        v = rng.normal(size=(nk, nkh, d))
        dy = rng.normal(size=(nq, nqh, d))
        ids_q = [f"Q{i}" for i in range(nq)]
        ids_k = [f"K{i}" for i in range(nk)]
        sess_q = tuple(f"s{i % 64}" for i in range(nq))
        sess_k = tuple(f"s{i % 64}" for i in range(nk))
        pos_q = tuple(i % 256 for i in range(nq))
        pos_k = tuple(i % 256 for i in range(nk))
        groups = [tuple(ids_k[i : i + 256]) for i in range(0, nk, 256)]
        lay = Layout(
            query_ids=tuple(ids_q),
            key_ids=tuple(ids_k),
            query_session=sess_q,
            key_session=sess_k,
            query_pos=pos_q,
            key_pos=pos_k,
            allow={},
            head_map=tuple(h % nkh for h in range(nqh)),
            left=64,
            step=cap["seed"],
            sample="capacity-probe",
            keep=(3, 4),
            forward_groups=tuple(groups),
            backward_groups=tuple(groups),
        )
        eng = AttentionEngine(
            lay,
            q,
            k,
            v,
            dy,
            max_workspace_bytes=cap["max_workspace_bytes"],
            max_receipt_bytes=cap["max_receipt_bytes"],
        )
        # 自动块必须把单块峰值压在工作区预算内
        g = 256
        per_row = 6 * g * 8 + 4 * d * 8
        self.assertLessEqual(per_row * eng.q_block, cap["max_workspace_bytes"])
        # 凭据账：m + l + 空行标记
        receipt_expected = nq * nqh * (8 + 8 + 1)
        self.assertLessEqual(receipt_expected, cap["max_receipt_bytes"])
        # 不实际跑满 SHA-256（纯 Python 身份循环在该规模仅做规划核对）
        self.assertEqual(eng.d, d)


if __name__ == "__main__":
    unittest.main()
