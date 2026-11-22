"""核对合成采集的交换记录与原始尺寸。"""
import json
from pathlib import Path
import unittest

import numpy as np

from caseio import load_case


class CaptureTests(unittest.TestCase):
    def setUp(self):
        data = Path(__file__).resolve().parents[1] / "data"
        self.records, self.tensors = load_case(data / "capture.json")
        self.capacity = json.loads((data / "capacity.json").read_text(encoding="utf-8"))

    def test_capture_contract(self):
        r, a = self.records, self.tensors
        self.assertEqual(a["q"].shape, a["dy"].shape)
        self.assertEqual(a["k"].shape, a["v"].shape)
        self.assertEqual(len(r["keys"]), a["k"].shape[0])
        self.assertEqual({x for group in r["forward_groups"] for x in group},
                         {x for group in r["backward_groups"] for x in group})
        self.assertNotEqual(r["step"], r["mismatched_gradient_version"]["step"])
