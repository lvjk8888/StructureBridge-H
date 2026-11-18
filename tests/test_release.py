"""Small synthetic interface checks, not scientific replication."""
from pathlib import Path
import json
import tempfile
import unittest
from experiments.main.synthetic_demo import make_fixture
from experiments.main.offline import select_file, rerank_file


class OfflineInterfaceTests(unittest.TestCase):
    def test_frozen_budgets_and_top10(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = make_fixture(Path(tmp) / "synthetic")
            self.assertEqual(select_file(d / "pool.csv", d / "selected.jsonl"), 1)
            record = json.loads((d / "selected.jsonl").read_text())
            self.assertEqual(set(record["by_budget"]), {"20", "50", "100"})
            for budget, methods in record["by_budget"].items():
                self.assertEqual(set(methods), {"raw", "edge_mean_touched", "anchor_ic_max"})
                for cell in methods.values():
                    self.assertEqual(len(set(cell["selected"])), int(budget))
            self.assertEqual(record["by_budget"]["20"]["raw"]["selected"],
                             [f"synthetic-d{i}" for i in range(1, 21)])
            self.assertEqual(rerank_file(d / "selected.jsonl", d / "scores.jsonl", d / "top.jsonl"), 1)
            top = json.loads((d / "top.jsonl").read_text())["top10"]
            scores = json.loads((d / "scores.jsonl").read_text())["scores"]
            for budget, methods in top.items():
                for method, ids in methods.items():
                    self.assertEqual(len(ids), 10)
                    self.assertTrue(set(ids) <= set(record["by_budget"][budget][method]["selected"]))
                    self.assertEqual([scores[d] for d in ids], sorted([scores[d] for d in ids], reverse=True))
            with self.assertRaises(FileExistsError):
                select_file(d / "pool.csv", d / "selected.jsonl")

    def test_missing_scores_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = make_fixture(Path(tmp) / "synthetic")
            select_file(d / "pool.csv", d / "selected.jsonl")
            (d / "empty.jsonl").write_text('{"query_id":"synthetic-q1","scores":{}}\n')
            with self.assertRaisesRegex(ValueError, "Missing selected-document"):
                rerank_file(d / "selected.jsonl", d / "empty.jsonl", d / "rejected.jsonl")


    def test_duplicate_columns_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = make_fixture(Path(tmp) / "synthetic")
            pool = d / "pool.csv"
            lines = pool.read_text().splitlines()
            lines[0] += ",query"
            pool.write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(ValueError, "unique documented columns"):
                select_file(pool, d / "invalid.jsonl")
            self.assertFalse((d / "invalid.jsonl").exists())

    def test_empty_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = make_fixture(Path(tmp) / "synthetic")
            (d / "empty_records.jsonl").write_text("")
            with self.assertRaisesRegex(ValueError, "Empty selection"):
                rerank_file(d / "empty_records.jsonl", d / "scores.jsonl", d / "invalid.jsonl")
            self.assertFalse((d / "invalid.jsonl").exists())
            (d / "pool.csv").write_text((d / "pool.csv").read_text().splitlines()[0] + "\n")
            with self.assertRaisesRegex(ValueError, "no candidate rows"):
                select_file(d / "pool.csv", d / "invalid.jsonl")


if __name__ == "__main__":
    unittest.main()
