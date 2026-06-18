from __future__ import annotations

import unittest

from agent.execution import ExecutionResult


class ExecutionRenderTests(unittest.TestCase):
    def test_render_truncates_long_cell_values(self) -> None:
        result = ExecutionResult(
            ok=True,
            rows=[("x" * 500,)],
            columns=["body"],
            row_count=1,
        )

        rendered = result.render(max_cell_chars=32)

        self.assertIn("xxx", rendered)
        self.assertLess(len(rendered), 120)
        self.assertNotIn("x" * 100, rendered)

    def test_render_applies_total_character_budget(self) -> None:
        result = ExecutionResult(
            ok=True,
            rows=[tuple(f"value-{i}-{j}" for j in range(20)) for i in range(20)],
            columns=[f"col_{j}" for j in range(20)],
            row_count=20,
        )

        rendered = result.render(max_rows=20, max_chars=300)

        self.assertLessEqual(len(rendered), 300)
        self.assertTrue(rendered.endswith("..."))


if __name__ == "__main__":
    unittest.main()
