from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from agent.graph import (
    AgentState,
    get_max_iterations,
    get_verify_mode,
    route_after_deterministic_verify,
    route_after_verify,
)


class GraphRoutingTests(unittest.TestCase):
    def state(self, *, issue: str = "", iteration: int = 1, ok: bool = False) -> AgentState:
        return AgentState(
            question="question",
            db_id="formula_1",
            deterministic_issue=issue,
            verify_issue=issue,
            verify_ok=ok,
            iteration=iteration,
        )

    def test_defaults_preserve_full_mode_and_three_attempts(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_verify_mode(), "full")
            self.assertEqual(get_max_iterations(), 3)

    def test_deterministic_pass_routes_to_llm_verifier_in_full_mode(self) -> None:
        with patch.dict(os.environ, {"AGENT_VERIFY_MODE": "full"}):
            self.assertEqual(route_after_deterministic_verify(self.state()), "verify")

    def test_deterministic_pass_ends_in_fast_mode(self) -> None:
        with patch.dict(os.environ, {"AGENT_VERIFY_MODE": "fast"}):
            self.assertEqual(route_after_deterministic_verify(self.state()), "end")

    def test_deterministic_fail_routes_to_revise_below_cap(self) -> None:
        with patch.dict(os.environ, {"AGENT_MAX_ITERATIONS": "2"}):
            self.assertEqual(
                route_after_deterministic_verify(self.state(issue="bad literal", iteration=1)),
                "revise",
            )

    def test_deterministic_fail_ends_at_two_attempt_cap(self) -> None:
        with patch.dict(os.environ, {"AGENT_MAX_ITERATIONS": "2"}):
            self.assertEqual(
                route_after_deterministic_verify(self.state(issue="bad literal", iteration=2)),
                "end",
            )

    def test_llm_verifier_ends_at_two_attempt_cap(self) -> None:
        with patch.dict(os.environ, {"AGENT_MAX_ITERATIONS": "2"}):
            self.assertEqual(route_after_verify(self.state(iteration=2, ok=False)), "end")


if __name__ == "__main__":
    unittest.main()
