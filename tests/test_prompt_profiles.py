from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from agent import prompts


class PromptProfileTests(unittest.TestCase):
    def test_default_profile_is_normal(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(prompts.get_prompt_profile(), "normal")
            self.assertIn("careful text-to-SQL", prompts.generate_sql_system())

    def test_invalid_profile_falls_back_to_normal(self) -> None:
        with patch.dict(os.environ, {"AGENT_PROMPT_PROFILE": "unknown"}):
            self.assertEqual(prompts.get_prompt_profile(), "normal")

    def test_short_profile_uses_compact_prompts(self) -> None:
        with patch.dict(os.environ, {"AGENT_PROMPT_PROFILE": "short"}):
            self.assertEqual(prompts.get_prompt_profile(), "short")
            self.assertNotIn("careful text-to-SQL", prompts.generate_sql_system())
            self.assertIn("Correct?", prompts.verify_user())


if __name__ == "__main__":
    unittest.main()
