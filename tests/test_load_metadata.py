from __future__ import annotations

import unittest

from load_test.driver import camel_run_metadata


class LoadMetadataTests(unittest.TestCase):
    def test_load_metadata_includes_agent_runtime_settings(self) -> None:
        metadata = camel_run_metadata({
            "environment": "prod",
            "inference_backend": "vllm",
            "prompt_version": "p0_baseline",
            "agent_version": "a0_llm_only",
            "verify_mode": "llm_only",
            "max_iterations": "3",
            "prompt_profile": "normal",
            "llm_cache_bust": "0",
            "load_run_id": "load-baseline",
        })

        self.assertEqual(metadata["verifyMode"], "llm_only")
        self.assertEqual(metadata["maxIterations"], "3")
        self.assertEqual(metadata["promptProfile"], "normal")
        self.assertEqual(metadata["llmCacheBust"], "0")
        self.assertEqual(metadata["loadRunId"], "load-baseline")
        self.assertEqual(metadata["sessionId"], "load-baseline")


if __name__ == "__main__":
    unittest.main()
