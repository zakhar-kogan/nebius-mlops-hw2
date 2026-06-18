from __future__ import annotations

import unittest

from load_test.driver import camel_run_metadata, summarize_agent_response


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
            "schema_profile": "compact",
            "llm_cache_bust": "0",
            "load_run_id": "load-baseline",
        })

        self.assertEqual(metadata["verifyMode"], "llm_only")
        self.assertEqual(metadata["maxIterations"], "3")
        self.assertEqual(metadata["promptProfile"], "normal")
        self.assertEqual(metadata["schemaProfile"], "compact")
        self.assertEqual(metadata["llmCacheBust"], "0")
        self.assertEqual(metadata["loadRunId"], "load-baseline")
        self.assertEqual(metadata["sessionId"], "load-baseline")

    def test_summarize_agent_response_keeps_lightweight_diagnostics(self) -> None:
        summary = summarize_agent_response({
            "ok": True,
            "error": None,
            "iterations": 2,
            "sql": "SELECT 1",
            "history": [
                {"node": "generate_sql", "duration_ms": 100.25},
                {"node": "execute", "duration_ms": 5},
                {"node": "revise", "duration_ms": 200.5},
                {"node": "execute", "duration_ms": 7},
            ],
        })

        self.assertEqual(summary["agent_ok"], True)
        self.assertEqual(summary["iterations"], 2)
        self.assertEqual(summary["sql_length"], 8)
        self.assertEqual(summary["history_event_count"], 4)
        self.assertEqual(summary["history_node_counts"]["execute"], 2)
        self.assertEqual(summary["history_node_duration_ms"]["execute"], 12.0)
        self.assertEqual(summary["history_node_max_duration_ms"]["revise"], 200.5)


if __name__ == "__main__":
    unittest.main()
