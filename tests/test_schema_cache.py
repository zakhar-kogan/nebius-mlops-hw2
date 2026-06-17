from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.schema import clear_schema_caches, prewarm_schemas, render_schema


class SchemaCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_schema_caches()

    def tearDown(self) -> None:
        clear_schema_caches()

    def test_render_schema_writes_cache_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"AGENT_SCHEMA_CACHE_DIR": tmp}):
                rendered = render_schema("superhero")
                path = Path(tmp) / "superhero.schema.txt"
                self.assertTrue(path.exists())
                self.assertEqual(path.read_text(), rendered)

    def test_render_schema_reads_existing_cache_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "superhero.schema.txt"
            path.write_text("-- cached schema")
            with patch.dict(os.environ, {"AGENT_SCHEMA_CACHE_DIR": tmp}):
                self.assertEqual(render_schema("superhero"), "-- cached schema")

    def test_prewarm_schemas_renders_requested_dbs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"AGENT_SCHEMA_CACHE_DIR": tmp}):
                warmed = prewarm_schemas(["superhero"])
                self.assertEqual(warmed, ["superhero"])
                self.assertTrue((Path(tmp) / "superhero.schema.txt").exists())


if __name__ == "__main__":
    unittest.main()
