# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host CLI regression for binary configuration path spelling and repeat runs."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(
    os.environ.get(
        "MERGE_CONFIG_SOURCE",
        str(Path(__file__).resolve().parents[1] / "merge_ops_config_json.py"),
    )
)
PATH_FORMS = ("op", "op/", "op//", "./op/./")


class TestMergeConfigPaths(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / "op"
        self.config.mkdir()
        self.entries = [{"bin_filename": "first"}, {"bin_filename": "second"}]
        for index, entry in enumerate(self.entries):
            (self.config / f"part{index}.json").write_text(
                json.dumps({"op_type": "Example", "op_list": [entry]}), encoding="utf-8"
            )
        (self.config / "metadata.json").write_text(
            '{"build": "metadata"}', encoding="utf-8"
        )

    def test_equivalent_directory_spellings_keep_output_name(self):
        for spelling in PATH_FORMS:
            with self.subTest(path=spelling):
                self._run(spelling)
                self._check_output()

    def test_repeat_runs_do_not_ingest_previous_output(self):
        for spelling in PATH_FORMS:
            with self.subTest(path=spelling):
                for _ in range(2):
                    self._run(spelling)
                    self._check_output()

    def test_absolute_path_with_trailing_separator(self):
        self._run(str(self.config) + os.sep)
        self._check_output()

    def test_conflicting_operator_types_still_fail(self):
        (self.config / "part1.json").write_text(
            '{"op_type": "Different", "op_list": []}', encoding="utf-8"
        )
        result = self._invoke("op/")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("different op config", result.stderr)
        self.assertFalse((self.config / "op_binary.json").exists())

    def test_glob_metacharacters_in_directory_name_are_literal(self):
        self.config = self.config.rename(self.root / "op[1]")
        neighbor = self.root / "op1"
        neighbor.mkdir()
        (neighbor / "other.json").write_text(
            '{"op_type": "Other", "op_list": [{"bin_filename": "wrong"}]}',
            encoding="utf-8",
        )
        self._run("op[1]")
        self._check_output()

    def test_literal_parent_directory_with_valid_operator_name(self):
        parent = self.root / "build[1]"
        parent.mkdir()
        self.config = self.config.rename(parent / "op")
        neighbor = self.root / "build1" / "op"
        neighbor.mkdir(parents=True)
        (neighbor / "other.json").write_text(
            '{"op_type": "Other", "op_list": [{"bin_filename": "wrong"}]}',
            encoding="utf-8",
        )
        self._run("build[1]/op")
        self._check_output()

    def _invoke(self, spelling):
        return subprocess.run(
            [sys.executable, str(SOURCE), spelling],
            cwd=self.root,
            text=True,
            capture_output=True,
            timeout=10,
        )

    def _run(self, spelling):
        result = self._invoke(spelling)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def _check_output(self):
        output = self.config / (self.config.name + "_binary.json")
        self.assertTrue(output.is_file(), list(self.config.iterdir()))
        value = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(value["op_type"], "Example")
        self.assertCountEqual(value["op_list"], self.entries)
        self.assertEqual(len(value["op_list"]), 2)
        self.assertFalse((self.config / "_binary.json").exists())


if __name__ == "__main__":
    unittest.main()
