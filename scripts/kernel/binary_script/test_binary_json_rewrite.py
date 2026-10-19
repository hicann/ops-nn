# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Host regressions for repeated binary-config generation and its JSON consumer."""

import json
import os
from pathlib import Path
import runpy
import stat
import subprocess
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parent
WRITE_JSON = runpy.run_path(str(SCRIPTS / "binary_util/util.py"))["wr_json"]


class BinaryJsonRewriteTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="binary-json-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def run_script(self, name, *arguments):
        return subprocess.run(
            [sys.executable, str(SCRIPTS / name), *map(str, arguments)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def test_shorter_rewrite_removes_previous_document(self):
        path = self.directory / "config.json"
        previous = {"op_type": "Fixture", "op_list": [{"bin_filename": "old" * 100}]}
        for replacement in (
            {"op_type": "Fixture", "op_list": []},
            {},
            {"message": "新配置"},
        ):
            with self.subTest(replacement=replacement):
                WRITE_JSON(previous, path)
                previous_size = path.stat().st_size
                WRITE_JSON(replacement, path)
                self.assertLess(path.stat().st_size, previous_size)
                self.assertEqual(
                    json.loads(path.read_text(encoding="utf-8")), replacement
                )

    def test_repeated_impl_mode_cli_remains_readable_by_splitter(self):
        original = self.directory / "original.json"
        original.write_text(
            json.dumps(
                {"op_type": "Fixture", "op_list": [{"bin_filename": "Fixture_0"}]}
            ),
            encoding="utf-8",
        )
        generated = self.directory / "generated.json"
        for mode in ("long_mode" * 50, "hp"):
            result = self.run_script(
                "gen_opc_json_with_impl_mode.py", original, generated, mode
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        consumed = self.run_script("gen_opc_json_with_threadnum.py", generated, "1")
        self.assertEqual(consumed.returncode, 0, consumed.stdout + consumed.stderr)
        split = Path(str(generated) + "_0")
        self.assertEqual(
            json.loads(split.read_text(encoding="utf-8")),
            {"op_type": "Fixture", "op_list": [{"bin_filename": "Fixture_0_hp"}]},
        )

    @unittest.skipUnless(os.name == "posix", "POSIX file mode checks")
    def test_new_file_mode_and_existing_permissions_are_preserved(self):
        path = self.directory / "permissions.json"
        WRITE_JSON({"value": 1}, path)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        path.chmod(0o640)
        WRITE_JSON({"value": 2}, path)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o640)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": 2})


if __name__ == "__main__":
    unittest.main()
