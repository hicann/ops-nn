# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Exercise production packaging paths with level and legacy compiler outputs."""

import copy
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gen_output_json as operator_index
import gen_binary_info_config as global_index
from binary_util.deterministic_levels import (
    finalize_binary_list,
    index_list_name,
    is_level_entry,
    validate_operator_index,
)


def support(levels=None):
    data = {
        "simplifiedKeyMode": 0,
        "inputs": [
            {
                "name": "x",
                "index": 0,
                "dtype": "float32",
                "format": "ND",
                "paramType": "required",
            }
        ],
        "outputs": [
            {
                "name": "y",
                "index": 0,
                "dtype": "float32",
                "format": "ND",
                "paramType": "required",
            }
        ],
        "attrs": [],
        "simplifiedKey": ["LpNormV2/d=0,p=1/0,2/0,2"],
    }
    if levels is not None:
        data["deterministicLevels"] = levels
        data["simplifiedKey"] = [
            f"LpNormV2/d={int(x != 0)},p=1,dt={x}/0,2/0,2" for x in levels
        ]
    return data


class PackagingLevels(unittest.TestCase):
    def test_local_dual_lists_and_global_unified_list(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bin"
            kernels = root / "ascend950/lp_norm_v2"
            config = root / "config/ascend950"
            kernels.mkdir(parents=True)
            config.mkdir(parents=True)
            source = Path(directory) / "task.json"
            source.write_text(
                json.dumps(
                    {
                        "op_type": "LpNormV2",
                        "op_list": [
                            {
                                "bin_filename": "LpNormV2_fp32",
                            }
                        ],
                    }
                )
            )
            legacy = support()
            legacy["deterministic"] = "ignore"
            legacy["simplifiedKey"] = [
                f"LpNormV2/d={mode},p=1/0,2/0,2" for mode in (0, 1)
            ]
            (kernels / "LpNormV2_fp32.json").write_text(
                json.dumps(
                    {
                        "binFileName": "LpNormV2_fp32",
                        "binFileSuffix": ".o",
                        "coreType": "VectorCore",
                        "supportInfo": legacy,
                    }
                )
            )
            (kernels / "LpNormV2_fp32.o").write_bytes(b"fixture")
            for suffix, levels in [
                ("_level0", [0]),
                ("_level1_2", [1, 2]),
                ("_level3", [3]),
            ]:
                name = "LpNormV2_fp32" + suffix
                (kernels / (name + ".json")).write_text(
                    json.dumps(
                        {
                            "binFileName": name,
                            "binFileSuffix": ".o",
                            "coreType": "VectorCore",
                            "supportInfo": support(levels),
                        }
                    )
                )
                (kernels / (name + ".o")).write_bytes(b"fixture")
            target = config / "lp_norm_v2_apt.json"
            operator_index.main(str(source), str(kernels) + "/", str(target), 0)
            global_index.gen_binary_info_config(str(root), "ascend950")
            op = json.loads(target.read_text())
            glob = json.loads((config / "binary_info_config.json").read_text())[
                "LpNormV2"
            ]
            self.assertEqual(len(op["binList"]), 1)
            self.assertEqual(len(op["binListWithLoadPriority"]), 3)
            self.assertTrue(all(not is_level_entry(x) for x in op["binList"]))
            self.assertTrue(
                all(is_level_entry(x) for x in op["binListWithLoadPriority"])
            )
            self.assertEqual(
                [
                    entry["deterministicLevels"]
                    for entry in op["binListWithLoadPriority"]
                ],
                [[0], [1, 2], [3]],
            )
            self.assertEqual(op["binList"][0]["deterministic"], "ignore")
            self.assertNotIn("deterministicLevelBinInfo", op["binList"][0])
            for data, name in [(glob, "binaryList")]:
                self.assertNotIn("levelBinaryList", data)
                self.assertNotIn("binListWithLoadPriority", data)
                entries = data[name]
                legacy = [x for x in entries if not is_level_entry(x)]
                grades = [x for x in entries if is_level_entry(x)]
                self.assertEqual(len(legacy), 1)
                self.assertEqual(len(grades), 3)
                self.assertEqual(entries, legacy + grades)
                self.assertTrue(
                    all(",dt=" not in k for x in legacy for k in x["simplifiedKey"])
                )
                self.assertEqual(
                    {
                        k.split(",dt=")[1].split("/")[0]
                        for x in grades
                        for k in x["simplifiedKey"]
                    },
                    {"0", "1", "2", "3"},
                )
                self.assertEqual(
                    sorted(len(x["simplifiedKey"]) for x in grades), [1, 1, 2]
                )

    def test_incomplete_split_fails_instead_of_packaging_partial(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "task.json"
            source.write_text(
                json.dumps(
                    {
                        "op_type": "LpNormV2",
                        "op_list": [
                            {
                                "bin_filename": "LpNormV2_fp32",
                            }
                        ],
                    }
                )
            )
            # One level output exists while the other group outputs are
            # missing; packaging must fail.
            (Path(d) / "LpNormV2_fp32_level0.json").write_text("{}")
            with self.assertRaises(FileNotFoundError):
                operator_index.generate_operator_json_content(
                    str(source), d + "/", {"binList": []}
                )

    def test_standalone_global_generator_rejects_incomplete_level_layout(self):
        # cmake invokes gen_binary_info_config standalone, so it must reject
        # a half-finished layout itself instead of publishing level0 only.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "bin"
            kernels = root / "ascend950/lp_norm_v2"
            config = root / "config/ascend950"
            kernels.mkdir(parents=True)
            config.mkdir(parents=True)
            (kernels / "LpNormV2_fp32_level0.json").write_text(
                json.dumps(
                    {
                        "binFileName": "LpNormV2_fp32_level0",
                        "binFileSuffix": ".o",
                        "coreType": "VectorCore",
                        "supportInfo": support([0]),
                    }
                )
            )
            (kernels / "LpNormV2_fp32_level0.o").write_bytes(b"fixture")
            with self.assertRaises(FileNotFoundError):
                global_index.gen_binary_info_config(str(root), "ascend950")
            self.assertFalse((config / "binary_info_config.json").exists())

    def build_level_family(self, d):
        root = Path(d) / "bin"
        kernels = root / "ascend950/lp_norm_v2"
        config = root / "config/ascend950"
        kernels.mkdir(parents=True)
        config.mkdir(parents=True)
        source = Path(d) / "task.json"
        source.write_text(
            json.dumps(
                {"op_type": "LpNormV2", "op_list": [{"bin_filename": "LpNormV2_fp32"}]}
            )
        )
        legacy = support()
        legacy["deterministic"] = "ignore"
        legacy["simplifiedKey"] = [f"LpNormV2/d={mode},p=1/0,2/0,2" for mode in (0, 1)]
        family = [("LpNormV2_fp32", legacy)]
        for suffix, levels in [
            ("_level0", [0]),
            ("_level1_2", [1, 2]),
            ("_level3", [3]),
        ]:
            family.append(("LpNormV2_fp32" + suffix, support(levels)))
        for name, info in family:
            (kernels / (name + ".json")).write_text(
                json.dumps(
                    {
                        "binFileName": name,
                        "binFileSuffix": ".o",
                        "coreType": "VectorCore",
                        "supportInfo": info,
                    }
                )
            )
            (kernels / (name + ".o")).write_bytes(b"fixture")
        return root, kernels, config, source

    def test_missing_level_object_fails_both_generators(self):
        for entry in ("task-local", "standalone-global"):
            with tempfile.TemporaryDirectory() as d:
                root, kernels, config, source = self.build_level_family(d)
                (kernels / "LpNormV2_fp32_level0.o").unlink()
                with self.subTest(entry=entry):
                    with self.assertRaisesRegex(FileNotFoundError, "companion"):
                        if entry == "task-local":
                            operator_index.main(
                                str(source),
                                str(kernels) + "/",
                                str(config / "lp_norm_v2_apt.json"),
                                0,
                            )
                        else:
                            global_index.gen_binary_info_config(str(root), "ascend950")

    def test_missing_legacy_companions_fail_level_family(self):
        def legacy_false_only(kernels):
            info = support()
            info["deterministic"] = "false"
            (kernels / "LpNormV2_fp32.json").write_text(
                json.dumps(
                    {
                        "binFileName": "LpNormV2_fp32",
                        "binFileSuffix": ".o",
                        "coreType": "VectorCore",
                        "supportInfo": info,
                    }
                )
            )

        mutations = [
            (
                lambda k: (k / "LpNormV2_fp32.o").unlink(),
                FileNotFoundError,
                "Missing legacy",
            ),
            (
                lambda k: (k / "LpNormV2_fp32.json").unlink(),
                FileNotFoundError,
                "Missing legacy",
            ),
            (legacy_false_only, ValueError, "both boolean"),
        ]
        for mutation, error, pattern in mutations:
            for entry in ("task-local", "standalone-global"):
                with tempfile.TemporaryDirectory() as d:
                    root, kernels, config, source = self.build_level_family(d)
                    mutation(kernels)
                    with self.subTest(entry=entry):
                        with self.assertRaisesRegex(error, pattern):
                            if entry == "task-local":
                                operator_index.main(
                                    str(source),
                                    str(kernels) + "/",
                                    str(config / "lp_norm_v2_apt.json"),
                                    0,
                                )
                            else:
                                global_index.gen_binary_info_config(
                                    str(root), "ascend950"
                                )

    def test_inferred_legacy_pair_covers_both_boolean_modes(self):
        for entry in ("task-local", "standalone-global"):
            with tempfile.TemporaryDirectory() as d:
                root, kernels, config, source = self.build_level_family(d)
                for name, flag in [
                    ("LpNormV2_fp32", "false"),
                    ("LpNormV2_fp32_deterministic", "true"),
                ]:
                    info = support()
                    info["deterministic"] = flag
                    info["simplifiedKey"] = [
                        f"LpNormV2/d={int(flag == 'true')},p=1/0,2/0,2"
                    ]
                    (kernels / (name + ".json")).write_text(
                        json.dumps(
                            {
                                "binFileName": name,
                                "binFileSuffix": ".o",
                                "coreType": "VectorCore",
                                "supportInfo": info,
                            }
                        )
                    )
                    (kernels / (name + ".o")).write_bytes(b"fixture")
                with self.subTest(entry=entry):
                    if entry == "task-local":
                        operator_index.main(
                            str(source),
                            str(kernels) + "/",
                            str(config / "lp_norm_v2_apt.json"),
                            0,
                        )
                    else:
                        global_index.gen_binary_info_config(str(root), "ascend950")

    def test_declared_groups_reject_missing_outputs_without_level_files(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "task.json"
            source.write_text(
                json.dumps(
                    {
                        "op_type": "LpNormV2",
                        "op_list": [{"bin_filename": "LpNormV2_fp32"}],
                    }
                )
            )
            # Groups are declared while all level outputs are missing;
            # packaging must fail.
            (Path(d) / "LpNormV2_fp32.json").write_text(
                json.dumps({"deterministicLevelGroups": [[0], [1, 2], [3]]})
            )
            with self.assertRaises(FileNotFoundError):
                operator_index.generate_operator_json_content(
                    str(source), d + "/", {"binList": []}
                )

    def test_unsupported_groups_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "task.json"
            source.write_text(
                json.dumps(
                    {
                        "op_type": "LpNormV2",
                        "op_list": [{"bin_filename": "LpNormV2_fp32"}],
                    }
                )
            )
            (Path(d) / "LpNormV2_fp32.json").write_text(
                json.dumps(
                    {"deterministicLevelGroups": [[3]], "supportInfo": support(None)}
                )
            )
            result = {"binList": []}
            with self.assertRaisesRegex(
                ValueError, "Unsupported deterministic level layout"
            ):
                operator_index.generate_operator_json_content(
                    str(source), d + "/", result
                )

    def test_obsolete_level_output_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "task.json"
            source.write_text(
                json.dumps(
                    {
                        "op_type": "LpNormV2",
                        "op_list": [{"bin_filename": "LpNormV2_fp32"}],
                    }
                )
            )
            (Path(d) / "LpNormV2_fp32_level0_1_2.json").write_text("{}")
            with self.assertRaisesRegex(
                ValueError, "Unsupported deterministic level output layout"
            ):
                operator_index.generate_operator_json_content(
                    str(source), d + "/", {"binList": []}
                )

    def test_obsolete_declared_layout_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "task.json"
            source.write_text(
                json.dumps(
                    {
                        "op_type": "LpNormV2",
                        "op_list": [{"bin_filename": "LpNormV2_fp32"}],
                    }
                )
            )
            (Path(d) / "LpNormV2_fp32.json").write_text(
                json.dumps({"deterministicLevelGroups": [[0, 1, 2], [3]]})
            )
            with self.assertRaisesRegex(
                ValueError, "Unsupported deterministic level layout"
            ):
                operator_index.generate_operator_json_content(
                    str(source), d + "/", {"binList": []}
                )

    def test_conflicting_or_incomplete_dt_is_rejected(self):
        variants = []
        data = support([1, 2])
        data["simplifiedKey"] = data["simplifiedKey"][:1]
        variants.append(data)
        data = support([0])
        data["simplifiedKey"][0] = data["simplifiedKey"][0].replace("d=0", "d=1")
        variants.append(data)
        data = support([1])
        del data["deterministicLevels"]
        variants.append(data)
        data = support([1])
        data["simplifiedKey"][0] = data["simplifiedKey"][0].replace(",dt=1", ",l=1")
        variants.append(data)
        data = support([1])
        data["simplifiedKey"][0] = data["simplifiedKey"][0].replace("p=1", "p=12")
        variants.append(data)
        for data in variants:
            with self.subTest(data=data), self.assertRaises(ValueError):
                index_list_name(copy.deepcopy(data), "binaryList")

    def test_alternate_generator_unifies_only_global_list(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "util"))
        import ascendc_ops_config as alternate

        config = {"binary_info_config.json": {}}
        for levels in (None, [0], [1, 2], [3]):
            info = support(levels)
            if levels is None:
                info["simplifiedKey"] = [
                    f"LpNormV2/d={mode},p=1/0,2/0,2" for mode in (0, 1)
                ]
            alternate.add_simplified_config(
                "LpNormV2", info, 2, None, "ascend950/lp_norm_v2/test.o", config
            )
            alternate.add_op_config("lp_norm_v2_apt", info, config)
        local = config["lp_norm_v2_apt"]
        self.assertEqual(len(local["binList"]), 1)
        self.assertEqual(len(local["binListWithLoadPriority"]), 3)
        self.assertTrue(all(not is_level_entry(x) for x in local["binList"]))
        self.assertTrue(
            all(is_level_entry(x) for x in local["binListWithLoadPriority"])
        )
        for data, name in [
            (config["binary_info_config.json"]["LpNormV2"], "binaryList"),
        ]:
            # These APIs collect intermediates; gen_all_config finalizes them.
            self.assertEqual(len(data[name]), 4)
            self.assertEqual(sum(is_level_entry(x) for x in data[name]), 3)
            self.assertNotIn("binListWithLoadPriority", data)
            self.assertNotIn("levelBinaryList", data)

    def test_local_lists_keep_separation_without_changing_layout(self):
        legacy = support()
        legacy["simplifiedKey"] += ["LpNormV2/d=1,p=1/0,2/0,2"]
        config = {
            "binList": [legacy],
            "binListWithLoadPriority": [support([3]), support([0])],
        }
        before = copy.deepcopy(config)
        validate_operator_index(config)
        self.assertEqual(config, before)
        self.assertEqual(
            index_list_name(support([3]), "binList"), "binListWithLoadPriority"
        )
        self.assertEqual(index_list_name(support([3]), "binaryList"), "binaryList")

    def test_local_lists_reject_contamination_and_invalid_types(self):
        for config in (
            {"binList": [support([3])], "binListWithLoadPriority": []},
            {"binList": [support()], "binListWithLoadPriority": [support()]},
            {"binList": {}, "binListWithLoadPriority": []},
            {"binList": [], "binListWithLoadPriority": {}},
            {"binList": [support([3])]},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_operator_index(config)

    def test_local_legacy_only_preserves_existing_order(self):
        config = {"binList": [support(), support()]}
        before = copy.deepcopy(config)
        validate_operator_index(config)
        self.assertEqual(config, before)

    def test_legacy_unchanged(self):
        self.assertEqual(index_list_name(support(), "binList"), "binList")


class LegacyCompatibilityPackage(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "bin"
        self.kernels = self.root / "ascend950/lp_norm_v2"
        self.config = self.root / "config/ascend950"
        self.kernels.mkdir(parents=True)
        self.config.mkdir(parents=True)
        self.task = Path(self.directory.name) / "task.json"
        self.task.write_text(
            json.dumps(
                {"op_type": "LpNormV2", "op_list": [{"bin_filename": "LpNormV2_fp32"}]}
            )
        )

    def descriptor(self, name, levels=None, flag="ignore", required=None):
        info = support(levels)
        info["deterministic"] = flag
        if levels is None:
            modes = (0, 1) if flag == "ignore" else (int(flag == "true"),)
            info["simplifiedKey"] = [f"LpNormV2/d={mode},p=1/0,2/0,2" for mode in modes]
        value = {
            "binFileName": name,
            "binFileSuffix": ".o",
            "coreType": "VectorCore",
            "supportInfo": info,
        }
        if required is not None:
            value["legacyCompatibilityOutputs"] = required
        (self.kernels / (name + ".json")).write_text(json.dumps(value))
        # Descriptor-level integration fixture; this is not device machine code.
        (self.kernels / (name + ".o")).write_bytes(b"fixture")

    def levels(self, required):
        for suffix, group in [("level0", [0]), ("level1_2", [1, 2]), ("level3", [3])]:
            self.descriptor(
                "LpNormV2_fp32_" + suffix,
                group,
                "false" if group == [0] else "true",
                required,
            )

    def package(self):
        target = self.config / "lp_norm_v2_apt.json"
        operator_index.main(str(self.task), str(self.kernels) + "/", str(target), 0)
        global_index.gen_binary_info_config(str(self.root), "ascend950")
        return json.loads(target.read_text()), json.loads(
            (self.config / "binary_info_config.json").read_text()
        )["LpNormV2"]

    def test_dual_protocol_package_for_old_and_new_consumers(self):
        required = ["LpNormV2_fp32", "LpNormV2_fp32_deterministic"]
        self.levels(required)
        self.descriptor(required[0], flag="false")
        self.descriptor(required[1], flag="true")
        op, glob = self.package()
        self.assertEqual(len(op["binList"]), 2)
        self.assertEqual(len(op["binListWithLoadPriority"]), 3)
        self.assertTrue(all(not is_level_entry(x) for x in op["binList"]))
        self.assertTrue(all(is_level_entry(x) for x in op["binListWithLoadPriority"]))
        self.assertEqual(
            [entry["deterministic"] for entry in op["binList"]], ["false", "true"]
        )
        for entry in op["binList"] + op["binListWithLoadPriority"]:
            self.assertNotIn("deterministicLevelBinInfo", entry)
        for data, name in [(glob, "binaryList")]:
            entries = data[name]
            legacy = [x for x in entries if not is_level_entry(x)]
            grades = [x for x in entries if is_level_entry(x)]
            self.assertEqual(len(legacy), 2)
            self.assertEqual(len(grades), 3)
            self.assertEqual(entries, legacy + grades)
            self.assertNotIn("binListWithLoadPriority", data)
            self.assertNotIn("levelBinaryList", data)
            self.assertTrue(
                all(
                    ",dt=" not in key
                    for item in legacy
                    for key in item["simplifiedKey"]
                )
            )
            self.assertEqual(
                {
                    int(key.split(",dt=")[1].split("/")[0])
                    for item in grades
                    for key in item["simplifiedKey"]
                },
                {0, 1, 2, 3},
            )
            self.assertEqual(
                {
                    key.split("/", 2)[1]
                    for item in legacy
                    for key in item["simplifiedKey"]
                },
                {"d=0,p=1", "d=1,p=1"},
            )

    def test_one_ignore_legacy_object_covers_both_boolean_modes(self):
        self.levels(["LpNormV2_fp32"])
        self.descriptor("LpNormV2_fp32", flag="ignore")
        op, glob = self.package()
        self.assertEqual(len(op["binList"]), 1)
        self.assertEqual(len(op["binListWithLoadPriority"]), 3)
        self.assertEqual(op["binList"][0]["deterministic"], "ignore")
        self.assertEqual(len(op["binList"][0]["simplifiedKey"]), 2)
        self.assertNotIn("deterministicLevelBinInfo", op["binList"][0])
        self.assertEqual(
            [entry["deterministicLevels"] for entry in op["binListWithLoadPriority"]],
            [[0], [1, 2], [3]],
        )
        for entries in (glob["binaryList"],):
            self.assertEqual(len(entries), 4)
            self.assertEqual(sum(not is_level_entry(x) for x in entries), 1)
            self.assertFalse(is_level_entry(entries[0]))
            self.assertEqual(len(entries[0]["simplifiedKey"]), 2)

    def test_repeated_local_production_is_idempotent(self):
        self.levels(["LpNormV2_fp32"])
        self.descriptor("LpNormV2_fp32", flag="ignore")
        target = self.config / "lp_norm_v2_apt.json"
        operator_index.main(str(self.task), str(self.kernels) + "/", str(target), 0)
        before = json.loads(target.read_text())
        operator_index.main(str(self.task), str(self.kernels) + "/", str(target), 0)
        self.assertEqual(json.loads(target.read_text()), before)

    def test_missing_legacy_object_rejects_partial_new_package(self):
        self.levels(["LpNormV2_fp32"])
        self.descriptor("LpNormV2_fp32")
        (self.kernels / "LpNormV2_fp32.o").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing legacy"):
            self.package()

    def test_missing_legacy_descriptor_rejects_partial_new_package(self):
        # The declared legacy companion keeps the three level outputs but
        # loses its own descriptor; packaging must fail.
        self.levels(["LpNormV2_fp32"])
        self.descriptor("LpNormV2_fp32")
        (self.kernels / "LpNormV2_fp32.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing legacy"):
            self.package()

    def test_legacy_must_cover_true_and_false(self):
        self.levels(["LpNormV2_fp32"])
        self.descriptor("LpNormV2_fp32", flag="false")
        with self.assertRaisesRegex(ValueError, "both boolean"):
            self.package()

    def test_legacy_descriptor_cannot_contain_level_metadata(self):
        self.levels(["LpNormV2_fp32"])
        self.descriptor("LpNormV2_fp32", [0])
        with self.assertRaisesRegex(ValueError, "deterministicLevels"):
            self.package()

    def test_legacy_key_cannot_contain_dt(self):
        self.levels(["LpNormV2_fp32"])
        self.descriptor("LpNormV2_fp32")
        target = self.kernels / "LpNormV2_fp32.json"
        value = json.loads(target.read_text())
        value["supportInfo"]["simplifiedKey"] = ["LpNormV2/d=0,p=1,dt=0/0,2/0,2"]
        target.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "contains dt"):
            self.package()

    def test_published_indexes_omit_compiler_metadata_and_preserve_descriptors(self):
        self.levels(["LpNormV2_fp32"])
        self.descriptor("LpNormV2_fp32", flag="ignore")
        for path in self.kernels.glob("*_level*.json"):
            data = json.loads(path.read_text())
            data["deterministicLevelGroups"] = [[0], [1, 2], [3]]
            data["supportInfo"]["deterministicLevelGroups"] = [[0], [1, 2], [3]]
            data["supportInfo"]["legacyCompatibilityOutputs"] = ["LpNormV2_fp32"]
            path.write_text(json.dumps(data))
        before = {path: path.read_bytes() for path in self.kernels.iterdir()}
        op, glob = self.package()

        def check(value, local=False):
            if isinstance(value, dict):
                self.assertTrue(
                    {
                        "deterministicLevelGroups",
                        "legacyCompatibilityOutputs",
                    }.isdisjoint(value)
                )
                if not local:
                    self.assertNotIn("deterministicLevels", value)
                for item in value.values():
                    check(item, local)
            elif isinstance(value, list):
                for item in value:
                    check(item, local)

        check(op, local=True)
        check(glob)
        self.assertEqual(len(op["binList"]), 1)
        self.assertEqual(len(op["binListWithLoadPriority"]), 3)
        self.assertEqual(len(glob["binaryList"]), 4)
        self.assertEqual(
            [entry["deterministicLevels"] for entry in op["binListWithLoadPriority"]],
            [[0], [1, 2], [3]],
        )
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_alternate_publication_omits_compiler_metadata(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "util"))
        import ascendc_ops_config as alternate

        self.levels(["LpNormV2_fp32"])
        self.descriptor("LpNormV2_fp32", flag="ignore")
        alternate.gen_all_config(
            str(self.root / "ascend950"), "ascend950", str(self.config), False
        )
        local = json.loads((self.config / "lp_norm_v2.json").read_text())
        glob = json.loads((self.config / "binary_info_config.json").read_text())
        self.assertEqual(len(local["binList"]), 1)
        self.assertEqual(len(local["binListWithLoadPriority"]), 3)
        self.assertEqual(len(glob["LpNormV2"]["binaryList"]), 4)
        self.assertEqual(
            [
                entry["deterministicLevels"]
                for entry in local["binListWithLoadPriority"]
            ],
            [[0], [1, 2], [3]],
        )
        for field in (
            "deterministicLevelGroups",
            "legacyCompatibilityOutputs",
        ):
            self.assertNotIn('"' + field + '"', json.dumps(local))
            self.assertNotIn('"' + field + '"', json.dumps(glob))
        self.assertNotIn("deterministicLevels", json.dumps(glob))
        self.assertEqual(
            {
                int(key.split(",dt=")[1].split("/")[0])
                for entry in local["binListWithLoadPriority"]
                for key in entry["simplifiedKey"]
            },
            {0, 1, 2, 3},
        )
        for path in self.kernels.glob("*_level*.json"):
            self.assertIn(
                "deterministicLevels", json.loads(path.read_text())["supportInfo"]
            )


class AlternateLevelOutputLayout(unittest.TestCase):
    """The alternate generator must reject partial level outputs like the
    task-driven generator instead of silently publishing them."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "bin"
        self.kernels = self.root / "ascend950/lp_norm_v2"
        self.config = self.root / "config/ascend950"
        self.kernels.mkdir(parents=True)
        self.config.mkdir(parents=True)
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "util"))
        import ascendc_ops_config as alternate

        self.alternate = alternate

    def descriptor(self, name, levels=None, flag="ignore", directory=None):
        target = self.kernels if directory is None else directory
        info = support(levels)
        info["deterministic"] = flag
        if levels is None:
            modes = (0, 1) if flag == "ignore" else (int(flag == "true"),)
            info["simplifiedKey"] = [f"LpNormV2/d={mode},p=1/0,2/0,2" for mode in modes]
        (target / (name + ".json")).write_text(
            json.dumps(
                {
                    "binFileName": name,
                    "binFileSuffix": ".o",
                    "coreType": "VectorCore",
                    "supportInfo": info,
                }
            )
        )
        # Descriptor-level integration fixture; this is not device machine code.
        (target / (name + ".o")).write_bytes(b"fixture")

    def levels(self):
        for suffix, group in [("level0", [0]), ("level1_2", [1, 2]), ("level3", [3])]:
            self.descriptor(
                "LpNormV2_fp32_" + suffix,
                group,
                "false" if group == [0] else "true",
            )

    def package(self):
        self.alternate.gen_all_config(
            str(self.root / "ascend950"), "ascend950", str(self.config), True
        )

    def test_full_level_layout_alternate_packaging(self):
        self.descriptor("LpNormV2_fp32")
        self.levels()
        self.package()
        local = json.loads((self.config / "lp_norm_v2.json").read_text())
        self.assertEqual(len(local["binList"]), 1)
        self.assertEqual(len(local["binListWithLoadPriority"]), 3)

    def test_partial_level_outputs_fail_alternate_packaging(self):
        # The compile only produced level0; the other group outputs are
        # missing and packaging must fail instead of indexing one level.
        self.descriptor("LpNormV2_fp32")
        self.descriptor("LpNormV2_fp32_level0", [0], "false")
        with self.assertRaisesRegex(FileNotFoundError, "Incomplete level"):
            self.package()

    def test_unreadable_level_descriptor_does_not_satisfy_layout(self):
        # A level file that cannot register still leaves its group missing.
        self.descriptor("LpNormV2_fp32")
        self.levels()
        (self.kernels / "LpNormV2_fp32_level0.json").write_text("{}")
        with self.assertRaisesRegex(FileNotFoundError, "Incomplete level"):
            self.package()

    def test_obsolete_level_output_name_is_rejected_alternate(self):
        self.descriptor("LpNormV2_fp32")
        self.descriptor("LpNormV2_fp32_level0_1_2", [1, 2], "true")
        with self.assertRaisesRegex(
            ValueError, "Unsupported deterministic level output layout"
        ):
            self.package()

    def test_declared_groups_must_match_layout_alternate(self):
        self.descriptor("LpNormV2_fp32")
        self.levels()
        path = self.kernels / "LpNormV2_fp32_level0.json"
        data = json.loads(path.read_text())
        data["deterministicLevelGroups"] = [[3]]
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(
            ValueError, "Unsupported deterministic level layout"
        ):
            self.package()

    def test_legacy_kernel_with_level_suffix_is_not_misread(self):
        # A legacy kernel whose name contains _level0 carries no level
        # metadata; it must not require level group companions.
        self.descriptor("Sqrt_level0")
        self.package()
        local = json.loads((self.config / "lp_norm_v2.json").read_text())
        self.assertEqual(len(local["binList"]), 1)
        self.assertNotIn("binListWithLoadPriority", local)

    def test_base_declaration_without_level_files_fails_alternate(self):
        # The base descriptor declares the level layout while no _level
        # output exists; the alternate generator must fail like the
        # task-driven one instead of publishing a legacy-only package.
        self.descriptor("LpNormV2_fp32")
        path = self.kernels / "LpNormV2_fp32.json"
        data = json.loads(path.read_text())
        data["deterministicLevelGroups"] = [[0], [1, 2], [3]]
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(FileNotFoundError, "Incomplete level"):
            self.package()

    def test_base_declaration_with_full_levels_alternate_packaging(self):
        # A declaration backed by the complete level layout still packages.
        self.descriptor("LpNormV2_fp32")
        path = self.kernels / "LpNormV2_fp32.json"
        data = json.loads(path.read_text())
        data["deterministicLevelGroups"] = [[0], [1, 2], [3]]
        path.write_text(json.dumps(data))
        self.levels()
        self.package()
        local = json.loads((self.config / "lp_norm_v2.json").read_text())
        self.assertEqual(len(local["binList"]), 1)
        self.assertEqual(len(local["binListWithLoadPriority"]), 3)

    def test_validation_failure_writes_no_partial_indexes(self):
        # Index validation runs entirely before the write loop: a package
        # rejected on a later op must not leave earlier indexes on disk.
        from unittest import mock

        self.descriptor("LpNormV2_fp32")
        second = self.root / "ascend950/sqrt"
        second.mkdir()
        self.descriptor("Sqrt", directory=second)
        with mock.patch.object(
            self.alternate,
            "validate_operator_index",
            side_effect=[None, ValueError("boom")],
        ):
            with self.assertRaisesRegex(ValueError, "boom"):
                self.package()
        self.assertEqual(list(self.config.glob("*.json")), [])

    def test_missing_companions_fail_alternate_packaging(self):
        def legacy_false_only(kernels):
            info = support()
            info["deterministic"] = "false"
            (kernels / "LpNormV2_fp32.json").write_text(
                json.dumps(
                    {
                        "binFileName": "LpNormV2_fp32",
                        "binFileSuffix": ".o",
                        "coreType": "VectorCore",
                        "supportInfo": info,
                    }
                )
            )

        mutations = [
            (
                lambda k: (k / "LpNormV2_fp32_level0.o").unlink(),
                FileNotFoundError,
                "companion",
            ),
            (
                lambda k: (k / "LpNormV2_fp32.o").unlink(),
                FileNotFoundError,
                "Missing legacy",
            ),
            (
                lambda k: (k / "LpNormV2_fp32.json").unlink(),
                FileNotFoundError,
                "Missing legacy",
            ),
            (legacy_false_only, ValueError, "both boolean"),
        ]
        for mutation, error, pattern in mutations:
            with tempfile.TemporaryDirectory() as d:
                with self.subTest(pattern=pattern):
                    root = Path(d) / "bin"
                    kernels = root / "ascend950/lp_norm_v2"
                    config = root / "config/ascend950"
                    kernels.mkdir(parents=True)
                    config.mkdir(parents=True)
                    self.descriptor("LpNormV2_fp32", directory=kernels)
                    for suffix, group in [
                        ("level0", [0]),
                        ("level1_2", [1, 2]),
                        ("level3", [3]),
                    ]:
                        self.descriptor(
                            "LpNormV2_fp32_" + suffix,
                            group,
                            "false" if group == [0] else "true",
                            directory=kernels,
                        )
                    mutation(kernels)
                    with self.assertRaisesRegex(error, pattern):
                        self.alternate.gen_all_config(
                            str(root / "ascend950"), "ascend950", str(config), True
                        )


class ConsumerBinListContract(unittest.TestCase):
    """Pin the packaged local index to the opbase reader contract.

    opbase resolves kernels through GetEffectiveBinList
    (src/nnopbase/composite_op/aclnn_engine/op_kernel.cpp): it reads
    BIN_LIST_WITH_LOAD_PRIORITY first and, when the field is missing,
    falls back to the legacy binList and disables deterministic level
    matching. Level candidates only register when the producer publishes
    them under the exact consumer field name, so the packaged output is
    asserted through an emulation of the consumer instead of a
    producer-side field name alone.
    """

    # opbase op_kernel.h: BIN_LIST_WITH_LOAD_PRIORITY
    BIN_LIST_WITH_LOAD_PRIORITY = "binListWithLoadPriority"

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "bin"
        self.kernels = self.root / "ascend950/lp_norm_v2"
        self.config = self.root / "config/ascend950"
        self.kernels.mkdir(parents=True)
        self.config.mkdir(parents=True)
        self.task = Path(self.directory.name) / "task.json"
        self.task.write_text(
            json.dumps(
                {"op_type": "LpNormV2", "op_list": [{"bin_filename": "LpNormV2_fp32"}]}
            )
        )

    def descriptor(self, name, levels=None, flag="ignore"):
        info = support(levels)
        info["deterministic"] = flag
        if levels is None:
            modes = (0, 1) if flag == "ignore" else (int(flag == "true"),)
            info["simplifiedKey"] = [f"LpNormV2/d={mode},p=1/0,2/0,2" for mode in modes]
        (self.kernels / (name + ".json")).write_text(
            json.dumps(
                {
                    "binFileName": name,
                    "binFileSuffix": ".o",
                    "coreType": "VectorCore",
                    "supportInfo": info,
                }
            )
        )
        # Descriptor-level integration fixture; this is not device machine code.
        (self.kernels / (name + ".o")).write_bytes(b"fixture")

    def packaged_levels(self):
        self.descriptor("LpNormV2_fp32")
        for suffix, group in [("level0", [0]), ("level1_2", [1, 2]), ("level3", [3])]:
            self.descriptor(
                "LpNormV2_fp32_" + suffix,
                group,
                "false" if group == [0] else "true",
            )
        target = self.config / "lp_norm_v2_apt.json"
        operator_index.main(str(self.task), str(self.kernels) + "/", str(target), 0)
        return json.loads(target.read_text())

    def effective_bin_list(self, op):
        """Mirror of consumer GetEffectiveBinList: prefer the load-priority
        list; a missing field falls back to binList and disables the
        deterministic level key."""
        if self.BIN_LIST_WITH_LOAD_PRIORITY in op:
            return op[self.BIN_LIST_WITH_LOAD_PRIORITY], True
        return op.get("binList", []), False

    def select_bin_by_level(self, entries, level, level_key_enabled):
        """Mirror of the consumer level-key match on the d,p,dt segment."""
        if not level_key_enabled:
            return None
        for entry in entries:
            for key in entry.get("simplifiedKey", []):
                fields = key.split("/", 2)
                match = (
                    re.fullmatch(r"d=([01]),p=([0-7]),dt=([0-3])", fields[1])
                    if len(fields) == 3
                    else None
                )
                if match is not None and int(match.group(3)) == level:
                    return entry
        return None

    def test_level_candidates_register_under_consumer_field(self):
        op = self.packaged_levels()
        self.assertEqual(len(op["binList"]), 1)
        self.assertNotIn("levelBinList", op)
        entries, level_key_enabled = self.effective_bin_list(op)
        self.assertTrue(level_key_enabled)
        expected_paths = {
            0: "LpNormV2_fp32_level0.json",
            1: "LpNormV2_fp32_level1_2.json",
            2: "LpNormV2_fp32_level1_2.json",
            3: "LpNormV2_fp32_level3.json",
        }
        for level, path in sorted(expected_paths.items()):
            with self.subTest(level=level):
                entry = self.select_bin_by_level(entries, level, level_key_enabled)
                self.assertIsNotNone(entry)
                self.assertTrue(entry["binInfo"]["jsonFilePath"].endswith(path))

    def test_levels_outside_consumer_field_never_register(self):
        op = self.packaged_levels()
        # Same package with level entries under a field the consumer does
        # not read (the pre-fix producer name): GetEffectiveBinList falls
        # back to the legacy binList and disables level matching, so no
        # level resolves to an object path.
        op["levelBinList"] = op.pop(self.BIN_LIST_WITH_LOAD_PRIORITY)
        entries, level_key_enabled = self.effective_bin_list(op)
        self.assertFalse(level_key_enabled)
        for level in range(4):
            with self.subTest(level=level):
                self.assertIsNone(
                    self.select_bin_by_level(entries, level, level_key_enabled)
                )


class GlobalUnifiedListOrder(unittest.TestCase):
    def entry(self, levels=None):
        value = support(levels)
        value["coreType"] = 2
        if levels is None:
            value["simplifiedKey"] = [
                f"LpNormV2/d={mode},p=1/0,2/0,2" for mode in (0, 1)
            ]
        return value

    def test_grade_first_is_stably_partitioned_without_mutating_input(self):
        legacy1, legacy2 = self.entry(), self.entry()
        grades = [self.entry([3]), self.entry([0]), self.entry([1, 2])]
        entries = [grades[0], legacy1, grades[1], legacy2, grades[2]]
        before = copy.deepcopy(entries)
        self.assertEqual(finalize_binary_list(entries), [legacy1, legacy2] + grades)
        self.assertEqual(entries, before)

    def test_level_metadata_must_stay_canonical_in_unified_list(self):
        legacy = self.entry()
        conflicting = self.entry([3])
        conflicting["simplifiedKey"] = ["LpNormV2/d=0,p=1,dt=3/0,2/0,2"]
        with self.assertRaisesRegex(ValueError, "Conflicting boolean"):
            finalize_binary_list([legacy, conflicting])
        mismatched = self.entry([3])
        mismatched["deterministicLevels"] = [1]
        with self.assertRaisesRegex(ValueError, "must agree"):
            finalize_binary_list([legacy, mismatched])
        mixed = self.entry()
        mixed["deterministicLevels"] = [1, 2]
        mixed["simplifiedKey"] = [
            "LpNormV2/d=1,p=1,dt=1/0,2/0,2",
            "LpNormV2/d=1,p=1/0,2/0,2",
        ]
        with self.assertRaisesRegex(ValueError, "canonical d,p,dt"):
            finalize_binary_list([mixed])

    def test_opaque_custom_tail_dt_is_not_a_level_declaration(self):
        legacy = self.entry()
        legacy["simplifiedKey"] = ["LpNormV2/d=1,p=1/custom,dt=3/tail"]
        self.assertFalse(is_level_entry(legacy))
        self.assertEqual(index_list_name(legacy, "binaryList"), "binaryList")
        self.assertEqual(finalize_binary_list([legacy]), [legacy])


if __name__ == "__main__":
    unittest.main()
