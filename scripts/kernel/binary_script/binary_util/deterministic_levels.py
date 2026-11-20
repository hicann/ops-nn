# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Classify deterministic level entries and keep legacy entries first."""

import json
import os
import re


LEVEL_GROUP_LAYOUT = [[0], [1, 2], [3]]

LEVEL_FILE_NAME_RE = re.compile(
    r"^(?P<base>.+)_level(?P<group>[0-9]+(?:_[0-9]+)*)\.json$"
)


def index_list_name(support_info, legacy_name):
    levels = support_info.get("deterministicLevels")
    keys = support_info.get("simplifiedKey", [])
    has_dt = any(
        ",dt=" in key.split("/", 2)[1]
        for key in keys
        if isinstance(key, str) and "/" in key
    )
    if levels is None:
        if has_dt:
            raise ValueError("dt keys require deterministicLevels metadata")
        return legacy_name
    if (
        not isinstance(levels, list)
        or not levels
        or any(type(level) is not int or level not in range(4) for level in levels)
        or len(set(levels)) != len(levels)
    ):
        raise ValueError("deterministicLevels must contain unique integers in 0..3")
    if 0 in levels and len(levels) != 1:
        raise ValueError("level 0 cannot share a group with deterministic levels")
    if not keys:
        raise ValueError("level binaries require simplifiedKey")
    observed = set()
    for key in keys:
        fields = key.split("/", 2)
        if len(fields) < 3:
            raise ValueError("invalid simplifiedKey")
        match = re.fullmatch(r"d=([01]),p=([0-7]),dt=([0-3])", fields[1])
        if not match:
            raise ValueError("level key must use d=...,p=...,dt=N")
        deterministic, _, level = map(int, match.groups())
        if deterministic != int(level != 0) or level not in levels:
            raise ValueError("conflicting d/dt/metadata")
        observed.add(level)
    if observed != set(levels):
        raise ValueError("not all supported levels have a matching key")
    # Split level entries out of the legacy binList only for the local
    # schema, under the opbase consumer field name binListWithLoadPriority
    # (op_kernel.h BIN_LIST_WITH_LOAD_PRIORITY): GetEffectiveBinList reads
    # that field first and falls back to binList with level matching
    # disabled, so level candidates only register under the exact consumer
    # name. Other list names keep a single combined list.
    return "binListWithLoadPriority" if legacy_name == "binList" else legacy_name


def is_level_entry(entry):
    """A level entry declares deterministicLevels or a dt field in the compile segment."""
    if "deterministicLevels" in entry:
        return True
    for key in entry.get("simplifiedKey") or []:
        if not isinstance(key, str):
            continue
        parts = key.split("/", 2)
        if (
            len(parts) > 1
            and parts[1].startswith(("d=0,", "d=1,"))
            and ",dt=" in parts[1]
        ):
            return True
    return False


def finalize_binary_list(entries):
    """Stable-partition so legacy entries precede level entries.

    Level entries must stay canonical: every simplifiedKey uses the
    d,p,dt compile segment, d agrees with dt, and deterministicLevels
    agrees with the declared keys. Lists without level entries are
    returned unchanged.
    """
    if not isinstance(entries, list):
        raise ValueError("binary list must be an array")
    if not any(is_level_entry(entry) for entry in entries):
        return entries
    legacy_entries = [entry for entry in entries if not is_level_entry(entry)]
    level_entries = [entry for entry in entries if is_level_entry(entry)]
    for entry in level_entries:
        keys = entry.get("simplifiedKey", [])
        if not isinstance(keys, list) or not keys:
            raise ValueError("Unified level entries require simplifiedKey")
        levels = set()
        for key in keys:
            if not isinstance(key, str):
                raise ValueError("Level simplifiedKey must be a string")
            parts = key.split("/", 2)
            match = (
                re.fullmatch(r"d=([01]),p=([0-7]),dt=([0-3])", parts[1])
                if len(parts) == 3
                else None
            )
            if match is None:
                raise ValueError("Level key must use canonical d,p,dt fields")
            boolean, _, level = map(int, match.groups())
            if boolean != int(level != 0):
                raise ValueError("Conflicting boolean and deterministic level")
            levels.add(level)
        declared = entry.get("deterministicLevels")
        if declared is not None:
            if (
                not isinstance(declared, list)
                or any(
                    type(level) is not int or level not in range(4)
                    for level in declared
                )
                or len(set(declared)) != len(declared)
                or set(declared) != levels
            ):
                raise ValueError(
                    "deterministicLevels must agree with unified index keys"
                )
    return legacy_entries + level_entries


def validate_operator_index(config):
    """Keep the local binList and binListWithLoadPriority separate.

    The two lists keep their separation and order: binList only carries
    legacy entries so old composite consumers never see level entries,
    binListWithLoadPriority only carries level entries and must use the
    opbase consumer field name. No nested publication is performed on
    the local index.
    """
    legacy = config.get("binList", [])
    if not isinstance(legacy, list):
        raise ValueError("Local binList must be an array")
    if any(is_level_entry(entry) for entry in legacy):
        raise ValueError("Local binList must not expose level entries to old composite")
    levels = config.get("binListWithLoadPriority")
    if levels is None:
        return
    if not isinstance(levels, list):
        raise ValueError("Local binListWithLoadPriority must be an array")
    if any(not is_level_entry(entry) for entry in levels):
        raise ValueError(
            "Local binListWithLoadPriority must contain only level entries"
        )


def validate_legacy_compatibility(kernel_output, names):
    """Fail packaging when declared legacy fallback outputs are incomplete."""
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("legacyCompatibilityOutputs must be a nonempty unique list")
    covered = set()
    for name in names:
        if not isinstance(name, str) or not name or os.path.basename(name) != name:
            raise ValueError("Invalid legacy compatibility basename")
        path = os.path.join(kernel_output, name)
        for suffix in (".json", ".o"):
            if not os.path.isfile(path + suffix):
                raise FileNotFoundError(
                    f"Missing legacy compatibility output: {path + suffix}"
                )
        with open(path + ".json", encoding="utf-8") as stream:
            info = json.load(stream).get("supportInfo", {})
        if "deterministicLevels" in info:
            raise ValueError(
                "Legacy compatibility descriptor contains deterministicLevels"
            )
        keys = info.get("simplifiedKey", [])
        if (
            not isinstance(keys, list)
            or not keys
            or any(not isinstance(key, str) for key in keys)
        ):
            raise ValueError("Legacy compatibility keys must be a nonempty string list")
        for key in keys:
            head = key.split("/", 2)[1] if "/" in key else ""
            if (head.startswith("d=0,") or head.startswith("d=1,")) and ",dt=" in head:
                raise ValueError("Legacy compatibility key contains dt")
        flag = info.get("deterministic")
        if flag == "ignore":
            covered.update((False, True))
        elif flag in ("false", "true"):
            covered.add(flag == "true")
        else:
            raise ValueError(
                "Legacy compatibility descriptor has invalid deterministic flag"
            )
    if covered != {False, True}:
        raise ValueError("Legacy compatibility outputs must cover both boolean modes")


def validate_companion_object(descriptor_path, contents):
    """The object a level descriptor indexes must exist next to it."""
    name = contents.get("binFileName")
    suffix = contents.get("binFileSuffix")
    if not isinstance(name, str) or not isinstance(suffix, str):
        raise ValueError(
            f"Descriptor misses binFileName/binFileSuffix: {descriptor_path}"
        )
    object_path = os.path.join(os.path.dirname(descriptor_path), name + suffix)
    if not os.path.isfile(object_path):
        raise FileNotFoundError(f"Missing level companion object: {object_path}")


def validate_family_legacy(directory, base, declared=None):
    """A level family must keep legacy companions for old composite consumers.

    Old consumers resolve through legacy boolean entries only, so a
    published level family validates its legacy companions: the declared
    legacyCompatibilityOutputs names when the compiler states them, and
    otherwise the base descriptor plus its optional _deterministic
    variant. Companions must exist with their objects and together cover
    both boolean deterministic modes.
    """
    if declared is not None:
        validate_legacy_compatibility(directory, declared)
        return
    names = [base]
    if os.path.isfile(os.path.join(directory, base + "_deterministic.json")):
        names.append(base + "_deterministic")
    validate_legacy_compatibility(directory, names)


def validate_level_descriptor_files(json_files):
    """Fail packaging when scanned level descriptors cover a partial layout.

    Deterministic level binaries publish one descriptor per level group and
    a kernel indexed with only part of the [[0], [1, 2], [3]] layout would
    leave the consumer without binary candidates at the missing levels, so
    incomplete outputs fail packaging instead of silently publishing a
    half-finished package. Any descriptor may declare deterministicLevelGroups
    and a declaration demands the complete layout even when no _level output
    was produced, mirroring the task-driven generator. Legacy kernels whose
    names merely contain a _level suffix carry no level metadata and stay
    untouched. Every level descriptor must keep its companion object, and
    every family must keep legacy companions (declared
    legacyCompatibilityOutputs, else the base descriptor and its optional
    _deterministic variant) that exist with their objects and cover both
    boolean deterministic modes.
    """
    suffixes = {"level" + "_".join(map(str, group)) for group in LEVEL_GROUP_LAYOUT}
    observed = {}
    declarations = {}
    declared_legacy = {}
    for path in json_files:
        base_name = os.path.basename(path)
        match = LEVEL_FILE_NAME_RE.fullmatch(base_name)
        with open(path, encoding="utf-8") as stream:
            contents = json.load(stream)
        if not isinstance(contents, dict):
            continue
        groups = contents.get("deterministicLevelGroups")
        if groups and sorted(groups) != LEVEL_GROUP_LAYOUT:
            raise ValueError(f"Unsupported deterministic level layout: {path}")
        key = (
            os.path.dirname(path),
            match.group("base") if match else os.path.splitext(base_name)[0],
        )
        required = contents.get("legacyCompatibilityOutputs")
        if required is not None:
            previous = declared_legacy.get(key)
            if previous is not None and previous != required:
                raise ValueError(f"Conflicting legacy compatibility outputs: {path}")
            declared_legacy[key] = required
        support_info = contents.get("supportInfo")
        if (
            match is not None
            and "binFileName" in contents
            and isinstance(support_info, dict)
            and is_level_entry(support_info)
        ):
            suffix = "level" + match.group("group")
            if suffix not in suffixes:
                raise ValueError(
                    f"Unsupported deterministic level output layout: {path}"
                )
            validate_companion_object(path, contents)
            observed.setdefault(key, set()).add(suffix)
        elif groups:
            declarations.setdefault(key, path)
    for (directory, base), found in sorted(observed.items()):
        missing = [
            os.path.join(directory, f"{base}_{suffix}.json")
            for suffix in sorted(suffixes - found)
        ]
        if missing:
            raise FileNotFoundError(f"Incomplete level outputs: {missing}")
        validate_family_legacy(directory, base, declared_legacy.get((directory, base)))
    for (directory, base), path in sorted(declarations.items()):
        missing = [
            os.path.join(directory, f"{base}_{suffix}.json")
            for suffix in sorted(suffixes - observed.get((directory, base), set()))
        ]
        if missing:
            raise FileNotFoundError(
                f"Incomplete level outputs declared in {path}: {missing}"
            )


def strip_compilation_metadata(value, keep_levels=False):
    """Omit bookkeeping; local indexes retain levels for composite registration."""
    if isinstance(value, dict):
        return {
            key: strip_compilation_metadata(item, keep_levels)
            for key, item in value.items()
            if key
            not in {
                "deterministicLevelGroups",
                "legacyCompatibilityOutputs",
            }
            and (keep_levels or key != "deterministicLevels")
        }
    if isinstance(value, list):
        return [strip_compilation_metadata(item, keep_levels) for item in value]
    return value
