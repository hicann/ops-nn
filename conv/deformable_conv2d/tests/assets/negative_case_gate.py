#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""DeformableConv2D 负向 ST 用例的预期失败判定器。

TTK geir 没有原生 expect-fail 机制，负向用例在 ST 结果里会记为 FAIL/ERROR，无法与
真正的失败区分。本判定器把两者区分开：

  * 负向用例（用例名含 ``_expect_fail``）必须失败，且失败原因必须命中该用例声明的
    预期拒绝签名（见 ``EXPECTED_REJECTION``）；
  * 正向用例必须 PASS；
  * 全部符合预期返回 0，否则返回 1 并列出不符合项。

用法::

    python3 -m ttk geir --pc=1 --devices=0 --no-memory-check --run 1 \
        --plugin <op>/tests/assets/golden.py --compare stat_rel_err \
        -i <op>/tests/st/arch35/ttk_geir_deformable_conv2d.csv -o /tmp/dfc_result.csv
    python3 <op>/tests/assets/negative_case_gate.py /tmp/dfc_result.csv
"""

import csv
import re
import sys

NEGATIVE_SUFFIX = "_expect_fail"

# 用例名 -> [(说明, 正则)]，全部条目需同时命中
EXPECTED_REJECTION = {
    "ttk_geir_dfc_n01_x_not4d": [("x 形状维度非法", r"shape dim|should be 4")],
    "ttk_geir_dfc_n02_filter_format_nhwc": [("filter 格式不在白名单", r"format")],
    "ttk_geir_dfc_n03_dtype_mismatch": [("dtype 不一致", r"dtype|data ?type")],
    "ttk_geir_dfc_n04_offsets_channel_mismatch": [("offsets 通道数不符", r"offset")],
    "ttk_geir_dfc_n05_modulated_false": [
        ("modulated=False 不支持", r"modulated|support|EZ9999|internal error")
    ],
    "ttk_geir_dfc_n06_groups_mismatch": [("x 与 filter 通道/组不匹配", r"x, filter")],
    "ttk_geir_dfc_n07_stride_h_zero": [("stride=0 非法", r"stride")],
    "ttk_geir_dfc_n08_pads_negative": [("pads 负值非法", r"pad")],
    "ttk_geir_dfc_n09_bias_size_mismatch": [("bias 长度不符", r"bias")],
    "ttk_geir_dfc_n10_dfm_groups_zero": [
        ("deformable_groups=0 非法", r"deformable|x_group")
    ],
    "ttk_geir_dfc_n11_filter_kh_zero": [("filter 高度为 0", r"filter|shape|kernel")],
    "ttk_geir_dfc_n12_strides_size3": [("strides 长度非 4", r"stride")],
    "ttk_geir_dfc_n13_dilations_size3": [("dilations 长度非 4", r"dilation")],
    "ttk_geir_dfc_n14_stride_n2": [("stride N 维非 1", r"stride")],
    "ttk_geir_dfc_n15_dilation_c2": [("dilation C 维非 1", r"dilation")],
    "ttk_geir_dfc_n16_dilation_h0": [("dilation 高度为 0", r"dilation")],
    "ttk_geir_dfc_n17_dfm_not_divide": [
        ("offsets 通道不被 dfm 整除", r"deformable|offset|divis")
    ],
    "ttk_geir_dfc_n18_bias_2d": [("bias 非 1D", r"bias")],
    "ttk_geir_dfc_n19_offsets_h_mismatch": [("offsets 空间维与输出不符", r"offset")],
    "ttk_geir_dfc_n20_input_pad": [("pad 后尺寸小于 kernel", r"shape|pad|kernel")],
    "ttk_geir_dfc_n22_x_format_nd": [("x 格式非法", r"format")],
    "ttk_geir_dfc_n23_filter_5d": [("filter 非 4D", r"filter|shape")],
    "ttk_geir_dfc_n24_pads_len3": [("pads 长度非 4", r"pad")],
    "ttk_geir_dfc_n25_kn_groups": [("kn 不被 groups 整除", r"divisib|group|channel")],
    "ttk_geir_dfc_n26_offsets_3d": [("offsets 非 4D", r"offset|shape")],
    "ttk_geir_dfc_n27_offsets_format_nd": [("offsets 格式非法", r"offset|format")],
    "ttk_geir_dfc_n28_y_format_nd": [("y 格式非法", r"format")],
    "ttk_geir_dfc_n29_filter_kw_zero": [("filter kw=0", r"filter|kernel|shape")],
    "ttk_geir_dfc_n30_stride_c2": [("stride C 维非 1", r"stride")],
    "ttk_geir_dfc_n31_dilation_negative": [("dilation 负值", r"dilation")],
    "ttk_geir_dfc_n32_dilation_n2": [("dilation N 维非 1", r"dilation")],
    "ttk_geir_dfc_n33_ic_dfm": [("ic 不被 dfm 整除", r"deformable|divisib|channel")],
    "ttk_geir_dfc_n34_offsets_n_mismatch": [("offsets N 不一致", r"offset|shape")],
    "ttk_geir_dfc_n35_offsets_w_mismatch": [("offsets W 不一致", r"offset|shape")],
    "ttk_geir_dfc_n36_zero_input_h": [
        ("零张量推导负值输出", r"shape|negative|zero|input")
    ],
    "ttk_geir_dfc_n37_zero_input_nonzero_out": [
        ("零张量输入与非零输出", r"zero|empty|shape|input")
    ],
    "ttk_geir_dfc_n38_strides_w_zero": [("stride W 维非正", r"stride")],
    "ttk_geir_dfc_n39_dilations_w_negative": [("dilation W 维非正", r"dilation")],
    "ttk_geir_dfc_n40_groups_zero": [
        ("groups 为 0 (EZ0026)", r"EZ0026"),
        ("groups 文案", r"group"),
    ],
}


def _load_rows(result_csv):
    with open(result_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _status_text(row):
    return " ".join(
        str(row.get(key) or "") for key in ("precision_status", "precision")
    ).strip()


def _is_negative(name):
    return name.endswith(NEGATIVE_SUFFIX)


def _expected_key(name):
    if name in EXPECTED_REJECTION:
        return name
    base = name[: -len(NEGATIVE_SUFFIX)]
    return base if base in EXPECTED_REJECTION else None


def evaluate(rows):
    failures = []
    print(f"{'RESULT':8s} {'TESTCASE':44s} DETAIL")
    print("-" * 110)
    for row in rows:
        name = str(row.get("testcase_name") or "").strip()
        if not name:
            continue
        status = _status_text(row)
        if _is_negative(name):
            key = _expected_key(name)
            if key is None:
                print(f"{'FAIL':8s} {name:44s} 未登记预期拒绝签名")
                failures.append(name)
                continue
            checks = EXPECTED_REJECTION[key]
            rejected = bool(status) and not re.match(r"^\s*PASS\b", status, re.I)
            # judge by both EZ error code presence and message patterns: the EZ code
            # anchors the rejection layer, the patterns constrain the reason.
            ez_hit = re.search(r"EZ\d{4}", status) is not None
            matched = [desc for desc, pat in checks if re.search(pat, status, re.I)]
            if rejected and matched and not ez_hit:
                matched = []  # rejection text without an EZ code is not trusted
            if rejected and len(matched) == len(checks):
                print(f"{'PASS':8s} {name:44s} 预期拒绝命中: {', '.join(matched)}")
            elif not rejected:
                print(f"{'FAIL':8s} {name:44s} 未按预期被拒绝(用例实际通过)")
                failures.append(name)
            else:
                missing = [d for d, p in checks if not re.search(p, status, re.I)]
                print(
                    f"{'FAIL':8s} {name:44s} 拒绝原因不符, 未命中: {', '.join(missing)}"
                )
                failures.append(name)
        else:
            ok = bool(re.match(r"^\s*PASS\b", status, re.I))
            print(
                f"{'PASS' if ok else 'FAIL':8s} {name:44s} {'正向用例通过' if ok else status[:60]}"
            )
            if not ok:
                failures.append(name)
    return failures


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    rows = _load_rows(sys.argv[1])
    failures = evaluate(rows)
    total = len(rows)
    print("-" * 110)
    if failures:
        print(
            f"SUMMARY: {total - len(failures)}/{total} 符合预期; 不符合: {', '.join(failures)}"
        )
        return 1
    print(f"SUMMARY: {total}/{total} 全部符合预期（负向均按预期拒绝，正向均通过）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
