#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================
"""
EmbeddingHashTableExport clamping ST: truncation and valid-prefix checks.
"""

import ctypes as ct
import struct
import sys

_ACL = ct.CDLL("libascendcl.so")
_ACL.aclInit.argtypes = [ct.c_void_p]
_ACL.aclrtSetDevice.argtypes = [ct.c_int32]
_ACL.aclrtMalloc.argtypes = [ct.POINTER(ct.c_void_p), ct.c_size_t, ct.c_int32]
_ACL.aclrtMemset.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_int32, ct.c_size_t]
_ACL.aclrtMemcpy.argtypes = [
    ct.c_void_p,
    ct.c_size_t,
    ct.c_void_p,
    ct.c_size_t,
    ct.c_int32,
]
_ACL.aclInit(None)
if _ACL.aclrtSetDevice(0) != 0:
    sys.exit("aclrtSetDevice failed")

from es_nn import EmbeddingHashTableExport, EmbeddingHashTableLookupOrInsert  # noqa: E402
from ge.es import GraphBuilder  # noqa: E402
from ge.graph.tensor import Tensor  # noqa: E402
from ge.graph.types import DataType, Format  # noqa: E402
from ge.session import Session  # noqa: E402

_w = ct.CDLL(
    "/usr/local/Ascend/cann-9.3.0/python/site-packages/ge/_capi/libge_runtime_wrapper.so"
)
_w.GeApiWrapper_IsGEInitialized.restype = ct.c_bool
if not _w.GeApiWrapper_IsGEInitialized():
    _w.GeApiWrapper_GEInitialize.argtypes = [
        ct.POINTER(ct.c_char_p),
        ct.POINTER(ct.c_char_p),
        ct.c_int32,
    ]
    _w.GeApiWrapper_GEInitialize.restype = ct.c_int32
    keys = (ct.c_char_p * 1)(ct.c_char_p(b"ge.exec.deviceId"))
    vals = (ct.c_char_p * 1)(ct.c_char_p(b"0"))
    _w.GeApiWrapper_GEInitialize(keys, vals, 1)

DIM, BUCKETS, LK = 8, 1024, [11, 22, 33]
failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


def dev_malloc(size):
    p = ct.c_void_p()
    assert _ACL.aclrtMalloc(ct.byref(p), size, 0) == 0
    return p.value


def d2h(addr, size):
    b = ct.create_string_buffer(size)
    assert _ACL.aclrtMemcpy(b, size, ct.c_void_p(addr), size, 2) == 0
    return bytes(b.raw)


def flatten(x):
    out = []
    for e in x:
        if isinstance(e, list):
            out.extend(flatten(e))
        else:
            out.append(e)
    return out


def make_handle(table_addr):
    p = dev_malloc(64)
    buf = ct.create_string_buffer(
        struct.pack("<8q", table_addr, 0, 0, 0, 0, 0, 0, 0), 64
    )
    _ACL.aclrtMemcpy(ct.c_void_p(p), 64, buf, 64, 1)
    return p


stride = (24 + DIM * 4 + 7) // 8 * 8
table = dev_malloc(BUCKETS * stride * 8)
_ACL.aclrtMemset(table, BUCKETS * stride * 8, 0, BUCKETS * stride * 8)
handle = make_handle(table)
session = Session()

gb = GraphBuilder()
h = gb.create_input(0, data_type=DataType.DT_INT64, shape=[1])
k = gb.create_input(1, data_type=DataType.DT_INT64, shape=[len(LK)])
res = EmbeddingHashTableLookupOrInsert(
    h,
    k,
    bucket_size=BUCKETS,
    embedding_dim=DIM,
    filter_mode="no_filter",
    default_key_or_value=True,
    default_key=-1,
    default_value=0.0,
)
gb.set_graph_output(res, 0)
session.add_graph(1, gb.build_and_reset())
session.run_graph(
    1,
    [
        Tensor(
            [handle], data_type=DataType.DT_INT64, format=Format.FORMAT_ND, shape=[1]
        ),
        Tensor(
            LK, data_type=DataType.DT_INT64, format=Format.FORMAT_ND, shape=[len(LK)]
        ),
    ],
)
session.remove_graph(1)


def run_export(table_sizes, mode="all", gid=2):
    gb = GraphBuilder()
    h = gb.create_input(0, data_type=DataType.DT_INT64, shape=[1])
    ts = gb.create_input(1, data_type=DataType.DT_INT64, shape=[1])
    ed = gb.create_input(2, data_type=DataType.DT_INT64, shape=[1])
    bs = gb.create_input(3, data_type=DataType.DT_INT64, shape=[1])
    r = EmbeddingHashTableExport(
        h,
        ts,
        ed,
        bs,
        keys_num=1,
        counters_num=1,
        filter_flags_num=1,
        values_num=1,
        export_mode=mode,
    )
    for i, oh in enumerate([r.keys[0], r.counters[0], r.filter_flags[0], r.values[0]]):
        gb.set_graph_output(oh, i)
    session.add_graph(gid, gb.build_and_reset())
    outs = session.run_graph(
        gid,
        [
            Tensor(
                [handle],
                data_type=DataType.DT_INT64,
                format=Format.FORMAT_ND,
                shape=[1],
            ),
            Tensor(
                [table_sizes],
                data_type=DataType.DT_INT64,
                format=Format.FORMAT_ND,
                shape=[1],
            ),
            Tensor(
                [DIM], data_type=DataType.DT_INT64, format=Format.FORMAT_ND, shape=[1]
            ),
            Tensor(
                [BUCKETS],
                data_type=DataType.DT_INT64,
                format=Format.FORMAT_ND,
                shape=[1],
            ),
        ],
    )
    session.remove_graph(gid)
    return [flatten(o.get_data()) for o in outs]


# 场景1 部分导出尾清：3 个有效 key，分配 1024
keys, counters, fflags, values = run_export(BUCKETS)
check(
    "部分导出：前缀恰好为插入 key 集合",
    sorted(keys[: len(LK)]) == sorted(LK),
    f"prefix={keys[: len(LK)]}",
)

# 场景3 截断：分配 1 小于实际 3 条
keys3, _, _, _ = run_export(1)
check("截断：恰导出 1 条且存活", len(keys3) == 1 and keys3[0] in LK, f"keys={keys3}")

sys.exit(f"FAILED: {failures}" if failures else 0)
