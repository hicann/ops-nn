# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""AddExamplePypto kernel — the PyPTO counterpart of examples/add_example.

Elementwise add/sub over a 2D FP16 tensor, written in the PyPTO DSL instead of AscendC C++.
The build picks this file up because the operator's CMakeLists.txt calls enable_pypto_kernel(add_example_pypto):
cmake runs scripts/util/pypto_codegen.py over it at configure time, which emits

  * AddExamplePyptoTiling_tiling.h        -- the tiling struct the host tiling fills in
  * AddExamplePyptoTilingKey_tilingkey.h  -- the ASCENDC_TPL tiling-key declarations
  * add_example_pypto_pypto_infer.cpp     -- the self-contained source used for .i infer

The first two are force-included into add_example_pypto_tiling.cpp, so the host side shares this file's
tiling layout and tiling-key bit assignment without a hand-written copy of either.
"""

from dataclasses import dataclass

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField


# ================================================================
#  Tiling data — mirrored on the host by AddExamplePyptoTilingData
# ================================================================
@dataclass
class AddExamplePyptoTiling:
    rows: int
    columns: int


# ================================================================
#  TilingKey — bit0 selects the elementwise operation
# ================================================================
class AddExamplePyptoTilingKey:
    Operation = TilingKeyField(bits=1, values=[0, 1])


# Tile shape processed per iteration. FP16, so 16 * 16 * 2 = 512 bytes per tile.
TILE_ROWS = 16
TILE_COLUMNS = 16
TILE_BYTES = TILE_ROWS * TILE_COLUMNS * 2


@pl.jit(auto_mutex=True, tiling_key=AddExamplePyptoTilingKey)
def add_example_pypto(
    x1: pl.Ptr[pl.DT_FP16],
    x2: pl.Ptr[pl.DT_FP16],
    y: pl.Ptr[pl.DT_FP16],
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: AddExamplePyptoTiling,
):
    # 注意：离线(OPC)编译下 kernel 参数用 pl.Ptr + pl.make_tensor，从 tiling 取 shape，
    # 不要用 pl.Tensor[[pl.DYNAMIC, ...]]——那会生成动态维度函数参数，而 OPC 框架
    # 只按算子原型下发输入输出/workspace/tiling 指针，不传动态维度，导致读到垃圾值。
    x1_tensor = pl.make_tensor(
        x1, [tiling.rows, tiling.columns], [tiling.columns, 1], dtype=pl.DT_FP16
    )
    x2_tensor = pl.make_tensor(
        x2, [tiling.rows, tiling.columns], [tiling.columns, 1], dtype=pl.DT_FP16
    )
    y_tensor = pl.make_tensor(
        y, [tiling.rows, tiling.columns], [tiling.columns, 1], dtype=pl.DT_FP16
    )

    # 循环内逐 tile 计算时，单一 make_tile 缓冲无法保证上一轮 TSTORE(MTE3) 完成后再被
    # 下一轮 TLOAD(MTE2) 覆盖（缺少跨迭代同步）。用 make_tile_group + mutex_ids 双缓冲，
    # 由 mutex 机制保证同一槽位被 next() 重新取用前其上的读写已完成。
    tile_type = pl.TileType(
        shape=[TILE_ROWS, TILE_COLUMNS],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    x1_group = pl.make_tile_group(
        type=tile_type, addrs=[0x0000, 0x0200], mutex_ids=[0, 1]
    )
    x2_group = pl.make_tile_group(
        type=tile_type, addrs=[0x0400, 0x0600], mutex_ids=[2, 3]
    )
    y_group = pl.make_tile_group(
        type=tile_type, addrs=[0x0800, 0x0A00], mutex_ids=[4, 5]
    )

    with pl.section_vector():
        for row in pl.range(0, tiling.rows, TILE_ROWS):
            for column in pl.range(0, tiling.columns, TILE_COLUMNS):
                tile_x1 = x1_group.next()
                tile_x2 = x2_group.next()
                tile_y = y_group.next()
                pl.load(tile_x1, x1_tensor, [row, column])
                pl.load(tile_x2, x2_tensor, [row, column])
                if Operation == 0:  # noqa: F821  (injected by the tiling key)
                    pl.add(tile_y, tile_x1, tile_x2)
                else:
                    pl.sub(tile_y, tile_x1, tile_x2)
                pl.store(y_tensor, tile_y, [row, column])
