#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
#
# golden(y) = saturate_cast_to(y_dtype)( rint( x / scales + zero_points ) ), computed in fp32 to mirror the
# ascend910b (DAV_2201) kernel numeric path (Cast x -> fp32, fp32 Div by scale, fp32 Add offset, rint, saturate).
#
# argv: <x_dtype> <scales_dtype> <zp_dtype> <y_dtype> <mode:pt|pc> <has_zp:0|1> <saturate:0|1> <shape_csv> <axis>

import sys
import os
import numpy as np
import ml_dtypes

DT = {
    "float32": np.float32,
    "float16": np.float16,
    "bfloat16": ml_dtypes.bfloat16,
    "int8": np.int8,
    "uint8": np.uint8,
    "int32": np.int32,
}
YRANGE = {
    "int8": (-128, 127),
    "uint8": (0, 255),
    "int32": (-2147483648, 2147483647),
}


def main():
    x_dt, s_dt, z_dt, y_dt, mode = (
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
        sys.argv[5],
    )
    has_zp, saturate = int(sys.argv[6]), int(sys.argv[7])
    shape = [int(v) for v in sys.argv[8].split(",")]
    axis = int(sys.argv[9])

    os.system("rm -rf *.bin")
    rank = len(shape)
    a = axis if axis >= 0 else axis + rank
    total = int(np.prod(shape)) if shape else 1

    if saturate:
        xv = np.random.uniform(-400.0, 400.0, total).astype(np.float32)
    else:
        xv = np.random.uniform(-6.0, 6.0, total).astype(np.float32)
    x = xv.reshape(shape).astype(DT[x_dt])

    chan = 1 if mode == "pt" else shape[a]
    scv = np.random.uniform(0.5, 2.0, chan).astype(np.float32)
    scales = scv.astype(DT[s_dt])

    zp = None
    if has_zp:
        if z_dt == "uint8":
            zpv = np.random.randint(0, 8, chan).astype(np.float32)
        else:
            zpv = np.random.randint(-8, 8, chan).astype(np.float32)
        zp = zpv.astype(DT[z_dt])

    # golden in fp32, using the dtype-rounded values (matches kernel Cast/ToFloat)
    xf = x.astype(np.float32)
    sf = scales.astype(np.float32)
    if mode == "pt":
        sfb = sf[0]
        zfb = zp.astype(np.float32)[0] if has_zp else np.float32(0.0)
    else:
        bshape = [1] * rank
        bshape[a] = chan
        sfb = sf.reshape(bshape)
        zfb = zp.astype(np.float32).reshape(bshape) if has_zp else np.float32(0.0)

    val = (xf / sfb + zfb).astype(np.float32)
    g = np.rint(val)
    lo, hi = YRANGE[y_dt]
    g = np.clip(g, lo, hi).astype(DT[y_dt])

    x.tofile("x.bin")
    scales.tofile("scales.bin")
    if has_zp:
        zp.tofile("zp.bin")
    g.tofile("golden.bin")
    print(
        "gen_data:",
        x_dt,
        s_dt,
        z_dt,
        y_dt,
        mode,
        "has_zp",
        has_zp,
        "sat",
        saturate,
        "shape",
        shape,
        "axis",
        axis,
    )


if __name__ == "__main__":
    main()
