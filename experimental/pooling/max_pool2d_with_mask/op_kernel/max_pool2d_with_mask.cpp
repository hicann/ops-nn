/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "max_pool2d_with_mask.h"
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ >= 220
#include "max_pool2d_with_mask_window.h"
#endif
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
#include "max_pool2d_with_mask_m200.h"
#include "max_pool2d_with_mask_m200_packed_intervals.h"
#include "max_pool2d_with_mask_m200_encoded.h"
#include "max_pool2d_with_mask_m200_transpose.h"
#include "max_pool2d_with_mask_m200_spatial.h"
#include "max_pool2d_with_mask_m200_spatial_half_input.h"
#include "max_pool2d_with_mask_m200_spatial_separable.h"
#endif

extern "C" __global__ __aicore__ void max_pool2d_with_mask(GM_ADDR x, GM_ADDR out, GM_ADDR indices, GM_ADDR workspace,
                                                           GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(MaxPool2dWithMaskTilingData);
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ >= 220
    if (TILING_KEY_IS(2) || TILING_KEY_IS(3) || TILING_KEY_IS(4) || TILING_KEY_IS(5) || TILING_KEY_IS(6) ||
        TILING_KEY_IS(7) || TILING_KEY_IS(8) || TILING_KEY_IS(9) || TILING_KEY_IS(10)) {
        GM_ADDR windowTiling = tiling + sizeof(MaxPool2dWithMaskTilingData);
        GET_TILING_DATA_WITH_STRUCT(MaxPool2dWithMaskWindowTilingData, windowData, windowTiling);
        TPipe pipe;
        if (TILING_KEY_IS(10)) {
#if ORIG_DTYPE_X == DT_BF16
            MaxPool2dWithMask::SpatialSmallKernel<false, true> op;
#else
            MaxPool2dWithMask::WindowKernel<DTYPE_X> op;
#endif
            op.Init(x, out, indices, &pipe, &windowData);
            op.Process();
        }
        if (TILING_KEY_IS(6)) {
#if ORIG_DTYPE_X == DT_BF16
            MaxPool2dWithMask::SpatialSmallKernel<> op;
#else
            MaxPool2dWithMask::WindowKernel<DTYPE_X> op;
#endif
            op.Init(x, out, indices, &pipe, &windowData);
            op.Process();
        }
        if (TILING_KEY_IS(4)) {
#if ORIG_DTYPE_X == DT_BF16
            MaxPool2dWithMask::WindowKernel<DTYPE_X, false, false, true> op;
#else
            MaxPool2dWithMask::WindowKernel<DTYPE_X> op;
#endif
            op.Init(x, out, indices, &pipe, &windowData);
            op.Process();
        }
        if (TILING_KEY_IS(5)) {
#if ORIG_DTYPE_X == DT_BF16
            MaxPool2dWithMask::WindowKernel<DTYPE_X, false, true> op;
#else
            MaxPool2dWithMask::WindowKernel<DTYPE_X> op;
#endif
            op.Init(x, out, indices, &pipe, &windowData);
            op.Process();
        }
        if (TILING_KEY_IS(8)) {
#if ORIG_DTYPE_X == DT_BF16
            MaxPool2dWithMask::SpatialSmallKernel<true> op;
#else
            MaxPool2dWithMask::WindowKernel<DTYPE_X> op;
#endif
            op.Init(x, out, indices, &pipe, &windowData);
            op.Process();
        }
        if (TILING_KEY_IS(7)) {
            MaxPool2dWithMask::WindowKernel<DTYPE_X, true, false, false, true> op;
            op.Init(x, out, indices, &pipe, &windowData);
            op.Process();
        }
        if (TILING_KEY_IS(3)) {
            MaxPool2dWithMask::WindowKernel<DTYPE_X, true> op;
            op.Init(x, out, indices, &pipe, &windowData);
            op.Process();
        }
        if (TILING_KEY_IS(9)) {
#if ORIG_DTYPE_X == DT_BF16
            MaxPool2dWithMask::WindowKernel<DTYPE_X, false, false, false, false, true> op;
#else
            MaxPool2dWithMask::WindowKernel<DTYPE_X> op;
#endif
            op.Init(x, out, indices, &pipe, &windowData);
            op.Process();
        }
        if (TILING_KEY_IS(2)) {
            MaxPool2dWithMask::WindowKernel<DTYPE_X> op;
            op.Init(x, out, indices, &pipe, &windowData);
            op.Process();
        }
        return;
    }
#endif
    GET_TILING_DATA_WITH_STRUCT(MaxPool2dWithMaskTilingData, tilingData, tiling);

#if !defined(__CCE_AICORE__) || __CCE_AICORE__ >= 220
    TPipe pipe;

    constexpr uint32_t PAD_DISABLE = 0;
    constexpr uint32_t PAD_ENABLE = 1;
    const bool hasPadding = tilingData.pt != 0 || tilingData.pd != 0 || tilingData.pl != 0 || tilingData.pr != 0;
    // Every dtype branch has to cover every key the host can return: the build emits code
    // per key per compilation unit, so a dtype that omits a key simply has no binary for it
    // and the runtime reports ACL error 361001 on the shapes that select it.
    (void)hasPadding;
    // Attribute specializations keep generic control flow out of common windows.
    // They share the existing tiling and exact value/index update semantics.
#if ORIG_DTYPE_X == DT_BF16
    using VectorComputeType = float;
#else
    using VectorComputeType = DTYPE_X;
#endif
    if (TILING_KEY_IS(11)) {
        MaxPool2dWithMaskVector<DTYPE_X, VectorComputeType, PAD_ENABLE, 3> op;
        op.Init(x, out, indices, &pipe, &tilingData);
        op.Process();
        return;
    }

#if ORIG_DTYPE_X == DT_BF16
    if (TILING_KEY_IS(1)) {
        MaxPool2dWithMaskVector<DTYPE_X, float, PAD_ENABLE> op;
        op.Init(x, out, indices, &pipe, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(0)) {
        MaxPool2dWithMaskVector<DTYPE_X, float, PAD_DISABLE> op;
        op.Init(x, out, indices, &pipe, &tilingData);
        op.Process();
    }
#else
    // One key per padding case, so each carries a single specialisation.
    if (TILING_KEY_IS(1)) {
        MaxPool2dWithMaskVector<DTYPE_X, DTYPE_X, PAD_ENABLE> op;
        op.Init(x, out, indices, &pipe, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(0)) {
        MaxPool2dWithMaskVector<DTYPE_X, DTYPE_X, PAD_DISABLE> op;
        op.Init(x, out, indices, &pipe, &tilingData);
        op.Process();
    }
#endif

#else
    // Compile each M200 algorithm separately. The host selects the same path
    // from its completed tiling, without retaining all NC specialisations in
    // the instruction stream used by small map/general kernels.
    if (TILING_KEY_IS(44)) {
        MaxPool2dWithMask::M200SpatialHalfInputKernel<DTYPE_X> op;
        op.Init(x, out, indices, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(40)) {
        MaxPool2dWithMask::M200SpatialSeparableKernel<DTYPE_X, 112, 112, 64> op;
        op.Init(x, out, indices, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(37)) {
        MaxPool2dWithMask::M200SpatialSeparableKernel<DTYPE_X> op;
        op.Init(x, out, indices, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(38)) {
        MaxPool2dWithMask::M200SpatialKernel<DTYPE_X, true, true, true> op;
        op.Init(x, out, indices, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(25)) {
        MaxPool2dWithMask::M200SpatialKernel<DTYPE_X, true, true> op;
        op.Init(x, out, indices, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(23)) {
        MaxPool2dWithMask::M200SpatialKernel<DTYPE_X, true> op;
        op.Init(x, out, indices, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(30)) {
        MaxPool2dWithMask::M200TransposeKernel<DTYPE_X, 16, true, true, true, false, true> op;
        op.Init(x, out, indices, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(28)) {
        MaxPool2dWithMask::M200TransposeKernel<DTYPE_X, 16, true, true, true> op;
        op.Init(x, out, indices, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(24)) {
        MaxPool2dWithMask::M200TransposeKernel<DTYPE_X, 16, true, true> op;
        op.Init(x, out, indices, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(27)) {
        MaxPool2dWithMask::M200TransposeKernel<DTYPE_X, 16, false, false, false, true> op;
        op.Init(x, out, indices, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(16)) {
        MaxPool2dWithMask::M200TransposeKernel<DTYPE_X, 16, true> op;
        op.Init(x, out, indices, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(15)) {
        MaxPool2dWithMask::M200SpatialKernel<DTYPE_X> op;
        op.Init(x, out, indices, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(14)) {
        MaxPool2dWithMask::M200TransposeKernel<DTYPE_X, 128> op;
        op.Init(x, out, indices, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(13)) {
        MaxPool2dWithMask::M200TransposeKernel<DTYPE_X, 64> op;
        op.Init(x, out, indices, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(12)) {
        MaxPool2dWithMask::M200TransposeKernel<DTYPE_X> op;
        op.Init(x, out, indices, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(11)) {
        MaxPool2dWithMask::M200VectorKernel<DTYPE_X> op;
        op.Init(x, out, indices, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(10)) {
        MaxPool2dWithMask::Kernel<DTYPE_X> op;
        op.Init(x, out, indices, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(49)) {
        MaxPool2dWithMask::M200EncodedMapKernel<DTYPE_X> op;
        op.Init(x, out, indices, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(48)) {
        MaxPool2dWithMask::M200TransposeKernel<DTYPE_X, 16, false, false, false, true, false, true> op;
        op.Init(x, out, indices, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(46)) {
        MaxPool2dWithMask::M200PackedIntervalsKernel<DTYPE_X, 64> op;
        op.Init(x, out, indices, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(47)) {
        MaxPool2dWithMask::M200PackedIntervalsKernel<DTYPE_X, 128> op;
        op.Init(x, out, indices, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(50)) {
        MaxPool2dWithMask::Kernel<DTYPE_X, true> op;
        op.Init(x, out, indices, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(45)) {
        MaxPool2dWithMask::M200VectorKernel<DTYPE_X, true> op;
        op.Init(x, out, indices, &tilingData);
        op.Process();
    }
#endif
}
