/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _TRANSPOSE_QUANT_BATCH_MAT_MUL_TILING_DEF_H_
#define _TRANSPOSE_QUANT_BATCH_MAT_MUL_TILING_DEF_H_

#include "kernel_tiling/kernel_tiling.h"
#include "../mat_mul_v3/arch35/mat_mul_tiling_data.h"

namespace AscendC {
namespace Te {
struct NDExtLayoutPtn {};
struct DNExtLayoutPtn {};
struct ZNLayoutPtn {};
struct NZLayoutPtn {};
} // namespace Te
} // namespace AscendC

namespace Blaze {
namespace Gemm {
enum class NoContiguousType { NON_CONTIGUOUS_TYPE_PERM_X1 = 0 };
} // namespace Gemm
} // namespace Blaze

template <class A_TYPE, class B_TYPE, class SCALE_TYPE, class C_TYPE, class BIAS_TYPE, class aLayout, class bLayout,
          class cLayout, uint64_t = 0, uint64_t = 0, uint64_t = 0>
inline void TqbmmMxTensorApiKernel(int, int, int, int, int, int, int)
{}

#ifndef POS_LOWEST
constexpr int32_t POS_LOWEST = 0;
#endif

#ifndef POS_HIGHEST
constexpr int32_t POS_HIGHEST = 1;
#endif

inline void InitTqbmmTilingData(void* tiling, void* const_data)
{
    memcpy(const_data, tiling, sizeof(BatchMatMulV3TilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    BatchMatMulV3TilingData tiling_data;         \
    InitTqbmmTilingData(tiling_arg, &tiling_data);
#endif

#define TQBMM_IMPL_CLASS_COMMON_TRANS(transposeX1, transposeX2, templateClass, ...)                         \
    do {                                                                                                    \
        templateClass<DTYPE_X1, DTYPE_X2, DTYPE_X2_SCALE, DTYPE_BIAS, DTYPE_X1_SCALE, DTYPE_Y, transposeX1, \
                      transposeX2, DTYPE_LOC_LOCAL, __VA_ARGS__>                                            \
            op;                                                                                             \
        op.Init(aGM, bGM, x2_scaleGM, x1_scaleGM, cGM, user, &tilingData, &pipe);                           \
        op.Process();                                                                                       \
    } while (0)
