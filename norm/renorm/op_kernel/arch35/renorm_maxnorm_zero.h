/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _RENORM_MAXNORM_ZERO_H_
#define _RENORM_MAXNORM_ZERO_H_

#include "kernel_operator.h"
#include "common/renorm_common.h"

/*
 * ===================== maxNorm = 0 场景（输出全零）=====================
 *
 * 【为什么 maxNorm=0 时不需要计算范数】
 *   renorm 的缩放公式为 scale = maxNorm / max(norm, eps) if norm > maxNorm else 1.0
 *   当 maxNorm = 0 时：任何 norm > 0 都会大于 maxNorm=0，触发缩放，
 *   scale = 0 / max(norm, eps) = 0，所以子张量所有元素乘以 0 后全部变为 0。
 *   既然结果一定是全零，就无需计算范数，直接把所有缩放因子设为 0 即可。
 *
 * 【FillScalesZero 的作用】
 *   将 scaleBuf（存储各子张量缩放因子的 buffer）中对应本核处理的切片范围全部置零。
 *   后续应用缩放阶段会读取这些 0 缩放因子，将输出全部写零。
 */

namespace NsRenorm {

using namespace AscendC;

// maxNorm = 0 场景：直接输出全零
// 不需要计算范数，所有缩放因子为 0
//
// 【FillScalesZero 的作用】
//   将本核负责的 [startSlice, startSlice+slicesThisCore) 范围内的缩放因子全部置零。
//   scaleBuf 是一个 FP32 的 LocalTensor，每个元素对应一个子张量的缩放因子。
//   Duplicate 用 0.0f 填充，长度对齐到 FP32_ALIGN(8) 以满足 Vector API 对齐要求。
template <typename D_T_X>
__aicore__ inline void FillScalesZero(LocalTensor<float>& scaleBuf, int64_t startSlice, int64_t slicesThisCore)
{
    // 将 scaleBuf 中 [0, slicesThisCore) 范围置零
    Duplicate(scaleBuf, 0.0f, static_cast<int32_t>(AlignUpFp32(slicesThisCore)));
}

} // namespace NsRenorm

#endif // _RENORM_MAXNORM_ZERO_H_
