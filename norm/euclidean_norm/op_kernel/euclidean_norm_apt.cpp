/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file euclidean_norm_apt.cpp
 * \brief EuclideanNorm 算子 kernel 入口分发（APT entry）。
 *
 * 模板参数 templateType / isEmptyTensor / isTailR（由 TilingKey TPL_BOOL_DECL 编译期展开）
 *   + 输入 dtype（由 DTYPE_X 编译期实例化，框架按 fp16/bf16/fp32/int32 自动生成 4 份实例）
 *   normal 模板：templateType=0, isEmptyTensor=0, isTailR ∈ {0,1} → 4 × 2 = 8 份
 *   empty  模板：templateType=0, isEmptyTensor=1, isTailR 固定 0      → 4 × 1 = 4 份
 *   group  模板：templateType=1, isEmptyTensor=0, isTailR ∈ {0,1} → 4 × 2 = 8 份
 *   合计 20 份 binary。
 */
#include "arch35/euclidean_norm.h"
#include "arch35/euclidean_norm_empty.h"

template <bool templateType, bool isEmptyTensor, bool isTailR>
__global__ __aicore__ void euclidean_norm(GM_ADDR x, GM_ADDR axes, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(EuclideanNormTilingData);
    GET_TILING_DATA_WITH_STRUCT(EuclideanNormTilingData, tilingData, tiling);

    if constexpr (isEmptyTensor) {
        NsEuclideanNorm::EuclideanNormEmptyKernel<DTYPE_X> op;
        op.Init(y, &tilingData);
        op.Process();
        (void)x;
    } else if constexpr (templateType) {
        // Group 模板：A×R 2D 分核 Phase 1 → SyncAll → Phase 2 RA mini-kernel
        NsEuclideanNorm::EuclideanNormKernel<DTYPE_X, isTailR> op;
        op.InitGroup(x, y, workspace, &tilingData);
        op.ProcessGroup();
    } else {
        NsEuclideanNorm::EuclideanNormKernel<DTYPE_X, isTailR> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    (void)axes;
}
