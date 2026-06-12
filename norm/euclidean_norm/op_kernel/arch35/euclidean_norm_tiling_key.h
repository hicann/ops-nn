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
 * \file euclidean_norm_tiling_key.h
 * \brief EuclideanNorm TilingKey 模板参数声明
 *
 * 三个 bool —— templateType / isEmptyTensor / isTailR。
 *   templateType == 0：
 *     isEmptyTensor == 0 → normal 模板：
 *       isTailR == 1 → 偶数轴 pattern，UB 内是 AR（尾 R）
 *       isTailR == 0 → 奇数轴 pattern，UB 内是 RA（尾 A）
 *     isEmptyTensor == 1 → 空 tensor 模板，isTailR 固定 0、kernel 不读。
 *   templateType == 1 → group 模板（A×R 2D 分核 + Phase 2 RA mini-kernel）：
 *     isEmptyTensor 固定 0（group 与 empty 互斥），isTailR ∈ {0,1}。
 *
 * dtype 走 DTYPE_X 编译期实例化（fp16 / bf16 / fp32 / int32 共 4 份），
 * 与 tilingkey 的 5 个组合（normal×2 + empty×1 + group×2）相乘 ⇒ 共 20 份 binary。
 */
#ifndef OPS_NORM_EUCLIDEAN_NORM_TILING_KEY_H_
#define OPS_NORM_EUCLIDEAN_NORM_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(
    EuclideanNorm,
    ASCENDC_TPL_BOOL_DECL(templateType, 0, 1),   // 0=normal/empty, 1=group
    ASCENDC_TPL_BOOL_DECL(isEmptyTensor, 0, 1),
    ASCENDC_TPL_BOOL_DECL(isTailR, 0, 1)
);

ASCENDC_TPL_SEL(
    // normal 模板：两种 tail 类型
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_BOOL_SEL(templateType, 0),
        ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(isTailR, 0, 1)
    ),
    // 空 tensor 模板：isTailR 固定 0（kernel 不读）
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_BOOL_SEL(templateType, 0),
        ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 1),
        ASCENDC_TPL_BOOL_SEL(isTailR, 0)
    ),
    // group 模板：isEmptyTensor 固定 0（group 与 empty 互斥）
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_BOOL_SEL(templateType, 1),
        ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 0),
        ASCENDC_TPL_BOOL_SEL(isTailR, 0, 1)
    )
);

#endif // OPS_NORM_EUCLIDEAN_NORM_TILING_KEY_H_
