/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */
/*!
 * \file apply_adam_d_tiling_data.h
 * \brief ApplyAdamD TilingData structure (host <-> kernel shared, name-identical)
 *
 * 仅承载 shape 切分信息。6 个标量参数(beta1_power/beta2_power/lr/beta1/beta2/epsilon)
 * 是运行期 GM [1] 张量输入，host tiling 无法读其数值，由 kernel 在 GM 侧读取(见 .h LoadScalars)。
 * use_nesterov 仅用于选 TilingKey(编译期模板参数)，不落 TilingData。
 */
#ifndef APPLY_ADAM_D_TILING_DATA_H_
#define APPLY_ADAM_D_TILING_DATA_H_

struct ApplyAdamDTilingData {
    int64_t totalNum = 0;      // var 元素总数(= GetShapeSize，rank 拍平后)
    int64_t blockFactor = 0;   // 每核元素数(按 32B/sizeof(T) 对齐)
    int64_t ubFactor = 0;      // 单次 UB loop 元素数(tileLen)
};

#endif  // APPLY_ADAM_D_TILING_DATA_H_
