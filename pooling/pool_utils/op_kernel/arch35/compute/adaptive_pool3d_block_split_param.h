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
 * \file adaptive_pool3d_block_split_param.h
 * \brief AdaptiveAvgPool3D / AdaptiveMaxPool3D parallel pool 模板共用的核内切分参数结构体。
 *        Avg/Max 两侧原本各自定义了内容完全一致的 BlockSplitParam，此处收编为唯一定义。
 *        仅为纯数据结构，不含任何归约语义，Avg/Max 的池化语义仍由各自 kernel 实现。
 */

#ifndef POOL_UTILS_ARCH35_COMPUTE_ADAPTIVE_POOL3D_BLOCK_SPLIT_PARAM_H_
#define POOL_UTILS_ARCH35_COMPUTE_ADAPTIVE_POOL3D_BLOCK_SPLIT_PARAM_H_

#include <cstdint>

namespace PoolUtils {
namespace Compute {

/*
 * 功能：3D Adaptive Pool parallel pool 模板的核内切分参数。
 * 说明：字段顺序与原算子侧定义保持一致，保证结构体布局与既有访问方式不变。
 */
struct AdaptivePool3dBlockSplitParam {
    int64_t ncIdx;
    int64_t doIdx;
    int64_t hoIdx;
    int64_t woIdx;
    int64_t ncNum;
    int64_t doNum;
    int64_t hoNum;
    int64_t woNum;

    int64_t kerDStartIdx;
    int64_t kerHStartIdx;
    int64_t kerWStartIdx;

    int64_t diDataLen;
    int64_t hiDataLen;
    int64_t wiDataLen;
    int64_t xOffset;
};

} // namespace Compute
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_COMPUTE_ADAPTIVE_POOL3D_BLOCK_SPLIT_PARAM_H_
