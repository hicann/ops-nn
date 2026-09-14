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
 * \file adaptive_avg_pool3d_grad_ncdhw_fields.h
 * \brief AdaptiveAvgPool3DGrad NCDHW big/small kernel 共用的成员字段。
 *        两个 kernel 原先各自声明了内容一致的轴切分 / 偏移 / 窗口边界字段，此处收编为唯一定义。
 *        仅为字段聚合，不含任何计算语义，窗口边界公式与梯度分配仍由各 kernel 自身实现。
 */

#ifndef POOL_UTILS_ARCH35_ADAPTIVE_AVG_POOL3D_GRAD_NCDHW_FIELDS_H_
#define POOL_UTILS_ARCH35_ADAPTIVE_AVG_POOL3D_GRAD_NCDHW_FIELDS_H_

#include <cstdint>

namespace PoolUtils {
namespace Fields {

/*
 * 功能：AdaptiveAvgPool3DGrad NCDHW 系 kernel 共用字段。
 * 说明：字段顺序与原 big/small kernel 中的声明顺序保持一致，避免改变既有访问方式；
 *       各 kernel 特有字段（如 big 的 wOutputAligned_、small 的 curLoopNum_）仍留在各自类中。
 */
class AdaptiveAvgPool3dGradNcdhwFields {
protected:
    uint32_t blockIdx_ = 0;

    int64_t dOutput_ = 1;
    int64_t hOutput_ = 1;
    int64_t wOutput_ = 1;

    int64_t dGradInput_ = 1;
    int64_t hGradInput_ = 1;
    int64_t wGradInput_ = 1;

    int64_t highAxisInner_ = 1;
    int64_t highAxisTail_ = 1;
    int64_t highAxisOuter_ = 1;
    int64_t highAxisActual_ = 1;

    int64_t dOutputInner_ = 1;
    int64_t dOutputTail_ = 1;
    int64_t dOutputOuter_ = 1;
    int64_t dOutputActual_ = 1;

    int64_t hOutputInner_ = 1;
    int64_t hOutputTail_ = 1;
    int64_t hOutputOuter_ = 1;
    int64_t hOutputActual_ = 1;

    int64_t wOutputInner_ = 1;
    int64_t wOutputTail_ = 1;
    int64_t wOutputOuter_ = 1;
    int64_t wOutputActual_ = 1;

    int64_t normalCoreProcessNum_ = 1;
    int64_t tailCoreProcessNum_ = 1;
    int64_t curCoreProcessNum_ = 1;
    int64_t usedCoreNum_ = 1;

    int64_t highAxisIndex_ = 0;
    int64_t hAxisIndex_ = 0;
    int64_t wAxisIndex_ = 0;
    int64_t dAxisIndex_ = 0;

    int64_t hGradInputActual_ = 0;
    int64_t dGradInputActual_ = 0;
    int64_t wGradInputActual_ = 0;

    int64_t gradInputPlaneSize_ = 0;

    int64_t highAxisGradInputOffset_ = 0;
    int64_t hAxisGradInputOffset_ = 0;
    int64_t dAxisGradInputOffset_ = 0;
    int64_t wAxisGradInputOffset_ = 0;
};

} // namespace Fields
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_ADAPTIVE_AVG_POOL3D_GRAD_NCDHW_FIELDS_H_
