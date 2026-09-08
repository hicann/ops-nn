/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3d_op_type_v2.h
 * \brief conv3d反传家族算子类型定义，供共享tiling框架区分算子
 */
#ifndef OPS_NN_CONV_COMMON_OP_TILING_CONV3D_OP_TYPE_V2_H
#define OPS_NN_CONV_COMMON_OP_TILING_CONV3D_OP_TYPE_V2_H

#include <cstddef>

namespace optiling {
// 枚举值会跨so传给legacy库(LegacyGenTbeConvBackwardTiling)，顺序不可变更
enum OpTypeV2 : size_t {
    kConv3DBackpropFilterV2,
    kConv3DBackpropInputV2,
    kConv3DTransposeV2,
    kExtendConvTranspose,
    kExtendConvTransposeV2,
};
} // namespace optiling

#endif // OPS_NN_CONV_COMMON_OP_TILING_CONV3D_OP_TYPE_V2_H
