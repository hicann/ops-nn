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
 * \file conv3d_transpose_v2_base_tiling.h
 * \brief
 */
#ifndef CONV3D_TRANSPOSE_V2_BASE_TILING_H
#define CONV3D_TRANSPOSE_V2_BASE_TILING_H

#include "conv/conv3d_backprop_input_v2/op_host/op_tiling/arch32/conv3d_backprop_input_v2_base_tiling.h"

namespace Ops {
namespace NN {
namespace Conv {

class Conv3DTransposeV2Tiling : public Conv3DBackpropInputV2Tiling {
public:
    explicit Conv3DTransposeV2Tiling(gert::TilingContext* context) : Conv3DBackpropInputV2Tiling(context)
    {
        Reset();
        opType_ = optiling::OpTypeV2::kConv3DTransposeV2;
    }
    ~Conv3DTransposeV2Tiling() override = default;
};

} // namespace Conv
} // namespace NN
} // namespace Ops

#endif // CONV3D_TRANSPOSE_V2_BASE_TILING_H
