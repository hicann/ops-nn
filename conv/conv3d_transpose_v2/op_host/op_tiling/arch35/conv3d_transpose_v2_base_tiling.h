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
#ifndef CONV3D_TRANSPOSE_V2_BASE_TILING_ARCH35_H
#define CONV3D_TRANSPOSE_V2_BASE_TILING_ARCH35_H

#include "conv/conv3d_backprop_input_v2/op_host/op_tiling/arch35/conv3d_backprop_input_v2_base_tiling.h"
#include "conv/conv3d_backprop_input_v2/op_host/op_tiling/arch35/conv3d_backprop_input_v2_fullLoad_tiling.h"
#include "conv/conv3d_backprop_input_v2/op_host/op_tiling/arch35/conv3d_backprop_input_v2_inner_product_tiling.h"
#include "conv/conv3d_backprop_input_v2/op_host/op_tiling/arch35/conv3d_backprop_input_v2_kernel_split_fullLoad_tiling.h"
#include "conv/conv3d_backprop_input_v2/op_host/op_tiling/arch35/conv3d_backprop_input_v2_kernel_split_tiling.h"
#include "conv/conv3d_backprop_input_v2/op_host/op_tiling/arch35/conv3d_backprop_input_v2_small_shape_tiling.h"
#include "conv/conv3d_transpose_v2/op_kernel/conv3d_transpose_v2_arch35_tiling_key.h"

namespace Ops {
namespace NN {
namespace Conv {

class Conv3DTransposeV2TilingArch35 : public Conv3DBackpropInputV2TilingArch35 {
public:
    explicit Conv3DTransposeV2TilingArch35(gert::TilingContext* context) : Conv3DBackpropInputV2TilingArch35(context)
    {
        Reset();
        opType_ = optiling::OpTypeV2::kConv3DTransposeV2;
    }
    ~Conv3DTransposeV2TilingArch35() override = default;
};

class Conv3DTransposeV2SmallShapeTiling : public Conv3DDXV2SmallShapeTiling {
public:
    explicit Conv3DTransposeV2SmallShapeTiling(gert::TilingContext* context) : Conv3DDXV2SmallShapeTiling(context)
    {
        Reset();
        opType_ = optiling::OpTypeV2::kConv3DTransposeV2;
    }
    ~Conv3DTransposeV2SmallShapeTiling() override = default;
};

class Conv3DTransposeV2FullLoadTiling : public Conv3DDXV2FullLoadTiling {
public:
    explicit Conv3DTransposeV2FullLoadTiling(gert::TilingContext* context) : Conv3DDXV2FullLoadTiling(context)
    {
        Reset();
        opType_ = optiling::OpTypeV2::kConv3DTransposeV2;
    }
    ~Conv3DTransposeV2FullLoadTiling() override = default;
};

class Conv3DTransposeV2InnerProductTiling : public Conv3DDXV2InnerProductTiling {
public:
    explicit Conv3DTransposeV2InnerProductTiling(gert::TilingContext* context) : Conv3DDXV2InnerProductTiling(context)
    {
        Reset();
        opType_ = optiling::OpTypeV2::kConv3DTransposeV2;
    }
    ~Conv3DTransposeV2InnerProductTiling() override = default;
};

class Conv3DTransposeV2KernelSplitTiling : public Conv3DDXV2KernelSplitTiling {
public:
    explicit Conv3DTransposeV2KernelSplitTiling(gert::TilingContext* context) : Conv3DDXV2KernelSplitTiling(context)
    {
        Reset();
        opType_ = optiling::OpTypeV2::kConv3DTransposeV2;
    }
    ~Conv3DTransposeV2KernelSplitTiling() override = default;
};

class Conv3DTransposeV2KernelSplitFullLoadTiling : public Conv3DDXV2KernelSplitFullLoadTiling {
public:
    explicit Conv3DTransposeV2KernelSplitFullLoadTiling(gert::TilingContext* context)
        : Conv3DDXV2KernelSplitFullLoadTiling(context)
    {
        Reset();
        opType_ = optiling::OpTypeV2::kConv3DTransposeV2;
    }
    ~Conv3DTransposeV2KernelSplitFullLoadTiling() override = default;
};

} // namespace Conv
} // namespace NN
} // namespace Ops

#endif // CONV3D_TRANSPOSE_V2_BASE_TILING_ARCH35_H
