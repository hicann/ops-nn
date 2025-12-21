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
 * \file entend_conv_transpose_tiling.h
 * \brief
 */
#ifndef EXTEND_CONV_TRANSPOSE_TILING_ADVANCE_H
#define EXTEND_CONV_TRANSPOSE_TILING_ADVANCE_H

#include "conv/conv3d_backprop_input_v2/op_host/op_tiling/arch35/conv3d_backprop_input_v2_base_tiling.h"
#include "conv/conv3d_backprop_input_v2/op_host/op_tiling/arch35/conv3d_backprop_input_v2_fullLoad_tiling.h"
#include "conv/conv3d_backprop_input_v2/op_host/op_tiling/arch35/conv3d_backprop_input_v2_inner_product_tiling.h"
#include "conv/conv3d_backprop_input_v2/op_host/op_tiling/arch35/conv3d_backprop_input_v2_kernel_split_fullLoad_tiling.h"
#include "conv/conv3d_backprop_input_v2/op_host/op_tiling/arch35/conv3d_backprop_input_v2_kernel_split_tiling.h"
#include "conv/conv3d_backprop_input_v2/op_host/op_tiling/arch35/conv3d_backprop_input_v2_small_shape_tiling.h"

namespace Ops {
namespace NN {
namespace Conv {

class ExtendConvTransposeTiling : public Conv3DBackpropInputV2TilingArch35 {
public:
    explicit ExtendConvTransposeTiling(gert::TilingContext *context) : Conv3DBackpropInputV2TilingArch35(context)
    {
        Reset();
        opType_ = optiling::OpTypeV2::kConv3DTransposeV2;
    }
    ~ExtendConvTransposeTiling() override = default;
};

class ExtendConvTransposeSmallShapeTiling : public Conv3DDXV2SmallShapeTiling {
    public:
        explicit ExtendConvTransposeSmallShapeTiling(gert::TilingContext *context) : Conv3DDXV2SmallShapeTiling(context)
        {
            Reset();
            opType_ = optiling::OpTypeV2::kConv3DTransposeV2;
        }
        ~ExtendConvTransposeSmallShapeTiling() override = default;
};

class ExtendConvTransposeFullLoadTiling : public Conv3DDXV2FullLoadTiling {
    public:
        explicit ExtendConvTransposeFullLoadTiling(gert::TilingContext *context) : Conv3DDXV2FullLoadTiling(context)
        {
            Reset();
            opType_ = optiling::OpTypeV2::kConv3DTransposeV2;
        }
        ~ExtendConvTransposeFullLoadTiling() override = default;
};

class ExtendConvTransposeInnerProductTiling : public Conv3DDXV2InnerProductTiling {
    public:
        explicit ExtendConvTransposeInnerProductTiling(gert::TilingContext *context) : Conv3DDXV2InnerProductTiling(context)
        {
            Reset();
            opType_ = optiling::OpTypeV2::kConv3DTransposeV2;
        }
        ~ExtendConvTransposeInnerProductTiling() override = default;
};

class ExtendConvTransposeKernelSplitTiling : public Conv3DDXV2KernelSplitTiling {
    public:
        explicit ExtendConvTransposeKernelSplitTiling(gert::TilingContext *context) : Conv3DDXV2KernelSplitTiling(context)
        {
            Reset();
            opType_ = optiling::OpTypeV2::kConv3DTransposeV2;
        }
        ~ExtendConvTransposeKernelSplitTiling() override = default;
};

class ExtendConvTransposeKernelSplitFullLoadTiling : public Conv3DDXV2KernelSplitFullLoadTiling {
    public:
        explicit ExtendConvTransposeKernelSplitFullLoadTiling(gert::TilingContext *context) : Conv3DDXV2KernelSplitFullLoadTiling(context)
        {
            Reset();
            opType_ = optiling::OpTypeV2::kConv3DTransposeV2;
        }
        ~ExtendConvTransposeKernelSplitFullLoadTiling() override = default;
};

} // namespace Conv
} // namespace NN
} // namespace Ops
#endif  // EXTEND_CONV_TRANSPOSE_TILING_ADVANCE_H