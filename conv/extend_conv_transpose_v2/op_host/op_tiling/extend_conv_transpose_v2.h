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
 * \file extend_conv_transpose_v2.h
 * \brief
 */
#ifndef EXTEND_CONV_TRANSPOSE_V2_TILING_ADVANCE_H
#define EXTEND_CONV_TRANSPOSE_V2_TILING_ADVANCE_H

#include "conv/conv3d_backprop_input_v2/op_host/op_tiling/arch35/conv3d_backprop_input_v2_fullLoad_tiling.h"
#include "conv/conv3d_backprop_input_v2/op_host/op_tiling/arch35/conv3d_backprop_input_v2_inner_product_tiling.h"
#include "conv/conv3d_backprop_input_v2/op_host/op_tiling/arch35/conv3d_backprop_input_v2_kernel_split_fullLoad_tiling.h"
#include "conv/conv3d_backprop_input_v2/op_host/op_tiling/arch35/conv3d_backprop_input_v2_kernel_split_tiling.h"
#include "conv/conv3d_backprop_input_v2/op_host/op_tiling/arch35/conv3d_backprop_input_v2_small_kernel_tiling.h"
#include "conv/conv3d_backprop_input_v2/op_host/op_tiling/arch35/conv3d_backprop_input_v2_small_shape_tiling.h"

namespace Ops {
namespace NN {
namespace Conv {

class ExtendConvTransposeV2Tiling : public Conv3DDXV2InnerProductTiling {
public:
    explicit ExtendConvTransposeV2Tiling(gert::TilingContext* context) : Conv3DDXV2InnerProductTiling(context)
    {
        Reset();
        opType_ = optiling::OpTypeV2::kExtendConvTransposeV2;
    }
    ~ExtendConvTransposeV2Tiling() override = default;
};

class ExtendConvTransposeV2SmallKernelTiling : public Conv3DDXV2SmallKernelTiling {
public:
    explicit ExtendConvTransposeV2SmallKernelTiling(gert::TilingContext* context) : Conv3DDXV2SmallKernelTiling(context)
    {
        Reset();
        opType_ = optiling::OpTypeV2::kExtendConvTransposeV2;
    }
    ~ExtendConvTransposeV2SmallKernelTiling() override = default;
};

class ExtendConvTransposeV2SmallShapeTiling : public Conv3DDXV2SmallShapeTiling {
public:
    explicit ExtendConvTransposeV2SmallShapeTiling(gert::TilingContext* context) : Conv3DDXV2SmallShapeTiling(context)
    {
        Reset();
        opType_ = optiling::OpTypeV2::kExtendConvTransposeV2;
    }
    ~ExtendConvTransposeV2SmallShapeTiling() override = default;
};

class ExtendConvTransposeV2FullLoadTiling : public Conv3DDXV2FullLoadTiling {
public:
    explicit ExtendConvTransposeV2FullLoadTiling(gert::TilingContext* context) : Conv3DDXV2FullLoadTiling(context)
    {
        Reset();
        opType_ = optiling::OpTypeV2::kExtendConvTransposeV2;
    }
    ~ExtendConvTransposeV2FullLoadTiling() override = default;
};

class ExtendConvTransposeV2InnerProductTiling : public Conv3DDXV2InnerProductTiling {
public:
    explicit ExtendConvTransposeV2InnerProductTiling(gert::TilingContext* context)
        : Conv3DDXV2InnerProductTiling(context)
    {
        Reset();
        opType_ = optiling::OpTypeV2::kExtendConvTransposeV2;
    }
    ~ExtendConvTransposeV2InnerProductTiling() override = default;
};

class ExtendConvTransposeV2KernelSplitTiling : public Conv3DDXV2KernelSplitTiling {
public:
    explicit ExtendConvTransposeV2KernelSplitTiling(gert::TilingContext* context) : Conv3DDXV2KernelSplitTiling(context)
    {
        Reset();
        opType_ = optiling::OpTypeV2::kExtendConvTransposeV2;
    }
    ~ExtendConvTransposeV2KernelSplitTiling() override = default;
};

class ExtendConvTransposeV2KernelSplitFullLoadTiling : public Conv3DDXV2KernelSplitFullLoadTiling {
public:
    explicit ExtendConvTransposeV2KernelSplitFullLoadTiling(gert::TilingContext* context)
        : Conv3DDXV2KernelSplitFullLoadTiling(context)
    {
        Reset();
        opType_ = optiling::OpTypeV2::kExtendConvTransposeV2;
    }
    ~ExtendConvTransposeV2KernelSplitFullLoadTiling() override = default;
};

} // namespace Conv
} // namespace NN
} // namespace Ops
#endif // EXTEND_CONV_TRANSPOSE_V2_TILING_ADVANCE_H
