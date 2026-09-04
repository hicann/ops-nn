/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3d_dx_v2_vec_tiling.h
 * \brief Conv3DBackpropInput vector 兜底 tiling（priority 2，不调用 GetTbeTiling）。
 */
#ifndef CONV3D_DX_V2_VEC_TILING_H
#define CONV3D_DX_V2_VEC_TILING_H

#include "conv3d_backprop_input_v2_base_tiling.h"

namespace Ops {
namespace NN {
namespace Conv {

class Conv3DDXV2VecTiling : public Conv3DBackpropInputV2Tiling {
public:
    explicit Conv3DDXV2VecTiling(gert::TilingContext* context) : Conv3DBackpropInputV2Tiling(context) { Reset(); }
    ~Conv3DDXV2VecTiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;

private:
    ge::graphStatus ComputeVecTiling();
    void FillShapeFields();
    void CalcReversePadding();
    ge::graphStatus CalcUbStrategy();
    void CalcB16UbBudget(uint64_t ubSize);
    void CalcFp32UbBudget(uint64_t ubSize);
    ge::graphStatus CheckWeightUbBudget(uint64_t ubSize);
    void SetupMultiCorePartition();
    // 原始输入 dtype，用于区分 FP16/BF16/FP32（SetRunInfoToV2 会把 BF16 归一为 FP16）
    ge::DataType vecDtype_ = ge::DT_FLOAT16;
};

} // namespace Conv
} // namespace NN
} // namespace Ops

#endif // CONV3D_DX_V2_VEC_TILING_H
