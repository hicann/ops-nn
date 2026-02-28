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
 * \file binary_cross_entropy_grad_tiling_arch35.h
 * \brief
 */
#ifndef OPS_OP_TILING_RUNTIME_BINARY_CORSS_ENTROPY_GRAD_TILING_H
#define OPS_OP_TILING_RUNTIME_BINARY_CORSS_ENTROPY_GRAD_TILING_H

#include "register/tilingdata_base.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "atvoss/reduce/reduce_tiling.h"
#include "op_host/tiling_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
struct BinaryCrossEntropyGradCompileInfo {
  ge::DataType dtype;
  bool reductionIsNone{false};
  uint64_t coreNum = 0;
  uint64_t ubSize = 0;
};

class BinaryCrossEntropyGradTiling : public Ops::NN::Optiling::TilingBaseClass
{
public:
    explicit BinaryCrossEntropyGradTiling(gert::TilingContext* context) : TilingBaseClass(context) {};

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    ge::graphStatus CalcOutputDtype();
    ge::graphStatus CalcInputDtype();
    ge::graphStatus CheckInputShape();
    float CalcMeanCof();
    ge::graphStatus RunFp16BroadcastTiling(float meanCof);
    ge::graphStatus RunFp32BroadcastTiling(float meanCof);
private:
    const char *reductionStr = "";
    bool isReductionNone = false;
    bool isReductionMean = false;
    bool isReductionSum = false;
    ge::DataType outputDtype;
    ge::DataType inputXDtype;
    ge::DataType inputYDtype;
    ge::DataType inputGradOutputDtype;
    uint64_t tilingKey_ = 0;
    uint64_t hasWeight_ = 0;
    float meanCof_ = 0.0f;
};
}  // namespace optiling
#endif  // OPS_OP_TILING_RUNTIME_BINARY_CORSS_ENTROPY_GRAD_TILING_H