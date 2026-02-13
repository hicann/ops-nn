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
 * \file binary_cross_entropy_tiling_arch35.h
 * \brief binary_cross_entropy_tiling head file
 */
#ifndef OPS_OP_TILING_RUNTIME_BINARY_CORSS_ENTROPY_TILING_H
#define OPS_OP_TILING_RUNTIME_BINARY_CORSS_ENTROPY_TILING_H

#include "register/tilingdata_base.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "atvoss/reduce/reduce_tiling.h"
#include "loss/binary_cross_entropy/op_kernel/arch35/binary_cross_entropy_dag.h"
#include "loss/binary_cross_entropy/op_kernel/arch35/binary_cross_entropy_tiling_key.h"
#include "loss/binary_cross_entropy/op_kernel/arch35/binary_cross_entropy_struct.h"

using namespace Ops::Base;

namespace optiling {
struct BinaryCrossEntropyCompileInfo {
  ge::DataType dtype;
  uint64_t coreNum = 0;
  uint64_t ubSize = 0;
  Ops::Base::ReduceOpCompileInfo opInfo;
};

struct BinaryCrossEntropyTilingKey
{
    ReduceTilingKey reduceTiling;
    uint32_t dType = 20;
    uint32_t reduction = 2;
    uint32_t hasWeight = 0;
};

class BinaryCrossEntropyTiling
{
public:
    explicit BinaryCrossEntropyTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling(const BinaryCrossEntropyCompileInfo *compileInfo);
    BinaryCrossEntropyTilingData* tiling;

protected:
    ge::graphStatus CalcOutputDtype();
    ge::graphStatus CalcInputDtype();
    ge::graphStatus CheckInputShape();
    ge::graphStatus DoElewiseTiling();
    ge::graphStatus DoReduceTiling(const BinaryCrossEntropyCompileInfo *compileInfo);
    ge::graphStatus SetTilingData();
    
    ge::graphStatus RunFp16ReduceTiling(ReduceOpInputParam& opInput, const BinaryCrossEntropyCompileInfo *compileInfo);
    ge::graphStatus RunFp32ReduceTiling(ReduceOpInputParam& opInput, const BinaryCrossEntropyCompileInfo *compileInfo);
private:
    gert::TilingContext* tilingContext;
    ge::DataType outputDtype;
    ge::DataType inputXDtype;
    ge::DataType inputYDtype;
    const char *reductionStr = "";
    bool isReductionNone = false;
    bool isReductionMean = false;
    bool isReductionSum = false;
    BinaryCrossEntropyTilingKey bceTilingKey;
};
}  // namespace optiling
#endif  // OPS_OP_TILING_RUNTIME_BINARY_CORSS_ENTROPY_TILING_H