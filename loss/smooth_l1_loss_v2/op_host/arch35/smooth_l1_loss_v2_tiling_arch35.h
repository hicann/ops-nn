/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file smooth_l1_loss_v2_tiling_arch35.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_SMOOTH_L1_LOSS_V2_TILING_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_SMOOTH_L1_LOSS_V2_TILING_H_

#include <string>
#include <vector>
#include "register/tilingdata_base.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "../../op_kernel/arch35/smooth_l1_loss_v2_tilingdata.h"
#include "atvoss/reduce/reduce_tiling.h"
#include "atvoss/broadcast/broadcast_tiling.h"

using namespace SmoothL1LossV2;
using namespace Ops::Base;
namespace optiling {

struct SmoothL1LossV2CompileInfo
{
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
    Ops::Base::ReduceOpCompileInfo opInfo;
};

struct SmoothL1LossV2TilingKey {
    Ops::Base::ReduceTilingKey ReduceTiling;
    uint32_t Reduction = 2;
    uint32_t Dtype = 20;
};
class SmoothL1LossV2Tiling {
public:
    explicit SmoothL1LossV2Tiling(gert::TilingContext *context) : tilingContext(context){};
    ge::graphStatus RunTiling(const SmoothL1LossV2CompileInfo *compileInfo);
    SmoothL1LossV2TilingData* tiling = nullptr;
protected:
    ge::graphStatus SetTilingData();
    ge::graphStatus CheckShape();
    ge::graphStatus TilingEle();
    ge::graphStatus TilingReduce(const SmoothL1LossV2CompileInfo *compileInfo);
private:
    ge::DataType outputDtype;
    gert::TilingContext *tilingContext;
    SmoothL1LossV2TilingKey key;
    uint32_t reduction;
};
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_SMOOTH_L1_LOSS_V2_H_