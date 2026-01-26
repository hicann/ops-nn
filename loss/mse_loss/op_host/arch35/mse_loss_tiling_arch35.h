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
 * \file mse_loss_tiling_arch35.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_MSE_LOSS_TILING_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_MSE_LOSS_TILING_H_

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "mse_loss_tiling.h"
#include "atvoss/elewise/elewise_base_struct.h"
#include "atvoss/reduce/reduce_tiling.h"
#include "mse_loss/op_kernel/arch35/mse_loss_tiling_def.h"

using namespace Ops::Base;
namespace optiling {

struct MseLossTilingKey {
    ReduceTilingKey ReduceTiling;
    uint32_t Reduction = 2;
    uint32_t Dtype = 20;
};
class MseLossTiling {
public:
    explicit MseLossTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling(const MseLossCompileInfo* compileInfo);

protected:
    ge::graphStatus SetTilingData();
    ge::graphStatus CheckShape();
    ge::graphStatus TilingEle();
    ge::graphStatus TilingReduce(const MseLossCompileInfo* compileInfo);

private:
    ge::DataType outputDtype;
    gert::TilingContext* tilingContext;
    MseLossTilingKey key;
    MseLossTilingData* tiling;
    uint32_t reduction;
};
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_MSE_LOSS_H_