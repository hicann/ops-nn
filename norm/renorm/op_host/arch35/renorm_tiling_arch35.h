/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file renorm_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_RENORM_TILING_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_RENORM_TILING_H_

#include <vector>
#include "register/tilingdata_base.h"
#include "atvoss/reduce/reduce_tiling.h"
#include "atvoss/reduce/reduce_tiling_data.h"
#include "../../op_kernel/arch35/renorm_tiling_struct.h"

using namespace Ops::Base;

namespace optiling
{
struct RenormTilingKey {
    ReduceTilingKey reduceTiling;
    uint32_t templateNum;
};

class RenormTiling
{
public:
    explicit RenormTiling(gert::TilingContext* context) : tilingContext_(context) {};
    ge::graphStatus RunTiling(const ReduceOpCompileInfo* compileInfo);

private:
    ge::graphStatus SetTilingData();
    ge::graphStatus TilingReduce(const ReduceOpCompileInfo* compileInfo);
    ge::graphStatus GetAndCheckOtherAttrs(ge::DataType xDtype);
    ge::graphStatus GetAndCheckReduceAxis();
    ge::graphStatus GetAndCheckDtypes();
    ge::graphStatus HandleP0(const ReduceOpCompileInfo* compileInfo, ReduceOpInputParam& opInput,
                             ReduceOpTilingData& reduceTiling);
    ge::graphStatus HandleP1(const ReduceOpCompileInfo* compileInfo, ReduceOpInputParam& opInput,
                             ReduceOpTilingData& reduceTiling);
    ge::graphStatus HandleP2(const ReduceOpCompileInfo* compileInfo, ReduceOpInputParam& opInput,
                             ReduceOpTilingData& reduceTiling);
    ge::graphStatus HandleP3(const ReduceOpCompileInfo* compileInfo, ReduceOpInputParam& opInput,
                             ReduceOpTilingData& reduceTiling);
    ge::graphStatus HandlePInf(const ReduceOpCompileInfo* compileInfo, ReduceOpInputParam& opInput,
                               ReduceOpTilingData& reduceTiling);
    ge::graphStatus HandlePOther(const ReduceOpCompileInfo* compileInfo, ReduceOpInputParam& opInput,
                                 ReduceOpTilingData& reduceTiling);
    bool CheckReduceAxisIsOne();

private:
    ge::DataType xDtype_;
    gert::TilingContext* tilingContext_;
    RenormTilingKey key_;
    RenormTilingData* tilingData_;
    float p_ = 0.0f;
    float recp_ = 0.0f;
    float epsilon_ = 0.0f;
    float maxnorm_ = 0.0f;
    std::vector<int64_t> reduceAxis_;
    uint32_t templateNum_ = 0;
    bool dtypeXEqualY = false;
};

}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_RENORM_H_