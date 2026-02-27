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
 * \file foreach_reduce_regbase_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_SOLO_REGBASE_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_SOLO_REGBASE_H_

#include "tiling/tiling_api.h"
#include "foreach_reduce_tiling_def.h"
#include "tiling_base/tiling_util.h"

namespace optiling {
class ForeachReduceRegbaseTiling : public ForeachBaseClass
{
public:
    explicit ForeachReduceRegbaseTiling(gert::TilingContext* context) : ForeachBaseClass(context)
    {}
    ~ForeachReduceRegbaseTiling() override = default;

    void Reset(gert::TilingContext* context) override
    {
        ForeachReduceRegbaseTiling::Reset(context);
    }

protected:
    bool IsCapable() override
    {
        return Ops::NN::OpTiling::IsRegbaseSocVersion(context_);
    }
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override; // 检查output shape
    ge::graphStatus CheckOutput();
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

    ge::DataType dataType_ = ge::DT_UNDEFINED;
    ge::DataType scalarDtype_ = ge::DT_UNDEFINED;
    int64_t numBlocks_ = 0;
    uint64_t tensorDataCountList_[MAX_TENSOR_CONT_950] = {0};
    uint16_t tensorStartList_[MAX_CORE_CONT_950] = {0};
    uint16_t tensorEndList_[MAX_CORE_CONT_950] = {0};
    uint64_t tensorStartOffsetList_[MAX_CORE_CONT_950] = {0};
    uint64_t tensorEndOffsetList_[MAX_CORE_CONT_950] = {0};
    uint16_t tensorMiddleCountList_[MAX_TENSOR_CONT_950] = {0};
    uint16_t tensorMiddleStartList_[MAX_TENSOR_CONT_950] = {0};
    uint16_t coreMiddleOffsetList_[MAX_CORE_CONT_950] = {0};
    int64_t totalDataCount_ = 0;
    uint16_t totalTensorCount_ = 0;
    uint16_t maxTensorNumPerCore_ = 0;
    ForeachReduceTilingDataRegbase foreachSoloTilingData_;

private:
    ge::graphStatus CheckScalar();
    void AssignDataToEachCore(int64_t needCoreNum, int64_t elementsPerBlock);
    ge::graphStatus CheckShapeAllPositive(const gert::Shape& shape, uint32_t idx);
    ge::graphStatus AssignTensorMiddleCountList(int64_t needCoreNum);
};
} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_SOLO_REGBASE_H_