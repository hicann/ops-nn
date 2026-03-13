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
 * \file broadcast_gradient_args_tiling_arch35.cpp
 * \brief
 */

#include "broadcast_gradient_args_tiling_arch35.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "op_common/op_host/util/platform_util.h"


namespace optiling {
constexpr size_t INDEX_X1 = 0;
constexpr size_t INDEX_X2 = 1;
constexpr int64_t SIZE_INT32 = 4;
constexpr int64_t SIZE_INT64 = 8;
constexpr int64_t UB_RESERVED_BYTE = 1024;
constexpr int64_t IN_OUT_NODE_NUM = 4;
constexpr int64_t FLAG_NODE_NUM = 4;
class BroadcastGradientArgsTilingImpl
{
public:
    explicit BroadcastGradientArgsTilingImpl(gert::TilingContext* context) : context_(context){};

    ge::graphStatus Init()
    {
        auto platformInfo = context_->GetPlatformInfo();
        if (platformInfo == nullptr) {
            auto compileInfoPtr = reinterpret_cast<const BroadcastGradientArgsCompileInfo*>(context_->GetCompileInfo());
            OP_CHECK_IF(
                compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "CompileInfo is nullptr."),
                return ge::GRAPH_FAILED);
            coreNum_ = compileInfoPtr->coreNum;
            ubSize_ = compileInfoPtr->ubSize;
        } else {
            auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
            coreNum_ = ascendcPlatform.GetCoreNumAiv();
            uint64_t ubSize = 0;
            ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
            ubSize_ = ubSize;
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoTiling()
    {
        const gert::StorageShape* x1_storage = context_->GetInputShape(INDEX_X1);
        const gert::StorageShape* x2_storage = context_->GetInputShape(INDEX_X2);
        OP_CHECK_NULL_WITH_CONTEXT(context_, x1_storage);
        OP_CHECK_NULL_WITH_CONTEXT(context_, x2_storage);
        const gert::Shape& shape1 = x1_storage->GetStorageShape();
        const gert::Shape& shape2 = x2_storage->GetStorageShape();
        auto x1Dtype = context_->GetInputDesc(INDEX_X1)->GetDataType();
        if (x1Dtype != ge::DT_INT32 && x1Dtype != ge::DT_INT64) {
            OP_LOGE(context_->GetNodeName(), "The dtype of x1 must be int32 or int64!");
            return ge::GRAPH_FAILED;
        }
        auto x2Dtype = context_->GetInputDesc(INDEX_X2)->GetDataType();
        if (x1Dtype != x2Dtype) {
            OP_LOGE(context_->GetNodeName(), "Inputs x1 and x2 dtypes must be same!");
            return ge::GRAPH_FAILED;
        }
        if (shape1.GetDimNum() != 1 || shape2.GetDimNum() != 1) {
            OP_LOGE(context_->GetNodeName(), "Inputs x1 and x2 shapes must be 1D!");
            return ge::GRAPH_FAILED;
        }
        int64_t x1Len = shape1.GetDim(0);
        int64_t x2Len = shape2.GetDim(0);
        if (x1Len < 0 || x2Len < 0) {
            OP_LOGE(context_->GetNodeName(), "Inputs x1 and x2 shape size must >= 0!");
            return ge::GRAPH_FAILED;
        }
        int64_t maxRank = x1Len > x2Len ? x1Len : x2Len;
        int64_t dtypeBytes = (x1Dtype == ge::DT_INT64) ? SIZE_INT64 : SIZE_INT32;
        int64_t blockSize = Ops::Base::GetUbBlockSize(context_);
        int64_t ubFactor = (ubSize_ - UB_RESERVED_BYTE - FLAG_NODE_NUM * blockSize) / (IN_OUT_NODE_NUM * dtypeBytes);
        int64_t ubMaxRank = Ops::Base::FloorAlign(ubFactor, 2 * (blockSize / SIZE_INT32));
        tiling_data_.set_x1Len(x1Len);
        tiling_data_.set_x2Len(x2Len);
        tiling_data_.set_maxRank(maxRank);
        tiling_data_.set_ubMaxRank(ubMaxRank);
        tiling_data_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(tiling_data_.GetDataSize());
        context_->SetBlockDim(1);
        OP_LOGD(
            context_->GetNodeName(),
            "Tiling: x1Len=%ld, x2Len=%ld, maxRank=%ld, ubMaxRank=%ld", tiling_data_.get_x1Len(),
            tiling_data_.get_x2Len(), tiling_data_.get_maxRank(), tiling_data_.get_ubMaxRank());
        int64_t tilingKey = maxRank <= ubMaxRank ? 0 : 1;
        context_->SetTilingKey(tilingKey);

        return ge::GRAPH_SUCCESS;
    }

private:
    gert::TilingContext* context_;
    BroadcastGradientArgsTilingData tiling_data_;
    int64_t ubSize_ = 0;
    int64_t coreNum_ = 0;
};

// ########################################################################
ge::graphStatus Tiling4BroadcastGradientArgs(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start BroadcastGradientArgsTiling!");
    BroadcastGradientArgsTilingImpl impl(context);
    ge::graphStatus status = impl.Init();
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Init failed!");
        return status;
    }
    status = impl.DoTiling();
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "DoTiling failed!");
        return status;
    }
    OP_LOGD(context->GetNodeName(), "End BroadcastGradientArgsTiling!");
    return status;
}

ge::graphStatus TilingPrepare4BroadcastGradientArgs(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4BroadcastGradientArgs running.");
    auto compileInfo = context->GetCompiledInfo<BroadcastGradientArgsCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        compileInfo->coreNum <= 0, OP_LOGE(context->GetNodeName(), "coreNum must be greater than 0."),
        return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = ubSize;
    OP_CHECK_IF(
        compileInfo->ubSize <= 0, OP_LOGE(context->GetNodeName(), "ubSize must be greater than 0."),
        return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "coreNum: %ld, ubSize: %ld", compileInfo->coreNum, compileInfo->ubSize);
    OP_LOGD(context->GetNodeName(), "TilingPrepare4BroadcastGradientArgs success.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BroadcastGradientArgs)
    .Tiling(Tiling4BroadcastGradientArgs)
    .TilingParse<BroadcastGradientArgsCompileInfo>(TilingPrepare4BroadcastGradientArgs);

} // namespace optiling