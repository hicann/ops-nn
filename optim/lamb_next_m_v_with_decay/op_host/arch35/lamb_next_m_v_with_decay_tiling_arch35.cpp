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
 * \file lamb_next_m_v_with_decay_tiling_arch35.cpp
 * \brief lamb_next_m_v_with_decay_tiling_arch35 source file
 */

#include "lamb_next_m_v_with_decay_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include <securec.h>
#include <algorithm>
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_templates_registry.h"

using namespace ge;

namespace optiling {

constexpr static uint64_t LAMB_NEXT_M_V_TILING_PRIORITY = 0;
constexpr static uint64_t TILING_KEY_FP32 = 100;
constexpr static uint64_t TILING_KEY_FP16 = 200;
constexpr static int32_t INPUT_NUM = 13;
constexpr static int32_t OUTPUT_NUM = 4;

static ge::graphStatus TilingPrepareForLambNextMVWithDecay(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<LambNextMVWithDecayCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LambNextMVWithDecayTiling::GetShapeAttrsInfo()
{
    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);
    ge::DataType input0DType = input0Desc->GetDataType();
    static const char* kInputNames[] = {"input_mul3",     "input_mul2", "input_realdiv1", "input_mul1", "input_mul0",
                                        "input_realdiv0", "input_mul4", "mul0_x",         "mul1_sub",   "mul2_x",
                                        "mul3_sub1",      "mul4_x",     "add2_y"};
    static const char* kOutputNames[] = {"y1", "y2", "y3", "y4"};
    for (int32_t inputIdx = 1; inputIdx < INPUT_NUM; inputIdx++) {
        auto inputDesc = context_->GetInputDesc(inputIdx);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);

        auto curDtype = inputDesc->GetDataType();
        if (curDtype != input0DType) {
            std::string paramNames = std::string(kInputNames[inputIdx]) + " and input0";
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context_->GetNodeName(), paramNames.c_str(),
                (Ops::Base::ToString(curDtype) + " and " + Ops::Base::ToString(input0DType)).c_str(),
                "Their dtypes should be the same");
            return ge::GRAPH_FAILED;
        }
    }
    for (int32_t outputIdx = 0; outputIdx < OUTPUT_NUM; outputIdx++) {
        auto outputDesc = context_->GetOutputDesc(outputIdx);
        OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);

        auto curDtype = outputDesc->GetDataType();
        if (curDtype != input0DType) {
            std::string paramNames = std::string(kOutputNames[outputIdx]) + " and input0";
            std::string incorrectDtypes = Ops::Base::ToString(curDtype) + " and " + Ops::Base::ToString(input0DType);
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), paramNames.c_str(), incorrectDtypes.c_str(),
                                                   "Their dtypes should be the same");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

bool LambNextMVWithDecayTiling::IsCapable() { return true; }

ge::graphStatus LambNextMVWithDecayTiling::DoOpTiling()
{
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData);
    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);

    ge::DataType input0DType = input0Desc->GetDataType();
    uint32_t dtSize = 0;
    if (input0DType == ge::DT_FLOAT) {
        tilingKey = TILING_KEY_FP32;
        dtSize = sizeof(float);
    } else if (input0DType == ge::DT_FLOAT16) {
        tilingKey = TILING_KEY_FP16;
        dtSize = sizeof(uint16_t);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "input_mul3", Ops::Base::ToString(input0DType).c_str(),
                                  "fp16 or fp32");
        return ge::GRAPH_FAILED;
    }

    using PlanTiling = LambBrcTilingData<13, 4>;
    td_ = PlanTiling{};
    // 空进空出: 输出 0 元素时 tiling 全 0, kernel 按 usedCoreNum=0 直接退出。
    auto outShape0 = context_->GetOutputShape(0);
    if (outShape0 == nullptr || outShape0->GetStorageShape().GetShapeSize() != 0) {
        // 先取返回值再判: 模板实参里的逗号会被预处理器当成 OP_CHECK_IF 的参数分隔符。
        ge::graphStatus planRet = BuildLambBrcPlan<13, 4>(context_, coreNum_, ubSize_, dtSize, td_);
        OP_CHECK_IF(planRet != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "build broadcast plan failed"),
                    return ge::GRAPH_FAILED);
    }

    auto ret = memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(), &td_, sizeof(td_));
    OP_CHECK_IF(ret != EOK, OP_LOGE(context_->GetNodeName(), "copy tiling data failed, ret %d", ret),
                return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(td_));
    context_->SetBlockDim(std::max<uint32_t>(td_.usedCoreNum, 1));
    size_t* ws = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ws);
    ws[0] = 0U;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LambNextMVWithDecayTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t LambNextMVWithDecayTiling::GetTilingKey() const { return tilingKey; }

ge::graphStatus LambNextMVWithDecayTiling::GetWorkspaceSize() { return ge::GRAPH_SUCCESS; }

ge::graphStatus LambNextMVWithDecayTiling::PostTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus LambNextMVWithDecayTiling::GetPlatformInfo()
{
    auto compileInfo = static_cast<const LambNextMVWithDecayCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    coreNum_ = compileInfo->coreNum;
    ubSize_ = compileInfo->ubSize;
    OP_CHECK_IF(coreNum_ == 0 || ubSize_ == 0,
                OP_LOGE(context_->GetNodeName(), "invalid platform info: coreNum %lu ubSize %lu", coreNum_, ubSize_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForLambNextMVWithDecay(gert::TilingContext* context)
{
    OP_LOGD("LambNextMVWithDecayTiling", "Enter TilingForLambNextMVWithDecay");
    if (context == nullptr) {
        OP_LOGE("LambNextMVWithDecayTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = static_cast<const LambNextMVWithDecayCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc LambNextMVWithDecayTiling");
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

IMPL_OP_OPTILING(LambNextMVWithDecay)
    .Tiling(TilingForLambNextMVWithDecay)
    .TilingParse<LambNextMVWithDecayCompileInfo>(TilingPrepareForLambNextMVWithDecay);

REGISTER_OPS_TILING_TEMPLATE(LambNextMVWithDecay, LambNextMVWithDecayTiling, LAMB_NEXT_M_V_TILING_PRIORITY);
} // namespace optiling
