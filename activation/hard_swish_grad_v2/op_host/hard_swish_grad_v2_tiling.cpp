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
 * \file hard_swish_grad_v2_tiling.cpp
 * \brief hard_swish_grad_v2_tiling source file
 */
#include <vector>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "hard_swish_grad_v2_tiling_def.h"
#include "log/log.h" 

namespace optiling {
constexpr int64_t UB_DIVIDE_NUM_EACH_CORE = 10;

constexpr int64_t UB_MEMORY_ALIGN = 256;

class HardSwishGradV2Tiling {
public:
    explicit HardSwishGradV2Tiling(gert::TilingContext* context) : tilingContext(context){};
    ge::graphStatus RunBigKernelTiling();

private:
    ge::DataType dataType = ge::DT_UNDEFINED;
    gert::TilingContext* tilingContext = nullptr;
    gert::Shape inputShape;
    HardSwishGradV2TilingData tilingData;

    int64_t inputShapeSize = 0;
    int64_t elementNumEachCore = 0;

    inline uint32_t CeilA2B(const int64_t a, const int64_t b) {
        if (b != 0) {
            return (a + b - 1) / b;
        } else {
            return a;
        }
    }

    int32_t GetNeedCoreNum(const int32_t coreNumPlatform) {
        int32_t needCoreNum = CeilA2B(inputShapeSize, elementNumEachCore);
        if (needCoreNum == 0) {
            needCoreNum = 1;
        }
        if (needCoreNum >= coreNumPlatform) {
            return coreNumPlatform;
        } else {
            return needCoreNum;
        }
    }
};

ge::graphStatus HardSwishGradV2Tiling::RunBigKernelTiling() {
    // 获取输入矩阵
    auto srcTensor = tilingContext->GetInputTensor(0);
    if (srcTensor == nullptr) {
        OP_LOGE(tilingContext, "HardSwishGradV2Tiling srcTensor nullptr");
        return ge::GRAPH_FAILED;
    }
    
    // 获取数据类型
    auto temp = tilingContext->GetInputDesc(0);
    if (temp == nullptr) {
        OP_LOGE(tilingContext, "HardSwishGradV2Tiling temp nullptr");
        return ge::GRAPH_FAILED;
    }

    uint64_t tilingKey = 0;
    tilingContext->SetTilingKey(tilingKey);

    // 获取输入的shape
    auto srcShape = tilingContext->GetInputShape(0);
    inputShape = srcShape->GetOriginShape();
    inputShapeSize = inputShape.GetShapeSize();

    auto platformInfo = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());

    uint64_t ubSize = 196608;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    elementNumEachCore = ubSize / UB_DIVIDE_NUM_EACH_CORE / sizeof(float)/ UB_MEMORY_ALIGN * UB_MEMORY_ALIGN;
    OP_LOGD(tilingContext, "HardSwishGradV2Tiling GetCoreMemSize ubSize = %d, elementNumEachCore = %d", ubSize, elementNumEachCore);
    if (ubSize <= 0 || elementNumEachCore <= 0) {
        OP_LOGE(tilingContext, "HardSwishGradV2Tiling GetCoreMemSize ubSize error");
        return ge::GRAPH_FAILED;
    }
    uint32_t needCoreNum = GetNeedCoreNum(platformInfo.GetCoreNumAiv());

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    uint32_t sysWorkspaceSize = platformInfo.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = sysWorkspaceSize;

    tilingData.set_elementNum(inputShapeSize);
    tilingData.set_needCoreNum(needCoreNum);
    tilingData.set_ubSize(ubSize);
    tilingData.set_elementNumEachCore(elementNumEachCore);

    tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                            tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    
    tilingContext->SetBlockDim(needCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4HardSwishGradV2Tiling([[maybe_unused]] gert::TilingParseContext* context) {
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingHardSwishGradV2Tiling(gert::TilingContext* context) {
    HardSwishGradV2Tiling tilingObject(context);
    return tilingObject.RunBigKernelTiling();
}

IMPL_OP_OPTILING(HardSwishGradV2)
    .Tiling(TilingHardSwishGradV2Tiling)
    .TilingParse<HardSwishGradV2CompileInfo>(TilingPrepare4HardSwishGradV2Tiling);
} // namespace optiling