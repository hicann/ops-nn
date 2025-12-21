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
 * \file cross_entropy_loss_grad_regbase_tiling.h
 * \brief cross_entropy_loss_grad_regbase_tiling
 */

#ifndef CROSS_ENTROPY_LOSS_GRAD_REGBASE_TILING_H
#define CROSS_ENTROPY_LOSS_GRAD_REGBASE_TILING_H

#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_util.h"
#include "util/math_util.h"

#include "../op_kernel/arch35/cross_entropy_loss_grad_tiling_data.h"
#include "../op_kernel/arch35/cross_entropy_loss_grad_tiling_key.h"
namespace optiling {

struct CrossEntropyLossGradRegbaseTilingKey
{
    uint32_t schId = 0;
    uint32_t reduction = 1;
    uint32_t isWeight = 0;
    uint32_t labelS = 0;
    uint32_t isIgnore = 0;
};
class CrossEntropyLossGradRegbaseTiling
{
public:
    explicit CrossEntropyLossGradRegbaseTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling();

protected:
    ge::graphStatus GetTilingInput();
    ge::graphStatus GetTilingAttr();
    void InitSplitInfo();
    bool CELossGradInnerCoreFullLoad();
    void CELossGradCoreSplitInfo();
    ge::graphStatus GetPlatform();
    ge::graphStatus CheckDtype();
    ge::graphStatus DoEmptyTensorTiling();
    ge::graphStatus SetTilingData();
    void PrintInfo();
    int64_t FindNearestPower2(const int64_t value);
    int64_t CalLog2(int64_t value);
    int64_t GetMod(int64_t l_value, int64_t r_value);
private:
    gert::TilingContext* tilingContext;
    ge::DataType outputDtype;
    ge::DataType inputXDtype;
    const char *reducationStr = "";
    int64_t reductionKey = 0;
    uint64_t dtypeSize = 4;
    uint64_t targetDtypeSize = 0;
    int32_t hasEmptyTensor = 0;
    int64_t totalCoreNum = 0;
    uint64_t totalUbSize = 0;
    uint64_t vectorLength = 0;
    int64_t blockSize = 0;
    int64_t targetSize = 0;
    int64_t rowVal = 1;
    int64_t colVal = 1;
    int64_t colLoopNum = 0;
    int64_t alignColLoopNum = 0;
    int64_t alignNCNum = 0;
    int64_t gradLossSize = 0;
    int64_t targetWeightSize = 0;
    int64_t tBuf2Size = 0;
    int64_t tBuf3Size = 0;
    int64_t targetCastSize = 0;
    int64_t ignoreSize = 0;
    int64_t smoothSize = 0;
    int64_t colLoop = 0;
    int64_t colLoopNumTail = 0;
    int64_t ubNNum = 0;
    // 二分
    int64_t kTimes = 0;
    int64_t cOnceNum = 0;
    int64_t cOnceNumTail = 0;
    int64_t kTimesTail = 0;
    int64_t cOnceNumTailAlign = 0;
    int64_t cacheStart = 0;
    uint64_t bufferNum = 1;

    CrossEntropyLossGradRegbaseTilingKey ceLossGradTilingKey;
    CrossEntropyLossGradRegbaseTilingData *ceLossGradRegbaseTiling{ nullptr };
};
// ge::graphStatus Tiling4CrossEntropyLossGradRegbase(gert::TilingContext* context);
}  // namespace optiling
#endif  // CROSS_ENTROPY_LOSS_GRAD_REGBASE_TILING_H
