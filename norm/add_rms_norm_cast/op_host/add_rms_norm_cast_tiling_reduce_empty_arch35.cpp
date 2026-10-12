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
 * \file add_rms_norm_cast_tiling_reduce_empty_arch35.cpp
 * \brief 空张量(reduce empty)兜底 tiling 模板：归一化维元素数 N=0 且行数 M>0 时，
 *        y1/y2/x 为空不写，把 rstd 当一维 M 个 fp32 做核间切分并计算每循环元素数，
 *        kernel 侧按行写 NaN。与 Regbase tiling 解耦，经 tiling 模板注册按优先级独立分发。
 */

#include "add_rms_norm_cast_tiling_reduce_empty_arch35.h"

namespace optiling {

bool AddRmsNormCastReduceEmptyTiling::CheckInputShapeValue()
{
    if (!AddRmsNormCastRegbaseTiling::CheckInputShapeValue()) {
        return false;
    }

    // Called before SetInputParams: inspect shapes directly; params like numM/numN are not filled yet.
    constexpr uint64_t X1_INDEX = 0;
    constexpr uint64_t GAMMA_INDEX = 2;
    constexpr uint64_t RSTD_INDEX = 2;
    const gert::StorageShape* x1Shape = context_->GetInputShape(X1_INDEX);
    const gert::StorageShape* gammaShape = context_->GetInputShape(GAMMA_INDEX);
    // When the norm dims (covered by gamma) of x1 contain 0, rstd is the only valid output:
    // one NaN is written per row, so its shape size must equal the row num of x1.
    const gert::StorageShape* rstdShape = context_->GetOutputShape(RSTD_INDEX);
    size_t x1DimNum = x1Shape->GetStorageShape().GetDimNum();
    size_t gammaDimNum = gammaShape->GetStorageShape().GetDimNum();
    bool normDimZero = false;
    for (size_t i = x1DimNum - gammaDimNum; i < x1DimNum; i++) {
        if (x1Shape->GetStorageShape().GetDim(i) == 0) {
            normDimZero = true;
            break;
        }
    }
    if (normDimZero) {
        uint64_t numM = 1;
        for (size_t i = 0; i < x1DimNum - gammaDimNum; i++) {
            numM *= static_cast<uint64_t>(x1Shape->GetStorageShape().GetDim(i));
        }
        OP_CHECK_IF(static_cast<uint64_t>(rstdShape->GetStorageShape().GetShapeSize()) != numM,
                    OP_LOGE(nodeName.c_str(),
                            "Shape size check invalid, rstd shape size is %lu, should be equal to the row num of x1 "
                            "%lu when the norm dims of x1 are 0.",
                            static_cast<uint64_t>(rstdShape->GetStorageShape().GetShapeSize()), numM),
                    return false);
    }

    return true;
}

bool AddRmsNormCastReduceEmptyTiling::IsCapable()
{
    // 直接按 shape 派生值判定，不经共享标志：归一化维为 0 且行数 > 0 时由本模板认领
    bool capable = (0 == tilingParams.numN) && (0 < tilingParams.numM);
    OP_LOGD(nodeName.c_str(), "AddRmsNormCastReduceEmptyTiling IsCapable: %d (numM: %lu, numN: %lu).", capable,
            tilingParams.numM, tilingParams.numN);
    return capable;
}

ge::graphStatus AddRmsNormCastReduceEmptyTiling::DoOpTiling()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormCastReduceEmptyTiling DoOpTiling.");
    // compile info 缺失时与 Regbase 模板一致地重取平台信息，保证 totalCoreNum/maxUbSize 有效，
    // 保证每循环处理的元素数不超过可用 UB 容量。
    if (tilingParams.needGetCompileInfo) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, tilingParams.maxUbSize);
        tilingParams.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    }
    // 每核活量不满 32KB 不单独开核，封顶物理核数
    constexpr int64_t SINGLE_CORE_MIN_BYTES = 32 * 1024;
    constexpr int64_t SINGLE_CORE_MIN_ELEMENTS = SINGLE_CORE_MIN_BYTES / static_cast<int64_t>(sizeof(float));
    int64_t rows = static_cast<int64_t>(tilingParams.numM);
    // ceil(rows / SINGLE_CORE_MIN_ELEMENTS), without a potentially overflowing byte product or addition.
    int64_t blockNum = rows / SINGLE_CORE_MIN_ELEMENTS + (rows % SINGLE_CORE_MIN_ELEMENTS != 0);
    if (blockNum > static_cast<int64_t>(tilingParams.totalCoreNum)) {
        blockNum = static_cast<int64_t>(tilingParams.totalCoreNum);
    }
    if (blockNum < 1) {
        blockNum = 1;
    }
    tilingParams.usedCoreNum = static_cast<uint64_t>(blockNum);
    tilingParams.mPerCore = static_cast<uint64_t>(rows / blockNum + (rows % blockNum != 0));
    tilingParams.mLastCore = static_cast<uint64_t>(rows - (blockNum - 1) * static_cast<int64_t>(tilingParams.mPerCore));

    // 每循环搬出的元素数：不超过单核行数，也不超过 UB 容量，并按 32B 块对齐
    int64_t perLoopMax = static_cast<int64_t>(tilingParams.maxUbSize - UB_RESERVE_FOR_RSTDALIGN) /
                         static_cast<int64_t>(sizeof(float));
    int64_t chunk = (perLoopMax < static_cast<int64_t>(tilingParams.mPerCore)) ?
                        perLoopMax :
                        static_cast<int64_t>(tilingParams.mPerCore);
    int64_t elemsPerBlock = static_cast<int64_t>(tilingParams.b32BlockNum);
    int64_t rowsPerLoop = chunk / elemsPerBlock * elemsPerBlock;
    if (rowsPerLoop < 1) {
        rowsPerLoop = static_cast<int64_t>(tilingParams.mPerCore);
    }

    reduceEmptyTilingData.set_numM(tilingParams.numM);
    reduceEmptyTilingData.set_mPerCore(tilingParams.mPerCore);
    reduceEmptyTilingData.set_mLastCore(tilingParams.mLastCore);
    reduceEmptyTilingData.set_rowsPerLoop(static_cast<uint64_t>(rowsPerLoop));
    OP_LOGI(nodeName.c_str(),
            "ReduceEmpty tiling numM: %lu, mPerCore: %lu, mLastCore: %lu, rowsPerLoop: %lu, usedCoreNum: %lu.",
            reduceEmptyTilingData.get_numM(), reduceEmptyTilingData.get_mPerCore(),
            reduceEmptyTilingData.get_mLastCore(), reduceEmptyTilingData.get_rowsPerLoop(), tilingParams.usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

uint64_t AddRmsNormCastReduceEmptyTiling::GetTilingKey() const
{
    OP_LOGI(nodeName.c_str(), "TilingKey is %lu.", TILING_KEY_REDUCE_EMPTY);
    return TILING_KEY_REDUCE_EMPTY;
}

ge::graphStatus AddRmsNormCastReduceEmptyTiling::PostTiling()
{
    OP_LOGD(nodeName.c_str(), "Tiling usedCoreNum is %lu.", tilingParams.usedCoreNum);
    context_->SetBlockDim(tilingParams.usedCoreNum);
    reduceEmptyTilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                                       context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(reduceEmptyTilingData.GetDataSize());

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    size_t sysWorkSpaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkSpaceSize;
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(AddRmsNormCast, AddRmsNormCastReduceEmptyTiling, TEMPLATE_REDUCE_EMPTY_PRIORITY);
} // namespace optiling
