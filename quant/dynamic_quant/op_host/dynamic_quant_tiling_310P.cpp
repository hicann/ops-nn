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
 * \file dynamic_quant_tiling_310P.cpp
 * \brief
 */
#include "dynamic_quant_tiling_310P.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "error_util.h"

using namespace ge;
using namespace AscendC;

namespace optiling {

const static uint32_t X_INDEX = 0;
const static int64_t ATTR_ASYNMMETRIC = 0;
const static uint32_t DYNAMIC_QUANT_BUFFER_NUM_ONE = 1;
const static uint32_t DYNAMIC_QUANT_BUFFER_NUM_TWO = 2;
const static uint32_t DYNAMIC_QUANT_SIZE_OF_INT8 = 1;
const static uint32_t DYNAMIC_QUANT_SIZE_OF_HALF = 2;
const static uint32_t DYNAMIC_QUANT_SIZE_OF_FLOAT = 4;
const static uint32_t DYNAMIC_QUANT_ALIGN_SIZE = 32;
const static uint32_t DYNAMIC_QUANT_ALIGN_NUM_X = 16;
const static uint32_t DYNAMIC_QUANT_ALIGN_NUM_SCALE = 8;
const static uint32_t DYNAMIC_QUANT_MAX_VALUE = 127;
const static int32_t DYNAMIC_QUANT_MIN_VALUE = -128;
const static uint32_t DYNAMIC_QUANT_ASYMMETRIC_VALUE = 255;
const static uint32_t DYNAMIC_QUANT_HEADSPACE = 10240;
const static uint32_t DYNAMIC_QUANT_COPY_ROW_SCALE = 5;
const static uint32_t DYNAMIC_QUANT_FP16_BUF_SCALE = 2;
const static uint32_t DYNAMIC_QUANT_COPY_ROW_LONG = 64;
const static uint32_t DYNAMIC_QUANT_COPY_ROW_SHORT = 192;
const static uint32_t DYNAMIC_QUANT_LEN_H_LONG = 512;
const static uint32_t DYNAMIC_QUANT_LEN_H_SHORT = 64;
const static uint32_t DYNAMIC_QUANT_ROW_SUIT_MUL = 2;
const static uint32_t DYNAMIC_QUANT_ROW_SUIT_DIV = 7;
const static uint32_t DYNAMIC_QUANT_ROW_SUIT_ADD = 213;
const static uint32_t DYNAMIC_QUANT_STATUS_ALIGN = 0;
const static uint32_t DYNAMIC_QUANT_STATUS_UNALIGN_310P = 1;
const static uint32_t DYNAMIC_QUANT_STATUS_UNALIGN_910B = 2;
const static float DYNAMIC_QUANT_EPSINON = 1e-12;
const static uint32_t DYNAMIC_QUANT_BF16_LAST_DIM_LIMITATION = 7552;      // This formula is summarized after testing.
const static uint32_t DYNAMIC_QUANT_FP16_LAST_DIM_LIMITATION_310P = 4096; // This formula is summarized after testing.
const static uint32_t DYNAMIC_QUANT_TILING_KEY_BASE = 10000;
const static uint32_t DYNAMIC_QUANT_TILING_KEY_UNALIGN_MODE = 1000; // 0: 800I unalign, 300I unalign
const static uint32_t DYNAMIC_QUANT_TILING_KEY_ALIGN = 100;         // 0:DynamicQuantUnAlign; 1:DynamicQuantAlign
const static uint32_t DYNAMIC_QUANT_TILING_KEY_DTYPE = 10;          // 0:fp16; 1:bf16
const static uint32_t DYNAMIC_QUANT_TILING_KEY_LARGE = 1;           // 0:small; 1:large

inline size_t RoundUp(size_t size, size_t divisor = 32)
{
    if (divisor == 0U || (size + divisor - 1U) < size) {
        return size;
    }
    return (size + divisor - 1U) / divisor * divisor;
}

template <typename T>
auto CeilDiv(T dividend, T divisor) -> T
{
    return (divisor == 0) ? 0 : ((dividend + divisor - 1) / divisor);
}

// todo 待修改
ge::graphStatus DynamicQuantTiling310P::ParseShape(const gert::TilingContext* context, uint64_t* rowNumTotal)
{
    size_t dims = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    for (size_t i = 0; i < dims; ++i) {
        if (i < dims - 1U) {
            *rowNumTotal *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
        } else {
            tiling.set_sizeH(context->GetInputShape(0)->GetStorageShape().GetDim(i));
        }
    }
    return ge::GRAPH_SUCCESS;
}

void DynamicQuantTiling310P::SetSuitNumCopyRow(const gert::TilingContext* context)
{
    tiling.set_sizeX(RoundUp(tiling.get_sizeH(), DYNAMIC_QUANT_ALIGN_NUM_X));
    tiling.set_sizeZOut(RoundUp(tiling.get_sizeH()));

    uint64_t ubSizePlatForm;
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    uint32_t ubSize = static_cast<uint32_t>(ubSizePlatForm);

    tiling.set_numCopyRow(
        (ubSize - tiling.get_sizeX() * DYNAMIC_QUANT_FP16_BUF_SCALE - DYNAMIC_QUANT_HEADSPACE) /
        (tiling.get_sizeX() * DYNAMIC_QUANT_COPY_ROW_SCALE));

    uint32_t rowSuit =
        DYNAMIC_QUANT_ROW_SUIT_ADD - tiling.get_sizeX() * DYNAMIC_QUANT_ROW_SUIT_MUL / DYNAMIC_QUANT_ROW_SUIT_DIV;

    rowSuit = rowSuit - rowSuit % DYNAMIC_QUANT_ALIGN_NUM_SCALE;

    if (tiling.get_numCopyRow() > DYNAMIC_QUANT_COPY_ROW_LONG && tiling.get_sizeX() >= DYNAMIC_QUANT_LEN_H_LONG) {
        tiling.set_numCopyRow(DYNAMIC_QUANT_COPY_ROW_LONG);
    } else if (
        tiling.get_numCopyRow() > rowSuit && rowSuit > DYNAMIC_QUANT_ALIGN_NUM_SCALE &&
        tiling.get_sizeX() >= DYNAMIC_QUANT_LEN_H_SHORT) {
        tiling.set_numCopyRow(rowSuit);
    } else if (
        tiling.get_numCopyRow() > DYNAMIC_QUANT_COPY_ROW_SHORT && tiling.get_sizeX() < DYNAMIC_QUANT_LEN_H_SHORT &&
        tiling.get_sizeX() > DYNAMIC_QUANT_ALIGN_NUM_SCALE) {
        tiling.set_numCopyRow(DYNAMIC_QUANT_COPY_ROW_SHORT);
    } else if (tiling.get_numCopyRow() > DYNAMIC_QUANT_ALIGN_NUM_SCALE) {
        tiling.set_numCopyRow(tiling.get_numCopyRow() - tiling.get_numCopyRow() % DYNAMIC_QUANT_ALIGN_NUM_SCALE);
    }
}

ge::graphStatus DynamicQuantTiling310P::CorrectNumCopyRow(uint64_t rowNumTotal)
{
    uint32_t perRowNum = CeilDiv(static_cast<uint32_t>(rowNumTotal), tiling.get_coreNum());
    uint32_t alignRowNum = RoundUp(perRowNum, DYNAMIC_QUANT_ALIGN_NUM_SCALE);

    tiling.set_alignType(DYNAMIC_QUANT_STATUS_UNALIGN_310P);

    if (tiling.get_numCopyRow() >= DYNAMIC_QUANT_ALIGN_NUM_SCALE && perRowNum <= DYNAMIC_QUANT_ALIGN_NUM_SCALE) {
        tiling.set_numCopyRow(DYNAMIC_QUANT_ALIGN_NUM_SCALE);
    } else if (tiling.get_numCopyRow() >= alignRowNum) {
        tiling.set_numCopyRow(alignRowNum);
    }
    if (tiling.get_sizeH() > DYNAMIC_QUANT_FP16_LAST_DIM_LIMITATION_310P) {
        tiling.set_numCopyRow(DYNAMIC_QUANT_ALIGN_NUM_SCALE);
        tiling.set_innerLoopEle(DYNAMIC_QUANT_FP16_LAST_DIM_LIMITATION_310P);
        tiling.set_innerLoopTimes(tiling.get_sizeH() / tiling.get_innerLoopEle());
        tiling.set_innerLoopTail(tiling.get_sizeH() % tiling.get_innerLoopEle());
    }

    if (tiling.get_sizeH() % DYNAMIC_QUANT_ALIGN_SIZE != 0 || tiling.get_numCopyRow() < DYNAMIC_QUANT_ALIGN_NUM_SCALE) {
        return ge::GRAPH_FAILED;
    }

    tiling.set_sizeCopyRow(RoundUp(tiling.get_numCopyRow(), DYNAMIC_QUANT_ALIGN_NUM_SCALE));
    return ge::GRAPH_SUCCESS;
}

// 待修改
ge::graphStatus DynamicQuantTiling310P::SetTilingData(uint64_t rowNumTotal, gert::TilingContext* context)
{
    SetSuitNumCopyRow(context);
    ge::graphStatus status = CorrectNumCopyRow(rowNumTotal);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    uint32_t patchTotal = rowNumTotal / tiling.get_numCopyRow();
    tiling.set_numLastTailRow(rowNumTotal % tiling.get_numCopyRow());
    tiling.set_numTailTimes(patchTotal / tiling.get_coreNum());
    tiling.set_numHeadCore(patchTotal % tiling.get_coreNum());
    tiling.set_numTailCore(tiling.get_numCopyRow() - tiling.get_numHeadCore());
    tiling.set_numHeadTimes(tiling.get_numTailTimes() + 1);

    if (tiling.get_numLastTailRow() == 0 && tiling.get_numCopyRow() % DYNAMIC_QUANT_ALIGN_NUM_SCALE == 0 &&
        tiling.get_sizeH() % DYNAMIC_QUANT_ALIGN_SIZE == 0 &&
        tiling.get_sizeH() <= DYNAMIC_QUANT_FP16_LAST_DIM_LIMITATION_310P) {
        tiling.set_alignType(DYNAMIC_QUANT_STATUS_ALIGN);
    }
    return ge::GRAPH_SUCCESS;
}

// 待修改
uint64_t DynamicQuantTiling310P::ComputeTilingKey(const gert::TilingContext* context)
{
    uint64_t tilingKey = DYNAMIC_QUANT_TILING_KEY_BASE;
    tilingKey += context->GetInputTensor(0)->GetDataType() == ge::DT_BF16 ? DYNAMIC_QUANT_TILING_KEY_DTYPE : 0;

    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto socVersion = ascendcPlatform.GetSocVersion();

    if (tiling.get_alignType() == 0) {
        tilingKey += DYNAMIC_QUANT_TILING_KEY_ALIGN;
    } else if (tiling.get_alignType() == 1) {
        tilingKey += DYNAMIC_QUANT_TILING_KEY_UNALIGN_MODE;
    }
    if (tiling.get_sizeH() > DYNAMIC_QUANT_FP16_LAST_DIM_LIMITATION_310P &&
        (socVersion == platform_ascendc::SocVersion::ASCEND310P ||
         socVersion == platform_ascendc::SocVersion::ASCEND910)) {
        tilingKey += DYNAMIC_QUANT_TILING_KEY_LARGE;
    }

    return tilingKey;
}

// 待修改
ge::graphStatus DynamicQuantTiling310P::CheckDtype(const gert::TilingContext* context) const
{
    auto x = context->GetInputDesc(X_INDEX);
    ge::DataType dataType = x->GetDataType();
    if (dataType != ge::DT_FLOAT16) {
        OP_LOGE(context->GetNodeName(), "DynamicQuan on Atlas 300I Duo only support float16.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicQuantTiling310P::RunFusionKernelTiling(gert::TilingContext* context)
{
    ge::graphStatus ret = CheckDtype(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t totalCoreNum = ascendcPlatform.GetCoreNumAiv(); // 总核数
    tiling.set_coreNum(totalCoreNum);
    uint64_t rowNumTotal = 1;
    ret = ParseShape(context, &rowNumTotal);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = SetTilingData(rowNumTotal, context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    uint64_t tilingKey = ComputeTilingKey(context);
    context->SetBlockDim(tiling.get_coreNum());
    context->SetTilingKey(tilingKey);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    size_t workspaceNum = 0;
    currentWorkspace[0] = workspaceNum;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling