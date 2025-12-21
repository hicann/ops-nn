/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __TILING_FUNCTION_DEF_H__
#define __TILING_FUNCTION_DEF_H__

#include <iostream>
#include <string>
#include <cstdint>
#include <map>
#include <cmath>
#include "ge_glu_grad_v2_tiling_def.h"
#include "exe_graph/runtime/shape.h"
#include "graph/types.h"
#include "graph/ge_error_codes.h"

namespace optiling {

enum class GeGluGradV2TilingKey : uint64_t
{
    TILING_KEY_TANH_101 = 101,
    TILING_KEY_TANH_102,
    TILING_KEY_TANH_103,
    TILING_KEY_TANH_201 = 201,
    TILING_KEY_TANH_202,
    TILING_KEY_TANH_203,
    TILING_KEY_TANH_301 = 301,
    TILING_KEY_TANH_302,
    TILING_KEY_TANH_303,
    TILING_KEY_ERF_701 = 701,
    TILING_KEY_ERF_702,
    TILING_KEY_ERF_703,
    TILING_KEY_ERF_801 = 801,
    TILING_KEY_ERF_802,
    TILING_KEY_ERF_803,
    TILING_KEY_ERF_901 = 901,
    TILING_KEY_ERF_902,
    TILING_KEY_ERF_903
};

/* Tanh */
constexpr int32_t TANH_BUF_CNT_FP16 = 5 * 2 + 6;
constexpr int32_t TANH_BUF_CNT_BFP16 = 7 * 2 + 4;
constexpr int32_t TANH_BUF_CNT_FP32 = 11;

/* Erf */
constexpr int32_t ERF_BUF_CNT_FP16 = 5 * 2 + 6;
constexpr int32_t ERF_BUF_CNT_BFP16 = 7 * 2 + 4;
constexpr int32_t ERF_BUF_CNT_FP32 = 11;

constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t TRANSPOSE_REPEAT_SIZE = 512;
constexpr int32_t NUM_ONE = 1;
constexpr int32_t NUM_TWO = 2;
constexpr int32_t NUM_HUNDRED = 100;

static const std::map<ge::DataType, int32_t> DTYPE_BUF_CNT_MAP_TANH = {
    {ge::DT_BF16, TANH_BUF_CNT_BFP16}, {ge::DT_FLOAT16, TANH_BUF_CNT_FP16}, {ge::DT_FLOAT, TANH_BUF_CNT_FP32}};
static const std::map<ge::DataType, int32_t> DTYPE_BUF_CNT_MAP_ERF = {
    {ge::DT_BF16, ERF_BUF_CNT_BFP16}, {ge::DT_FLOAT16, ERF_BUF_CNT_FP16}, {ge::DT_FLOAT, ERF_BUF_CNT_FP32}};

namespace platform_ascendc {
enum class SocVersion
{
    ASCEND910 = 0,
    ASCEND910B,
    ASCEND310P,
    ASCEND310B,
    RESERVED_VERSION = 99999
};
}

struct GeGluGradV2CompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
    platform_ascendc::SocVersion curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
};

class GeGluGradV2Tiling
{
public:
    GeGluGradV2Tiling(
        const std::vector<std::vector<int64_t>> shapesInfo, const std::vector<int64_t> attsInfo,
        ge::DataType dyDtypeIn = ge::DT_FLOAT)
        : dimAttr(attsInfo[0]), approximateAttr(attsInfo[1])
    {
        dyDtype = dyDtypeIn;

        dtypeSize = 4;
        activateLeftAttr = attsInfo[2] ? true : false;
        if (dyDtype == ge::DT_FLOAT16 || dyDtype == ge::DT_BF16) {
            dtypeSize = 2;
        }

        for (const auto& dim : shapesInfo[0]) {
            dyShape.AppendDim(dim);
        }
        for (const auto& dim : shapesInfo[1]) {
            xShape.AppendDim(dim);
        }
        for (const auto& dim : shapesInfo[0]) {
            geluShape.AppendDim(dim);
        }
        for (const auto& dim : shapesInfo[1]) {
            dxShape.AppendDim(dim);
        }
    };

    ge::graphStatus RunTiling4GeGluGradV2();
    void FillTilingData(uint8_t* tilingPtr);

    int32_t GetNeedCoreNum()
    {
        return needCoreNum;
    };

    int32_t GetTilingKey()
    {
        return static_cast<int32_t>(tilingKey);
    };

private:
    ge::graphStatus Init();

    template <typename T1, typename T2>
    inline T1 CeilDiv(T1 a, T2 b)
    {
        a = int64_t(a);
        b = int64_t(b);
        return T1(b == 0 ? a : (a + b - 1) / b);
    };

    template <typename T1, typename T2>
    inline T1 AlignA2B(T1 a, T2 b)
    {
        a = int64_t(a);
        b = int64_t(b);
        return T1(b == 0 ? a : (a / b) * b);
    };

    void CalcValueNM();
    ge::graphStatus CaclMaxProcessCount();
    void ProcessTilingCore();

private:
    GeGluGradV2TilingData tilingData;
    const GeGluGradV2CompileInfo ptrCompileInfo = {48, 196608, platform_ascendc::SocVersion::ASCEND910B};
    GeGluGradV2TilingKey tilingKey;

    // input output infos
    gert::Shape dyShape;
    gert::Shape xShape;
    gert::Shape geluShape;
    gert::Shape dxShape;
    int64_t dimAttr = -1;
    int64_t approximateAttr = 1;
    bool activateLeftAttr = false;

    /**
     * The meanings of valueN and valueM are as follows:
     * Shape(A, B, C) of input x, dim=1 ==> valueN=A, valueM=B*C//2
     * Shape(A, B, C) of input x, dim=-1 ==> valueN=A*B, valueM=C//2
     * Shape(A, B, C, D) of input x, dim=2 ==> valueN=A*B, valueM=C*D//2
     */
    int64_t valueN = 1;
    int64_t valueM = 1;
    ge::DataType dyDtype = ge::DT_UNDEFINED;
    int32_t dtypeSize = 0;

    // tiling params
    int64_t maxProcCount = 0;
    int32_t needCoreNum = 0;
    int64_t loopNumPerCore = 0;
    int64_t tailCoreIndex = 0;
    int64_t tailUbLoopNum = 0;
    int64_t groupNum = 0;
};

ge::graphStatus GeGluGradV2Tiling::RunTiling4GeGluGradV2()
{
    Init();
    CalcValueNM();
    CaclMaxProcessCount();
    ProcessTilingCore();
    return ge::GRAPH_SUCCESS;
}

void GeGluGradV2Tiling::FillTilingData(uint8_t* tilingPtr)
{
    tilingData.approximate = int32_t(approximateAttr);
    tilingData.activateLeft = int32_t(activateLeftAttr);
    tilingData.maxProcCount = maxProcCount;
    tilingData.valueN = valueN;
    tilingData.valueM = valueM;
    tilingData.needCoreNum = needCoreNum;
    tilingData.loopNumPerCore = loopNumPerCore;
    tilingData.tailCoreIndex = tailCoreIndex;
    tilingData.tailUbLoopNum = tailUbLoopNum;
    tilingData.groupNum = groupNum;
    memcpy(tilingPtr, &tilingData, sizeof(GeGluGradV2TilingData));
    printf(
        "Tiling data is maxProcCount:%ld, valueN:%ld, valueM:%ld, needCoreNum:%d, loopNumPerCore:%ld, "
        "tailCoreIndex:%ld, tailUbLoopNum:%ld, groupNum:%ld, tilingKey:%lu.\n",
        maxProcCount, valueN, valueM, needCoreNum, loopNumPerCore, tailCoreIndex, tailUbLoopNum, groupNum,
        static_cast<uint64_t>(tilingKey));
}

ge::graphStatus GeGluGradV2Tiling::Init()
{
    size_t xDimNum = xShape.GetDimNum();
    dimAttr = dimAttr < 0 ? xDimNum + dimAttr : dimAttr;
    printf(
        "Attr info: dimAttr:%ld, approximateAttr:%ld, activateLeftAttr:%s, dyDtype: %d.\n", dimAttr, approximateAttr,
        activateLeftAttr ? "true" : "false", static_cast<int32_t>(dyDtype));

    return ge::GRAPH_SUCCESS;
}

void GeGluGradV2Tiling::CalcValueNM()
{
    for (int64_t i = 0; i < dimAttr; ++i) {
        valueN *= dyShape.GetDim(i);
    }
    for (int64_t i = dimAttr; i < int64_t(dyShape.GetDimNum()); ++i) {
        valueM *= dyShape.GetDim(i);
    }
}

ge::graphStatus GeGluGradV2Tiling::CaclMaxProcessCount()
{
    if (approximateAttr == NUM_ONE) {
        const auto iter = DTYPE_BUF_CNT_MAP_TANH.find(dyDtype);
        maxProcCount = AlignA2B(ptrCompileInfo.ubSizePlatForm / iter->second, BLOCK_SIZE) / dtypeSize;
        tilingKey = GeGluGradV2TilingKey::TILING_KEY_TANH_101;
    } else {
        const auto iter = DTYPE_BUF_CNT_MAP_ERF.find(dyDtype);
        maxProcCount = AlignA2B(ptrCompileInfo.ubSizePlatForm / iter->second, BLOCK_SIZE) / dtypeSize;
        tilingKey = GeGluGradV2TilingKey::TILING_KEY_ERF_701;
    }

    if (dyDtype == ge::DT_FLOAT16) {
        tilingKey = static_cast<GeGluGradV2TilingKey>(static_cast<int32_t>(tilingKey) + NUM_HUNDRED);
    } else if (dyDtype == ge::DT_FLOAT) {
        tilingKey = static_cast<GeGluGradV2TilingKey>(static_cast<int32_t>(tilingKey) + NUM_TWO * NUM_HUNDRED);
    }

    return ge::GRAPH_SUCCESS;
}

void GeGluGradV2Tiling::ProcessTilingCore()
{
    int64_t ubLoopNum = 0;
    int64_t repeatDataCount = TRANSPOSE_REPEAT_SIZE / dtypeSize;
    int64_t maxPerfCount = maxProcCount / repeatDataCount;
    if (ptrCompileInfo.curSocVersion == platform_ascendc::SocVersion::ASCEND910B && valueM <= maxPerfCount) {
        tilingKey = static_cast<GeGluGradV2TilingKey>(static_cast<int32_t>(tilingKey) + NUM_TWO);
        groupNum = AlignA2B(maxProcCount / valueM, repeatDataCount);
        ubLoopNum = CeilDiv(valueN, groupNum);
        tailUbLoopNum = valueN % groupNum;
    } else if (valueM <= maxProcCount) {
        int64_t alignValueM = CeilDiv(valueM, (BLOCK_SIZE / dtypeSize)) * (BLOCK_SIZE / dtypeSize);
        groupNum = maxProcCount / alignValueM;
        ubLoopNum = CeilDiv(valueN, groupNum);
        tailUbLoopNum = valueN % groupNum;
    } else {
        groupNum = CeilDiv(valueM, maxProcCount);
        ubLoopNum = valueN * groupNum;
        tilingKey = static_cast<GeGluGradV2TilingKey>(static_cast<int32_t>(tilingKey) + NUM_ONE);
    }

    needCoreNum = ubLoopNum < ptrCompileInfo.totalCoreNum ? ubLoopNum : ptrCompileInfo.totalCoreNum;
    if (needCoreNum < ptrCompileInfo.totalCoreNum) {
        loopNumPerCore = 0;
        tailCoreIndex = tailUbLoopNum != 0 ? needCoreNum - 1 : needCoreNum;
    } else {
        loopNumPerCore = ubLoopNum / ptrCompileInfo.totalCoreNum;
        int64_t modValue = ubLoopNum % ptrCompileInfo.totalCoreNum;
        if (modValue != 0) {
            tailCoreIndex = tailUbLoopNum != 0 ? modValue - 1 : modValue;
        } else {
            loopNumPerCore -= 1;
            tailCoreIndex = tailUbLoopNum != 0 ? ptrCompileInfo.totalCoreNum - 1 : ptrCompileInfo.totalCoreNum;
        }
    }
}

} // namespace optiling

#endif // __TILING_FUNCTION_DEF_H__