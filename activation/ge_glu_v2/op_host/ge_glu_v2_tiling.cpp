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
 * \file ge_glu_v2_tiling.cpp
 * \brief
 */
#include "ge_glu_v2_tiling.h"
#include "op_common/op_host/util/op_const_def.h"
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include <string>
#include <vector>
#include "base/err_msg.h"

namespace optiling {
using namespace Ops::NN::OpTiling;
static const uint32_t INPUT_IDX = 0;
static const uint32_t DIM_0 = 0;
static const uint32_t DIM_1 = 1;
static const uint32_t DIM_2 = 2;
static const uint32_t DIM_3 = 3;
static const uint32_t ATTR_DIM_INDEX = 0;
static const uint32_t ATTR_APPROXIMATE_INDEX = 1;
static const uint32_t ATTR_ACTIVATE_LEFT_INDEX = 2;
static const uint32_t FP16_DTYPE_BYTES = 2;
static const uint32_t BF16_DTYPE_BYTES = 2;
static const uint32_t FP32_DTYPE_BYTES = 4;
static const uint32_t WORK_SPACE_SIZE = 16U * 1024U * 1024U;
static const uint32_t SPLIT_FACTOR = 2;
static const uint32_t SPLIT_ERROR_STATUS = 10000;
static const int64_t APPROXIMATE_USING_TANH = 1;
static const int64_t APPROXIMATE_USING_ERF = 0;
static const int64_t BYTES_ONE_BLOCK = 32;

static const int64_t FP16_BLOCK_SIZE = 16;
static const int64_t BFP16_BLOCK_SIZE = 16;
static const int64_t FP32_BLOCK_SIZE = 8;

static const int64_t COMMON_B2_USE_UB_BYTE_COFF = 36;
static const int64_t COMMON_B4_USE_UB_BYTE_COFF = 40;
static const int64_t NY1_B4_USE_UB_BYTE_COFF = 48;

static const int64_t COMMON_BUFFER_SIZE_LIMIT_REGBASE = 4880;
static const int64_t FP32_BUFFER_SIZE_VREDUCE_ERF_REGBASE = 4224;
static const int64_t COMMON_BUFFER_SIZE_LIMIT = 6144;
static const int64_t FP32_BUFFER_SIZE_VREDUCE_ERF = 4912;

// RegBase per-template perElementBytes: bytes per bufferSize element, computed from InitBuffer allocations
// tanh (approximate=1) align/last_axis_big
static const int64_t REGBASE_TANH_ALIGN_PER_ELEM_FP16 = 16;
static const int64_t REGBASE_TANH_ALIGN_PER_ELEM_BF16 = 16;
static const int64_t REGBASE_TANH_ALIGN_PER_ELEM_FP32 = 32;
// tanh (approximate=1) vreduce (ny==1)
static const int64_t REGBASE_TANH_VREDUCE_PER_ELEM_FP16 = 20;
static const int64_t REGBASE_TANH_VREDUCE_PER_ELEM_BF16 = 32;
static const int64_t REGBASE_TANH_VREDUCE_PER_ELEM_FP32 = 40;
// erf (approximate=0) align/last_axis_big
static const int64_t REGBASE_ERF_ALIGN_PER_ELEM_FP16 = 32;
static const int64_t REGBASE_ERF_ALIGN_PER_ELEM_BF16 = 32;
static const int64_t REGBASE_ERF_ALIGN_PER_ELEM_FP32 = 40;
// erf (approximate=0) vreduce (ny==1)
static const int64_t REGBASE_ERF_VREDUCE_PER_ELEM_FP16 = 36;
static const int64_t REGBASE_ERF_VREDUCE_PER_ELEM_BF16 = 36;
static const int64_t REGBASE_ERF_VREDUCE_PER_ELEM_FP32 = 48;

static int64_t GetRegbasePerElem(int64_t typeSize, int64_t approximate, int64_t ny)
{
    bool isVreduce = (ny == 1);
    if (approximate == APPROXIMATE_USING_TANH) {
        if (isVreduce) {
            if (typeSize == FP32_DTYPE_BYTES) {
                return REGBASE_TANH_VREDUCE_PER_ELEM_FP32;
            }
            if (typeSize == BF16_DTYPE_BYTES) {
                return REGBASE_TANH_VREDUCE_PER_ELEM_BF16;
            }
            return REGBASE_TANH_VREDUCE_PER_ELEM_FP16;
        } else {
            if (typeSize == FP32_DTYPE_BYTES) {
                return REGBASE_TANH_ALIGN_PER_ELEM_FP32;
            }
            if (typeSize == BF16_DTYPE_BYTES) {
                return REGBASE_TANH_ALIGN_PER_ELEM_BF16;
            }
            return REGBASE_TANH_ALIGN_PER_ELEM_FP16;
        }
    } else {
        if (isVreduce) {
            if (typeSize == FP32_DTYPE_BYTES) {
                return REGBASE_ERF_VREDUCE_PER_ELEM_FP32;
            }
            if (typeSize == BF16_DTYPE_BYTES) {
                return REGBASE_ERF_VREDUCE_PER_ELEM_BF16;
            }
            return REGBASE_ERF_VREDUCE_PER_ELEM_FP16;
        } else {
            if (typeSize == FP32_DTYPE_BYTES) {
                return REGBASE_ERF_ALIGN_PER_ELEM_FP32;
            }
            if (typeSize == BF16_DTYPE_BYTES) {
                return REGBASE_ERF_ALIGN_PER_ELEM_BF16;
            }
            return REGBASE_ERF_ALIGN_PER_ELEM_FP16;
        }
    }
}

static const int64_t REGBASE_MIN_BUFFER_SIZE = 2048;
static const int64_t REGBASE_MAX_SHAPE_BUFFER_SIZE_MULTI_TIMES = 2;
static int64_t CalcRegbaseBufSize(uint64_t ubSize, int64_t typeSize, int64_t blockSize, int64_t approximate, int64_t ny,
                                  int64_t x, int64_t coreNum)
{
    int64_t maxPerElem = GetRegbasePerElem(typeSize, approximate, ny);
    int64_t raw = static_cast<int64_t>(ubSize) / maxPerElem;
    int64_t floored = raw / blockSize * blockSize;
    int64_t bufSize = floored > 0 ? floored : blockSize;

    // 大尾轴模板均匀切分，避免小尾块导致流水无法覆盖
    if (ny > bufSize) {
        int64_t divide = (ny + bufSize - 1) / bufSize;
        bufSize = (ny + divide - 1) / divide;
        bufSize = (bufSize + blockSize - 1) / blockSize * blockSize;
    }

    // 调整bufSize 适应小shape场景，尽量均匀分到多核
    int64_t perCoreElem = ((x * ny) + coreNum - 1) / coreNum;
    perCoreElem = (perCoreElem + blockSize - 1) / blockSize * blockSize;
    if (bufSize <= REGBASE_MIN_BUFFER_SIZE) {
        return bufSize;
    }
    if (perCoreElem <= REGBASE_MIN_BUFFER_SIZE) {
        return REGBASE_MIN_BUFFER_SIZE;
    }
    if (perCoreElem <= bufSize) {
        return perCoreElem;
    }
    // 大于 bufSize
    if (perCoreElem <= bufSize * REGBASE_MAX_SHAPE_BUFFER_SIZE_MULTI_TIMES) {
        int64_t balanceDiv = (perCoreElem + bufSize - 1) / bufSize;
        int64_t balanceBufSize = perCoreElem / balanceDiv;
        balanceBufSize = (balanceBufSize + blockSize - 1) / blockSize * blockSize;
        balanceBufSize = balanceBufSize < REGBASE_MIN_BUFFER_SIZE ? REGBASE_MIN_BUFFER_SIZE : balanceBufSize;
        return balanceBufSize;
    }
    // 大于 REGBASE_MAX_SHAPE_BUFFER_SIZE_MULTI_TIMES * bufSize
    return bufSize;
}

constexpr uint32_t BATCH_MODE = 1;

inline static ge::graphStatus SetTilingDataForGeGluV2(gert::TilingContext* context, GeGluV2TilingData& tilingData)
{
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static void GetTilingDataSmall(GeGluV2TilingData& tilingData, TilingParam& tilingParam, bool isVreduce)
{
    int64_t splitSizeAlign{tilingParam.ny};
    int64_t blockSize = tilingParam.blockSize;
    int64_t group{1};
    if (isVreduce || tilingParam.ny % blockSize == 0) {
        group = tilingParam.bufSize / tilingParam.ny;
    } else {
        splitSizeAlign = (tilingParam.ny + blockSize - 1) / blockSize * blockSize;
        group = tilingParam.bufSize / splitSizeAlign;
    }

    int64_t numPerCore = Ops::Base::CeilDiv(tilingParam.x, tilingParam.coreNum);
    int64_t realCoreNum = Ops::Base::CeilDiv(tilingParam.x, numPerCore);
    // numPerCore对group向下取整，剩余的用尾核处理，直接一把copy即可，不用担心覆盖
    if (tilingParam.isAscend310P && tilingParam.ny % blockSize != 0 &&
        (numPerCore % group) * tilingParam.ny < blockSize) {
        // 判断只有在nLastTailGroup * splitSize小于blockSize时才生效，在使用workspace转存需要向前借几个数据
        numPerCore = numPerCore / group * group;
    }

    int64_t numOneBlock = Ops::Base::CeilDiv(blockSize, tilingParam.ny);
    if (tilingParam.ny < blockSize && numPerCore <= numOneBlock && tilingParam.isAscend310P) {
        // 一个核处理不满32B时，合并多行
        numPerCore = tilingParam.x;
        realCoreNum = Ops::Base::CeilDiv(tilingParam.x, numPerCore);
    }

    tilingData.set_splitSize(tilingParam.ny);
    tilingData.set_group(group);
    tilingData.set_numPerCore(numPerCore);
    tilingData.set_loopNum(numPerCore / group);
    tilingData.set_nLastTailGroup(numPerCore % group);
    tilingData.set_realCoreNum(realCoreNum);

    if (numPerCore * realCoreNum != tilingParam.x) {
        int64_t tailTotalNum = tilingParam.x - numPerCore * (realCoreNum - 1);
        tilingData.set_tailLoopNum(tailTotalNum / group);
        tilingData.set_lastTailGroup(tailTotalNum % group);
    } else {
        tilingData.set_tailLoopNum(0);
        tilingData.set_lastTailGroup(0);
    }
}

static void GetTilingDataBig(GeGluV2TilingData& tilingData, TilingParam& tilingParam)
{
    int64_t group = tilingParam.ny / tilingParam.bufSize;
    int64_t tailNum = tilingParam.ny - group * tilingParam.bufSize;

    tilingData.set_realCoreNum(tilingParam.coreNum);
    tilingData.set_splitSize(tilingParam.bufSize);
    tilingData.set_group(group);
    tilingData.set_tailLoopNum(tailNum);
    if (tailNum == 0) {
        tilingData.set_loopNum(tilingParam.x * group);
    } else {
        tilingData.set_loopNum(tilingParam.x * (group + 1));
    }
}

static void GetFp16TilingData(GeGluV2TilingData& tilingData, TilingParam& tilingParam)
{
    int64_t bufSizeLimit;
    if (tilingParam.isRegbase) {
        bufSizeLimit = CalcRegbaseBufSize(tilingParam.ubSize, FP16_DTYPE_BYTES, FP16_BLOCK_SIZE,
                                          tilingParam.approximate, tilingParam.ny, tilingParam.x, tilingParam.coreNum);
    } else {
        bufSizeLimit = COMMON_BUFFER_SIZE_LIMIT;
    }
    tilingParam.blockSize = FP16_BLOCK_SIZE;

    if (tilingParam.ny == 1) {
        tilingParam.bufSize = bufSizeLimit;
        tilingParam.useUbCoeff = COMMON_B2_USE_UB_BYTE_COFF;
        GetTilingDataSmall(tilingData, tilingParam, true);
        int64_t tilingkey = tilingParam.approximate == 1 ? static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_102) :
                                                           static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_112);
        tilingData.set_tilingKey(tilingkey);
    } else if (tilingParam.ny <= bufSizeLimit) {
        tilingParam.bufSize = bufSizeLimit;
        tilingParam.useUbCoeff = COMMON_B2_USE_UB_BYTE_COFF;
        GetTilingDataSmall(tilingData, tilingParam, false);
        int64_t tilingkey = tilingParam.approximate == 1 ? static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_101) :
                                                           static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_111);
        tilingData.set_tilingKey(tilingkey);
    } else {
        tilingParam.bufSize = bufSizeLimit;
        tilingParam.useUbCoeff = COMMON_B2_USE_UB_BYTE_COFF;
        GetTilingDataBig(tilingData, tilingParam);
        int64_t tilingkey = tilingParam.approximate == 1 ? static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_103) :
                                                           static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_113);
        tilingData.set_tilingKey(tilingkey);
    }

    tilingData.set_blockSize(FP16_BLOCK_SIZE);
    tilingData.set_bufSize(tilingParam.bufSize);
}

static void GetBf16TilingData(GeGluV2TilingData& tilingData, TilingParam& tilingParam)
{
    int64_t bufSizeLimit;
    if (tilingParam.isRegbase) {
        bufSizeLimit = CalcRegbaseBufSize(tilingParam.ubSize, BF16_DTYPE_BYTES, BFP16_BLOCK_SIZE,
                                          tilingParam.approximate, tilingParam.ny, tilingParam.x, tilingParam.coreNum);
    } else {
        bufSizeLimit = COMMON_BUFFER_SIZE_LIMIT;
    }
    tilingParam.blockSize = BFP16_BLOCK_SIZE;

    if (tilingParam.ny == 1) {
        tilingParam.bufSize = bufSizeLimit;
        tilingParam.useUbCoeff = COMMON_B2_USE_UB_BYTE_COFF;
        GetTilingDataSmall(tilingData, tilingParam, true);
        int64_t tilingkey = tilingParam.approximate == 1 ? static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_202) :
                                                           static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_212);
        tilingData.set_tilingKey(tilingkey);
    } else if (tilingParam.ny <= bufSizeLimit) {
        tilingParam.bufSize = bufSizeLimit;
        tilingParam.useUbCoeff = COMMON_B2_USE_UB_BYTE_COFF;
        GetTilingDataSmall(tilingData, tilingParam, false);
        int64_t tilingkey = tilingParam.approximate == 1 ? static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_201) :
                                                           static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_211);
        tilingData.set_tilingKey(tilingkey);
    } else {
        tilingParam.bufSize = bufSizeLimit;
        tilingParam.useUbCoeff = COMMON_B2_USE_UB_BYTE_COFF;
        GetTilingDataBig(tilingData, tilingParam);
        int64_t tilingkey = tilingParam.approximate == 1 ? static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_203) :
                                                           static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_213);
        tilingData.set_tilingKey(tilingkey);
    }

    tilingData.set_blockSize(BFP16_BLOCK_SIZE);
    tilingData.set_bufSize(tilingParam.bufSize);
}

static void GetFp32TilingData(GeGluV2TilingData& tilingData, TilingParam& tilingParam)
{
    int64_t commBufSizeLimit;
    int64_t vReduceErfBufSizeLimit;
    if (tilingParam.isRegbase) {
        commBufSizeLimit = CalcRegbaseBufSize(tilingParam.ubSize, FP32_DTYPE_BYTES, FP32_BLOCK_SIZE,
                                              tilingParam.approximate, tilingParam.ny, tilingParam.x,
                                              tilingParam.coreNum);
        vReduceErfBufSizeLimit = commBufSizeLimit;
    } else {
        commBufSizeLimit = COMMON_BUFFER_SIZE_LIMIT;
        vReduceErfBufSizeLimit = FP32_BUFFER_SIZE_VREDUCE_ERF;
    }

    tilingParam.blockSize = FP32_BLOCK_SIZE;

    if (tilingParam.ny == 1) {
        tilingParam.bufSize = tilingParam.approximate == 1 ? commBufSizeLimit : vReduceErfBufSizeLimit;
        tilingParam.useUbCoeff = NY1_B4_USE_UB_BYTE_COFF;
        GetTilingDataSmall(tilingData, tilingParam, true);
        int64_t tilingkey = tilingParam.approximate == 1 ? static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_302) :
                                                           static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_312);
        tilingData.set_tilingKey(tilingkey);
    } else if (tilingParam.ny <= commBufSizeLimit) {
        tilingParam.bufSize = commBufSizeLimit;
        tilingParam.useUbCoeff = COMMON_B4_USE_UB_BYTE_COFF;
        GetTilingDataSmall(tilingData, tilingParam, false);
        int64_t tilingkey = tilingParam.approximate == 1 ? static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_301) :
                                                           static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_311);
        tilingData.set_tilingKey(tilingkey);
    } else {
        tilingParam.bufSize = commBufSizeLimit;
        tilingParam.useUbCoeff = COMMON_B4_USE_UB_BYTE_COFF;
        GetTilingDataBig(tilingData, tilingParam);
        int64_t tilingkey = tilingParam.approximate == 1 ? static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_303) :
                                                           static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_313);
        tilingData.set_tilingKey(tilingkey);
    }

    tilingData.set_blockSize(FP32_BLOCK_SIZE);
    tilingData.set_bufSize(tilingParam.bufSize);
}

static void GetFp16TilingDataAscend310P(GeGluV2TilingData& tilingData, TilingParam& tilingParam)
{
    static const int64_t FP16_BUFFER_SIZE_LIMIT = 8192;
    tilingParam.blockSize = FP16_BLOCK_SIZE;

    if (tilingParam.ny <= FP16_BUFFER_SIZE_LIMIT) {
        static const int64_t FP16_BUFFER_SIZE_SMALL = 8192;
        tilingParam.bufSize = FP16_BUFFER_SIZE_SMALL;
        GetTilingDataSmall(tilingData, tilingParam, false);
        tilingData.set_tilingKey(static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_101));
    } else {
        static const int64_t FP16_BUFFER_SIZE_BIG = 8192;
        tilingParam.bufSize = FP16_BUFFER_SIZE_BIG;
        GetTilingDataBig(tilingData, tilingParam);
        tilingData.set_tilingKey(static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_103));
    }

    tilingData.set_blockSize(FP16_BLOCK_SIZE);
    tilingData.set_bufSize(tilingParam.bufSize);
}

static void GetFp32TilingDataAscend310P(GeGluV2TilingData& tilingData, TilingParam& tilingParam)
{
    static const int64_t FP32_BUFFER_SIZE_LIMIT = 8192;
    tilingParam.blockSize = FP32_BLOCK_SIZE;

    if (tilingParam.ny <= FP32_BUFFER_SIZE_LIMIT) {
        static const int64_t FP32_BUFFER_SIZE_SMALL = 8192;
        tilingParam.bufSize = FP32_BUFFER_SIZE_SMALL;
        GetTilingDataSmall(tilingData, tilingParam, false);
        tilingData.set_tilingKey(static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_301));
    } else {
        static const int64_t FP32_BUFFER_SIZE_BIG = 6144;
        tilingParam.bufSize = FP32_BUFFER_SIZE_BIG;
        GetTilingDataBig(tilingData, tilingParam);
        tilingData.set_tilingKey(static_cast<int64_t>(GeGluV2TilingKey::TILINGKEY_303));
    }

    tilingData.set_blockSize(FP32_BLOCK_SIZE);
    tilingData.set_bufSize(tilingParam.bufSize);
}

static ge::graphStatus CheckInputParams(gert::TilingContext* context)
{
    auto input = context->GetInputTensor(INPUT_IDX);
    OP_CHECK_IF(
        input == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "ge_glu_v2" : (context)->GetNodeName()),
            "x"),
        return ge::GRAPH_FAILED);

    auto dtype = context->GetInputDesc(INPUT_IDX)->GetDataType();
    int32_t typeSize = ge::GetSizeByDataType(dtype);

    OP_CHECK_IF(dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16 && dtype != ge::DT_FLOAT,
                OP_LOGE(context, "input dtype only support fp16, fp32, bf16 currently, please check."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((typeSize <= 0), OP_LOGE(context, "typeSize is invalid %d, please check.", typeSize),
                return ge::GRAPH_FAILED);

    // How to check split dim is -1.
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4GeGluV2(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepare4GeGluV2 enter.");

    auto compileInfo = context->GetCompiledInfo<GeGluV2CompileInfo>();
    if (compileInfo == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "ge_glu_v2" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "Tiling framework object compileInfo is unavailable; check the operator registration "
                               "and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "Tiling framework object compileInfo is unavailable; check "
                                                         "the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "ge_glu_v2" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "Tiling framework object platformInfo is unavailable; check the operator registration "
                               "and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "Tiling framework object platformInfo is unavailable; check "
                                                         "the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_LOGD(context, "Tiling totalCoreNum: %d", compileInfo->totalCoreNum);
    if (ascendcPlatform.GetCurNpuArch() == Ops::Base::DAV_2002) {
        compileInfo->totalCoreNum = compileInfo->totalCoreNum + ascendcPlatform.GetCoreNumVector();
    }
    OP_LOGD(context, "Tiling totalCoreNum: %d", compileInfo->totalCoreNum);

    if ((compileInfo->totalCoreNum <= 0)) {
        const std::string msgOpName = std::string(Ops::Base::GetOpInfo(context));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", "TilingPrepare4GeGluV2 fail to get core num.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "TilingPrepare4GeGluV2 fail to get core num."}));
        return ge::GRAPH_FAILED;
    };

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    if ((compileInfo->ubSizePlatForm <= 0)) {
        const std::string msgOpName = std::string(Ops::Base::GetOpInfo(context));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", "TilingPrepare4GeGluV2 fail to get ub size.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "TilingPrepare4GeGluV2 fail to get ub size."}));
        return ge::GRAPH_FAILED;
    };

    compileInfo->isAscend310P = ascendcPlatform.GetCurNpuArch() == Ops::Base::DAV_2002;
    compileInfo->isRegbase = IsRegbaseSocVersion(context);
    OP_LOGD(context, "TilingPrepare4GeGluV2 exit. coreNum: %d ubSize: %lu", compileInfo->totalCoreNum,
            compileInfo->ubSizePlatForm);
    return ge::GRAPH_SUCCESS;
}

static size_t GetAttrSplitDim(const gert::TilingContext* context)
{
    auto input = context->GetInputTensor(INPUT_IDX);
    auto inputShape = input->GetStorageShape();
    size_t inputShapeSize = static_cast<int64_t>(inputShape.GetDimNum());

    auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "ge_glu_v2" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(
            msgOpName.c_str(), "%s",
            "Tiling framework object attrs is unavailable; check the operator registration and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "Tiling framework object attrs is unavailable; check the "
                                                         "operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    auto* attrDim = attrs->GetAttrPointer<int64_t>(ATTR_DIM_INDEX);
    if (attrDim == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "ge_glu_v2" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "Tiling framework object attrDim is unavailable; check the operator registration and "
                               "context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "Tiling framework object attrDim is unavailable; check the "
                                                         "operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    auto splitDim = static_cast<int64_t>(*attrDim);
    OP_LOGD(context, "splitDim is %ld, inputShapeSize is %zu", splitDim, inputShapeSize);

    size_t splitDimU = splitDim < 0 ? splitDim + inputShapeSize : splitDim;

    OP_CHECK_IF((splitDimU >= inputShapeSize),
                OP_LOGE(context, "The value of attr [dim] must be in the range [-%zu, %zu], but got [%zu].",
                        inputShapeSize, inputShapeSize, splitDimU),
                return SPLIT_ERROR_STATUS);

    OP_CHECK_IF((inputShape.GetDim(splitDimU) % SPLIT_FACTOR != 0),
                OP_LOGE(context, "The dim of: %zu can not be split with factor: %u on value %ld.", splitDimU,
                        SPLIT_FACTOR, inputShape.GetDim(splitDimU)),
                return SPLIT_ERROR_STATUS);

    return splitDimU;
}

static ge::graphStatus GetTilingAttr(const gert::TilingContext* context, TilingParam& tilingParam)
{
    auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "ge_glu_v2" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(
            msgOpName.c_str(), "%s",
            "Tiling framework object attrs is unavailable; check the operator registration and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "Tiling framework object attrs is unavailable; check the "
                                                         "operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };

    auto* attrActivateLeft = attrs->GetAttrPointer<bool>(ATTR_ACTIVATE_LEFT_INDEX);
    if (attrActivateLeft == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "ge_glu_v2" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "Tiling framework object attrActivateLeft is unavailable; check the operator "
                               "registration and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "Tiling framework object attrActivateLeft is unavailable; "
                                                         "check the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    auto activateLeft = *attrActivateLeft;
    int64_t activateLeftInt = activateLeft ? 1 : 0;
    tilingParam.activateLeft = activateLeftInt;

    auto* attrApproximate = attrs->GetAttrPointer<int64_t>(ATTR_APPROXIMATE_INDEX);
    if (attrApproximate == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "ge_glu_v2" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "Tiling framework object attrApproximate is unavailable; check the operator "
                               "registration and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "Tiling framework object attrApproximate is unavailable; "
                                                         "check the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    auto approximate = static_cast<int64_t>(*attrApproximate);
    OP_CHECK_IF(
        (approximate != 0 && approximate != 1),
        OP_LOGE(context, "The value of attr [approximate] must be in the enum [0, 1], but got [%ld].", approximate),
        return ge::GRAPH_FAILED);
    tilingParam.approximate = approximate;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetTillingParam(const gert::TilingContext* context, TilingParam& tilingParam)
{
    // get attrs of dim and activateLeft
    size_t splitDim = GetAttrSplitDim(context);
    OP_CHECK_IF((splitDim == SPLIT_ERROR_STATUS), OP_LOGE_WITHOUT_REPORT(context, "Get Split Dim Failed."),
                return ge::GRAPH_FAILED);

    auto inputShape = context->GetInputTensor(INPUT_IDX)->GetStorageShape();
    // fuse dims
    int64_t x{1}, y{1}, n{1};
    for (size_t i = 0; i < inputShape.GetDimNum(); i++) {
        if (i < splitDim) {
            x *= inputShape.GetDim(i);
        } else if (i > splitDim) {
            y *= inputShape.GetDim(i);
        } else {
            n = inputShape.GetDim(i) / SPLIT_FACTOR;
        }
    }
    int64_t ny = n * y;
    OP_CHECK_IF((x == 0 || ny == 0), OP_LOGE(context, "Get input tensor is empty."), return ge::GRAPH_FAILED);

    auto compileInfo = context->GetCompileInfo<GeGluV2CompileInfo>();
    tilingParam.x = x;
    tilingParam.ny = ny;
    tilingParam.coreNum = compileInfo->totalCoreNum;
    tilingParam.ubSize = compileInfo->ubSizePlatForm;
    tilingParam.isAscend310P = compileInfo->isAscend310P;
    tilingParam.isRegbase = compileInfo->isRegbase;

    OP_CHECK_IF(GetTilingAttr(context, tilingParam) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context, "GeGluV2SetTilingData set tiling data fail."), return ge::GRAPH_FAILED);

    OP_LOGI(context,
            "tilingParm is x:%ld, coreNum: %ld, ubSize: %lu splitDim: %zu, activate_left: %ld, approximate: %ld, \
           isAscend310P: %d isRegbase: %d.",
            tilingParam.x, tilingParam.coreNum, tilingParam.ubSize, splitDim, tilingParam.activateLeft,
            tilingParam.approximate, tilingParam.isAscend310P, tilingParam.isRegbase);

    return ge::GRAPH_SUCCESS;
}

static void GetTillingData(ge::DataType dtype, TilingParam& tilingParam, GeGluV2TilingData& tilingData)
{
    if (tilingParam.isAscend310P) {
        if (dtype == ge::DT_FLOAT16) {
            GetFp16TilingDataAscend310P(tilingData, tilingParam);
        } else {
            GetFp32TilingDataAscend310P(tilingData, tilingParam);
        }
    } else {
        if (dtype == ge::DT_FLOAT16) {
            GetFp16TilingData(tilingData, tilingParam);
        } else if (dtype == ge::DT_BF16) {
            GetBf16TilingData(tilingData, tilingParam);
        } else {
            GetFp32TilingData(tilingData, tilingParam);
        }
    }
}

static ge::graphStatus Tiling4GeGluV2(gert::TilingContext* context)
{
    OP_LOGD(context, "Tiling4GeGluV2 enter.");
    context->SetScheduleMode(BATCH_MODE);
    OP_CHECK_IF(CheckInputParams(context) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context, "InputParams not valid."), return ge::GRAPH_FAILED);

    TilingParam tilingParam;
    OP_CHECK_IF(GetTillingParam(context, tilingParam) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context, "Get Tiling Param Failed."), return ge::GRAPH_FAILED);

    auto dtype = context->GetInputDesc(INPUT_IDX)->GetDataType();
    GeGluV2TilingData tilingData;

    GetTillingData(dtype, tilingParam, tilingData);

    tilingData.set_ny(tilingParam.ny);
    tilingData.set_activateLeft(tilingParam.activateLeft);
    tilingData.set_approximate(tilingParam.approximate);

    OP_CHECK_IF(SetTilingDataForGeGluV2(context, tilingData) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context, "GeGluV2SetTilingData set tiling data fail."), return ge::GRAPH_FAILED);

    context->SetBlockDim(tilingData.get_realCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());
    size_t* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "ge_glu_v2" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "Tiling framework object workspaces is unavailable; check the operator registration and "
                               "context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "Tiling framework object workspaces is unavailable; check the "
                                                         "operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    workspaces[0] = WORK_SPACE_SIZE + tilingParam.coreNum * BYTES_ONE_BLOCK * FP32_DTYPE_BYTES;

    OP_LOGD(context, "tilingData is splitSize:%ld, group:%ld, realCoreNum:%ld, numPerCore:%ld, loopNum:%ld, \
           tailLoopNum:%ld,nLastTailGroup:%ld, lastTailGroup:%ld, tilingKey:%ld, blockSize:%ld, \
           activateLeft: %ld, ny: %ld, approximate: %ld ",
            tilingData.get_splitSize(), tilingData.get_group(), tilingData.get_realCoreNum(),
            tilingData.get_numPerCore(), tilingData.get_loopNum(), tilingData.get_tailLoopNum(),
            tilingData.get_nLastTailGroup(), tilingData.get_lastTailGroup(), tilingData.get_tilingKey(),
            tilingData.get_blockSize(), tilingData.get_activateLeft(), tilingData.get_ny(),
            tilingData.get_approximate());

    OP_LOGD(context, "Tiling4GeGluV2 exit.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GeGluV2).Tiling(Tiling4GeGluV2).TilingParse<GeGluV2CompileInfo>(TilingPrepare4GeGluV2);
} // namespace optiling
