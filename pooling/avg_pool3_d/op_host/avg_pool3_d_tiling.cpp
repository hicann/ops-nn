/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file avg_pool3_d_tiling.cpp
 * \brief tiling function of avg_pool3d
 */
#include "avg_pool3_d_tiling.h"
#include <string>
#include <vector>
#include <cstdint>
#include <type_traits>

#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "cube_tiling_runtime.h"
#include "error_util.h"

#include "platform/platform_infos_def.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "pool_tiling_templates_registry.h"

using optiling::PoolTilingRegistry;

namespace {
constexpr int32_t SHAPE_SIZE_6D = 6;
constexpr size_t X_INDEX = 0;
constexpr size_t Y_INDEX = 0;
constexpr size_t KSIZE_INDEX = 0;
constexpr size_t STRIDES_INDEX = 1;
constexpr size_t PADS_INDEX = 2;
constexpr size_t CEIL_MODE_INDEX = 3;
constexpr size_t COUNT_INCLUDE_PAD_INDEX = 4;
constexpr size_t DIVISOR_OVERRIDE_INDEX = 5;
constexpr size_t DATA_FORMAT_INDEX = 6;
constexpr size_t X_DIMS = 5;
constexpr size_t Y_DIMS = 5;
constexpr size_t DIM0 = 0;
constexpr size_t DIM1 = 1;
constexpr size_t DIM2 = 2;
constexpr size_t DIM3 = 3;
constexpr size_t DIM4 = 4;
constexpr size_t OUTPUT_SIZE_DIMS = 3;
constexpr size_t COMPATIABLE_PAD_DIM = 6;
constexpr size_t LEN1 = 1;
constexpr size_t LEN3 = 3;

constexpr int64_t WORK_SPACE_SIZE = 16777216;
constexpr int64_t CORE_SYNC_SPACE_SIZE = 2048;
constexpr uint32_t RESERVED_UB_SIZE = 10U * 1024U;
constexpr uint32_t INDEX_BUF_SIZE = 9U * 1024U;
constexpr uint32_t INDEX_BUF_NUM = 9;

constexpr int32_t FP32_DTYPE_KEY = 0;
constexpr int32_t FP16_DTYPE_KEY = 1;
constexpr int32_t BF16_DTYPE_KEY = 2;

constexpr int32_t MODE_SPLIT_C = 1;
constexpr int32_t MODE_SPLIT_W = 2;
constexpr int32_t MODE_MULTI_W = 3;
constexpr int32_t MODE_REDUCE_D = 4;
constexpr int32_t MODE_NORMAL = 5;

constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t MAX_TILE_NUM = 4095;

constexpr uint64_t KERNEL_SIZE_LIMIT = 128;
constexpr uint64_t BLOCK_LEN = 256;
constexpr uint64_t KERNEL_W_LIMIT = 16;

struct TilingParams {
    uint64_t inN = 0;
    uint64_t inC = 0;
    uint64_t tileC = 0;
    uint64_t inD = 0;
    uint64_t inH = 0;
    uint64_t inW = 0;
    uint64_t outD = 0;
    uint64_t outH = 0;
    uint64_t outW = 0;
    uint64_t kD = 0;
    uint64_t kH = 0;
    uint64_t kW = 0;
    uint64_t dD = 0;
    uint64_t dH = 0;
    uint64_t dW = 0;
    uint64_t pD = 0;
    uint64_t pH = 0;
    uint64_t pW = 0;
    int64_t divisorOverride = 0;
    uint64_t countIncludePad = 0;
    uint64_t ceilMode = 0;
    uint64_t formerLength = 0;
    uint64_t formerNum = 0;
    uint64_t tailLength = 0;
    uint64_t tailNum = 0;
    uint64_t indexBufLen = 0;
    uint64_t windowWNum = 0;
    uint64_t tileInput = 0;
    uint64_t tileHW = 0;
    uint64_t atomicAddNum = 0;

    uint32_t ubSize = 0;
    uint64_t coreNum = 0;
    uint64_t usedCoreNum = 0;
    int32_t dataTypeKey = 0;
    std::string dataFormat = "NDHWC";
    bool isonlyT = false;
    uint64_t useCoreNum = 0;
    uint64_t ncFactor = 0;
    uint64_t doFactor = 0;
    uint64_t hoFactor = 0;
    uint64_t woFactor = 0;
    uint64_t ncOuter = 0;
    uint64_t doOuter = 0;
    uint64_t hoOuter = 0;
    uint64_t woOuter = 0;
    uint64_t ncTail = 0;
    uint64_t doTail = 0;
    uint64_t hoTail = 0;
    uint64_t woTail = 0;

    uint64_t blockFactor = 0;
    uint64_t blockTail = 0;
    uint64_t totalIdx = 0;
    uint64_t coreNums = 0;

    platform_ascendc::SocVersion socVersion;
};

static inline uint64_t FindDivisorWindowNum(uint64_t len, uint64_t initWindowNum) {
    std::vector<uint64_t> divisors;
    uint64_t maxWindowNum = 0;

    for (uint64_t i = 1; i <= static_cast<uint64_t>(std::sqrt(len)) + 1; ++i) {
        if (len % i == 0) {
            divisors.push_back(i);
            if (i != len / i) {
                divisors.push_back(len / i);
            }
        }
    }

    for (auto w: divisors) {
        if (w <= initWindowNum && maxWindowNum < w) {
            maxWindowNum = w;
        }
    }

    return maxWindowNum;
}
} // namespace

namespace optiling {
namespace avg_pool3_d {
using namespace avg_pool3_d_tiling_compile_info;
static bool IsRegbaseSocVersion(platform_ascendc::SocVersion version)
{
    const static std::set<platform_ascendc::SocVersion> regbaseSocVersions = {
        platform_ascendc::SocVersion::ASCEND910_95
    };

    return regbaseSocVersions.find(version) != regbaseSocVersions.end();
}

bool IsRegbaseSocVersion(const gert::TilingParseContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    return IsRegbaseSocVersion(socVersion);
}

bool IsRegbaseSocVersion(const gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    return IsRegbaseSocVersion(socVersion);
}

inline std::unique_ptr<nlohmann::json> GetCompileInfoJson(gert::TilingParseContext* context) {
  auto json_str = context->GetCompiledJson();
  OP_CHECK_IF(json_str == nullptr, OP_LOGE(context->GetNodeName(), "json_str is nullptr!"), return nullptr);
  std::unique_ptr<nlohmann::json> parsed_object_cinfo =
      std::make_unique<nlohmann::json>(nlohmann::json::parse(json_str));
  return parsed_object_cinfo;
}

ge::graphStatus Tiling4AvgPool3DNNRegBase(gert::TilingContext* context)
{
    return PoolTilingRegistry::GetInstance().DoTilingImpl(context);
}

static void ComputeCoreTilingStrategy(TilingParams& params, int32_t& usedCoreNum) {
    uint64_t outputNum = 0;

    if (params.dataFormat == "NDHWC") {
        outputNum = params.inN * params.outD * params.outH * params.outW;
    } else if (params.dataFormat == "NCDHW" && params.isonlyT){
        outputNum = params.inN * params.inC * params.outD;
    } else{
        outputNum = params.inN * params.inC * params.outD * params.outH * params.outW;
    }

    if (outputNum < params.coreNum) {
        params.formerNum = outputNum;
        params.tailNum = 0UL;
        params.formerLength = 1UL;
        params.tailLength = 0UL;
        usedCoreNum = static_cast<int32_t>(outputNum);
    } else if (outputNum % params.coreNum == 0UL) {
        params.formerNum = params.coreNum;
        params.tailNum = 0UL;
        params.formerLength = outputNum / params.coreNum;
        params.tailLength = 0UL;
        usedCoreNum = static_cast<int32_t>(params.coreNum);
    } else {
        params.formerNum = outputNum % params.coreNum;
        params.tailNum = params.coreNum - params.formerNum;
        params.formerLength = outputNum / params.coreNum + 1UL;
        params.tailLength = outputNum / params.coreNum;
        usedCoreNum = static_cast<int32_t>(params.coreNum);
    }
    params.usedCoreNum = static_cast<uint64_t>(usedCoreNum);
}

static bool IsCapbale(TilingParams& params)
{
    // 按照搬运对齐的大小全载UB 和 params.kW<=16, 判断是否走当前模板
    auto kernelWMaxAlign =
        (params.kW + static_cast<unsigned long>(7)) / static_cast<unsigned long>(8) * static_cast<unsigned long>(8);
    bool limitSizeMax = (params.kD * params.kH * kernelWMaxAlign <= KERNEL_SIZE_LIMIT);
    bool limitWMax = (params.kW <= KERNEL_W_LIMIT);
    bool isCapable = limitSizeMax && limitWMax;
    return isCapable;
}

static void SetNormalTilingParams(TilingParams& params)
{
    params.ncFactor = BLOCK_LEN / sizeof(float);
    params.woFactor = params.outW;
    params.hoFactor = params.outH;
    params.doFactor = params.outD;

    params.ncOuter = (params.inN * params.inC + params.ncFactor - static_cast<unsigned long>(1)) / params.ncFactor;
    params.ncTail = params.inN * params.inC - (params.ncOuter - static_cast<unsigned long>(1)) * params.ncFactor;

    if (params.ncOuter < params.coreNum) {
        auto doCntNeed = (params.coreNum + params.ncOuter - static_cast<unsigned long>(1)) / params.ncOuter;
        if (static_cast<unsigned long>(0) != doCntNeed) {
            params.doFactor = params.outD / doCntNeed < static_cast<unsigned long>(1) ? static_cast<unsigned long>(1) :
                              params.outD / doCntNeed;
        }
    }
    params.doOuter = (params.outD + params.doFactor - static_cast<unsigned long>(1)) / params.doFactor;
    params.doTail = params.outD - (params.doOuter - static_cast<unsigned long>(1)) * params.doFactor;

    if (params.ncOuter * params.doOuter < params.coreNum) {
        auto hoCntNeed = (params.coreNum + params.ncOuter * params.doOuter - static_cast<unsigned long>(1)) /
                         (params.ncOuter * params.doOuter);
        if (static_cast<unsigned long>(0) != hoCntNeed) {
            params.hoFactor = params.outH / hoCntNeed < static_cast<unsigned long>(1) ? static_cast<unsigned long>(1) :
                              params.outH / hoCntNeed;
        }
    }
    params.hoOuter = (params.outH + params.hoFactor - static_cast<unsigned long>(1)) / params.hoFactor;
    params.hoTail = params.outH - (params.hoOuter - static_cast<unsigned long>(1)) * params.hoFactor;
    if(params.ncOuter * params.doOuter * params.hoOuter < params.coreNum) {
        auto woCntNeed =
            (params.coreNum + params.ncOuter * params.doOuter * params.hoOuter - static_cast<unsigned long>(1)) /
            (params.ncOuter * params.doOuter * params.hoOuter);
        if (static_cast<unsigned long>(0) != woCntNeed) {
            params.woFactor = params.outW / woCntNeed < static_cast<unsigned long>(1) ? static_cast<unsigned long>(1) :
                              params.outW / woCntNeed;
        }
    }
    params.woOuter = (params.outW + params.woFactor - static_cast<unsigned long>(1)) / params.woFactor;
    params.woTail = params.outW - (params.woOuter - static_cast<unsigned long>(1)) * params.woFactor;
    params.totalIdx = params.ncOuter * params.woOuter * params.hoOuter * params.doOuter;  // 总共的UB块
    params.blockFactor = (params.totalIdx + params.coreNum - static_cast<unsigned long>(1)) / params.coreNum;
    params.useCoreNum = (params.totalIdx + params.blockFactor - static_cast<unsigned long>(1)) / params.blockFactor;
    params.blockTail = params.totalIdx - (params.useCoreNum - static_cast<unsigned long>(1)) * params.blockFactor;
}

static void Tilling4NCDHWNormal(TilingParams& params)
{
    SetNormalTilingParams(params);
}

static void ComputeUBTilingStrategy(TilingParams& params, int32_t& mode) {
    int32_t dataTypeSize = params.dataTypeKey == FP32_DTYPE_KEY ? 4 : 2;

    uint64_t alignNum = static_cast<uint64_t>(BLOCK_SIZE) * 2UL / dataTypeSize;
    uint64_t tileLen = params.ubSize / (static_cast<uint64_t>(dataTypeSize) + sizeof(float) * 2UL) / alignNum *
                       alignNum;

    if (params.dataFormat == "NCDHW" && params.isonlyT) {
        mode = MODE_REDUCE_D;
        uint64_t alignHW = (params.inH * params.inW + alignNum - 1UL) / alignNum * alignNum;
        params.tileHW = alignHW > tileLen ? tileLen : alignHW;
        uint64_t tileTailLen = (params.inH * params.inW) % params.tileHW;
        params.atomicAddNum = (tileTailLen < alignNum) && (tileTailLen != 0UL) ? 1UL : 0UL;
        return;
    } else if (params.dataFormat == "NCDHW") {
        //走Nornal模板
        bool isNormal = IsCapbale(params);
        if(isNormal){
            Tilling4NCDHWNormal(params);
            mode = MODE_NORMAL;
            return;
        }
    }
    uint64_t alignC = (params.inC + alignNum - 1UL) / alignNum * alignNum;

    uint64_t doubleC = 2UL * alignC;
    if (doubleC > tileLen) {
        mode = MODE_SPLIT_C;
        params.tileC = alignC > tileLen ? tileLen : alignC;
        uint64_t tileTailLen = params.inC % params.tileC;
        params.atomicAddNum = (tileTailLen < alignNum) && (tileTailLen != 0UL) ? 1UL : 0UL;
        return;
    }

    uint64_t alignNumSingle = static_cast<uint64_t>(BLOCK_SIZE) / static_cast<uint64_t>(dataTypeSize);
    uint64_t alignCSingle = (params.inC + alignNumSingle - 1UL) / alignNumSingle * alignNumSingle;
    if (params.inC < alignNumSingle) {
        params.atomicAddNum = (alignCSingle - 1UL) / params.inC;
    }

    uint64_t tileInput =
        (static_cast<uint64_t>(params.ubSize) / alignC - static_cast<uint64_t>(dataTypeSize) - sizeof(float)) /
        sizeof(float);
    if (tileInput < params.kW) {
        mode = MODE_SPLIT_W;
        params.tileInput = tileInput < static_cast<uint64_t>(MAX_TILE_NUM) ? tileInput :
                           static_cast<uint64_t>(MAX_TILE_NUM);
        return;
    }

    if (params.dW > params.kW) {
        mode = MODE_SPLIT_W;
        params.tileInput = params.kW;
        return;
    }

    uint64_t windowWNum = (params.ubSize / alignC - params.kW * sizeof(float)) /
        ((params.dW + 1UL) * sizeof(float) + static_cast<uint64_t>(dataTypeSize));

    mode = MODE_MULTI_W;
    // when multiW kw over 1024 and  datacopy step by step(avoid DataCopyParams), skip MAX_TILE_NUM limit to reduce loop iterations.
    if (!(params.socVersion == platform_ascendc::SocVersion::ASCEND310P && params.kW > 1024 &&
          alignCSingle != params.inC)) {
        windowWNum = windowWNum * params.kW <= static_cast<uint64_t>(MAX_TILE_NUM) ? windowWNum :
        static_cast<uint64_t>(MAX_TILE_NUM) / params.kW;
    }
    windowWNum = windowWNum < params.outW ? windowWNum : params.outW;
    params.windowWNum = FindDivisorWindowNum(params.indexBufLen, windowWNum);

    if (windowWNum == 0UL) {
        mode = MODE_SPLIT_C;
        params.tileC = alignC > tileLen ? tileLen : alignC;
    }
}

static void SetTiling(const TilingParams& params, AvgPool3DTilingData& tiling) {
    tiling.set_inN(params.inN);
    tiling.set_inC(params.inC);
    tiling.set_tileC(params.tileC);
    tiling.set_inD(params.inD);
    tiling.set_inH(params.inH);
    tiling.set_inW(params.inW);
    tiling.set_outD(params.outD);
    tiling.set_outH(params.outH);
    tiling.set_outW(params.outW);
    tiling.set_kD(params.kD);
    tiling.set_kH(params.kH);
    tiling.set_kW(params.kW);
    tiling.set_dD(params.dD);
    tiling.set_dH(params.dH);
    tiling.set_dW(params.dW);
    tiling.set_pD(params.pD);
    tiling.set_pH(params.pH);
    tiling.set_pW(params.pW);
    tiling.set_divisorOverride(params.divisorOverride);
    tiling.set_countIncludePad(params.countIncludePad);
    tiling.set_ceilMode(params.ceilMode);
    tiling.set_formerLength(params.formerLength);
    tiling.set_formerNum(params.formerNum);
    tiling.set_tailLength(params.tailLength);
    tiling.set_tailNum(params.tailNum);
    tiling.set_indexBufLen(params.indexBufLen);
    tiling.set_windowWNum(params.windowWNum);
    tiling.set_tileInput(params.tileInput);
    tiling.set_tileHW(params.tileHW);
    tiling.set_atomicAddNum(params.atomicAddNum);
    tiling.set_useCoreNum(params.useCoreNum);
    tiling.set_ncFactor(params.ncFactor);
    tiling.set_doFactor(params.doFactor);
    tiling.set_hoFactor(params.hoFactor);
    tiling.set_woFactor(params.woFactor);
    tiling.set_ncOuter(params.ncOuter);
    tiling.set_doOuter(params.doOuter);
    tiling.set_hoOuter(params.hoOuter);
    tiling.set_woOuter(params.woOuter);
    tiling.set_ncTail(params.ncTail);
    tiling.set_doTail(params.doTail);
    tiling.set_hoTail(params.hoTail);
    tiling.set_woTail(params.woTail);
    tiling.set_blockFactor(params.blockFactor);
    tiling.set_blockTail(params.blockTail);
    tiling.set_totalIdx(params.totalIdx);
    tiling.set_coreNums(params.coreNums);
    tiling.set_usedCoreNum(params.usedCoreNum);
}

static void PrintTiling(const gert::TilingContext* context, AvgPool3DTilingData& tiling, uint32_t tilingKey) {
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "tilingKey:          %d.", tilingKey);
    OP_LOGD(nodeName, "usedCoreNum:        %ld.", tiling.get_usedCoreNum());
    OP_LOGD(nodeName, "inN:                %ld.", tiling.get_inN());
    OP_LOGD(nodeName, "inC:                %ld.", tiling.get_inC());
    OP_LOGD(nodeName, "tileC:              %ld.", tiling.get_tileC());
    OP_LOGD(nodeName, "inD:                %ld.", tiling.get_inD());
    OP_LOGD(nodeName, "inH:                %ld.", tiling.get_inH());
    OP_LOGD(nodeName, "inW:                %ld.", tiling.get_inW());
    OP_LOGD(nodeName, "outD:               %ld.", tiling.get_outD());
    OP_LOGD(nodeName, "outH:               %ld.", tiling.get_outH());
    OP_LOGD(nodeName, "outW:               %ld.", tiling.get_outW());
    OP_LOGD(nodeName, "kD:                 %ld.", tiling.get_kD());
    OP_LOGD(nodeName, "kH:                 %ld.", tiling.get_kH());
    OP_LOGD(nodeName, "kW:                 %ld.", tiling.get_kW());
    OP_LOGD(nodeName, "dD:                 %ld.", tiling.get_dD());
    OP_LOGD(nodeName, "dH:                 %ld.", tiling.get_dH());
    OP_LOGD(nodeName, "dW:                 %ld.", tiling.get_dW());
    OP_LOGD(nodeName, "pD:                 %ld.", tiling.get_pD());
    OP_LOGD(nodeName, "pH:                 %ld.", tiling.get_pH());
    OP_LOGD(nodeName, "pW:                 %ld.", tiling.get_pW());
    OP_LOGD(nodeName, "divisorOverride:    %ld.", tiling.get_divisorOverride());
    OP_LOGD(nodeName, "countIncludePad:    %ld.", tiling.get_countIncludePad());
    OP_LOGD(nodeName, "ceilMode:           %ld.", tiling.get_ceilMode());
    OP_LOGD(nodeName, "formerLength:       %ld.", tiling.get_formerLength());
    OP_LOGD(nodeName, "formerNum:          %ld.", tiling.get_formerNum());
    OP_LOGD(nodeName, "tailLength:         %ld.", tiling.get_tailLength());
    OP_LOGD(nodeName, "tailNum:            %ld.", tiling.get_tailNum());
    OP_LOGD(nodeName, "indexBufLen:        %ld.", tiling.get_indexBufLen());
    OP_LOGD(nodeName, "windowWNum:         %ld.", tiling.get_windowWNum());
    OP_LOGD(nodeName, "tileInput:          %ld.", tiling.get_tileInput());
    OP_LOGD(nodeName, "tileHW:             %ld.", tiling.get_tileHW());
    OP_LOGD(nodeName, "atomicAddNum:       %ld.", tiling.get_atomicAddNum());
    OP_LOGD(nodeName, "useCoreNum:         %ld.", tiling.get_useCoreNum());
    OP_LOGD(nodeName, "ncFactor:           %ld.", tiling.get_ncFactor());
    OP_LOGD(nodeName, "doFactor:           %ld.", tiling.get_doFactor());
    OP_LOGD(nodeName, "hoFactor:           %ld.", tiling.get_hoFactor());
    OP_LOGD(nodeName, "woFactor:           %ld.", tiling.get_woFactor());
    OP_LOGD(nodeName, "ncOuter:            %ld.", tiling.get_ncOuter());
    OP_LOGD(nodeName, "doOuter:            %ld.", tiling.get_doOuter());
    OP_LOGD(nodeName, "hoOuter:            %ld.", tiling.get_hoOuter());
    OP_LOGD(nodeName, "woOuter:            %ld.", tiling.get_woOuter());
    OP_LOGD(nodeName, "ncTail:             %ld.", tiling.get_ncTail());
    OP_LOGD(nodeName, "doTail:             %ld.", tiling.get_doTail());
    OP_LOGD(nodeName, "hoTail:             %ld.", tiling.get_hoTail());
    OP_LOGD(nodeName, "woTail:             %ld.", tiling.get_woTail());
    OP_LOGD(nodeName, "blockFactor:        %ld.", tiling.get_blockFactor());
    OP_LOGD(nodeName, "blockTail:          %ld.", tiling.get_blockTail());
    OP_LOGD(nodeName, "totalIdx:           %ld.", tiling.get_totalIdx());
    OP_LOGD(nodeName, "coreNums:           %ld.", tiling.get_coreNums());
}

static bool GetDataTypeKey(ge::DataType dataType, int32_t& dataTypeKey) {
    switch (dataType) {
        case ge::DT_FLOAT16:
            dataTypeKey = FP16_DTYPE_KEY;
            break;
        case ge::DT_BF16:
            dataTypeKey = BF16_DTYPE_KEY;
            break;
        case ge::DT_FLOAT:
            dataTypeKey = FP32_DTYPE_KEY;
            break;
        default:
            return false;
    }

    return true;
}

static ge::graphStatus KernelTiling(gert::TilingContext* context, TilingParams& params) {
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "Tiling start.");

    int32_t usedCoreNum = 0;
    ComputeCoreTilingStrategy(params, usedCoreNum);

    int32_t modeKey = MODE_SPLIT_C;
    ComputeUBTilingStrategy(params, modeKey);

    AvgPool3DTilingData tiling;
    SetTiling(params, tiling);

    uint32_t tilingKey = static_cast<uint32_t>(modeKey * 10 + params.dataTypeKey);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(usedCoreNum);

    PrintTiling(context, tiling, tilingKey);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = params.atomicAddNum == 0UL ? static_cast<size_t>(WORK_SPACE_SIZE) :
                          static_cast<size_t>(WORK_SPACE_SIZE + CORE_SYNC_SPACE_SIZE);

    OP_LOGD(nodeName, "Tiling end.");

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4AvgPool3DVec(gert::TilingContext* context)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "Tiling4AvgPool3D start.");

    auto compileInfo = static_cast<const Conv3DCompileInfo*>(context->GetCompileInfo());

    const gert::Shape xShape = context->GetInputShape(X_INDEX)->GetStorageShape();
    OP_CHECK_IF(
        xShape.GetDimNum() != X_DIMS,
        OP_LOGE(nodeName, "Check x shape failed, the dims of x not equal 5."),
        return ge::GRAPH_FAILED);

    auto dataType = context->GetInputDesc(X_INDEX)->GetDataType();
    int32_t dataTypeKey = FP32_DTYPE_KEY;
    OP_CHECK_IF(
        GetDataTypeKey(dataType, dataTypeKey) == false,
        OP_LOGE(nodeName, "The dtype of input must be in [float32, float16, bfloat16]."),
        return ge::GRAPH_FAILED);

    const gert::Shape yShape = context->GetOutputShape(Y_INDEX)->GetStorageShape();
    OP_CHECK_IF(
        yShape.GetDimNum() != Y_DIMS,
        OP_LOGE(nodeName, "Check y shape failed, the dims of y not equal 5."),
        return ge::GRAPH_FAILED);

    auto attrPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrPtr);

    auto ksizePtr = attrPtr->GetAttrPointer<gert::ContinuousVector>(KSIZE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, ksizePtr);
    OP_CHECK_IF(
        ksizePtr->GetSize() != 5 && ksizePtr->GetSize() != 3 && ksizePtr->GetSize() != 1,
        OP_LOGE(nodeName, "Check kernel_size failed, the size of kernel_size not equal 5, 3 or 1."),
        return ge::GRAPH_FAILED);
    auto ksize = static_cast<const int64_t*>(ksizePtr->GetData());

    auto stridesPtr = attrPtr->GetAttrPointer<gert::ContinuousVector>(STRIDES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, stridesPtr);
    OP_CHECK_IF(
        stridesPtr->GetSize() != 5 && stridesPtr->GetSize() != 3 && stridesPtr->GetSize() != 1,
        OP_LOGE(nodeName, "Check stride failed, the size of strides not equal 5, 3 or 1."),
        return ge::GRAPH_FAILED);
    auto strides = static_cast<const int64_t*>(stridesPtr->GetData());

    auto padsPtr = attrPtr->GetAttrPointer<gert::ContinuousVector>(PADS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, padsPtr);
    OP_CHECK_IF(
        padsPtr->GetSize() != COMPATIABLE_PAD_DIM && padsPtr->GetSize() != 3 && padsPtr->GetSize() != 1,
        OP_LOGE(nodeName, "Check pad failed, the size of pad not equal 6, 3 or 1."),
        return ge::GRAPH_FAILED);
    auto pads = static_cast<const int64_t*>(padsPtr->GetData());

    const bool* ceilMode = attrPtr->GetAttrPointer<bool>(CEIL_MODE_INDEX);
    const bool* countIncludePad = attrPtr->GetAttrPointer<bool>(COUNT_INCLUDE_PAD_INDEX);
    const int64_t* divisorOverride = attrPtr->GetAttrPointer<int64_t>(DIVISOR_OVERRIDE_INDEX);

    const std::string dataFormat = static_cast<std::string>(attrPtr->GetStr(DATA_FORMAT_INDEX));
    OP_CHECK_IF(
        dataFormat != "NDHWC" && dataFormat != "NCDHW",
        OP_LOGE(nodeName, "Check data_format failed, the value of data_format should be NDHWC or NCDHW."),
        return ge::GRAPH_FAILED);

    TilingParams params;
    params.dataFormat = dataFormat;

    if (dataFormat == "NCDHW") {
        params.inN = xShape.GetDim(DIM0);
        params.inC = xShape.GetDim(DIM1);
        params.inD = xShape.GetDim(DIM2);
        params.inH = xShape.GetDim(DIM3);
        params.inW = xShape.GetDim(DIM4);

        params.outD = yShape.GetDim(DIM2);
        params.outH = yShape.GetDim(DIM3);
        params.outW = yShape.GetDim(DIM4);
    } else {
        params.inN = xShape.GetDim(DIM0);
        params.inC = xShape.GetDim(DIM4);
        params.inD = xShape.GetDim(DIM1);
        params.inH = xShape.GetDim(DIM2);
        params.inW = xShape.GetDim(DIM3);

        params.outD = yShape.GetDim(DIM1);
        params.outH = yShape.GetDim(DIM2);
        params.outW = yShape.GetDim(DIM3);
    }

    if (ksizePtr->GetSize() == LEN1 || ksizePtr->GetSize() == LEN3){
        params.kD = ksize[DIM0];
        params.kH = ksizePtr->GetSize() == LEN1 ? ksize[DIM0] : ksize[DIM1];
        params.kW = ksizePtr->GetSize() == LEN1 ? ksize[DIM0] : ksize[DIM2];
    }else{
        params.kD = dataFormat == "NCDHW" ? ksize[DIM2] : ksize[DIM1];
        params.kH = dataFormat == "NCDHW" ? ksize[DIM3] : ksize[DIM2];
        params.kW = dataFormat == "NCDHW" ? ksize[DIM4] : ksize[DIM3];
    }

    if (stridesPtr->GetSize() == LEN1 || stridesPtr->GetSize() == LEN3){
        params.dD = strides[DIM0];
        params.dH = stridesPtr->GetSize() == LEN1 ? strides[DIM0] : strides[DIM1];
        params.dW = stridesPtr->GetSize() == LEN1 ? strides[DIM0] : strides[DIM2];
    }else{
        params.dD = dataFormat == "NCDHW" ? strides[DIM2] : strides[DIM1];
        params.dH = dataFormat == "NCDHW" ? strides[DIM3] : strides[DIM2];
        params.dW = dataFormat == "NCDHW" ? strides[DIM4] : strides[DIM3];
    }

    if (padsPtr->GetSize() != COMPATIABLE_PAD_DIM) {
        params.pD = pads[DIM0];
        params.pH = padsPtr->GetSize() == 1 ? pads[DIM0] : pads[DIM1];
        params.pW = padsPtr->GetSize() == 1 ? pads[DIM0] : pads[DIM2];
    } else {
        params.pD = pads[DIM0];
        params.pH = pads[DIM2];
        params.pW = pads[DIM4];
    }
    params.isonlyT = (params.kH == static_cast<unsigned long>(1)) && (params.kW == static_cast<unsigned long>(1)) &&
                     (params.dH == static_cast<unsigned long>(1)) && (params.dW == static_cast<unsigned long>(1));
    params.countIncludePad = countIncludePad == nullptr ? 0 : static_cast<uint64_t>(*countIncludePad);
    params.divisorOverride = divisorOverride == nullptr ? 0 : static_cast<int64_t>(*divisorOverride);
    params.ceilMode = ceilMode == nullptr ? 0 : static_cast<uint64_t>(*ceilMode);

    params.indexBufLen = static_cast<uint64_t>(INDEX_BUF_SIZE) / static_cast<uint64_t>(INDEX_BUF_NUM) / sizeof(int64_t);
    params.coreNum = compileInfo->core_num;
    params.ubSize = compileInfo->ub_size - RESERVED_UB_SIZE - INDEX_BUF_SIZE;
    params.dataTypeKey = dataTypeKey;
    params.socVersion = compileInfo->platform_soc_version;

    ge::graphStatus tilingStatus = KernelTiling(context, params);

    OP_LOGD(nodeName, "Tiling4AvgPool3d end.");
    return tilingStatus;
}

static ge::graphStatus TilingPrepare4AvgPool3DVec(gert::TilingParseContext* context)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "TilingPrepare4AvgPool3D Vector start.");
    auto compileInfo = context->GetCompiledInfo<Conv3DCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->platform_soc_version = ascendcPlatform.GetSocVersion();
    compileInfo->core_num = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->core_num <= 0), OP_LOGE(nodeName, "Failed to get core num."), return ge::GRAPH_FAILED);
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ub_size = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF((compileInfo->ub_size <= 0), OP_LOGE(nodeName, "Failed to get ub size."), return ge::GRAPH_FAILED);
    compileInfo->is_ascend_c = true;
    compileInfo->is_regbase = IsRegbaseSocVersion(context);
    OP_LOGD(nodeName, "TilingPrepare4AvgPool3D Vector end.");

    return ge::GRAPH_SUCCESS;
}

// remove rt1.0 op register, rt2.0 include rt1.0 tiling REGISTER_OP_TILING_FUNC_BUFFERED_V2(AvgPool3D, AvgPool3DTiling)

ge::graphStatus Tiling4AvgPool3D(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, CUBE_INNER_ERR_REPORT("AvgPool3D", "context is null"), return ge::GRAPH_FAILED);
    auto compileInfo = static_cast<const Conv3DCompileInfo *>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    if (compileInfo->is_regbase) {
        return Tiling4AvgPool3DNNRegBase(context);
    } else if (compileInfo->is_ascend_c) {
        return Tiling4AvgPool3DVec(context);
    } else {
        const auto x_shape = context->GetInputShape(0);
        const auto y_shape = context->GetOutputShape(0);
        return TilingForConv3D(context, x_shape, y_shape, false);
    }
}

static ge::graphStatus TilingPrepare4AvgPool3D(gert::TilingParseContext* context)
{
    std::unique_ptr<nlohmann::json> parsedObjectInfo = GetCompileInfoJson(context);
    const nlohmann::json& compileInfoJson = (*parsedObjectInfo)["_pattern"]; // old version has keyword "_pattern"
    bool isAscendC = compileInfoJson.empty();
    if (isAscendC) {
        return TilingPrepare4AvgPool3DVec(context);
    } else {
        return ParseCubeCompileInfo<Conv3DCompileInfo, 4>(context);  // 4: ndhw
    }
}

IMPL_OP_OPTILING(AvgPool3D)
    .Tiling(Tiling4AvgPool3D)
    .TilingParse<Conv3DCompileInfo>(TilingPrepare4AvgPool3D);

IMPL_OP_OPTILING(AvgPool3DV2)
.Tiling(Tiling4AvgPool3D)
.TilingParse<Conv3DCompileInfo>(TilingPrepare4AvgPool3D);
} // namespace avg_pool3_d
} // namespace optiling
