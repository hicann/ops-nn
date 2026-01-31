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
 * \file matmul_fp32_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/matmul_fp32_tiling_data.h"
#include "../op_kernel/matmul_fp32_tiling_key.h"

using namespace matmul_tiling;

namespace optiling {
constexpr uint64_t WS_SYS_SIZE =  16U * 1024U* 1024U; // 16MB
constexpr uint64_t DB_SIZE = 2UL;
constexpr uint64_t BLOCK_BYTE_SIZE = 32UL;
constexpr uint64_t H_ALIGNED = 16UL;
constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256UL;
constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128UL;
constexpr uint64_t BASIC_BLOCK_K_128_BYTE = 128UL;
constexpr uint64_t DB_OFF_SIZE = 1UL;
constexpr uint64_t ITER_COL_FIRST = 0UL;
constexpr uint64_t DATA_SIZE_FP32 = 4UL;
constexpr uint64_t CO_SIZE_FP32 = 8UL;
constexpr uint64_t BIAS_TABLE_NUM = 256UL;
constexpr uint64_t NUM_HALF = 2UL;
constexpr uint64_t BASE_STEP = 1UL;
constexpr uint64_t BASIC_ALIGN_FP32 = 128UL;
constexpr uint64_t BASIC_ALIGN_16 = 16UL;
constexpr uint64_t BASIC_ALIGN_256 = 256UL;
constexpr uint64_t BASIC_ALIGN_512 = 512UL;
constexpr uint64_t MIN_BASE_M_VALUE = 16;

uint64_t FULL_LOAD_TYPE = MAT_MUL_FP32_NO_FULLLOAD;
bool is_support_bl1 = false;

struct MatmulFp32CompileInfo{
    uint64_t aicNum{0UL};
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
    uint64_t l1Size{0UL};
    uint64_t l2Size{0UL};
    uint64_t l0ASize{0UL};
    uint64_t l0BSize{0UL};
    uint64_t l0CSize{0UL};
    uint64_t btSize{0UL};
    platform_ascendc::SocVersion socVersion;
};

struct MatmulFp32RunInfo
{
    uint64_t usedCoreNum = 1;
    uint64_t singleCoreM = 1;
    uint64_t singleCoreN = 1;
    uint64_t singleCoreK = 1;
    uint64_t baseM = 1;
    uint64_t baseN = 1;
    uint64_t baseK = 1;
    uint64_t stepM = 1;
    uint64_t stepN = 1;
    uint64_t stepKa = 1;
    uint64_t stepKb = 1;
    uint64_t depthA1 = 1;
    uint64_t depthB1 = 1;
    uint64_t iterateOrder = 0;
    uint64_t dbL0c = 0;
};

struct MatmulFp32Args{
    const char* opName = nullptr;
    bool isATrans = false;
    bool isBTrans = false;
    bool hasBias = false;
    ge::DataType aType = ge::DT_FLOAT;
    ge::DataType bType = ge::DT_FLOAT;
    ge::DataType cType = ge::DT_FLOAT;
    ge::DataType biasType = ge::DT_FLOAT;
    ge::Format aFormat = ge::FORMAT_ND;
    ge::Format bFormat = ge::FORMAT_ND;
    ge::Format outFormat = ge::FORMAT_ND;
    uint64_t mValue = 0;
    uint64_t nValue = 0;
    uint64_t kValue = 0;
};

inline uint64_t CeilDiv(uint64_t x, uint64_t align)
{
    if (align == 0) {
        return x;
    }
    return (x + align - 1) / align;
}

inline uint64_t CeilAlign(uint64_t x, uint64_t align)
{
    return CeilDiv(x, align) * align;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, MatmulFp32CompileInfo* compileInfoPtr)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context->GetNodeName(), "context is null"), return ge::GRAPH_FAILED);
    fe::PlatFormInfos *platFormInfos = context->GetPlatformInfo();
    OP_CHECK_IF(platFormInfos == nullptr, OP_LOGE(context->GetNodeName(), "platform info is null"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platFormInfos);
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    if (compileInfoPtr->aicNum == 0) {
        OP_LOGE(context->GetNodeName(), "aicNum is zero.");
        return ge::GRAPH_FAILED;
    }
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->btSize = 1024UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);
    OP_LOGI(context->GetNodeName(),
        "parse compile info soc:%d, l1Size:%lu, l2Size:%lu, coreNum:%lu",
        static_cast<int32_t>(compileInfoPtr->socVersion), compileInfoPtr->l1Size, compileInfoPtr->l2Size, compileInfoPtr->aicNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, MatmulFp32Args* argsPtr)
{
    argsPtr->opName = context->GetNodeName();
    OP_CHECK_IF(argsPtr->opName == nullptr, OP_LOGE("matmul_fp32", "get op name invalid"), return ge::GRAPH_FAILED);
    // 输入张亮shape与描述信息
    auto shapeX1 = context->GetInputShape(0);
    auto shapeX2 = context->GetInputShape(1);
    auto shapeBias = context->GetInputShape(2);
    auto descX1 = context->GetInputDesc(0);
    auto descX2 = context->GetInputDesc(1);
    auto descOut = context->GetOutputDesc(0);
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(
        shapeX1 == nullptr || shapeX2 == nullptr || shapeBias == nullptr || descX1 == nullptr || descX2 == nullptr ||
            descOut == nullptr || attrs == nullptr,
        OP_LOGE(argsPtr->opName, "the input is invalid"), return ge::GRAPH_FAILED);
    // 输入属性信息
    argsPtr->isATrans = *attrs->GetAttrPointer<bool>(0);
    argsPtr->isBTrans = *attrs->GetAttrPointer<bool>(1);

    // 输入、输出格式信息
    argsPtr->aFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(descX1->GetStorageFormat()));
    argsPtr->bFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(descX2->GetStorageFormat()));
    argsPtr->outFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(descOut->GetStorageFormat()));
    OP_CHECK_IF(
        argsPtr->aFormat != ge::FORMAT_ND || argsPtr->bFormat != ge::FORMAT_ND || argsPtr->outFormat != ge::FORMAT_ND,
        OP_LOGE(argsPtr->opName, "input and output only support nd format"), return ge::GRAPH_FAILED);
    // 输入、输出数据类型
    argsPtr->aType = descX1->GetDataType();
    argsPtr->bType = descX2->GetDataType();
    argsPtr->cType = descOut->GetDataType();
    OP_CHECK_IF(
        argsPtr->aType != ge::DT_FLOAT || argsPtr->bType != ge::DT_FLOAT || argsPtr->cType != ge::DT_FLOAT,
        OP_LOGE(argsPtr->opName, "input and output only support float32"), return ge::GRAPH_FAILED);

    // 获取input的m/n/k
    const size_t x1DimNum = shapeX1->GetStorageShape().GetDimNum();
    const size_t x2DimNum = shapeX2->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        x1DimNum != 2 || x2DimNum != 2, OP_LOGE(argsPtr->opName, "the input shape dimensions are not equal to 2"),
        return ge::GRAPH_FAILED);
    argsPtr->hasBias = shapeBias == nullptr ? false : true;
    argsPtr->mValue = argsPtr->isATrans ? shapeX1->GetStorageShape().GetDim(1) : shapeX1->GetStorageShape().GetDim(0);
    argsPtr->kValue = argsPtr->isATrans ? shapeX1->GetStorageShape().GetDim(0) : shapeX1->GetStorageShape().GetDim(1);
    argsPtr->nValue = argsPtr->isBTrans ? shapeX2->GetStorageShape().GetDim(0) : shapeX2->GetStorageShape().GetDim(1);
    OP_LOGD(context->GetNodeName(),
        "parse op shape info m:%d, n:%d, k:%d",
        static_cast<int>(argsPtr->mValue), static_cast<int>(argsPtr->nValue), static_cast<int>(argsPtr->kValue));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InitTilingData(
    AscendC::tiling::TCubeTiling& tCubeTiling, MatmulFp32CompileInfo* compileInfoPtr, MatmulFp32Args* argsPtr)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    MultiCoreMatmulTiling tilingApi(*ascendcPlatform);
    tilingApi.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, argsPtr->isATrans);
    tilingApi.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, argsPtr->isBTrans);
    tilingApi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tilingApi.SetDim(compileInfoPtr->aicNum);
    tilingApi.SetShape(argsPtr->mValue, argsPtr->nValue, argsPtr->kValue);
    tilingApi.SetOrgShape(argsPtr->mValue, argsPtr->nValue, argsPtr->kValue);
    if (argsPtr->hasBias) {
        tilingApi.SetBias(true);
        tilingApi.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    }
    tilingApi.SetBufferSpace(compileInfoPtr->l1Size, compileInfoPtr->l0CSize, compileInfoPtr->ubSize);
    if (tilingApi.GetTiling(tCubeTiling) == -1) {
        OP_LOGE(argsPtr->opName, "failed to init tilingdata.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

inline void ResetBase(MatmulFp32Args* argsPtr, MatmulFp32RunInfo* runInfoPtr)
{
    runInfoPtr->baseM = BASIC_BLOCK_SIZE_128;
    runInfoPtr->baseN = BASIC_BLOCK_SIZE_256; // 256 is better base
    runInfoPtr->baseK = BASIC_BLOCK_K_128_BYTE / DATA_SIZE_FP32;
    runInfoPtr->stepM = BASE_STEP;
    runInfoPtr->stepN = BASE_STEP;
    runInfoPtr->iterateOrder = ITER_COL_FIRST;
    runInfoPtr->dbL0c = DB_OFF_SIZE;
    runInfoPtr->singleCoreK = argsPtr->kValue;
}


static void CalL1Tiling(MatmulFp32CompileInfo* compileInfoPtr, MatmulFp32Args* argsPtr, MatmulFp32RunInfo* runInfoPtr)
{
    runInfoPtr->baseM = CeilAlign(std::min(argsPtr->mValue, runInfoPtr->baseM), BASIC_ALIGN_16);
    runInfoPtr->baseN = CeilAlign(std::min(argsPtr->nValue, runInfoPtr->baseN), BASIC_ALIGN_16);
    uint64_t reserveBTSize = argsPtr->hasBias ? BIAS_TABLE_NUM * DATA_SIZE_FP32 : 0;
    runInfoPtr->depthA1 = compileInfoPtr->l1Size / NUM_HALF / runInfoPtr->baseM / runInfoPtr->baseK / DATA_SIZE_FP32; // 2: half of l1
    runInfoPtr->depthB1 = compileInfoPtr->l1Size / NUM_HALF / runInfoPtr->baseN / runInfoPtr->baseK / DATA_SIZE_FP32; // 2: half of l1

    uint64_t depthASize = runInfoPtr->depthA1 * runInfoPtr->baseM * runInfoPtr->baseK * DATA_SIZE_FP32;
    uint64_t depthBSize = runInfoPtr->depthB1 * runInfoPtr->baseN * runInfoPtr->baseK * DATA_SIZE_FP32;
    if (depthASize + depthBSize > compileInfoPtr->l1Size - reserveBTSize) {
        if (runInfoPtr->baseM <= runInfoPtr->baseN) {
            runInfoPtr->depthA1 = runInfoPtr->depthA1 / NUM_HALF; // 2: adjust deptch for l1 buffer
        } else {
            runInfoPtr->depthB1 = runInfoPtr->depthB1 / NUM_HALF; // 2: adjust deptch for l1 buffer
        }
    }
    runInfoPtr->stepKa = runInfoPtr->depthA1 / DB_SIZE;
    runInfoPtr->stepKb = runInfoPtr->depthB1 / DB_SIZE;

    if (runInfoPtr->stepKa >= runInfoPtr->stepKb) {
        runInfoPtr->stepKa = runInfoPtr->stepKa / runInfoPtr->stepKb * runInfoPtr->stepKb;
    } else {
        runInfoPtr->stepKb = runInfoPtr->stepKb / runInfoPtr->stepKa * runInfoPtr->stepKa;
    }
    runInfoPtr->depthA1 = runInfoPtr->stepKa * DB_SIZE; // depth % (stepKa * stepM) == 0
    runInfoPtr->depthB1 = runInfoPtr->stepKb * DB_SIZE; // depth % (stepKb * stepN) == 0
    runInfoPtr->singleCoreM = runInfoPtr->baseM;
    runInfoPtr->singleCoreN = runInfoPtr->baseN;
    return;
} 

static ge::graphStatus DoBL1FullLoadTiling(MatmulFp32CompileInfo* compileInfoPtr, MatmulFp32Args* argsPtr, MatmulFp32RunInfo* runInfoPtr)
{
    // mValue should be 16 times more than max of k/nValue, and kValue should be no more than 256
    bool validMK = argsPtr->mValue > 16 * std::max(argsPtr->kValue, argsPtr->nValue) && argsPtr->kValue <= 256;
    uint64_t biasSize = argsPtr->hasBias ? runInfoPtr->baseN * DATA_SIZE_FP32 : 0; // 默认最高精度保证BF16
    bool bl1SizeValid =
        (compileInfoPtr->l1Size / NUM_HALF - biasSize) > argsPtr->kValue * argsPtr->nValue * DATA_SIZE_FP32;
    if (!validMK || !bl1SizeValid) {
        return ge::GRAPH_FAILED;
    }
    FULL_LOAD_TYPE = MAT_MUL_FP32_BL1_FULLLOAD;
    // fine tune tiling
    runInfoPtr->stepM = 1;
    runInfoPtr->baseN = std::min(argsPtr->nValue, runInfoPtr->baseN);
    // BaseN need to do Block alignment
    runInfoPtr->baseN = argsPtr->isBTrans ? CeilAlign(runInfoPtr->baseN, H_ALIGNED) : CeilAlign(runInfoPtr->baseN, CO_SIZE_FP32);
    runInfoPtr->stepN = CeilDiv(argsPtr->nValue, runInfoPtr->baseN);
    runInfoPtr->stepKb = CeilDiv(argsPtr->kValue, runInfoPtr->baseK);
    runInfoPtr->stepKa = runInfoPtr->stepKb;
    runInfoPtr->depthA1 = DB_SIZE * runInfoPtr->stepKa;
    runInfoPtr->depthB1 = runInfoPtr->stepN * runInfoPtr->stepKb;
    uint64_t loadSize = static_cast<uint64_t>(runInfoPtr->baseK) *
                        (runInfoPtr->depthA1 * runInfoPtr->baseM + runInfoPtr->depthB1 * runInfoPtr->baseN) * DATA_SIZE_FP32;
    loadSize += argsPtr->hasBias ? runInfoPtr->baseN * DATA_SIZE_FP32 : 0;
    // Check L1 load size
    uint64_t totalBSizeL1 = loadSize - (runInfoPtr->baseM - MIN_BASE_M_VALUE) * runInfoPtr->baseK * runInfoPtr->depthA1 * DATA_SIZE_FP32;
    if (totalBSizeL1 > compileInfoPtr->l1Size) {
        OP_LOGI(argsPtr->opName, "min A size in L1 and total B size in L1 is larger than total L1 size, cannot be fullLoad.");
        FULL_LOAD_TYPE = MAT_MUL_FP32_NO_FULLLOAD;
        return ge::GRAPH_FAILED;
    }
    while (loadSize > compileInfoPtr->l1Size) {
        loadSize -= runInfoPtr->depthA1 * runInfoPtr->baseM * runInfoPtr->baseK * DATA_SIZE_FP32;
        runInfoPtr->baseM = runInfoPtr->baseM / NUM_HALF;
        loadSize += runInfoPtr->depthA1 * runInfoPtr->baseM * runInfoPtr->baseK * DATA_SIZE_FP32;
    }
    runInfoPtr->singleCoreM = DB_SIZE * runInfoPtr->baseM;
    runInfoPtr->singleCoreN = argsPtr->nValue;
    runInfoPtr->dbL0c = runInfoPtr->baseM * runInfoPtr->baseN * DATA_SIZE_FP32 * DB_SIZE <= compileInfoPtr->l0CSize ? DB_SIZE : 1;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DoOpTiling(
    AscendC::tiling::TCubeTiling& tCubeTiling, MatmulFp32CompileInfo* compileInfoPtr, MatmulFp32Args* argsPtr,
    MatmulFp32RunInfo* runInfoPtr)
{
    // 高阶api初始化tilingdata
    auto ret = InitTilingData(tCubeTiling, compileInfoPtr, argsPtr);

    // 重置基本块
    ResetBase(argsPtr, runInfoPtr);

    // 计算L1的tiling参数
    CalL1Tiling(compileInfoPtr, argsPtr, runInfoPtr);

    if (is_support_bl1) {
        // BL1全载tiling调整
        ret = DoBL1FullLoadTiling(compileInfoPtr, argsPtr, runInfoPtr);
    }

    return ret;
}

static ge::graphStatus PostTiling(
    gert::TilingContext* context, MatmulFp32CompileInfo* compileInfoPtr, MatmulFp32TilingData* tilingDataPtr,
    MatmulFp32Args* argsPtr, MatmulFp32RunInfo* runInfoPtr)
{
    // 设置tilingdata
    tilingDataPtr->matmulFp32RunInfo.transA = static_cast<uint32_t>(argsPtr->isATrans);
    tilingDataPtr->matmulFp32RunInfo.transB = static_cast<uint32_t>(argsPtr->isBTrans);
    tilingDataPtr->tCubeTiling.usedCoreNum = std::min(CeilDiv(argsPtr->mValue, runInfoPtr->singleCoreM) *
        CeilDiv(argsPtr->nValue, runInfoPtr->singleCoreN), compileInfoPtr->aicNum);
    tilingDataPtr->tCubeTiling.singleCoreM = static_cast<uint32_t>(runInfoPtr->singleCoreM);
    tilingDataPtr->tCubeTiling.singleCoreN = static_cast<uint32_t>(runInfoPtr->singleCoreN);
    tilingDataPtr->tCubeTiling.singleCoreK = static_cast<uint32_t>(runInfoPtr->singleCoreK);
    tilingDataPtr->tCubeTiling.baseM = static_cast<uint32_t>(runInfoPtr->baseM);
    tilingDataPtr->tCubeTiling.baseN = static_cast<uint32_t>(runInfoPtr->baseN);
    tilingDataPtr->tCubeTiling.baseK = static_cast<uint32_t>(runInfoPtr->baseK);
    tilingDataPtr->tCubeTiling.depthA1 = static_cast<uint32_t>(runInfoPtr->depthA1);
    tilingDataPtr->tCubeTiling.depthB1 = static_cast<uint32_t>(runInfoPtr->depthB1);
    tilingDataPtr->tCubeTiling.stepM = static_cast<uint32_t>(runInfoPtr->stepM);
    tilingDataPtr->tCubeTiling.stepN = static_cast<uint32_t>(runInfoPtr->stepN);
    tilingDataPtr->tCubeTiling.stepKa = static_cast<uint32_t>(runInfoPtr->stepKa);
    tilingDataPtr->tCubeTiling.stepKb = static_cast<uint32_t>(runInfoPtr->stepKb);
    tilingDataPtr->tCubeTiling.iterateOrder = static_cast<uint32_t>(runInfoPtr->iterateOrder);
    tilingDataPtr->tCubeTiling.dbL0C = static_cast<uint32_t>(runInfoPtr->dbL0c);
    size_t tilingDataSize = sizeof(MatmulFp32TilingData);
    auto ret = memcpy_s(
        context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity(),
        reinterpret_cast<void*>(tilingDataPtr), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context->GetRawTilingData()->SetDataSize(tilingDataSize);
    context->SetBlockDim(tilingDataPtr->tCubeTiling.usedCoreNum);
    context->SetScheduleMode(1);
    // 生成tilingkey
    uint64_t tilingKey = GET_TPL_TILING_KEY(
        FULL_LOAD_TYPE, MAT_MUL_FP32_BASE_SPLIT_K, MAT_MUL_FP32_BASE_FIXOPTI, MAT_MUL_FP32_MIXND2NZ_TRUE);
    OP_LOGI(context->GetNodeName(), "Tiling Key is 0x%x", tilingKey);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static void GetWorkspaceSize(gert::TilingContext* context, MatmulFp32Args* argsPtr)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = WS_SYS_SIZE + sysWorkspaceSize;
}

// 1.Tiling需要获取的运行环境信息，包括可用核数、UB大小等，并将获取到的信息传递给'CompileInfo'
// 由于自动生成的aclnn接口实现不调用该函数，直接返回'ge::GRAPH_SUCCESS'即可
static ge::graphStatus TilingPrepareForMatmulFp32([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// 2. Tiling计算主入口
static ge::graphStatus MatmulFp32TilingFunc(gert::TilingContext* context)
{
    // 2.1 平台信息
    MatmulFp32CompileInfo* compileInfoPtr = new MatmulFp32CompileInfo;
    auto ret =GetPlatformInfo(context, compileInfoPtr);
    if (ret !=ge::GRAPH_SUCCESS){
        return ret;
    }

    // 2.2 输入信息
    MatmulFp32Args* argsPtr = new MatmulFp32Args;
    ret = GetShapeAttrsInfo(context, argsPtr);
    if (ret !=ge::GRAPH_SUCCESS){
        return ret;
    }

    // 2.3 计算Tiling参数
    MatmulFp32TilingData* tilingDataPtr = context->GetTilingData<MatmulFp32TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingDataPtr);
    MatmulFp32RunInfo* runInfoPtr = new MatmulFp32RunInfo;
    ret = DoOpTiling(tilingDataPtr->tCubeTiling, compileInfoPtr, argsPtr, runInfoPtr);
    if (ret !=ge::GRAPH_SUCCESS){
        return ret;
    }

    // 2.4 设置TillingData与生成tilingkey
    ret = PostTiling(context, compileInfoPtr, tilingDataPtr, argsPtr, runInfoPtr);
    if (ret !=ge::GRAPH_SUCCESS){
        return ret;
    }

    // 2.5设置workspaceSize
    GetWorkspaceSize(context, argsPtr);
    return ge::GRAPH_SUCCESS;
}


// 3. Tiling注册入口
IMPL_OP_OPTILING(MatmulFp32)
    .Tiling(MatmulFp32TilingFunc)
    .TilingParse<MatmulFp32CompileInfo>(TilingPrepareForMatmulFp32);
}
