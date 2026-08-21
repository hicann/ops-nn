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
 * \file cross_entropy_sum_exp_and_index_logit_tiling_arch35.cpp
 * \brief A5 (ascend950) tiling logic — 核间 floor+remainder 均衡 + 每核独立内循环，单 TilingKey 100
 */

#include <algorithm>
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "op_common/atvoss/reduce/reduce_tiling.h"
#include "cross_entropy_sum_exp_and_index_logit_tiling_arch35.h"

using namespace std;

namespace {
constexpr int64_t NUM_ZERO = 0;
constexpr int64_t NUM_ONE = 1;

// TilingKey：单 key 100（CE_REGBASE），dtype 差异由 kernel 侧 DTYPE_VOCAB_PARALLEL_LOGITS 编译宏裁分支
constexpr uint32_t CE_REGBASE = 100;

// Inputs Index
constexpr int64_t LOGITS_INDEX = 0;
constexpr int64_t TARGET_INDEX = 1;
constexpr int64_t GLOBAL_MAX_INDEX = 2;

// Attrs Index
constexpr int64_t ATTR_VOCAB_START_INDEX = 0;
constexpr int64_t ATTR_VOCAB_END_INDEX = 1;

// Outputs Index
constexpr int64_t OUT_PREDICTED_INDEX = 0;
constexpr int64_t OUT_SUMEXP_INDEX = 1;
constexpr int64_t OUT_EXP_INDEX = 2;
constexpr int64_t OUT_OFFSET_INDEX = 3;
constexpr int64_t OUT_MASK_INDEX = 4;

// Dim Num
constexpr int64_t DIM_NUM_ONE = 1;
constexpr int64_t DIM_NUM_TWO = 2;
constexpr int64_t DIM_NUM_THREE = 3;

// 入参限制常量
constexpr int64_t N_SIZE_UP_LIMIT = 32 * 1024;   // N 上限 32K
constexpr int64_t N_SIZE_DOWN_LIMIT = 1;         // N 下限 1
constexpr int64_t V_LOCAL_UP_LIMIT = 200 * 1024; // V_local 上限 200K
constexpr int64_t V_LOCAL_DOWN_LIMIT = 16;       // V_local 下限 16

// 切分常量
constexpr int64_t V_TILE = 2048; // 核内 V_local tile 长度，2048 = 32×64（三级块算 2048→256→32→1）
constexpr int64_t FP32_SIZE = 4; // sizeof(float)
constexpr int64_t FP32_PER_BLOCK = 8;         // FP32 每 32B datablock 元素数
constexpr int64_t BF16_SIZE = 2;              // sizeof(bfloat16_t)
constexpr int64_t BUFFER_NUM = 2;             // 双缓冲
constexpr int64_t ALIGN_BF16 = 16;            // BF16 32B = 16 elem
constexpr int64_t ALIGN_FP32 = 8;             // FP32 32B = 8 elem
constexpr int64_t ROW_BLOCK_MIN = 4;          // rowBlock 下界
constexpr int64_t ROW_BLOCK_MAX = 40;         // rowBlock 上界（与 kernel MAX_ROW_BLOCK 一致）
constexpr int64_t SMALL_BUF_UB_A5 = 3 * 1024; // A5 纯标量/中间缓存预留（含 10 个 TBuf 小 buffer）

// workspace 预留（本算子无 workspace，占位）
constexpr int64_t DEFAULT_WORKSPACE_SIZE = 16 * 1024 * 1024;

// 官方接口查询 ReduceSum(AR,float) 在 shape=[rows, vTile] 下所需 sharedTmpBuffer 最小字节。
//   min==max（文档保证），完全以 GetReduceSumMaxMinTmpSize 为准，不依赖任何自推公式。
int64_t QueryReduceSumTmpBytes(const int64_t rows, const int64_t vTile)
{
    ge::Shape srcShape({rows, vTile});
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetReduceSumMaxMinTmpSize(srcShape, ge::DataType::DT_FLOAT, AscendC::ReducePattern::AR,
                                       true,  /* isSrcInnerPad= */
                                       false, /* isReuseSource= */
                                       maxValue, minValue);
    return static_cast<int64_t>(minValue);
}
} // namespace

namespace optiling {
class CrossEntropySumExpAndIndexLogitTiling {
public:
    explicit CrossEntropySumExpAndIndexLogitTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();

private:
    void TilingDataSet();
    void TilingDataPrint() const;
    bool CheckParam();
    bool CoreSplit();
    bool UbSplit();
    CrossEntropySumExpAndIndexLogitRegBaseTilingData* tilingData_{nullptr};
    gert::TilingContext* tilingContext = nullptr;
    uint32_t tilingKey = 0;
    int32_t coreNum = 1;
    uint64_t ubSize = 0;
    uint64_t workspaceSize = 0;

    // 解析参数
    ge::DataType logitsType_ = ge::DT_FLOAT;
    ge::DataType globalMaxType_ = ge::DT_FLOAT;
    int64_t N_ = 0;
    int64_t vLocal_ = 0;
    int64_t vocabStart_ = 0;
    int64_t vocabEnd_ = 0;

    // 切分结果
    int64_t usedCores_ = 0;
    int64_t headCoreNum_ = 0;
    int64_t tokensPerCore_ = 0;
    int64_t tokensPerCoreTail_ = 0;
    int64_t headBlockNum_ = 0;
    int64_t tailBlockNum_ = 0;
    int64_t rowBlockMax_ = 0;
    int64_t reduceTmpBytes_ = 0;
    int64_t vTile_ = 0;
    int64_t vLoopNum_ = 0;
    int64_t lastVTile_ = 0;
};

bool CrossEntropySumExpAndIndexLogitTiling::CheckParam()
{
    if (tilingContext->GetInputShape(LOGITS_INDEX) == nullptr || tilingContext->GetInputDesc(LOGITS_INDEX) == nullptr ||
        tilingContext->GetInputShape(TARGET_INDEX) == nullptr || tilingContext->GetInputDesc(TARGET_INDEX) == nullptr ||
        tilingContext->GetInputShape(GLOBAL_MAX_INDEX) == nullptr ||
        tilingContext->GetInputDesc(GLOBAL_MAX_INDEX) == nullptr ||
        tilingContext->GetOutputDesc(OUT_PREDICTED_INDEX) == nullptr ||
        tilingContext->GetOutputDesc(OUT_SUMEXP_INDEX) == nullptr ||
        tilingContext->GetOutputDesc(OUT_EXP_INDEX) == nullptr ||
        tilingContext->GetOutputDesc(OUT_OFFSET_INDEX) == nullptr ||
        tilingContext->GetOutputDesc(OUT_MASK_INDEX) == nullptr || tilingContext->GetRawTilingData() == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(tilingContext->GetNodeName(), "tilingContext inputs/outputs",
                                                 "shape or desc or rawTilingData is nullptr");
        return false;
    }
    auto attrs = tilingContext->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE_WITH_INVALID_ATTR(tilingContext->GetNodeName(), "attrs", "null", "non_empty_value");
        return false;
    }
    const int64_t* vocabStartPtr = attrs->GetAttrPointer<int64_t>(ATTR_VOCAB_START_INDEX);
    const int64_t* vocabEndPtr = attrs->GetAttrPointer<int64_t>(ATTR_VOCAB_END_INDEX);
    if (vocabStartPtr == nullptr || vocabEndPtr == nullptr) {
        OP_LOGE_WITH_INVALID_ATTR(tilingContext->GetNodeName(), "vocab_start_index/vocab_end_index", "nullptr",
                                  "not nullptr");
        return false;
    }
    vocabStart_ = *vocabStartPtr;
    vocabEnd_ = *vocabEndPtr;
    if (vocabEnd_ <= vocabStart_) {
        OP_LOGE_WITH_INVALID_ATTR(tilingContext->GetNodeName(), "vocab_end_index", std::to_string(vocabEnd_).c_str(),
                                  "larger than vocab_start_index");
        return false;
    }

    // vocab_parallel_logits：dtype + 维度（2/3D）+ N/V_local 范围 + 对齐 + vocab 区间一致性
    logitsType_ = tilingContext->GetInputDesc(LOGITS_INDEX)->GetDataType();
    if (logitsType_ != ge::DT_FLOAT && logitsType_ != ge::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "vocab_parallel_logits",
                                  Ops::Base::ToString(logitsType_).c_str(), "FLOAT or BF16");
        return false;
    }
    auto& logitsShape = tilingContext->GetInputShape(LOGITS_INDEX)->GetStorageShape();
    int64_t logitsDim = logitsShape.GetDimNum();
    if (logitsDim != DIM_NUM_TWO && logitsDim != DIM_NUM_THREE) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(tilingContext->GetNodeName(), "vocab_parallel_logits",
                                     std::to_string(logitsDim).c_str(), "2D or 3D");
        return false;
    }
    // 空 tensor 校验：本算子不支持任一输入任一维为 0 的空 tensor（现有 N/V_local 范围检查
    //   虽会间接拦截，但此处显式校验并给出明确错误信息）
    auto CheckInputNotEmpty = [&](const auto& shape, const char* name) -> bool {
        for (size_t i = 0; i < shape.GetDimNum(); ++i) {
            if (shape.GetDim(i) <= NUM_ZERO) {
                std::string shapeMsg = Ops::Base::ToString(shape);
                std::string reasonMsg = "is an empty tensor, dim " + std::to_string(i) + " must be greater than 0";
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(tilingContext->GetNodeName(), name, shapeMsg.c_str(),
                                                      reasonMsg.c_str());
                return false;
            }
        }
        return true;
    };
    if (!CheckInputNotEmpty(logitsShape, "vocab_parallel_logits")) {
        return false;
    }
    int64_t vLocal = logitsShape.GetDim(logitsDim - NUM_ONE);
    int64_t n = NUM_ONE;
    for (int64_t i = 0; i < logitsDim - NUM_ONE; ++i) {
        n *= logitsShape.GetDim(i);
    }
    vLocal_ = vLocal;
    N_ = n;
    if (n < N_SIZE_DOWN_LIMIT || n > N_SIZE_UP_LIMIT) {
        OP_LOGE_FOR_INVALID_SHAPESIZE(tilingContext->GetNodeName(), "vocab_parallel_logits", std::to_string(n).c_str(),
                                      "in [1, 32768]");
        return false;
    }
    if (vLocal < V_LOCAL_DOWN_LIMIT || vLocal > V_LOCAL_UP_LIMIT) {
        OP_LOGE_FOR_INVALID_SHAPESIZE(tilingContext->GetNodeName(), "vocab_parallel_logits",
                                      std::to_string(vLocal).c_str(), "in [16, 204800]");
        return false;
    }
    int64_t alignReq = (logitsType_ == ge::DT_BF16) ? ALIGN_BF16 : ALIGN_FP32;
    if (vLocal % alignReq != NUM_ZERO) {
        std::string reasonMsg = "must be multiple of " + std::to_string(alignReq) + " for the dtype";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(tilingContext->GetNodeName(), "vocab_parallel_logits",
                                              std::to_string(vLocal).c_str(), reasonMsg.c_str());
        return false;
    }
    if (vocabEnd_ - vocabStart_ != vLocal) {
        OP_LOGE_WITH_INVALID_ATTR(tilingContext->GetNodeName(), "vocab_end_index", std::to_string(vocabEnd_).c_str(),
                                  "vocab_start_index + V_local");
        return false;
    }

    // target：INT32，元素数等于 N
    auto targetDtype = tilingContext->GetInputDesc(TARGET_INDEX)->GetDataType();
    if (targetDtype != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "target", Ops::Base::ToString(targetDtype).c_str(),
                                  "INT32");
        return false;
    }
    auto& targetShape = tilingContext->GetInputShape(TARGET_INDEX)->GetStorageShape();
    if (!CheckInputNotEmpty(targetShape, "target")) {
        return false;
    }
    int64_t targetNum = NUM_ONE;
    for (size_t i = 0; i < targetShape.GetDimNum(); ++i) {
        targetNum *= targetShape.GetDim(i);
    }
    if (targetNum != N_) {
        std::string shapeMsg = std::to_string(targetNum) + " and " + std::to_string(N_);
        std::string reasonMsg = "target element num must equal N (prod of logits leading dims)";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(tilingContext->GetNodeName(), "target and vocab_parallel_logits",
                                               shapeMsg.c_str(), reasonMsg.c_str());
        return false;
    }

    // global_logits_max：dtype 与 logits 一致，元素数等于 N
    globalMaxType_ = tilingContext->GetInputDesc(GLOBAL_MAX_INDEX)->GetDataType();
    if (globalMaxType_ != ge::DT_FLOAT && globalMaxType_ != ge::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "global_logits_max",
                                  Ops::Base::ToString(globalMaxType_).c_str(), "FLOAT or BF16");
        return false;
    }
    if (globalMaxType_ != logitsType_) {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "global_logits_max",
                                  Ops::Base::ToString(globalMaxType_).c_str(), "same as vocab_parallel_logits dtype");
        return false;
    }
    auto& maxShape = tilingContext->GetInputShape(GLOBAL_MAX_INDEX)->GetStorageShape();
    if (!CheckInputNotEmpty(maxShape, "global_logits_max")) {
        return false;
    }
    int64_t maxNum = NUM_ONE;
    for (size_t i = 0; i < maxShape.GetDimNum(); ++i) {
        maxNum *= maxShape.GetDim(i);
    }
    if (maxNum != N_) {
        std::string shapeMsg = std::to_string(maxNum) + " and " + std::to_string(N_);
        std::string reasonMsg = "global_logits_max element num must equal N";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(tilingContext->GetNodeName(),
                                               "global_logits_max and vocab_parallel_logits", shapeMsg.c_str(),
                                               reasonMsg.c_str());
        return false;
    }

    // 输出 dtype 校验（输出 dtype 固定，不随输入推导：predicted/sum_exp/exp 为 FLOAT，
    // offset/mask 为 INT32；tiling 侧显式校验与 OpDef 声明一致）
    auto predictedDtype = tilingContext->GetOutputDesc(OUT_PREDICTED_INDEX)->GetDataType();
    if (predictedDtype != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "predicted_logits",
                                  Ops::Base::ToString(predictedDtype).c_str(), "FLOAT");
        return false;
    }
    auto sumExpDtype = tilingContext->GetOutputDesc(OUT_SUMEXP_INDEX)->GetDataType();
    if (sumExpDtype != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "sum_exp_logits",
                                  Ops::Base::ToString(sumExpDtype).c_str(), "FLOAT");
        return false;
    }
    auto expDtype = tilingContext->GetOutputDesc(OUT_EXP_INDEX)->GetDataType();
    if (expDtype != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "exp_logits", Ops::Base::ToString(expDtype).c_str(),
                                  "FLOAT");
        return false;
    }
    auto offsetDtype = tilingContext->GetOutputDesc(OUT_OFFSET_INDEX)->GetDataType();
    if (offsetDtype != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "target_offset",
                                  Ops::Base::ToString(offsetDtype).c_str(), "INT32");
        return false;
    }
    auto maskDtype = tilingContext->GetOutputDesc(OUT_MASK_INDEX)->GetDataType();
    if (maskDtype != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "target_mask", Ops::Base::ToString(maskDtype).c_str(),
                                  "INT32");
        return false;
    }
    return true;
}

ge::graphStatus CrossEntropySumExpAndIndexLogitTiling::Init()
{
    if (tilingContext == nullptr) {
        OP_LOGE(tilingContext, "tilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = tilingContext->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum = static_cast<int32_t>(ascendcPlatform.GetCoreNumAiv());
        uint64_t ubSizePlatform = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize = ubSizePlatform;
    } else {
        auto compileInfo = static_cast<const CrossEntropySumExpAndIndexLogitCompileInfo*>(
            tilingContext->GetCompileInfo());
        OP_CHECK_NULL_WITH_CONTEXT(tilingContext, compileInfo);
        coreNum = static_cast<int32_t>(compileInfo->totalCoreNum);
        ubSize = compileInfo->ubSizePlatForm;
    }
    if (coreNum <= 0) {
        OP_LOGE(tilingContext, "coreNum must greater than 0.");
        return ge::GRAPH_FAILED;
    }
    if (ubSize == 0) {
        OP_LOGE(tilingContext, "ubSize is 0.");
        return ge::GRAPH_FAILED;
    }
    workspaceSize = DEFAULT_WORKSPACE_SIZE;
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tilingContext->GetWorkspaceSizes(1));
    if (!CheckParam()) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(tilingContext, "Tiling inited.");
    return ge::GRAPH_SUCCESS;
}

bool CrossEntropySumExpAndIndexLogitTiling::CoreSplit()
{
    // Step1: rowBlockMax（UB 反推 vTile 下最大行数）
    int64_t inTSize = (logitsType_ == ge::DT_BF16) ? BF16_SIZE : FP32_SIZE;
    // perRow：logitsInQue(BUFFER_NUM×vTile×inT) + expOutQue(BUFFER_NUM×vTile×FP32)
    int64_t perRow = BUFFER_NUM * V_TILE * inTSize + BUFFER_NUM * V_TILE * FP32_SIZE;
    int64_t avail = static_cast<int64_t>(ubSize) - SMALL_BUF_UB_A5;
    // ReduceSum(AR) 的 sharedTmpBuffer 随 rows 增大 → 存在 "rowBlock ↔ tmp" 循环依赖。
    //   tmp 完全以官方 GetReduceSumMaxMinTmpSize 为准（QueryReduceSumTmpBytes）。
    //   tmp 对 rows 单调不减 → 从 ROW_BLOCK_MAX 递减找第一个自洽 R：perRow×R + tmp(R) ≤ avail。
    // 反推固定 vTile=2048（详设 CE_REGBASE）：blkTmp = r*(2048/8)*4 = 1024r。
    int64_t vTileForTmp = V_TILE;
    rowBlockMax_ = 0;
    for (int64_t r = ROW_BLOCK_MAX; r >= ROW_BLOCK_MIN; --r) {
        // reduceTmp 需同时覆盖两条规约路径，取二者较大值：
        //   ① 块算路径（vTile%64==0 时走）：中间结果 [r, vTile/8]，需 r*(vTile/8)*4 字节；
        //   ② fallback 路径（vTile 非 64 倍数）：高阶 ReduceSum<AR> 官方 GetReduceSumMaxMinTmpSize。
        int64_t blkTmp = r * (vTileForTmp / FP32_PER_BLOCK) * FP32_SIZE;
        int64_t redTmp = QueryReduceSumTmpBytes(r, vTileForTmp);
        int64_t tmpBytes = std::max(blkTmp, redTmp);
        if (perRow * r + tmpBytes <= avail) {
            rowBlockMax_ = r;
            reduceTmpBytes_ = tmpBytes;
            break;
        }
    }
    if (rowBlockMax_ < ROW_BLOCK_MIN) {
        OP_LOGE(tilingContext, "computed rowBlockMax[%ld] < min[%ld], UB too small.", rowBlockMax_, ROW_BLOCK_MIN);
        return false;
    }

    // Step2: 核间均衡分配 token（floor+remainder）
    //   usedCores=min(N,aivNum),  base=tokens/核,  rem=N%usedCores 前 rem 核多 1 token
    usedCores_ = (N_ < coreNum) ? N_ : coreNum;
    int64_t base = N_ / usedCores_;
    int64_t rem = N_ % usedCores_;
    headCoreNum_ = (rem == 0) ? 0 : rem;
    tokensPerCore_ = (rem > 0) ? (base + NUM_ONE) : base;
    tokensPerCoreTail_ = base;

    // Step3: 每核内循环块数（kernel 自行均衡行数 = floor(tokens/blockNum) + 前 rem 块多 1 行）
    headBlockNum_ = (tokensPerCore_ > 0) ? (tokensPerCore_ + rowBlockMax_ - NUM_ONE) / rowBlockMax_ : 0;
    tailBlockNum_ = (tokensPerCoreTail_ > 0) ? (tokensPerCoreTail_ + rowBlockMax_ - NUM_ONE) / rowBlockMax_ : 0;
    return true;
}

bool CrossEntropySumExpAndIndexLogitTiling::UbSplit()
{
    // 核内切分：V_local 按固定 vTile=2048（16(BF16)/8(FP32) 倍数，两 dtype 32B 对齐通用）
    vTile_ = std::min(V_TILE, vLocal_);
    int64_t alignReq = (logitsType_ == ge::DT_BF16) ? ALIGN_BF16 : ALIGN_FP32;
    vTile_ = vTile_ / alignReq * alignReq;
    if (vTile_ <= 0) {
        OP_LOGE(tilingContext, "computed vTile <= 0.");
        return false;
    }
    vLoopNum_ = (vLocal_ + vTile_ - NUM_ONE) / vTile_;     // ceil(vLocal / vTile)
    lastVTile_ = vLocal_ - vTile_ * (vLoopNum_ - NUM_ONE); // 尾块
    return true;
}

ge::graphStatus CrossEntropySumExpAndIndexLogitTiling::RunKernelTiling()
{
    OP_LOGD(tilingContext, "Tiling start.");
    tilingKey = CE_REGBASE;
    if (!CoreSplit()) {
        return ge::GRAPH_FAILED;
    }
    if (!UbSplit()) {
        return ge::GRAPH_FAILED;
    }
    TilingDataSet();
    OP_LOGD(tilingContext, "Tiling end.");
    return ge::GRAPH_SUCCESS;
}

void CrossEntropySumExpAndIndexLogitTiling::TilingDataSet()
{
    tilingData_ = tilingContext->GetTilingData<CrossEntropySumExpAndIndexLogitRegBaseTilingData>();
    tilingData_->N = N_;
    tilingData_->vLocal = vLocal_;
    tilingData_->usedCores = usedCores_;
    tilingData_->headCoreNum = headCoreNum_;
    tilingData_->tokensPerCore = tokensPerCore_;
    tilingData_->tokensPerCoreTail = tokensPerCoreTail_;
    tilingData_->headBlockNum = headBlockNum_;
    tilingData_->tailBlockNum = tailBlockNum_;
    tilingData_->rowBlockMax = rowBlockMax_;
    tilingData_->vTile = vTile_;
    tilingData_->vLoopNum = vLoopNum_;
    tilingData_->lastVTile = lastVTile_;
    tilingData_->reduceTmpBytes = reduceTmpBytes_;
    tilingData_->vocabStart = vocabStart_;
    tilingData_->vocabEnd = vocabEnd_;

    TilingDataPrint();

    tilingContext->SetBlockDim(static_cast<uint32_t>(usedCores_));
    tilingContext->SetTilingKey(static_cast<uint64_t>(tilingKey));
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize;
}

void CrossEntropySumExpAndIndexLogitTiling::TilingDataPrint() const
{
    OP_LOGI(tilingContext, "N: %u.", tilingData_->N);
    OP_LOGI(tilingContext, "vLocal: %u.", tilingData_->vLocal);
    OP_LOGI(tilingContext, "usedCores: %u.", tilingData_->usedCores);
    OP_LOGI(tilingContext, "headCoreNum: %u.", tilingData_->headCoreNum);
    OP_LOGI(tilingContext, "tokensPerCore: %u.", tilingData_->tokensPerCore);
    OP_LOGI(tilingContext, "tokensPerCoreTail: %u.", tilingData_->tokensPerCoreTail);
    OP_LOGI(tilingContext, "headBlockNum: %u.", tilingData_->headBlockNum);
    OP_LOGI(tilingContext, "tailBlockNum: %u.", tilingData_->tailBlockNum);
    OP_LOGI(tilingContext, "rowBlockMax: %u.", tilingData_->rowBlockMax);
    OP_LOGI(tilingContext, "reduceTmpBytes: %u.", tilingData_->reduceTmpBytes);
    OP_LOGI(tilingContext, "vTile: %u.", tilingData_->vTile);
    OP_LOGI(tilingContext, "vLoopNum: %u.", tilingData_->vLoopNum);
    OP_LOGI(tilingContext, "lastVTile: %u.", tilingData_->lastVTile);
    OP_LOGI(tilingContext, "vocabStart: %ld.", tilingData_->vocabStart);
    OP_LOGI(tilingContext, "vocabEnd: %ld.", tilingData_->vocabEnd);
    OP_LOGI(tilingContext, "tilingKey: %u.", tilingKey);
    OP_LOGI(tilingContext, "blockDim: %u.", tilingData_->usedCores);
}

ge::graphStatus TilingCrossEntropySumExpAndIndexLogit(gert::TilingContext* context)
{
    CrossEntropySumExpAndIndexLogitTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunKernelTiling();
}

ge::graphStatus TilingPrepareForCrossEntropySumExpAndIndexLogit(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepareForCrossEntropySumExpAndIndexLogit start.");
    auto compileInfo = context->GetCompiledInfo<CrossEntropySumExpAndIndexLogitCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF((compileInfo->ubSizePlatForm <= 0), OP_LOGE(context, "Failed to get ub size."),
                return ge::GRAPH_FAILED);
    OP_LOGD(context, "ub_size_platform is %lu.", compileInfo->ubSizePlatForm);
    OP_LOGD(context, "TilingPrepareForCrossEntropySumExpAndIndexLogit end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CrossEntropySumExpAndIndexLogit)
    .Tiling(TilingCrossEntropySumExpAndIndexLogit)
    .TilingParse<CrossEntropySumExpAndIndexLogitCompileInfo>(TilingPrepareForCrossEntropySumExpAndIndexLogit);
} // namespace optiling
