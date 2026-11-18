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
 * \file qbmm_streamk_tiling_960.cpp
 * \brief DAV_9201/DAV_9202 StreamK tiling,
 *        与3510版qbmm_streamk_tiling的差异：面向CMCT BasicAPI StreamK kernel，
 *        tiling data复用QuantBatchMatmulV3StreamKBasicAPITilingData（与3510 Blaze
 * StreamK共用布局），按NpuArch注册调度。
 */
#include "qbmm_streamk_tiling_960.h"

#include "op_host/tiling_templates_registry.h"
#include "error_util.h"
#include "log/log.h"
#include "quant_batch_matmul_v3_checker_arch_9201.h"
#include "quant_batch_matmul_v3_tiling_strategy.h"
#include "base_block_calculator.h"
#include "../quant_batch_matmul_v3_compile_info.h"
#include "l1_tiling_data_calculator.h"
#include "tiling_cost_evaluator.h"

namespace {
constexpr uint32_t DB_SIZE = 2;
constexpr uint32_t DATA_SIZE_L0C = 4;
constexpr uint64_t DATA_SIZE_FP32 = 4UL;
constexpr uint32_t BASIC_BLOCK_SIZE_256 = 256;
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
constexpr uint64_t EXTRA_WORKSPACE = 16UL;
constexpr float SK_VALID_THRESHOLD = 0.3; // 所估计出的SK耗时比ASWT小0.3us及以上才走入SK模板

const static std::vector<int32_t> supportedNpuArch = {static_cast<int32_t>(Ops::Base::DAV_9201),
                                                      static_cast<int32_t>(Ops::Base::DAV_9202)};
} // namespace

using Ops::NN::MathUtil;
using namespace QuantBatchMatmulV3Arch35TilingKey;
using optiling::strategy::MX_STREAMK;

namespace optiling {
REGISTER_TILING_TEMPLATE_WITH_ARCH(QuantBatchMatmulV3, QBMMV3StreamKTiling960, supportedNpuArch, MX_STREAMK);

QBMMV3StreamKTiling960::QBMMV3StreamKTiling960(gert::TilingContext* context)
    : QuantBatchMatmulV3TilingBase(context, false), tilingData_(tilingDataSelf_)
{
    Reset();
    InitCompileInfo();
    inputParams_.Reset();
}

QBMMV3StreamKTiling960::QBMMV3StreamKTiling960(gert::TilingContext* context,
                                               DequantBmm::QuantBatchMatmulV3StreamKBasicAPITilingData* out)
    : QuantBatchMatmulV3TilingBase(context, true), tilingData_(*out)
{
    Reset();
    InitCompileInfo();
    inputParams_.Reset();
}

bool QBMMV3StreamKTiling960::IsCapable()
{
    // 量化场景分流：9201/9202 可走 SK 的量化场景一致（MX per-group + 非MX per-tensor + HIF4 per-group），均走 Blaze。
    bool isQuantCapable = inputParams_.isMxPerGroup || inputParams_.isHif4PerGroup || IsPertensorStreamKInput();
    if (compileInfo_.aicNum != compileInfo_.aivNum || !isQuantCapable || inputParams_.batchC > 1) {
        return false;
    }
    // Blaze 路径依赖运行时 tensorapi，不可用时本模板放弃。
    if (!IsTensorapiCapable()) {
        OP_LOGD(inputParams_.opName, "QBMM StreamK requires tensorapi runtime for Blaze path.");
        return false;
    }
    return true;
}

bool QBMMV3StreamKTiling960::IsPertensorStreamKInput() const
{
    const bool isSupportedFormat = inputParams_.aFormat == ge::FORMAT_ND && inputParams_.cFormat == ge::FORMAT_ND &&
                                   (inputParams_.bFormat == ge::FORMAT_ND ||
                                    inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ);
    // 本 StreamK 扩展只覆盖非 MX 的 per-tensor 量化：x2Scale 为 {1}；
    // x1Scale(接口名 pertokenScaleOptional)若存在，也必须是 {1}，对应 isDoubleScale。
    // shape 为 {M} 的真正 per-token 行广播场景暂不纳入该模板，避免误走 StreamK。
    const bool isSupportedQuantMode = inputParams_.isPerTensor && !inputParams_.isPertoken &&
                                      !inputParams_.isPerChannel && !inputParams_.isMxPerGroup &&
                                      !inputParams_.isPerBlock && !inputParams_.isPerBlockPerToken;
    if (!isSupportedFormat || !isSupportedQuantMode) {
        return false;
    }

    // 非 MX StreamK 的 SK block 把 raw accumulator 写入 workspace，最终在 AIV/UB 上做 scale/bias 反量化。
    // X2 scale 按 Fixpipe 乘法字段掩码后再乘；uint64/int64 输入从低 32bit 的 deq_scale 编码中取该字段。
    const bool isIntScale = inputParams_.scaleDtype == ge::DT_UINT64 || inputParams_.scaleDtype == ge::DT_INT64;
    const bool isFloatScale = inputParams_.scaleDtype == ge::DT_FLOAT || inputParams_.scaleDtype == ge::DT_BF16;

    const bool isInt8 = inputParams_.aDtype == ge::DT_INT8 && inputParams_.bDtype == ge::DT_INT8;
    if (isInt8) {
        const bool isSupportedScaleAndOutput = !inputParams_.isDoubleScale &&
                                               ((isIntScale && (inputParams_.cDtype == ge::DT_FLOAT16 ||
                                                                inputParams_.cDtype == ge::DT_BF16)) ||
                                                (isFloatScale && inputParams_.cDtype == ge::DT_BF16));
        // INT32 bias stays in the MMAD accumulation domain. Matching FP32/BF16 scale and bias are
        // applied by the AIV epilogue; IsCapable limits that combination to an all-SK schedule.
        const bool isSupportedBias = !inputParams_.hasBias || inputParams_.biasDtype == ge::DT_INT32 ||
                                     IsPostDequantBiasInput();
        return isSupportedScaleAndOutput && isSupportedBias;
    }

    const auto isFp8 = [](ge::DataType dtype) { return dtype == ge::DT_FLOAT8_E4M3FN || dtype == ge::DT_FLOAT8_E5M2; };
    const bool isHif8Pair = inputParams_.aDtype == ge::DT_HIFLOAT8 && inputParams_.bDtype == ge::DT_HIFLOAT8;
    const bool isFp8Pair = isFp8(inputParams_.aDtype) && isFp8(inputParams_.bDtype);
    if (!isHif8Pair && !isFp8Pair) {
        return false;
    }

    const bool isSupportedOutput = inputParams_.cDtype == ge::DT_FLOAT16 || inputParams_.cDtype == ge::DT_BF16 ||
                                   inputParams_.cDtype == ge::DT_FLOAT;
    const bool isSupportedScale = (!inputParams_.isDoubleScale && isIntScale) ||
                                  (inputParams_.isDoubleScale && inputParams_.scaleDtype == ge::DT_FLOAT &&
                                   inputParams_.perTokenScaleDtype == ge::DT_FLOAT);
    // Encoded integer scale keeps FP32 bias in MMAD. Double-FP32 scale applies FP32 bias after both scale
    // multiplications in the dedicated AIV epilogue; IsCapable limits that case to an all-SK schedule.
    const bool isSupportedBias = !inputParams_.hasBias || (isIntScale && inputParams_.biasDtype == ge::DT_FLOAT) ||
                                 IsPostDequantBiasInput();
    return isSupportedScale && isSupportedOutput && isSupportedBias;
}

bool QBMMV3StreamKTiling960::IsPostDequantBiasInput() const
{
    const bool isInt8 = inputParams_.aDtype == ge::DT_INT8 && inputParams_.bDtype == ge::DT_INT8;
    const bool isInt8SingleScale = isInt8 && !inputParams_.isDoubleScale && inputParams_.hasBias &&
                                   (inputParams_.scaleDtype == ge::DT_FLOAT ||
                                    inputParams_.scaleDtype == ge::DT_BF16) &&
                                   inputParams_.biasDtype == inputParams_.scaleDtype;
    const bool isDoubleFp32Scale = inputParams_.isDoubleScale && inputParams_.scaleDtype == ge::DT_FLOAT &&
                                   inputParams_.perTokenScaleDtype == ge::DT_FLOAT && inputParams_.hasBias &&
                                   inputParams_.biasDtype == ge::DT_FLOAT;
    return isInt8SingleScale || isDoubleFp32Scale;
}

ge::graphStatus QBMMV3StreamKTiling960::DoOpTiling()
{
    OP_LOGD(inputParams_.opName, "DoOpTiling of qbmm streamk tiling strategy.");
    if (!CalcBasicBlockV2()) {
        OP_LOGD(inputParams_.opName, "StreamkTiling CalcBasicBlock failed, continue next tiling mode");
        return ge::GRAPH_PARAM_INVALID;
    }

    OP_LOGD(inputParams_.opName, "run into DoOpTiling after");
    CalL1Tiling();

    if (!IsProfilingBetter()) {
        OP_LOGD(inputParams_.opName, "streamk profiling gains is not enough, continue next tiling mode");
        return ge::GRAPH_PARAM_INVALID;
    }
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

bool QBMMV3StreamKTiling960::CalcBasicBlockV2()
{
    BaseBlockCalculator baseBlockSk(inputParams_, compileInfo_, 1UL);
    bool isValid = baseBlockSk.Compute(BaseBlockMode::STREAMK);
    if (!isValid) {
        return false;
    }
    BaseBlockRes baseBlockStreamk = baseBlockSk.GetOutput();

    tilingData_.matmulTiling.baseM = baseBlockStreamk.baseM;
    tilingData_.matmulTiling.baseN = baseBlockStreamk.baseN;
    tilingData_.matmulTiling.baseK = baseBlockStreamk.baseK;
    tilingData_.streamKTiling.kCnt = static_cast<uint32_t>(baseBlockStreamk.streamKCnt);
    tilingData_.streamKTiling.singleCoreK = baseBlockStreamk.singleCoreK;

    uint64_t mCnt = MathUtil::CeilDivision(inputParams_.mSize, tilingData_.matmulTiling.baseM);
    uint64_t nCnt = MathUtil::CeilDivision(inputParams_.nSize, tilingData_.matmulTiling.baseN);
    uint64_t totalCnt = mCnt * nCnt;

    tilingData_.streamKTiling.usedCoreNum = totalCnt * tilingData_.streamKTiling.kCnt > compileInfo_.aicNum ?
                                                static_cast<uint32_t>(compileInfo_.aicNum) :
                                                static_cast<uint32_t>(totalCnt * tilingData_.streamKTiling.kCnt);

    tilingData_.matmulTiling.nBufferNum = static_cast<uint8_t>(DB_SIZE);
    tilingData_.matmulTiling.isBias = 0;
    return true;
}

bool QBMMV3StreamKTiling960::IsProfilingBetter()
{
    if (tilingData_.streamKTiling.kCnt <= 1) {
        return false;
    }

    BaseBlockCalculator baseBlockDatabalance(inputParams_, compileInfo_, 1UL);
    bool isLoadBalanceValid = baseBlockDatabalance.Compute(BaseBlockMode::LOAD_BALANCE);
    if (!isLoadBalanceValid) {
        return true;
    }
    BaseBlockRes baseBlockAwst = baseBlockDatabalance.GetOutput();
    L1TilingDataCalculator l1Tiling(inputParams_, compileInfo_, baseBlockAwst.baseM, baseBlockAwst.baseN,
                                    baseBlockAwst.baseK);
    l1Tiling.Compute(L1TilingMode::DEFAULT);
    L1TilingData l1Res = l1Tiling.GetOutput();
    uint64_t awstKl1 = std::min(l1Res.stepKa_, l1Res.stepKb_) * baseBlockAwst.baseK;

    SocSpecifyParams socParams;
    bool isValidInfo = baseBlockDatabalance.GetSocSpecAdjustBalanceInfo(socParams);
    if (!isValidInfo) {
        return false;
    }

    // Evaluate ASWT cost
    uint64_t aswtUsedCore = MathUtil::CeilDivision(inputParams_.mSize, baseBlockAwst.baseM) *
                            MathUtil::CeilDivision(inputParams_.nSize, baseBlockAwst.baseN);
    aswtUsedCore = std::min(aswtUsedCore, static_cast<uint64_t>(compileInfo_.aicNum));
    uint64_t cDtypeBytes = static_cast<uint64_t>(ge::GetSizeByDataType(inputParams_.cDtype));
    TilingCostEvaluator aswtEvaluator(socParams);
    CaseInfo aswtCaseInfo{inputParams_.mSize,  inputParams_.nSize,  inputParams_.kSize, baseBlockAwst.baseM,
                          baseBlockAwst.baseN, baseBlockAwst.baseK, aswtUsedCore,       awstKl1,
                          cDtypeBytes};

    AicTimeCost aswtResult;
    aswtEvaluator.EvaluateAicTime(aswtCaseInfo, inputParams_, compileInfo_, aswtResult);

    // Evaluate SK cost
    TilingCostEvaluator skEvaluator(socParams);
    const auto& skMt = tilingData_.matmulTiling;
    uint64_t skMCnt = MathUtil::CeilDivision(inputParams_.mSize, skMt.baseM);
    uint64_t skNCnt = MathUtil::CeilDivision(inputParams_.nSize, skMt.baseN);
    uint64_t skUsedCoreNum = std::min(skMCnt * skNCnt * tilingData_.streamKTiling.kCnt,
                                      static_cast<uint64_t>(compileInfo_.aicNum));
    CaseInfo skCaseInfo{inputParams_.mSize, inputParams_.nSize, inputParams_.kSize, skMt.baseM,
                        skMt.baseN,         skMt.baseK,         skUsedCoreNum,      tilingData_.streamKTiling.kL1,
                        cDtypeBytes};

    StreamKTimeCost skResult = skEvaluator.EvaluateStreamkCost(skCaseInfo, tilingData_.streamKTiling.singleCoreK,
                                                               inputParams_, compileInfo_);
    if (skResult.totalTime + SK_VALID_THRESHOLD < aswtResult.totalTime) {
        return true;
    } else {
        return false;
    }
}

void QBMMV3StreamKTiling960::CalL1Tiling()
{
    tilingData_.matmulTiling.dbL0C = (tilingData_.matmulTiling.baseM * tilingData_.matmulTiling.baseN * DATA_SIZE_L0C *
                                          DB_SIZE <=
                                      aicoreParams_.l0cSize) ?
                                         static_cast<uint8_t>(DB_SIZE) :
                                         1;
    L1TilingDataCalculator l1Tiling(inputParams_, compileInfo_, tilingData_.matmulTiling.baseM,
                                    tilingData_.matmulTiling.baseN, tilingData_.matmulTiling.baseK);

    l1Tiling.Compute(L1TilingMode::DEFAULT);
    L1TilingData l1Res = l1Tiling.GetOutput();

    tilingData_.streamKTiling.kL1 = std::min(l1Res.stepKa_, l1Res.stepKb_) * tilingData_.matmulTiling.baseK;
    if (inputParams_.isMxPerGroup || inputParams_.isHif4PerGroup) {
        tilingData_.matmulTiling.scaleKL1 = std::min(l1Res.stepKa_ * l1Res.scaleFactorA_,
                                                     l1Res.stepKb_ * l1Res.scaleFactorB_) *
                                            tilingData_.matmulTiling.baseK;
    }
}

uint64_t QBMMV3StreamKTiling960::GetBatchMode() const { return TPL_WITHOUT_BATCH; }

uint64_t QBMMV3StreamKTiling960::GetKernelType() const { return TPL_VEC_EPILOGUE_STREAMK_WITH_MMAPI; }

uint64_t QBMMV3StreamKTiling960::GetApiLevel(NpuArch npuArch) const
{
    // DAV_9201/DAV_9202所有StreamK量化场景统一走Blaze路径：直接复用3510的
    // QbmmMxStreamKBasicApiKernel（tiling data已统一）。
    return static_cast<uint64_t>(QMMApiLevel::BLAZE_LEVEL);
}

uint64_t QBMMV3StreamKTiling960::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(inputParams_.transA), static_cast<uint64_t>(inputParams_.transB),
                              GetBatchMode(), GetKernelType(), GetApiLevel(compileInfo_.npuArch));
}

ge::graphStatus QBMMV3StreamKTiling960::DoLibApiTiling()
{
    tilingData_.matmulTiling.m = inputParams_.mSize;
    tilingData_.matmulTiling.n = inputParams_.nSize;
    tilingData_.matmulTiling.k = inputParams_.kSize;

    tilingData_.matmulTiling.isBias = inputParams_.hasBias ? 1 : 0;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QBMMV3StreamKTiling960::GetWorkspaceSize()
{
    workspaceSize_ = compileInfo_.aicNum * BASIC_BLOCK_SIZE_256 * BASIC_BLOCK_SIZE_256 * DATA_SIZE_FP32 +
                     MB_SIZE * EXTRA_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

void QBMMV3StreamKTiling960::Reset()
{
    isBf16Opt_ = false;
    isUbQuant_ = false;

    if (!isTilingOut_) {
        tilingData_ = DequantBmm::QuantBatchMatmulV3StreamKBasicAPITilingData();
        OP_TILING_CHECK(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                                 0, context_->GetRawTilingData()->GetCapacity()) != EOK,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Fail to clear tiling data"), return);
    }
}

ge::graphStatus QBMMV3StreamKTiling960::GetShapeAttrsInfo()
{
    inputParams_.Reset();
    tilingDataSize_ = sizeof(DequantBmm::QuantBatchMatmulV3StreamKBasicAPITilingData);
    return QuantBatchMatmulV3TilingBase::GetShapeAttrsInfo();
}

bool QBMMV3StreamKTiling960::CheckDtype() const
{
    QuantBatchMatmulV3CheckerArch9201 qmmV3Checker(context_, inputParams_);
    OP_TILING_CHECK(!qmmV3Checker.CheckDtype(), CUBE_INNER_ERR_REPORT(inputParams_.opName, "CheckDtype fail"),
                    return false);
    return true;
}

bool QBMMV3StreamKTiling960::CheckShape(const std::vector<gert::Shape*>& mandatoryShape,
                                        const gert::StorageShape* biasShape, const gert::StorageShape* pertokenShape,
                                        const std::vector<int64_t>& dimValueOfMKN) const
{
    QuantBatchMatmulV3CheckerArch9201 qmmV3Checker(context_, inputParams_);
    OP_TILING_CHECK(!qmmV3Checker.CheckShape(mandatoryShape, biasShape, pertokenShape, dimValueOfMKN),
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "CheckShape fail"), return false);
    return true;
}

ge::graphStatus QBMMV3StreamKTiling960::PostTiling()
{
    OP_TILING_CHECK(
        tilingDataSize_ % sizeof(uint64_t) != 0UL,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Tiling data size[%zu] is not aligned to 8.", tilingDataSize_),
        return ge::GRAPH_PARAM_INVALID);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void*>(&tilingData_), tilingDataSize_);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_PARAM_INVALID;
    }
    context_->SetBlockDim(tilingData_.streamKTiling.usedCoreNum);
    context_->SetScheduleMode(1); // StreamK kernel uses cross-core synchronization and requires batch mode.
    context_->GetRawTilingData()->SetDataSize(tilingDataSize_);
    size_t* workspaces = context_->GetWorkspaceSizes(1); // set workspace
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

void QBMMV3StreamKTiling960::SetTilingData()
{
    tilingData_.params.batchA = inputParams_.batchA;
    tilingData_.params.batchB = inputParams_.batchB;
    tilingData_.params.batchC = inputParams_.batchC;
    tilingData_.params.batchA1 = inputParams_.batchA1;
    tilingData_.params.batchA2 = inputParams_.batchA2;
    tilingData_.params.batchA3 = inputParams_.batchA3;
    tilingData_.params.batchA4 = inputParams_.batchA4;
    tilingData_.params.batchB1 = inputParams_.batchB1;
    tilingData_.params.batchB2 = inputParams_.batchB2;
    tilingData_.params.batchB3 = inputParams_.batchB3;
    tilingData_.params.batchB4 = inputParams_.batchB4;
    tilingData_.params.batchC1 = inputParams_.batchC1;
    tilingData_.params.batchC2 = inputParams_.batchC2;
    tilingData_.params.batchC3 = inputParams_.batchC3;
    tilingData_.params.batchC4 = inputParams_.batchC4;
    tilingData_.params.biasDtype = static_cast<uint32_t>(inputParams_.biasDtype);
    tilingData_.params.biasThreeDim = static_cast<uint32_t>(inputParams_.batchBias > 1U);
    tilingData_.params.groupSizeM = static_cast<uint32_t>(inputParams_.groupSizeM);
    tilingData_.params.groupSizeN = static_cast<uint32_t>(inputParams_.groupSizeN);
    tilingData_.params.groupSizeK = static_cast<uint32_t>(inputParams_.groupSizeK);
    tilingData_.matmulTiling.weightMustHitL2 = static_cast<uint8_t>(
        IsWeightMustHitL2(inputParams_, tilingData_.matmulTiling.baseM) ? 1 : 0);
    // adaptiveSlidingWin仅作BasicAPI布局兼容保留，除windowLen外StreamK kernel不消费，填中性占位值。
    // windowLen 按架构规格计算并下发给 kernel，供 ASW block scheduler 使用。
    tilingData_.adaptiveSlidingWin.mTailTile = 1U;
    tilingData_.adaptiveSlidingWin.nTailTile = 1U;
    tilingData_.adaptiveSlidingWin.mBaseTailSplitCnt = 1U;
    tilingData_.adaptiveSlidingWin.nBaseTailSplitCnt = 1U;
    tilingData_.adaptiveSlidingWin.mTailMain = 0U;
    tilingData_.adaptiveSlidingWin.nTailMain = 0U;
    tilingData_.adaptiveSlidingWin.windowLen = static_cast<uint32_t>(
        GetAswWindowLen(compileInfo_.npuArch, aicoreParams_.aicNum));

    if (inputParams_.isDoubleScale) {
        tilingData_.params.x1QuantMode = static_cast<uint32_t>(BasicQuantMode::PERTENSOR_MODE);
        tilingData_.params.x2QuantMode = static_cast<uint32_t>(BasicQuantMode::PERTENSOR_MODE);
    } else if (inputParams_.isPerTensor) {
        tilingData_.params.x2QuantMode = static_cast<uint32_t>(BasicQuantMode::PERTENSOR_MODE);
    } else if (inputParams_.isPerChannel) {
        tilingData_.params.x2QuantMode = static_cast<uint32_t>(BasicQuantMode::PERCHANNEL_MODE);
    } else if (inputParams_.isMxPerGroup) {
        tilingData_.params.x1QuantMode = static_cast<uint32_t>(BasicQuantMode::MX_PERGROUP_MODE);
        tilingData_.params.x2QuantMode = static_cast<uint32_t>(BasicQuantMode::MX_PERGROUP_MODE);
    } else if (inputParams_.isHif4PerGroup) {
        tilingData_.params.x1QuantMode = static_cast<uint32_t>(BasicQuantMode::HIF4_PERGROUP_MODE);
        tilingData_.params.x2QuantMode = static_cast<uint32_t>(BasicQuantMode::HIF4_PERGROUP_MODE);
    }
}

ge::graphStatus QBMMV3StreamKTiling960::GetPlatformInfo()
{
    OP_LOGE_IF(!SetPlatformInfoForTiling(), ge::GRAPH_FAILED, inputParams_.opName, "SetPlatformInfo fail");
    if (aicoreParams_.aicNum == 0UL || aicoreParams_.l1Size == 0UL || aicoreParams_.l0cSize == 0UL) {
        OP_LOGE(inputParams_.opName, "coreNum/L1Size/L0cSize should not be 0. coreNum: %lu, L1Size: %lu, L0cSize: %lu",
                aicoreParams_.aicNum, aicoreParams_.l1Size, aicoreParams_.l0cSize);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
