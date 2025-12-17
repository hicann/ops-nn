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
 * \file transpose_batch_mat_mul_base_tiling.cc
 * \brief
 */
#include "transpose_batch_mat_mul_base_tiling.h"

#include "util/math_util.h"
#include "log/log.h"
#include "tiling_base/tiling_key.h"
#include "error_util.h"
#include "op_cache_tiling.h"
#include "runtime_kb_api.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_tuning.h"
#include "matmul/common/op_host/math_util.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"
#include "platform/platform_infos_def.h"
#include "../../op_kernel/transpose_batch_mat_mul_tiling_key.h"

using namespace optiling::transpose_batch_mat_mul;

namespace {
constexpr size_t ATTR_NUM = 5;
constexpr size_t HF32_ATTR_INDEX = 3;
constexpr size_t ALLOW_DIM = 3;
constexpr size_t BATCH_IDX = 0;
constexpr size_t M_IDX = 1;
constexpr size_t KA_IDX = 2;
constexpr size_t KB_IDX = 1;
constexpr size_t N_IDX = 2;
constexpr size_t BIAS_IDX = 2;
constexpr uint64_t BLOCK_CUBE = 16;
constexpr uint64_t NO_BATCH_SHAPE_DIM = 2;
constexpr uint64_t ONE_BATCH_SHAPE_DIM = 3;
constexpr uint64_t TWO_BATCH_SHAPE_DIM = 4;
constexpr uint64_t THREE_BATCH_SHAPE_DIM = 5;
constexpr uint64_t FOUR_BATCH_SHAPE_DIM = 6;
constexpr uint64_t DEFAULT_SIZE = 32;
constexpr uint64_t NUM_TWO = 2;
constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256;
}  // namespace


namespace optiling {
namespace transpose_batch_mat_mul {
static inline uint64_t LastPower2(uint64_t n)
{
    n |= n >> 1; // 前2位为1
    n |= n >> 2; // 前4位为1
    n |= n >> 4; // 前8位为1
    n |= n >> 8; // 前16位为1
    n |= n >> 16; // 前32位为1
    n |= n >> 32; // 前64位为1
    return (n & ~(n >> 1));
}

static inline uint64_t NextPower2(uint64_t n)
{
    uint64_t lastPower2 = LastPower2(n);
    return (n == lastPower2) ? lastPower2 : (lastPower2 << 1);
}

ge::graphStatus TransposeBatchMatMulBaseTiling::GetShapeAttrsInfo()  // 检查输入属性是否支持
{
    args_.opName = context_->GetNodeName();
    OP_TILING_CHECK(args_.opName == nullptr, CUBE_INNER_ERR_REPORT("TransposeBatchMatMul", "get op name invalid"),
                    return ge::GRAPH_FAILED);
    OP_LOGI(args_.opName, "TilingContext: %s", Ops::NN::DebugTilingContext(context_).c_str());
    OP_TILING_CHECK((CheckArgs() != ge::GRAPH_SUCCESS) || (GetArgs() != ge::GRAPH_SUCCESS),
                    CUBE_INNER_ERR_REPORT(args_.opName, "invalid context"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeBatchMatMulBaseTiling::DoLibApiTiling()
{
    auto ret = MatmulV3BaseTiling::DoLibApiTiling();
    tbmmTilingData_.multiBatchInfo.set_batchUsedCoreNum(tbmmTilingData_.matmulTiling.matmulTiling.get_usedCoreNum());
    tbmmTilingData_.multiBatchInfo.set_aBatchDim3(static_cast<uint32_t>(batchInfo_.batchA3));
    tbmmTilingData_.multiBatchInfo.set_aBatchDim2(static_cast<uint32_t>(batchInfo_.batchA2));
    tbmmTilingData_.multiBatchInfo.set_aBatchDim1(static_cast<uint32_t>(batchInfo_.batchA1));
    tbmmTilingData_.multiBatchInfo.set_aBatchDim0(static_cast<uint32_t>(batchInfo_.batchA0));
    tbmmTilingData_.multiBatchInfo.set_bBatchDim3(static_cast<uint32_t>(batchInfo_.batchB3));
    tbmmTilingData_.multiBatchInfo.set_bBatchDim2(static_cast<uint32_t>(batchInfo_.batchB2));
    tbmmTilingData_.multiBatchInfo.set_bBatchDim1(static_cast<uint32_t>(batchInfo_.batchB1));
    tbmmTilingData_.multiBatchInfo.set_bBatchDim0(static_cast<uint32_t>(batchInfo_.batchB0));
    tbmmTilingData_.multiBatchInfo.set_cBatchDim3(static_cast<uint32_t>(batchInfo_.batchC3));
    tbmmTilingData_.multiBatchInfo.set_cBatchDim2(static_cast<uint32_t>(batchInfo_.batchC2));
    tbmmTilingData_.multiBatchInfo.set_cBatchDim1(static_cast<uint32_t>(batchInfo_.batchC1));
    tbmmTilingData_.multiBatchInfo.set_cBatchDim0(static_cast<uint32_t>(batchInfo_.batchC0));

    aBatchDimAll_ = batchInfo_.batchA0 * batchInfo_.batchA1 * batchInfo_.batchA2 * batchInfo_.batchA3;
    bBatchDimAll_ = batchInfo_.batchB0 * batchInfo_.batchB1 * batchInfo_.batchB2 * batchInfo_.batchB3;
    cBatchDimAll_ = batchInfo_.batchC0 * batchInfo_.batchC1 * batchInfo_.batchC2 * batchInfo_.batchC3;
    tbmmTilingData_.multiBatchInfo.set_aBatchDimAll(static_cast<uint32_t>(aBatchDimAll_));
    tbmmTilingData_.multiBatchInfo.set_bBatchDimAll(static_cast<uint32_t>(bBatchDimAll_));
    tbmmTilingData_.multiBatchInfo.set_cBatchDimAll(static_cast<uint32_t>(cBatchDimAll_));
    tbmmTilingData_.multiBatchInfo.set_batchTileBlock(static_cast<uint32_t>(cBatchDimAll_));
    if (CheckBMMTilingDataIsVaild()) {
        return ge::GRAPH_FAILED;
    }
    tbmmTilingData_.multiBatchInfo.set_biasWithBatch(static_cast<uint32_t>(batchInfo_.biasWithBatch));
    tbmmTilingData_.multiBatchInfo.set_mOri(static_cast<uint32_t>(args_.mOriValue));
    DoCommonTiling();
    tbmmTilingData_.matmulTiling.tileL2cacheTiling.set_mTileCntL2(1);
    tbmmTilingData_.matmulTiling.tileL2cacheTiling.set_nTileCntL2(1);

    tbmmTilingData_.matmulTiling.matmulRunInfo.set_transA(permA_);
    tbmmTilingData_.matmulTiling.matmulRunInfo.set_transB(permB_);
    tbmmTilingData_.set_batchSplitFactor(batchSplitFactor_);
    return ret;
}

/*
 * Algorithm to calculate the best (baseM, baseN) that gives even workload amongst iterations.
 * Parameter `divisor` is used to control the starting point of the algorithm.
 * Choosing different starting point can sometimes get better performence.
 * The starting point of `divisor = 2` is half of that of `divisor = 1`
 */
static void CalcBaseMN(uint64_t &baseM, uint64_t &baseN, const matmul_v3::MatmulV3Args &args, uint64_t divisor = 1UL)
{
    uint64_t dtypeSize = GetSizeByDataType(args.aType);
    // step 1: calc baseM
    auto getBestBaseM = [dtypeSize, divisor](bool transX1, uint64_t m, uint64_t n, uint64_t k) -> uint64_t {
        if (!transX1) {
            // 防止baseK太小,限制baseM最大为2048B, k大于32时进一步限制为1024B
            uint64_t maxM = k >= 32UL? 1024UL : 2048UL;
            maxM /= (dtypeSize * divisor);
            uint64_t mTimes = ops::CeilDiv(m, maxM);
            uint64_t singleTimeM = ops::CeilDiv(m, mTimes);
            return ops::CeilAlign(singleTimeM, BLOCK_CUBE);
        }
        uint64_t nCalc = std::min(n, 512UL);    // baseN大小不会超过512
        uint64_t minBaseM = std::max(32768UL / dtypeSize / nCalc, 32UL); // 最小32, n小时限制baseM*n>=32768B(经验值)
        uint64_t maxBaseM = std::min(NextPower2(m), (2048UL / dtypeSize/ divisor)); // L0限制最大2048B, fp16尝试1024B
        uint64_t bestBaseM = std::max(maxBaseM, 16UL); // 最小16
        for (uint64_t candidate = bestBaseM; candidate >= minBaseM; candidate >>= uint64_t(1)) {
            if (m % candidate == uint64_t(0)) { break; }
            if (ops::CeilAlign(m, candidate) < ops::CeilAlign(m, bestBaseM)) {
                bestBaseM = candidate;
            }
        }
        return bestBaseM;
    };
    baseM = getBestBaseM(args.isATrans, args.mValue, args.nValue, args.kValue);
    // step 2: calc baseN
    auto getBestBaseN = [dtypeSize, baseM, divisor](bool transX2, uint64_t n, uint64_t k) -> uint64_t {
        uint64_t minBaseN = std::max(std::min(32768UL / dtypeSize / baseM, 512UL / dtypeSize), 32UL); // 32768
        // 最大2048，k为内轴且大于32时取1024
        uint64_t maxBaseN = (transX2 && k >= 32UL) ? 1024UL : 2048UL / dtypeSize / divisor;
        uint64_t bestBaseN = std::max(std::min(NextPower2(n), maxBaseN), 16UL); // 同时最小设为16
        for (uint64_t candidate = bestBaseN; candidate >= minBaseN; candidate >>= uint64_t(1)) {
            if (n % candidate == uint64_t(0)) { break; }
            if (ops::CeilAlign(n, candidate) < ops::CeilAlign(n, bestBaseN)) {
                bestBaseN = candidate;
            }
        }
        return std::min(LastPower2(32768UL / baseM), bestBaseN);    //L0C大小限制baseM * baseN <= 32768;
    };
    baseN = getBestBaseN(args.isBTrans, args.nValue, args.kValue);
}

static void TuneBaseMN(matmul_v3::MatmulV3RunInfo &runInfo,
                       const matmul_v3::MatmulV3Args &args,
                       uint64_t batchC,
                       uint64_t aicNum)
{
    uint64_t &oriBaseM = runInfo.baseM;
    uint64_t &oriBaseN = runInfo.baseN;
    CalcBaseMN(oriBaseM, oriBaseN, args);
    if (args.aType == ge::DT_FLOAT) {
        return;
    }
    // for bf16 fp16 try a different set of thresholds
    uint64_t newBaseM;
    uint64_t newBaseN;
    CalcBaseMN(newBaseM, newBaseN, args, 2UL);  // choose half of the starting point by setting divisor = 2
    // evaluate core utilization
    auto getCoreUtilization = [args, batchC, aicNum](uint64_t baseM, uint64_t baseN) -> double {
        uint64_t cnt = ops::CeilDiv(args.mValue, baseM) * ops::CeilDiv(args.nValue, baseN) * batchC;
        return static_cast<double>(cnt) / ops::CeilAlign(cnt, aicNum);
    };
    double oriCoreUtil = getCoreUtilization(oriBaseM, oriBaseN);
    if ((oriCoreUtil < 0.6) && (getCoreUtilization(newBaseM, newBaseN) > oriCoreUtil)) {    // 0.6为经验值
        oriBaseM = newBaseM;
        oriBaseN = newBaseN;
        return;
    }
    // evaluate data copy workload
    auto getDataCopyRatio = [](uint64_t baseM, uint64_t baseN) -> double {
        return 1.0 / static_cast<double>(baseM) + 1.0 / static_cast<double>(baseN);
    };
    if (getDataCopyRatio(newBaseM, newBaseN) < getDataCopyRatio(oriBaseM, oriBaseN)) {
        oriBaseM = newBaseM;
        oriBaseN = newBaseN;
    }
}

static void TuneBaseMKN(matmul_v3::MatmulV3RunInfo &runInfo,
                        const matmul_v3::MatmulV3Args &args,
                        uint64_t batchC,
                        uint64_t aicNum)
{
    uint64_t &baseM = runInfo.baseM;
    uint64_t &baseN = runInfo.baseN;
    uint64_t &baseK = runInfo.baseK;
    OP_LOGD(args.opName, "before DoCommonTiling baseM, baseN, baseK[%lu, %lu, %lu]", baseM, baseN, baseK);
    bool isSmallShape = (args.mValue < BASIC_BLOCK_SIZE_256 || args.nValue <= BASIC_BLOCK_SIZE_256);
    bool useBaseBlock = ((baseM == BASIC_BLOCK_SIZE_128 || baseM == BASIC_BLOCK_SIZE_256) &&
                         (baseN == BASIC_BLOCK_SIZE_128 || baseN == BASIC_BLOCK_SIZE_256));
    bool isL2Tile = (runInfo.l2Info.mTile > 1 || runInfo.l2Info.nTile > 1);
    if ((!isSmallShape && useBaseBlock) || isL2Tile) {
        return;
    }

    TuneBaseMN(runInfo, args, batchC, aicNum);
    uint64_t maxBaseK = 32768UL / GetSizeByDataType(args.aType) / std::max(baseM, baseN);   // L0AB大小限制32768B
    baseK = std::min(ops::FloorAlign(maxBaseK, BLOCK_CUBE), ops::CeilAlign(args.kValue, BLOCK_CUBE));
    OP_LOGD(args.opName, "after DoCommonTiling baseM, baseN, baseK[%lu, %lu, %lu]", baseM, baseN, baseK);
}


void TransposeBatchMatMulBaseTiling::DoCommonTiling()
{
    TuneBaseMKN(runInfo_, args_, cBatchDimAll_, compileInfo_.aicNum);
    uint64_t baseM = runInfo_.baseM;
    uint64_t baseN = runInfo_.baseN;
    uint64_t baseK = runInfo_.baseK;
    constexpr uint64_t reserveSize = 256;
    bool hasQuant = context_->GetOptionalInputShape(3) != nullptr;  // scale 为第3入参
    uint64_t totalL1Size = compileInfo_.l1Size + reserveSize; // 256B为预留给rpc使用，单算子不涉及
    if (hasQuant) {
        totalL1Size -= cBatchDimAll_ * args_.nValue * sizeof(uint64_t);
        OP_LOGI("tbmm", "Quant function is activated.");
    }
    if (args_.hasBias) {
        totalL1Size -= reserveSize * 4; // 1024: 256 * 4, biasTable 空间
        baseN = std::min(reserveSize, baseN);
    } // 带bias， baseN最大值为256
    // DB后FP32下，L1可以存65536个数值
    uint64_t depthA1 = (totalL1Size / NUM_TWO / aDtypeSize_ / (baseM * baseK) / uint64_t(4)) * uint64_t(4);
    // DB后FP32下，L1可以存65536个数值
    uint64_t depthB1 = (totalL1Size / NUM_TWO / aDtypeSize_ / (baseN * baseK) / uint64_t(4)) * uint64_t(4);
    depthA1 = std::max(NUM_TWO, depthA1);
    depthB1 = std::max(NUM_TWO, depthB1);
    uint64_t stepKa = depthA1 / NUM_TWO;
    uint64_t stepKb = depthB1 / NUM_TWO;
    if (stepKa > stepKb) {
        stepKa = stepKa / stepKb * stepKb;
        depthA1 = stepKa * NUM_TWO;
    } else {
        stepKb = stepKb / stepKa * stepKa;
        depthB1 = stepKb * NUM_TWO;
    }
    tbmmTilingData_.matmulTiling.matmulTiling.set_depthA1(depthA1);
    tbmmTilingData_.matmulTiling.matmulTiling.set_depthB1(depthB1);
    tbmmTilingData_.matmulTiling.matmulTiling.set_stepKa(stepKa);
    tbmmTilingData_.matmulTiling.matmulTiling.set_stepKb(stepKb);
    tbmmTilingData_.matmulTiling.matmulTiling.set_stepM(1);
    tbmmTilingData_.matmulTiling.matmulTiling.set_stepN(1);
    tbmmTilingData_.matmulTiling.matmulTiling.set_singleCoreM(std::min(args_.mValue, baseM));
    tbmmTilingData_.matmulTiling.matmulTiling.set_singleCoreN(std::min(args_.nValue, baseN));
    tbmmTilingData_.matmulTiling.matmulTiling.set_singleCoreK(args_.kValue);
    tbmmTilingData_.matmulTiling.matmulTiling.set_baseM(baseM);
    tbmmTilingData_.matmulTiling.matmulTiling.set_baseN(baseN);
    tbmmTilingData_.matmulTiling.matmulTiling.set_baseK(baseK);
    uint64_t batchSplitMode = batchSplitFactor_ > 1 ? 1 : 0;
    uint64_t ppMatmulMode = 0;
    uint64_t permX1 = 0;
    uint64_t permX2 = 0;
    if (permA_ == 213UL) {permX1 = 2;} // 2 是 permList 的字典序, 2 -> [1,0,2]
    else if(permA_ == 123UL) {permX1 = 0;}
    tilingKey_ = GET_TPL_TILING_KEY(batchSplitMode, ppMatmulMode, permX1, permX2,);
}

bool TransposeBatchMatMulBaseTiling::CheckBMMTilingDataIsVaild() const {
    return (optiling::matmul_v3::CheckNumberIsValid(batchInfo_.batchA3, args_.opName, "batchInfo_.batchA3") ||
            optiling::matmul_v3::CheckNumberIsValid(batchInfo_.batchA2, args_.opName, "batchInfo_.batchA2") ||
            optiling::matmul_v3::CheckNumberIsValid(batchInfo_.batchA1, args_.opName, "batchInfo_.batchA1") ||
            optiling::matmul_v3::CheckNumberIsValid(batchInfo_.batchA0, args_.opName, "batchInfo_.batchA0") ||
            optiling::matmul_v3::CheckNumberIsValid(batchInfo_.batchB3, args_.opName, "batchInfo_.batchB3") ||
            optiling::matmul_v3::CheckNumberIsValid(batchInfo_.batchB2, args_.opName, "batchInfo_.batchB2") ||
            optiling::matmul_v3::CheckNumberIsValid(batchInfo_.batchB1, args_.opName, "batchInfo_.batchB1") ||
            optiling::matmul_v3::CheckNumberIsValid(batchInfo_.batchB0, args_.opName, "batchInfo_.batchB0") ||
            optiling::matmul_v3::CheckNumberIsValid(batchInfo_.batchC3, args_.opName, "batchInfo_.batchC3") ||
            optiling::matmul_v3::CheckNumberIsValid(batchInfo_.batchC2, args_.opName, "batchInfo_.batchC2") ||
            optiling::matmul_v3::CheckNumberIsValid(batchInfo_.batchC1, args_.opName, "batchInfo_.batchC1") ||
            optiling::matmul_v3::CheckNumberIsValid(batchInfo_.batchC0, args_.opName, "batchInfo_.batchC0") ||
            optiling::matmul_v3::CheckNumberIsValid(aBatchDimAll_, args_.opName, "batchInfo_.aBatchDimAll") ||
            optiling::matmul_v3::CheckNumberIsValid(bBatchDimAll_, args_.opName, "batchInfo_.bBatchDimAll") ||
            optiling::matmul_v3::CheckNumberIsValid(cBatchDimAll_, args_.opName, "batchInfo_.cBatchDimAll"));
}

ge::graphStatus TransposeBatchMatMulBaseTiling::PostTiling()
{
    OP_TILING_CHECK(tilingData_.GetDataSize() % sizeof(uint64_t) != 0,
                    OP_LOGE(args_.opName, "tiling data size[%zu] is not aligned to 8", tilingData_.GetDataSize()),
                    return ge::GRAPH_FAILED);
    tbmmTilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
        context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tbmmTilingData_.GetDataSize());
    context_->SetBlockDim(compileInfo_.aicNum);
    workspaceSize_ = std::max(workspaceSize_, DEFAULT_SIZE);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr,
                    CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "workspaces is null"),
                    return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

uint64_t TransposeBatchMatMulBaseTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus TransposeBatchMatMulBaseTiling::CheckArgs()
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    size_t idx = 0;
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(idx));
    idx++;
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(idx));
    idx++;
    if (context_->GetOptionalInputShape(2)!= nullptr) { // bias是第2入参
        args_.hasBias = true;
    }
    if (attrs->GetAttrNum() >= ATTR_NUM) {
        OPS_CHECK_NULL_WITH_CONTEXT(context_,
                                    attrs->GetAttrPointer<bool>(ATTR_NUM - 2));  // 检查倒数第2个属性
        OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<int32_t>(ATTR_NUM - 1));
    }
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(0));
    return ge::GRAPH_SUCCESS;
}

static inline void GetFormat(const gert::TilingContext &context, optiling::matmul_v3::MatmulV3Args &args)
{
    args.aFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(context.GetInputDesc(0)->GetStorageFormat()));
    args.bFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(context.GetInputDesc(1)->GetStorageFormat()));
    args.outFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(context.GetOutputDesc(0)->GetStorageFormat()));
}

static inline void GetDtype(const gert::TilingContext &context, optiling::matmul_v3::MatmulV3Args &args)
{
    // op_impl_mode_enum: 0x1: default 0x2: high_performance 0x4: high_precision 0x8: super_performance
    // 0x10: support_of_bound_index 0x20: enable_float_32_execution 0x40: enable_hi_float_32_execution
    if (context.GetAttrs()->GetAttrNum() >= HF32_ATTR_INDEX + 1) {
        args.isHf32 = *context.GetAttrs()->GetAttrPointer<bool>(HF32_ATTR_INDEX) ? 1 : 0;
    }
    OP_LOGD(args.opName, "Hf32 flag is: %d", args.isHf32);

    args.aType = context.GetInputDesc(0)->GetDataType();
    args.bType = context.GetInputDesc(1)->GetDataType();
    args.cType = context.GetOutputDesc(0)->GetDataType();
    if (args.hasBias) {
        args.biasType = context.GetInputDesc(BIAS_IDX)->GetDataType();
    }
}

static ge::graphStatus OpSpecificCheck(const optiling::matmul_v3::MatmulV3Args &args)
{
    // format check
    OP_TILING_CHECK(
        (args.aFormat == ge::FORMAT_FRACTAL_NZ) || (args.bFormat == ge::FORMAT_FRACTAL_NZ) || (args.outFormat == ge::FORMAT_FRACTAL_NZ),
        CUBE_INNER_ERR_REPORT(args.opName, "invalid input/output format"), return ge::GRAPH_FAILED);

    // dtype check
    std::vector<ge::DataType> dtype = {args.aType, args.bType, args.cType};
    if (args.hasBias) {
        dtype.push_back(args.biasType);
    }

    auto isValidDtype = [&args](const std::vector<ge::DataType> &dtypeList) -> ge::graphStatus {
        const std::vector<std::vector<ge::DataType> > dtypeSuportList = {
            // x1,              x2,             y,              bias
            {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
            {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT},
            {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT},
            {ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT},
            {ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_FLOAT}};
        for (auto &supported : dtypeSuportList) {
            if (std::equal(dtypeList.begin(), dtypeList.end(), supported.begin())) {
                return ge::GRAPH_SUCCESS;
            }
        }
        OP_LOGE(args.opName, "Unsupported data type: x1[%d], x2[%d], y[%d], bias[%d], hasBias[%d]", args.aType,
                args.bType, args.cType, args.biasType, args.hasBias);
        return ge::GRAPH_FAILED;
    };
    return isValidDtype(dtype);
}

ge::graphStatus TransposeBatchMatMulBaseTiling::GetShapeMKN(const gert::Shape &aShape, const gert::Shape &bShape)
{
    int64_t m = aShape[M_IDX];
    int64_t kA = aShape[KA_IDX];
    int64_t kB = bShape[KB_IDX];
    int64_t n = bShape[N_IDX];

    if (aPermList_ != nullptr && aPermList_->GetSize() == ALLOW_DIM) {
        const int64_t *aPerm = reinterpret_cast<const int64_t *>(aPermList_->GetData());
        m = aShape[aPerm[M_IDX]];
        kA = aShape[aPerm[KA_IDX]];
        args_.isATrans = aPerm[M_IDX] > aPerm[KA_IDX];
    }
    if (bPermList_ != nullptr && bPermList_->GetSize() == ALLOW_DIM) {
        const int64_t *bPerm = reinterpret_cast<const int64_t *>(bPermList_->GetData());
        kB = bShape[bPerm[KB_IDX]];
        n = bShape[bPerm[N_IDX]];
        args_.isBTrans = bPerm[KB_IDX] > bPerm[N_IDX];
    }
    OP_TILING_CHECK(kA != kB, CUBE_INNER_ERR_REPORT(args_.opName, "unequal input kDim values"), return ge::GRAPH_FAILED);

    auto isValidDimValue = [](int64_t dim) -> bool { return (dim > 0) && (dim <= INT32_MAX); };
    if (!isValidDimValue(m) || !isValidDimValue(kA) || !isValidDimValue(n)) {
        OP_LOGE(args_.opName, "illegal value: m[%ld], k[%ld], n[%ld]", m, kA, n);
        return ge::GRAPH_FAILED;
    }
    args_.mValue = static_cast<uint64_t>(m);
    args_.kValue = static_cast<uint64_t>(kA);
    args_.nValue = static_cast<uint64_t>(n);
    args_.mOriValue = args_.mValue;
    args_.nOriValue = args_.nValue;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeBatchMatMulBaseTiling::GetShapeBatch(const gert::Shape &aShape, const gert::Shape &bShape)
{
    int64_t batchA = aShape[BATCH_IDX];
    int64_t batchB = bShape[BATCH_IDX];
    if (aPermList_ != nullptr && aPermList_->GetSize() == ALLOW_DIM) {
        const int64_t *aPerm = reinterpret_cast<const int64_t *>(aPermList_->GetData());
        batchA = aShape[aPerm[BATCH_IDX]];
    }
    if (bPermList_ != nullptr && bPermList_->GetSize() == ALLOW_DIM) {
        const int64_t *bPerm = reinterpret_cast<const int64_t *>(bPermList_->GetData());
        batchB = bShape[bPerm[BATCH_IDX]];
    }
    OP_TILING_CHECK(batchA != batchB, CUBE_INNER_ERR_REPORT(args_.opName, "unequal input batchDim values"),
                    return ge::GRAPH_FAILED);
    batchInfo_.batchA3 = batchA;
    batchInfo_.batchB3 = batchA;
    batchInfo_.batchC3 = batchA;
    batchInfo_.batchA = batchA;
    batchInfo_.batchB = batchA;
    batchInfo_.batchC = batchA;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeBatchMatMulBaseTiling::GetShapeBias()
{
    if (args_.hasBias) {
        const gert::Shape &biasShape = context_->GetInputShape(BIAS_IDX)->GetOriginShape();
        const int64_t biasValue = biasShape[biasShape.GetDimNum() - 1];
        if (args_.nOriValue != static_cast<uint64_t>(biasValue)) {
            OP_LOGE(args_.opName, "illegal value: bias[%ld], n[%lu]", biasValue, args_.nOriValue);
            return ge::GRAPH_FAILED;
        }
    }

    batchInfo_.biasWithBatch = false;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeBatchMatMulBaseTiling::GetShape()
{
    const gert::Shape &aShape = context_->GetInputShape(0)->GetOriginShape();
    const gert::Shape &bShape = context_->GetInputShape(1)->GetOriginShape();
    const gert::Shape &cShape = context_->GetOutputShape(0)->GetOriginShape();
    const size_t aDimNum = aShape.GetDimNum();
    const size_t bDimNum = bShape.GetDimNum();
    const size_t cDimNum = cShape.GetDimNum();
    OP_TILING_CHECK((aDimNum != ALLOW_DIM) || (bDimNum != ALLOW_DIM) || (cDimNum != ALLOW_DIM),
                    CUBE_INNER_ERR_REPORT(args_.opName, "invalid input/output dim num"), return ge::GRAPH_FAILED);

    auto attrs = context_->GetAttrs();
    size_t idx = 0;
    aPermList_ = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    bPermList_ = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);

    OP_TILING_CHECK(GetShapeMKN(aShape, bShape) != ge::GRAPH_SUCCESS, CUBE_INNER_ERR_REPORT(args_.opName,
                    "get m/k/n failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(GetShapeBatch(aShape, bShape) != ge::GRAPH_SUCCESS, CUBE_INNER_ERR_REPORT(args_.opName,
                    "get batch failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(GetShapeBias() != ge::GRAPH_SUCCESS, CUBE_INNER_ERR_REPORT(args_.opName,
                    "get bias failed"), return ge::GRAPH_FAILED);

    // 通过permA 传递perm， 将 perm 转成整数， 为避免零出现perm每位加一， 例 {1,0,2}  加一-> {2,1,3} -> 213
    const int64_t* perm_x1 = reinterpret_cast<const int64_t*>(aPermList_->GetData());
    permA_ = 0UL;
    for (uint32_t i = 0; i < aPermList_->GetSize(); i++) {
        permA_ = permA_ * 10 + perm_x1[i] + 1;   // 乘10 移位
    }
    const int64_t* perm_x2 = reinterpret_cast<const int64_t*>(bPermList_->GetData());
    permB_ = 0UL;
    for (uint32_t i = 0; i < bPermList_->GetSize(); i++) {
        permB_ = permB_ * 10 + perm_x2[i] + 1; // 乘10 移位
    }
    OP_LOGI(args_.opName, "permA_: %lu, permB_: %lu", permA_, permB_);

    uint64_t dtypeSize = GetSizeByDataType(args_.aType);
    uint64_t input_batch = batchInfo_.batchC;
    uint64_t input_k = args_.kValue;
    uint64_t input_n = args_.nValue;
    uint64_t alignLen = 256UL / dtypeSize;
    bool support_k_n = (input_k % alignLen == 0UL) && (input_n % alignLen == 0UL);
    bool support_batch_m = true;

    if (permA_ == 213UL) { //213 指的是 {1,0,2} 转置
        support_batch_m = (input_batch * input_k < 65536UL); // 65536 随路ND2NZ转换上限
    } else if (permA_ == 123UL) { //123 指的是 {0,1,2} 转置
        support_batch_m = (input_k < 65536UL); // 65536 随路ND2NZ转换上限
    } else {
        OP_LOGE(args_.opName, "perm is not supported");
        return ge::GRAPH_FAILED;
    }

    OP_TILING_CHECK(!(support_k_n && support_batch_m), CUBE_INNER_ERR_REPORT(args_.opName,
                    "only support shape inner axis < 65536, input shape b, m, n, k = %lu, %lu, %lu, %lu.",
                    input_batch, args_.mValue, input_n, input_k ), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeBatchMatMulBaseTiling::GetArgs()
{
    GetFormat(*context_, args_);
    GetDtype(*context_, args_);
    // get batch_split_factor
    if (context_->GetAttrs()->GetAttrNum() >= ATTR_NUM) {
        batchSplitFactor_ = std::max(*(context_->GetAttrs()->GetAttrPointer<int32_t>(ATTR_NUM - 1)), 1);
    }
    OP_TILING_CHECK((OpSpecificCheck(args_) != ge::GRAPH_SUCCESS),
                    CUBE_INNER_ERR_REPORT(args_.opName, "format and dtype check failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((GetShape() != ge::GRAPH_SUCCESS),
                    CUBE_INNER_ERR_REPORT(args_.opName, "get shape failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

}  // namespace transpose_batch_mat_mul
}  // namespace optiling