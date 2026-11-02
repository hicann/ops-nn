/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file matmul_v3_basic_streamk_tiling.cpp
 * \brief
 */
#include "matmul_v3_basic_streamk_tiling.h"
#include "matmul_v3_tiling_strategy.h"
#include "matmul_tiling_registry.h"
#include "matmul/common/op_host/math_util_nn.h"

using Ops::NN::MathUtil;
namespace {
using namespace optiling;
using namespace optiling::matmul_v3_advanced;

constexpr uint64_t FIXPIPE_N_ALIGN_ELEMENTS = 32UL;

// ------------------------------ CheckStreamKDPSKTiling -------------------------------------------//
bool CheckStreamKDPSKTilingDefault(const MatmulV3CompileInfo& /* compileInfo */, const MatMulV3Args& /* args */,
                                   MatMulV3RunInfo& /* runInfo */)
{
    return false;
}

bool CheckStreamKDPSKTilingDav3510(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args,
                                   MatMulV3RunInfo& runInfo)
{
    MatMulV3TilingHelper::ResetBase(compileInfo, args, runInfo);
    constexpr uint64_t STREAM_K_MIN_K_THRESHOLD = 8192UL;
    // 如果k轴小于32*256/DtypeSize_ 或 mn轴不是256对齐 或 输入是fp32类型，不走stream-k-dpsk
    if (args.mValue % runInfo.baseM != 0UL || args.nValue % runInfo.baseN != 0UL ||
        args.kValue <
            std::max(STREAM_K_MIN_K_THRESHOLD, compileInfo.aicNum * BASIC_BLOCK_K_128_BYTE) / args.aDtypeSize) {
        return false;
    }
    // 如果mn用256切分的份数小于核数 或者 取余核数为0或大于一半的核数，则不使用stream-k-dpsk
    uint64_t mCnt = MathUtil::CeilDivision(args.mValue, runInfo.baseM);
    uint64_t nCnt = MathUtil::CeilDivision(args.nValue, runInfo.baseN);
    uint64_t totalMNCnt = mCnt * nCnt;
    return (totalMNCnt >= compileInfo.aicNum) && (totalMNCnt % compileInfo.aicNum != 0UL) &&
           (totalMNCnt % compileInfo.aicNum <= compileInfo.aicNum / NUM_TWO);
}

using CheckStreamKDPSKTilingFunc = bool (*)(const MatmulV3CompileInfo&, const MatMulV3Args&, MatMulV3RunInfo&);

const static std::map<NpuArch, CheckStreamKDPSKTilingFunc> CheckStreamKDPSKTilingFuncMap = {
    {NpuArch::DAV_3510, CheckStreamKDPSKTilingDav3510},
    {NpuArch::DAV_9201, CheckStreamKDPSKTilingDav3510},
    {NpuArch::DAV_9202, CheckStreamKDPSKTilingDav3510},
};

// ------------------------------ CheckStreamKSKTiling -------------------------------------------//
bool CheckStreamKSKTilingDefault(const MatmulV3CompileInfo& /* compileInfo */, const MatMulV3Args& /* args */,
                                 MatMulV3RunInfo& /* runInfo */, const gert::TilingContext* /* context */)
{
    return false;
}

double GetMte2Perf(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args, int64_t usedCoreNum,
                   MatMulV3RunInfo& runInfo, fe::PlatFormInfos* platformInfo)
{
    int64_t hbmSize = (args.mValue + args.nValue) * args.kValue * args.aDtypeSize;
    int64_t l2Size = (MathUtil::CeilDivision(args.mValue, runInfo.baseM) * args.nValue +
                      MathUtil::CeilDivision(args.nValue, runInfo.baseN) * args.mValue) *
                         args.kValue * args.aDtypeSize -
                     hbmSize;
    // 单位换算为B/us
    double timeConvert = TIME_CONVERT_UNIT * TIME_CONVERT_UNIT;
    double mte2Perf = static_cast<double>(hbmSize) / (MatMulV3TilingHelper::GetHbmBW(platformInfo) * timeConvert) +
                      static_cast<double>(l2Size) / (MatMulV3TilingHelper::GetL2BW(platformInfo) * timeConvert);
    if (compileInfo.npuArch == NpuArch::DAV_9202) {
        return mte2Perf;
    }
    return mte2Perf / (static_cast<double>(usedCoreNum) / compileInfo.aicNum);
}

double GetNegPerf(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args, int64_t kCnt, int64_t usedCoreNum,
                  fe::PlatFormInfos* platformInfo)
{
    double hbmBw = MatMulV3TilingHelper::GetHbmBW(platformInfo);
    double timeConvert = TIME_CONVERT_UNIT * TIME_CONVERT_UNIT;
    // y=60.0*x+1000.0 为经验值
    double vecPerf = (kCnt * 60.0 + 1000.0) / (MatMulV3TilingHelper::GetCoreFreq(platformInfo) * TIME_CONVERT_UNIT) +
                     args.mValue * args.nValue * kCnt * DATA_SIZE_FP32 / (hbmBw * timeConvert);
    double dataTransferSize = args.mValue * args.nValue * (kCnt * DATA_SIZE_FP32 - args.aDtypeSize);
    double fixpPerf = static_cast<double>(dataTransferSize) / (hbmBw * timeConvert);
    if (compileInfo.npuArch == NpuArch::DAV_9202) {
        return vecPerf + fixpPerf;
    }
    return (vecPerf + fixpPerf) / (static_cast<double>(usedCoreNum) / compileInfo.aicNum);
}

// 纯FP32场景SK切K承担保精度职责(走SK模板的SK_SPLIT_K切分提升精度)，沿用原硬边界准入判定：
// MN小K大强制准入(isSKSupport) + mn份数门限。调用方需保证args为纯FP32且已过k轴门限；
// 准入成功后的tiling配置由DoSkOpTilingDefault启发式生成。
bool CheckStreamKSKFp32SplitKSupport(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args,
                                     MatMulV3RunInfo& runInfo)
{
    uint64_t mAlignedValue = ops::CeilAlign(args.mValue, BASIC_BLOCK_SIZE_16);
    uint64_t nAlignedValue = ops::CeilAlign(args.nValue, BASIC_BLOCK_SIZE_16);
    uint64_t kAlignedValue = ops::CeilAlign(args.kValue, BASIC_BLOCK_SIZE_8);
    MatMulV3TilingHelper::ResetBase(compileInfo, args, runInfo);
    uint64_t alignValue = runInfo.baseN;

    if (args.aDtypeSize == DATA_SIZE_FP32 && !args.isHf32) {
        if (kAlignedValue >= BASIC_BLOCK_SIZE_16 * compileInfo.aicNum * compileInfo.aicNum) {
            alignValue = FP32_MIN_BASE_BLOCK;
        } else {
            alignValue = BLOCK_BYTE_SIZE;
        }
    }
    // FP32场景MN小K大走SK模板提升精度
    bool isNdFormat = args.aFormat == ge::FORMAT_ND && args.bFormat == ge::FORMAT_ND;
    bool isFp32 = args.aType == ge::DT_FLOAT && args.bType == ge::DT_FLOAT && !args.isHf32;
    bool isSKSupport = mAlignedValue * nAlignedValue < FP32_MN_SHAPE_THRESHOLD &&
                       kAlignedValue >= BASIC_BLOCK_SIZE_16 * compileInfo.aicNum * compileInfo.aicNum && isFp32 &&
                       isNdFormat;
    // 判断mn是否需要已经能切32份及以上
    uint64_t mCnt = MathUtil::CeilDivision(args.mValue, alignValue);
    uint64_t nCnt = MathUtil::CeilDivision(args.nValue, alignValue);
    if (mCnt * nCnt > compileInfo.aicNum / NUM_TWO && !isSKSupport) {
        OP_LOGD(args.opName, "MatMulV3 tiling unenable state is DoStreamK mCnt[%lu], nCnt[%lu]", mCnt, nCnt);
        return false;
    }
    OP_LOGI(args.opName, "MatMulV3 tiling enable state is DoBasicApiSplitK.");
    return true;
}

// 候选baseM/baseN合法性校验：SK执行路径要求候选baseM/baseN不超过256，超出时DoOpTiling尾部的
// kValueMax(L0A容量/max(base)按baseK对齐)会退化为0使baseK=0，并在CalL1Tiling中触发除零，此类候选直接跳过；
// 3510在非32核(如28核)配置下GetRebalanceBlock可能给出base>256的候选，9201/9202沿用原行为直通
bool IsStreamKSKBaseValidDefault(const MatMulV3RunInfo& /* runInfoTmp */, const MatMulV3Args& /* args */)
{
    return true;
}

bool IsStreamKSKBaseValidDav3510(const MatMulV3RunInfo& runInfoTmp, const MatMulV3Args& args)
{
    uint64_t baseKAlignValue = !args.isATrans || args.isBTrans ? BASIC_BLOCK_SIZE_128 / args.aDtypeSize :
                                                                 BASIC_BLOCK_SIZE_16;
    uint64_t kValueMax = ops::FloorAlign(
        L0A_SIZE_2 / DB_SIZE / args.aDtypeSize / std::max(runInfoTmp.baseM, runInfoTmp.baseN), baseKAlignValue);
    return kValueMax != 0UL;
}

using IsStreamKSKBaseValidFunc = bool (*)(const MatMulV3RunInfo&, const MatMulV3Args&);

const static std::map<NpuArch, IsStreamKSKBaseValidFunc> IsStreamKSKBaseValidFuncMap = {
    {NpuArch::DAV_3510, IsStreamKSKBaseValidDav3510},
    {NpuArch::DAV_9201, IsStreamKSKBaseValidDefault},
    {NpuArch::DAV_9202, IsStreamKSKBaseValidDefault},
};

// 性能模型SK准入主体（3510非FP32与9201/9202共用）：按k切分候选逐一评估cube/mem边界收益并采纳最优配置
bool CheckStreamKSKPerfModelAdmission(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args,
                                      MatMulV3RunInfo& runInfo, const gert::TilingContext* context)
{
    MatMulV3TilingHelper::ResetBase(compileInfo, args, runInfo);
    MatMulV3TilingHelper::GetRebalanceBlock(compileInfo, args, runInfo, context);
    if (runInfo.cubeBoundParam <= runInfo.cubeBoundEdge) {
        OP_LOGD(args.opName, "Already CubeBound, no need streamK.");
        return false;
    }
    uint64_t mCnt = MathUtil::CeilDivision(args.mValue, runInfo.baseM);
    uint64_t nCnt = MathUtil::CeilDivision(args.nValue, runInfo.baseN);
    uint64_t singleCoreMaxRound = MathUtil::CeilDivision(mCnt * nCnt, compileInfo.aicNum);
    if (singleCoreMaxRound > 1) {
        OP_LOGD(args.opName, "SingleCore run multi round, no need streamK.");
        return false;
    }
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(args.opName, "platformInfo is null");
        return false;
    }
    MatMulV3RunInfo runInfoTmp;
    MatMulV3TilingHelper::ResetBase(compileInfo, args, runInfoTmp);
    runInfo.usedCoreNum = mCnt * nCnt;
    double bestPerf = GetMte2Perf(compileInfo, args, runInfo.usedCoreNum, runInfo, platformInfo);
    for (uint64_t kCnt = 2; kCnt <= compileInfo.aicNum; kCnt++) {
        runInfoTmp.usedCoreNum = compileInfo.aicNum / kCnt;
        MatMulV3TilingHelper::GetRebalanceBlock(compileInfo, args, runInfoTmp, context);
        uint64_t mCntTmp = MathUtil::CeilDivision(args.mValue, runInfoTmp.baseM);
        uint64_t nCntTmp = MathUtil::CeilDivision(args.nValue, runInfoTmp.baseN);
        uint64_t singleCoreMaxRoundTmp = MathUtil::CeilDivision(mCntTmp * nCntTmp, runInfoTmp.usedCoreNum);
        if (singleCoreMaxRoundTmp > 1) {
            continue;
        }
        auto baseValidIter = (IsStreamKSKBaseValidFuncMap.find(compileInfo.npuArch) ==
                              IsStreamKSKBaseValidFuncMap.end()) ?
                                 IsStreamKSKBaseValidDefault :
                                 IsStreamKSKBaseValidFuncMap.at(compileInfo.npuArch);
        if (!baseValidIter(runInfoTmp, args)) {
            continue;
        }
        double curMte2Perf = GetMte2Perf(compileInfo, args, mCntTmp * nCntTmp * kCnt, runInfoTmp, platformInfo);
        double negPerf = GetNegPerf(compileInfo, args, kCnt, mCntTmp * nCntTmp * kCnt, platformInfo);
        double curCubePerf = runInfoTmp.baseM * runInfoTmp.baseN *
                             ops::CeilAlign(MathUtil::CeilDivision(args.kValue, kCnt), BASIC_BLOCK_SIZE_16) /
                             (BASIC_BLOCK_SIZE_16 * BASIC_BLOCK_SIZE_16 * (BLOCK_BYTE_SIZE / args.aDtypeSize) *
                              MatMulV3TilingHelper::GetCoreFreq(platformInfo) *
                              TIME_CONVERT_UNIT); // 单核算力及单位换算
        bool isCubeBoundBetter = runInfoTmp.cubeBoundParam <= runInfoTmp.cubeBoundEdge &&
                                 (curCubePerf + negPerf) < bestPerf;
        bool isMemBoundBetter = runInfoTmp.cubeBoundParam > runInfoTmp.cubeBoundEdge &&
                                (curMte2Perf + negPerf) < bestPerf;
        if (isCubeBoundBetter || isMemBoundBetter) {
            runInfo.baseM = runInfoTmp.baseM;
            runInfo.baseN = runInfoTmp.baseN;
            runInfo.baseK = runInfoTmp.baseK;
            runInfo.tailInfo.kCnt = kCnt;
            runInfo.usedCoreNum = runInfoTmp.usedCoreNum * kCnt;
            if (isCubeBoundBetter) {
                bestPerf = curCubePerf + negPerf;
                if (mCntTmp * nCntTmp * kCnt == compileInfo.aicNum) {
                    break;
                }
            } else {
                bestPerf = curMte2Perf + negPerf;
            }
        }
    }
    return runInfo.tailInfo.kCnt > 1;
}

// DAV_3510的SK准入：k轴门限(纯FP32保精度与性能模型路径共用) + FP32走保精度保留路径(SK_SPLIT_K)，
// 其余(fp16/bf16/hf32，NZ与ND同路径)走性能模型准入
bool CheckStreamKSKTilingDav3510(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args,
                                 MatMulV3RunInfo& runInfo, const gert::TilingContext* context)
{
    // k轴门限：k过小时streamK切K无并行收益，仅引入fixup/结构开销，沿用主线硬边界准入
    constexpr uint64_t STREAM_K_MIN_K_THRESHOLD = 8192UL;
    if (ops::CeilAlign(static_cast<uint64_t>(args.kValue), BASIC_BLOCK_SIZE_256) <
        std::max(STREAM_K_MIN_K_THRESHOLD, compileInfo.aicNum * BASIC_BLOCK_K_256_BYTE) / args.aDtypeSize) {
        OP_LOGD(args.opName, "MatMulV3 tiling unenable state is DoStreamK value[%lu]", args.kValue);
        return false;
    }
    bool isFp32 = args.aType == ge::DT_FLOAT && args.bType == ge::DT_FLOAT && !args.isHf32;
    if (isFp32) {
        return CheckStreamKSKFp32SplitKSupport(compileInfo, args, runInfo);
    }
    return CheckStreamKSKPerfModelAdmission(compileInfo, args, runInfo, context);
}

bool CheckStreamKSKTilingDav9201(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args,
                                 MatMulV3RunInfo& runInfo, const gert::TilingContext* context)
{
    return CheckStreamKSKPerfModelAdmission(compileInfo, args, runInfo, context);
}

using CheckStreamKSKTilingFunc = bool (*)(const MatmulV3CompileInfo&, const MatMulV3Args&, MatMulV3RunInfo&,
                                          const gert::TilingContext*);

const static std::map<NpuArch, CheckStreamKSKTilingFunc> CheckStreamKSKTilingFuncMap = {
    {NpuArch::DAV_3510, CheckStreamKSKTilingDav3510},
    {NpuArch::DAV_9201, CheckStreamKSKTilingDav9201},
    {NpuArch::DAV_9202, CheckStreamKSKTilingDav9201},
};

// ------------------------------ GetL0C2OutFlag -------------------------------------------//
MatMulV3L0C2Out GetL0C2OutFlagDefault(const MatMulV3Args& /* args */) { return MatMulV3L0C2Out::ON_THE_FLY; }

MatMulV3L0C2Out GetL0C2OutFlagDav3510(const MatMulV3Args& args)
{
    if (args.nValue > BASIC_BLOCK_SIZE_64 && args.nValue % BASIC_BLOCK_SIZE_16 != 0 && args.mValue > NUM_TWO &&
        args.mValue * args.nValue >= BASIC_BLOCK_SIZE_256) {
        return MatMulV3L0C2Out::ND_FIXPIPE_1_2;
    }
    return MatMulV3L0C2Out::ON_THE_FLY;
}

using GetL0C2OutFlagFunc = MatMulV3L0C2Out (*)(const MatMulV3Args&);

const static std::map<NpuArch, GetL0C2OutFlagFunc> GetL0C2OutFlagFuncMap = {
    {NpuArch::DAV_3510, GetL0C2OutFlagDav3510},
};

// ND_FIXPIPE_1_2场景partial在workspace按32元素(128B)对齐宽度读写，计算n方向cnt个tile的对齐宽度总和
static uint64_t GetPaddedFixWidth(uint64_t nValue, uint64_t nCnt)
{
    uint64_t baseN = ops::CeilAlign(MathUtil::CeilDivision(nValue, nCnt), BASIC_BLOCK_SIZE_16);
    // baseN向上对齐后满块数可能少于nCnt-1，按实际块数推导，防止tailN无符号下溢
    uint64_t fullCnt = MathUtil::CeilDivision(nValue, baseN) - 1;
    uint64_t tailN = nValue - fullCnt * baseN;
    return fullCnt * ops::CeilAlign(baseN, BLOCK_BYTE_SIZE) + ops::CeilAlign(tailN, BLOCK_BYTE_SIZE);
}

// 减半base的盈亏门禁：halveN为true表示nCnt*2(减半baseN，A矩阵被重复读的份数翻倍)，false对应mCnt*2(B矩阵)
// 收益 = fixup流量减少(每份partial写m*n*4B、AIV按fixWidth宽度读m*fixWidth*4B，份数为kCnt)
// 代价 = 被重复读操作数新增一份流量，一律按HBM全价计：实测L2对跨tile重复读的吸收在21%/35%/81%占用率下均失效
// ND_FIXPIPE_1_2路径另存在流量模型未覆盖的按tile结构代价，要求2倍收益裕量
static bool IsHalveBaseProfitable(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args, uint64_t totalMNCnt,
                                  uint64_t mCnt, uint64_t nCnt, bool halveN, bool isNdFixpipe)
{
    uint64_t kCntOld = ops::FloorDiv(compileInfo.aicNum, totalMNCnt);
    uint64_t kCntNew = ops::FloorDiv(compileInfo.aicNum, totalMNCnt * NUM_TWO);
    if (kCntNew == 0UL || kCntNew >= kCntOld) {
        return false;
    }
    // fp32场景singleCoreK翻倍跨过精度切K阈值会切换到SK_SPLIT_K原子加路径
    if (args.aDtypeSize == DATA_SIZE_FP32 && !args.isHf32 &&
        MathUtil::CeilDivision(args.kValue, kCntNew) >= compileInfo.aicNum * BASIC_BLOCK_SIZE_16) {
        return false;
    }
    uint64_t fixWidthOld = args.nValue;
    uint64_t fixWidthNew = args.nValue;
    if (isNdFixpipe) {
        fixWidthOld = GetPaddedFixWidth(args.nValue, nCnt);
        fixWidthNew = halveN ? GetPaddedFixWidth(args.nValue, nCnt * NUM_TWO) : fixWidthOld;
    }
    uint64_t fixupOld = kCntOld * (args.nValue + fixWidthOld);
    uint64_t fixupNew = kCntNew * (args.nValue + fixWidthNew);
    if (fixupNew >= fixupOld) {
        return false; // ND_FIXPIPE_1_2对齐padding侵蚀后fixup流量不降反升
    }
    uint64_t saveBytes = NUM_FOUR * args.mValue * (fixupOld - fixupNew);
    uint64_t dupBytes = halveN ? nCnt * args.mValue * args.kValue * args.aDtypeSize :
                                 mCnt * args.nValue * args.kValue * args.bDtypeSize;
    if (isNdFixpipe) {
        return saveBytes > dupBytes * NUM_TWO;
    }
    return saveBytes > dupBytes;
}

void DoSkOpTilingDefault(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args, MatMulV3RunInfo& runInfo)
{
    uint64_t singleCoreKAlignValue = !args.isATrans || args.isBTrans ? BASIC_BLOCK_SIZE_256 / args.aDtypeSize :
                                                                       BASIC_BLOCK_SIZE_16;
    uint64_t mCnt = MathUtil::CeilDivision(args.mValue, runInfo.baseM);
    uint64_t nCnt = MathUtil::CeilDivision(args.nValue, runInfo.baseN);
    if (mCnt > compileInfo.aicNum / NUM_THREE && mCnt < compileInfo.aicNum / NUM_TWO) {
        mCnt = compileInfo.aicNum / NUM_TWO;
    }
    if (nCnt > compileInfo.aicNum / NUM_THREE && nCnt < compileInfo.aicNum / NUM_TWO) {
        nCnt = compileInfo.aicNum / NUM_TWO;
    }
    uint64_t totalMNCnt = mCnt * nCnt;
    if (totalMNCnt <= NUM_TWO) {
        // kCnt过大时减半baseM/baseN中更大者以降低k方向切分，base减半后仍需16对齐，base已为16时无法再减半；
        // 减半会使另一操作数被重复读的份数翻倍，需通过盈亏门禁避免新增MTE2流量超过fixup节省流量
        uint64_t baseMTmp = ops::CeilAlign(MathUtil::CeilDivision(args.mValue, mCnt), BASIC_BLOCK_SIZE_16);
        uint64_t baseNTmp = ops::CeilAlign(MathUtil::CeilDivision(args.nValue, nCnt), BASIC_BLOCK_SIZE_16);
        bool isNdFixpipe = args.nValue > BASIC_BLOCK_SIZE_64 && args.nValue % BASIC_BLOCK_SIZE_16 != 0 &&
                           args.mValue > NUM_TWO && args.mValue * args.nValue >= BASIC_BLOCK_SIZE_256;
        if (baseNTmp >= baseMTmp && baseNTmp > BASIC_BLOCK_SIZE_16) {
            if (IsHalveBaseProfitable(compileInfo, args, totalMNCnt, mCnt, nCnt, true, isNdFixpipe)) {
                nCnt *= NUM_TWO;
            }
        } else if (baseMTmp > BASIC_BLOCK_SIZE_16) {
            if (IsHalveBaseProfitable(compileInfo, args, totalMNCnt, mCnt, nCnt, false, isNdFixpipe)) {
                mCnt *= NUM_TWO;
            }
        }
        totalMNCnt = mCnt * nCnt;
    }

    // m、n、k轴在对齐基础上尽量均分
    runInfo.baseM = ops::CeilAlign(MathUtil::CeilDivision(args.mValue, mCnt), BASIC_BLOCK_SIZE_16);
    runInfo.baseN = ops::CeilAlign(MathUtil::CeilDivision(args.nValue, nCnt), BASIC_BLOCK_SIZE_16);
    runInfo.tailInfo.kCnt = ops::FloorDiv(compileInfo.aicNum, totalMNCnt);
    runInfo.singleCoreK = MathUtil::CeilDivision(args.kValue, runInfo.tailInfo.kCnt);
}

void DoSkOpTilingDav9201(const MatmulV3CompileInfo& /* compileInfo */, const MatMulV3Args& args,
                         MatMulV3RunInfo& runInfo)
{
    uint64_t singleCoreKAlignValue = !args.isATrans || args.isBTrans ? BASIC_BLOCK_SIZE_256 / args.aDtypeSize :
                                                                       BASIC_BLOCK_SIZE_16;
    runInfo.singleCoreK = ops::CeilAlign(MathUtil::CeilDivision(args.kValue, runInfo.tailInfo.kCnt),
                                         singleCoreKAlignValue);
    runInfo.tailInfo.kCnt = MathUtil::CeilDivision(args.kValue, runInfo.singleCoreK);
}

using DoSkOpTilingFunc = void (*)(const MatmulV3CompileInfo&, const MatMulV3Args&, MatMulV3RunInfo&);

const static std::map<NpuArch, DoSkOpTilingFunc> DoSkOpTilingFuncMap = {
    {NpuArch::DAV_3510, DoSkOpTilingDav9201}, // 3510性能模型准入与9201同构，复用薄实现
    {NpuArch::DAV_9201, DoSkOpTilingDav9201},
    {NpuArch::DAV_9202, DoSkOpTilingDav9201},
};

} // namespace

namespace optiling {
namespace matmul_v3_advanced {
using namespace strategy;
MM_REGISTER_TILING_TEMPLATE(MatMulV3, MatMulV3BasicStreamKTiling, DAV_3510, BASIC_STREAM_K);
MM_REGISTER_TILING_TEMPLATE(MatMulV3, MatMulV3BasicStreamKTiling, DAV_9201, BASIC_STREAM_K);
MM_REGISTER_TILING_TEMPLATE(MatMulV3, MatMulV3BasicStreamKTiling, DAV_9202, BASIC_STREAM_K);

bool MatMulV3BasicStreamKTiling::CheckStreamKSKTiling()
{
    auto iter = (CheckStreamKSKTilingFuncMap.find(compileInfo_.npuArch) == CheckStreamKSKTilingFuncMap.end()) ?
                    CheckStreamKSKTilingDefault :
                    CheckStreamKSKTilingFuncMap.at(compileInfo_.npuArch);
    return iter(compileInfo_, args_, runInfo_, context_);
}

bool MatMulV3BasicStreamKTiling::CheckStreamKDPSKTiling()
{
    auto iter = (CheckStreamKDPSKTilingFuncMap.find(compileInfo_.npuArch) == CheckStreamKDPSKTilingFuncMap.end()) ?
                    CheckStreamKDPSKTilingDefault :
                    CheckStreamKDPSKTilingFuncMap.at(compileInfo_.npuArch);
    return iter(compileInfo_, args_, runInfo_);
}

MatMulV3L0C2Out MatMulV3BasicStreamKTiling::GetL0C2OutFlag() const
{
    auto iter = (GetL0C2OutFlagFuncMap.find(compileInfo_.npuArch) == GetL0C2OutFlagFuncMap.end()) ?
                    GetL0C2OutFlagDefault :
                    GetL0C2OutFlagFuncMap.at(compileInfo_.npuArch);
    return iter(args_);
}

void MatMulV3BasicStreamKTiling::DoSkOpTiling()
{
    OP_LOGI(args_.opName, "MatMulV3 tiling enable state is DoStreamK.");
    // 性能模型准入已选定配置(tailInfo.kCnt>=2)走薄实现；
    // 纯FP32保精度/非ND等保留路径准入(kCnt未定)走Default启发式生成配置
    auto iter = (runInfo_.tailInfo.kCnt > 1UL &&
                 DoSkOpTilingFuncMap.find(compileInfo_.npuArch) != DoSkOpTilingFuncMap.end()) ?
                    DoSkOpTilingFuncMap.at(compileInfo_.npuArch) :
                    DoSkOpTilingDefault;
    return iter(compileInfo_, args_, runInfo_);
}

bool MatMulV3BasicStreamKTiling::IsCapable()
{
    // batch一致性控制，当开关等级为2或3时，拒绝切k模板，达到强一致性和batch一致性
    OP_LOGD(args_.opName, "deterministic_level=%d", context_->GetDeterministicLevel());
    if (context_->GetDeterministicLevel() > 1) {
        return false;
    }
    if (args_.aFormat != ge::FORMAT_ND) {
        OP_LOGD(args_.opName, "ND is the only supported format for tensor_a in basic api");
        return false;
    }
    if (MatMulV3TilingHelper::IsSelfNonContiguous(context_)) {
        OP_LOGD(args_.opName, "NonContiguous self does not support StreamK");
        return false;
    }
    uint64_t aivToAicRatio = MatMulV3TilingHelper::GetAivToAicRatio(compileInfo_);
    if (compileInfo_.aivNum != compileInfo_.aicNum * aivToAicRatio) {
        OP_LOGD(args_.opName, "streamk only support aivNum == aicNum * %lu", aivToAicRatio);
        return false;
    }
    return CheckStreamKSKTiling() || CheckStreamKDPSKTiling();
}

ge::graphStatus MatMulV3BasicStreamKTiling::DoOpTiling()
{
    mCnt_ = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
    nCnt_ = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
    totalMNCnt_ = mCnt_ * nCnt_;
    if (totalMNCnt_ <= compileInfo_.aicNum / NUM_TWO) {
        DoSkOpTiling();
        l0C2Out_ = GetL0C2OutFlag();
    } else {
        runInfo_.tailInfo.kCnt = compileInfo_.aicNum / (totalMNCnt_ % compileInfo_.aicNum);
        uint64_t skSingleCoreK = MathUtil::CeilDivision(args_.kValue, runInfo_.tailInfo.kCnt);
        runInfo_.tailInfo.kCnt = MathUtil::CeilDivision(args_.kValue, skSingleCoreK);
        runInfo_.singleCoreK = skSingleCoreK;
    }
    if (args_.bFormat != ge::FORMAT_ND) {
        if (args_.bDtypeSize == DATA_SIZE_FP16 || (args_.bDtypeSize == DATA_SIZE_FP32 && !args_.isBTrans)) {
            runInfo_.singleCoreK = ops::CeilAlign(runInfo_.singleCoreK, BASIC_BLOCK_SIZE_16);
        } else {
            runInfo_.singleCoreK = ops::CeilAlign(runInfo_.singleCoreK, BASIC_BLOCK_SIZE_16 / NUM_TWO);
        }
    }
    if (compileInfo_.npuArch != NpuArch::DAV_9201 && compileInfo_.npuArch != NpuArch::DAV_9202) {
        uint64_t baseKAlignValue = !args_.isATrans || args_.isBTrans ? BASIC_BLOCK_SIZE_128 / args_.aDtypeSize :
                                                                       BASIC_BLOCK_SIZE_16;
        uint64_t kValueMax = ops::FloorAlign(
            L0A_SIZE_2 / DB_SIZE / args_.aDtypeSize / std::max(runInfo_.baseM, runInfo_.baseN), baseKAlignValue);
        runInfo_.baseK = std::min(runInfo_.singleCoreK, kValueMax);
    }
    MatMulV3TilingHelper::CalL1Tiling(compileInfo_, args_, runInfo_);
    // depthb1 is less than deptha1
    if (runInfo_.baseM == runInfo_.baseN && runInfo_.depthB1 == runInfo_.depthA1 * NUM_TWO) {
        runInfo_.depthA1 = runInfo_.depthA1 * NUM_TWO;
        runInfo_.depthB1 = runInfo_.depthB1 / NUM_TWO;
        runInfo_.stepKb = runInfo_.depthB1 / DB_SIZE;
        runInfo_.stepKa = runInfo_.depthA1 / DB_SIZE;
    }
    if ((totalMNCnt_ > compileInfo_.aicNum) && args_.hasBias) {
        runInfo_.stepKb = NUM_THREE; // reserve L1 space for bias
        runInfo_.stepKa = NUM_THREE;
    }
    bool isFusedMatMul = std::string_view(context_->GetNodeType()) == "FusedMatMul";
    bool isFusedMatMulBlaze = false;
    if (isFusedMatMul) {
        auto attrs = context_->GetAttrs();
        if (attrs != nullptr) {
            constexpr size_t FUSED_MATMUL_OP_TYPE_IDX = 3UL;
            std::string opType = attrs->GetAttrPointer<char>(FUSED_MATMUL_OP_TYPE_IDX);
            isFusedMatMulBlaze = ((opType == "relu" || opType == "") && compileInfo_.npuArch != NpuArch::DAV_RESV);
        }
    }
    // fusedMatMul(empty + relu) checkout to tensor api
    basicApi_ = (isFusedMatMul && !isFusedMatMulBlaze) || args_.isAvoidTensorApi;
    isSplitK_ = MatMulV3TilingHelper::IsSplitSingleCoreK(compileInfo_, args_, !basicApi_ && !isFusedMatMul,
                                                         runInfo_.singleCoreK);
    if (isSplitK_) {
        model_ = MatMulV3Model::SK_SPLIT_K;
        runInfo_.splitKBase = MatMulV3TilingHelper::SelectSplitKBase(context_, args_.mValue, args_.nValue,
                                                                     runInfo_.singleCoreK, runInfo_.baseK);
    }
    return ge::GRAPH_SUCCESS;
}

std::vector<size_t> MatMulV3BasicStreamKTiling::GetWorkspaceSize() const
{
    size_t workspaceSize = compileInfo_.aicNum * runInfo_.baseM *
                           ops::CeilAlign(runInfo_.baseN, FIXPIPE_N_ALIGN_ELEMENTS) * DATA_SIZE_FP32;
    // 9201/9202申请workspacesize时先恢复额外RPC申请
    if (compileInfo_.npuArch == NpuArch::DAV_9201 || compileInfo_.npuArch == NpuArch::DAV_9202) {
        workspaceSize += RPC_WORKSIZE * MB_SIZE;
    }
    OP_LOGI(args_.opName, "MatMulV3 tiling workspace size is %lu", workspaceSize);
    return {workspaceSize};
}

uint64_t MatMulV3BasicStreamKTiling::GetTilingKey() const
{
    MatMulV3TilingKey tmp = MatMulV3TilingKey();
    MatMulV3TilingKey& tilingKey = tilingKeyObj == nullptr ? tmp : *tilingKeyObj;
    return tilingKey.SetTrans(args_.isATrans, args_.isBTrans)
        .SetModel(model_)
        .SetL0C2Out(l0C2Out_)
        .SetApiLevel(basicApi_ ? MatMulV3ApiLevel::BASIC_LEVEL : MatMulV3ApiLevel::TENSOR_LEVEL)
        .GetTilingKey();
}

ge::graphStatus MatMulV3BasicStreamKTiling::GetTilingData(TilingResult& tiling) const
{
    return GetTilingDataImpl<MatMulV3BasicTilingData>(tiling);
}
} // namespace matmul_v3_advanced
} // namespace optiling
