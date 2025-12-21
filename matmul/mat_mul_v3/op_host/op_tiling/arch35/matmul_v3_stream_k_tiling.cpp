/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/* !
 * \file matmul_v3_stream_k_tiling.cc
 * \brief
 */

#include "matmul_v3_stream_k_tiling.h"
#include "matmul_tiling_registry.h"
#include "matmul_v3_tiling_strategy.h"
#include "matmul/common/op_host/math_util.h"

using Ops::NN::MathUtil;
namespace {
using namespace optiling;
using namespace optiling::matmul_v3_advanced;

// ------------------------------ CheckStreamKSKTiling -------------------------------------------//
bool CheckStreamKSKTilingDefault(const MatmulV3CompileInfo & /* compileInfo */, const MatMulV3Args & /* args */)
{
    return false;
}

bool CheckStreamKSKTiling91095(const MatmulV3CompileInfo &compileInfo, const MatMulV3Args &args)
{
    constexpr uint64_t STREAM_K_MIN_K_THRESHOLD = 8192UL;
    // 判断k轴是否大于32*256 / DtypeSize_, 小于就不走stream-k
    if (ops::CeilAlign(static_cast<uint64_t>(args.kValue), BASIC_BLOCK_SIZE_256) <
        std::max(STREAM_K_MIN_K_THRESHOLD, compileInfo.aicNum * BASIC_BLOCK_K_256_BYTE) / args.aDtypeSize) {
        OP_LOGD(args.opName, "MatMulV3 tiling unenable state is DoStreamK value[%lu]", args.kValue);
        return false;
    }

    uint64_t alignValue = BASIC_BLOCK_SIZE_256;
    if (args.aDtypeSize == DATA_SIZE_FP32 && !args.isHf32) {
        alignValue = BLOCK_BYTE_SIZE;  // 如果是Fp32 基本块判断要用32
    }
    // 判断mn是否需要已经能切32份及以上
    uint64_t mCnt = MathUtil::CeilDivision(args.mValue, alignValue);
    uint64_t nCnt = MathUtil::CeilDivision(args.nValue, alignValue);
    if (mCnt * nCnt > compileInfo.aicNum / NUM_TWO) {
        OP_LOGD(args.opName, "MatMulV3 tiling unenable state is DoStreamK mCnt[%lu], nCnt[%lu]", mCnt, nCnt);
        return false;
    }
    return true;
}

using CheckStreamKSKTilingFunc = bool (*)(const MatmulV3CompileInfo &, const MatMulV3Args &);

const static std::map<platform_ascendc::SocVersion, CheckStreamKSKTilingFunc> CheckStreamKSKTilingFuncMap = {
    {platform_ascendc::SocVersion::ASCEND910_95, CheckStreamKSKTiling91095},
};

// ------------------------------ CheckStreamKDPSKTiling -------------------------------------------//
bool CheckStreamKDPSKTilingDefault(const MatmulV3CompileInfo & /* compileInfo */, const MatMulV3Args & /* args */)
{
    return false;
}

bool CheckStreamKDPSKTiling91095(const MatmulV3CompileInfo &compileInfo, const MatMulV3Args &args)
{
    constexpr uint64_t STREAM_K_MIN_K_THRESHOLD = 8192UL;
    // 如果k轴小于32*256/DtypeSize_ 或 mn轴不是256对齐 或 输入是fp32类型，不走stream-k-dpsk
    if (args.mValue % BASIC_BLOCK_SIZE_256 != 0UL || args.nValue % BASIC_BLOCK_SIZE_256 != 0UL ||
        args.kValue <
            std::max(STREAM_K_MIN_K_THRESHOLD, compileInfo.aicNum * BASIC_BLOCK_K_256_BYTE) / args.aDtypeSize ||
        (args.aDtypeSize == DATA_SIZE_FP32 && !args.isHf32)) {
        return false;
    }
    // 如果mn用256切分的份数小于核数 或者 取余核数为0或大于一半的核数，则不使用stream-k-dpsk
    uint64_t mCnt = MathUtil::CeilDivision(args.mValue, BASIC_BLOCK_SIZE_256);
    uint64_t nCnt = MathUtil::CeilDivision(args.nValue, BASIC_BLOCK_SIZE_256);
    uint64_t totalMNCnt = mCnt * nCnt;
    return (totalMNCnt >= compileInfo.aicNum) && (totalMNCnt % compileInfo.aicNum != 0UL) &&
           (totalMNCnt % compileInfo.aicNum <= compileInfo.aicNum / NUM_TWO);
}

using CheckStreamKDPSKTilingFunc = bool (*)(const MatmulV3CompileInfo &, const MatMulV3Args &);

const static std::map<platform_ascendc::SocVersion, CheckStreamKDPSKTilingFunc> CheckStreamKDPSKTilingFuncMap = {
    {platform_ascendc::SocVersion::ASCEND910_95, CheckStreamKDPSKTiling91095},
};

// ------------------------------ GetL0C2OutFlag -------------------------------------------//
MatMulV3L0C2Out GetL0C2OutFlagDefault(const MatMulV3Args & /* args */)
{
    return MatMulV3L0C2Out::ON_THE_FLY;
}

MatMulV3L0C2Out GetL0C2OutFlag91095(const MatMulV3Args &args)
{
    if (args.nValue > BASIC_BLOCK_SIZE_64 && args.nValue % BASIC_BLOCK_SIZE_16 != 0 && args.mValue > NUM_TWO &&
        args.mValue * args.nValue >= BASIC_BLOCK_SIZE_256) {
        return MatMulV3L0C2Out::ND_FIXPIPE_1_2;
    }
    return MatMulV3L0C2Out::ON_THE_FLY;
}

using GetL0C2OutFlagFunc = MatMulV3L0C2Out (*)(const MatMulV3Args &);

const static std::map<platform_ascendc::SocVersion, GetL0C2OutFlagFunc> GetL0C2OutFlagFuncMap = {
    {platform_ascendc::SocVersion::ASCEND910_95, GetL0C2OutFlag91095},
};
}  // namespace

namespace optiling {
namespace matmul_v3_advanced {
using namespace strategy;

MM_REGISTER_TILING_TEMPLATE(MatMulV3, MatMulV3StreamKTiling, ASCEND910_95, STREAM_K);

constexpr uint64_t STREAM_K_MAX_K_THRESHOLD = 2000000UL;

bool MatMulV3StreamKTiling::CheckStreamKSKTiling() const
{
    auto iter = (CheckStreamKSKTilingFuncMap.find(compileInfo_.socVersion) == CheckStreamKSKTilingFuncMap.end())
                    ? CheckStreamKSKTilingDefault
                    : CheckStreamKSKTilingFuncMap.at(compileInfo_.socVersion);
    return iter(compileInfo_, args_);
}

bool MatMulV3StreamKTiling::CheckStreamKDPSKTiling() const
{
    auto iter = (CheckStreamKDPSKTilingFuncMap.find(compileInfo_.socVersion) == CheckStreamKDPSKTilingFuncMap.end())
                    ? CheckStreamKDPSKTilingDefault
                    : CheckStreamKDPSKTilingFuncMap.at(compileInfo_.socVersion);
    return iter(compileInfo_, args_);
}

MatMulV3L0C2Out MatMulV3StreamKTiling::GetL0C2OutFlag() const
{
    auto iter = (GetL0C2OutFlagFuncMap.find(compileInfo_.socVersion) == GetL0C2OutFlagFuncMap.end())
                    ? GetL0C2OutFlagDefault
                    : GetL0C2OutFlagFuncMap.at(compileInfo_.socVersion);
    return iter(args_);
}

bool MatMulV3StreamKTiling::IsCapable()
{
    // 如果dtype是fp32且k轴大于200万 则走基础模板来保证fp32的精度
    if (args_.aDtypeSize == DATA_SIZE_FP32 && !args_.isHf32 &&
        static_cast<uint64_t>(args_.kValue) > STREAM_K_MAX_K_THRESHOLD) {
        OP_LOGD(args_.opName, "Due to the requirement of binary accumulation, current fp32 does not support StreamK");
        return false;
    }
    return (CheckStreamKSKTiling() || CheckStreamKDPSKTiling());
}

ge::graphStatus MatMulV3StreamKTiling::DoOpTiling()
{
    OP_LOGI(args_.opName, "MatMulV3 tiling enable state is DoSplitK.");
    MatMulV3TilingHelper::ResetBase(compileInfo_, args_, runInfo_);
    mCnt_ = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
    nCnt_ = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
    totalMNCnt_ = mCnt_ * nCnt_;
    // 首地址256B对齐
    uint64_t singleCoreKAlignValue =
        !args_.isATrans || args_.isBTrans ? BASIC_BLOCK_SIZE_256 / args_.aDtypeSize : BASIC_BLOCK_SIZE_16;
    if (totalMNCnt_ <= compileInfo_.aicNum / NUM_TWO) {
        if (mCnt_ > compileInfo_.aicNum / NUM_THREE && mCnt_ < compileInfo_.aicNum / NUM_TWO) {
            mCnt_ = compileInfo_.aicNum / NUM_TWO;
        }
        if (nCnt_ > compileInfo_.aicNum / NUM_THREE && nCnt_ < compileInfo_.aicNum / NUM_TWO) {
            nCnt_ = compileInfo_.aicNum / NUM_TWO;
        }
        totalMNCnt_ = mCnt_ * nCnt_;
        runInfo_.tailInfo.kCnt = ops::FloorDiv(compileInfo_.aicNum, totalMNCnt_);
        OP_LOGI(args_.opName, "MatMulV3 tiling enable state is DoStreamK.");
        // m、n、k轴在对齐基础上尽量均分
        runInfo_.baseM = ops::CeilAlign(MathUtil::CeilDivision(args_.mValue, mCnt_), BASIC_BLOCK_SIZE_16);
        runInfo_.singleCoreM = std::min(runInfo_.baseM, args_.mValue);
        runInfo_.baseN = ops::CeilAlign(MathUtil::CeilDivision(args_.nValue, nCnt_), BASIC_BLOCK_SIZE_16);
        runInfo_.singleCoreN = std::min(runInfo_.baseN, args_.nValue);
        runInfo_.singleCoreK =
            ops::CeilAlign(MathUtil::CeilDivision(args_.kValue, runInfo_.tailInfo.kCnt), singleCoreKAlignValue);
        runInfo_.tailInfo.kCnt = MathUtil::CeilDivision(args_.kValue, runInfo_.singleCoreK);
        // baseK 128B对齐 ，step为偶数就可以让一次burstLen为256B对齐
        uint64_t baseKAlignValue =
            !args_.isATrans || args_.isBTrans ? BASIC_BLOCK_SIZE_128 / args_.aDtypeSize : BASIC_BLOCK_SIZE_16;
        uint64_t kValueMax = ops::FloorAlign(
            L0A_SIZE_2 / DB_SIZE / args_.aDtypeSize / std::max(runInfo_.baseM, runInfo_.baseN), baseKAlignValue);
        runInfo_.baseK = std::min(runInfo_.singleCoreK, kValueMax);
        l0C2Out_ = GetL0C2OutFlag();
    } else {
        runInfo_.tailInfo.kCnt = compileInfo_.aicNum / (totalMNCnt_ % compileInfo_.aicNum);
        uint64_t skSingleCoreK =
            ops::CeilAlign(MathUtil::CeilDivision(args_.kValue, runInfo_.tailInfo.kCnt), singleCoreKAlignValue);
        runInfo_.tailInfo.kCnt = MathUtil::CeilDivision(args_.kValue, skSingleCoreK);
    }
    MatMulV3TilingHelper::CalL1Tiling(compileInfo_, args_, runInfo_);
    // depthb1 is less than deptha1
    if (runInfo_.baseM == runInfo_.baseN && runInfo_.depthB1 == runInfo_.depthA1 * NUM_TWO) {
        runInfo_.depthA1 = runInfo_.depthA1 * NUM_TWO;
        runInfo_.depthB1 = runInfo_.depthB1 / NUM_TWO;
        runInfo_.stepKb = runInfo_.depthB1 / DB_SIZE;
        runInfo_.stepKa = runInfo_.depthA1 / DB_SIZE;
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t MatMulV3StreamKTiling::GetTilingKey() const
{
    MatMulV3TilingKey tmp = MatMulV3TilingKey();
    MatMulV3TilingKey& tilingKey = tilingKeyObj == nullptr ? tmp : *tilingKeyObj;
    return tilingKey.SetTrans(args_.isATrans, args_.isBTrans)
        .SetModel(MatMulV3Model::STREAM_K)
        .SetL0C2Out(l0C2Out_)
        .GetTilingKey();
}

std::vector<size_t> MatMulV3StreamKTiling::GetWorkspaceSize() const
{
    size_t workspaceSize =
        compileInfo_.aicNum * BASIC_BLOCK_SIZE_256 * BASIC_BLOCK_SIZE_256 * DATA_SIZE_FP32 + RPC_WORKSIZE * MB_SIZE;
    OP_LOGI(args_.opName, "MatMulV3 tiling workspace size is %lu", workspaceSize);
    return { workspaceSize };
}
} // namespace matmul_v3
} // namespace optiling
