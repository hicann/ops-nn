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
 * \file matmul_v3_tiling_helper.h
 * \brief
 */
#pragma once

#include "matmul_v3_common_advanced.h"
#include "matmul_v3_compile_info_advanced.h"
#include "matmul_v3_tiling_key.h"
#include "../../../op_kernel/arch35/mat_mul_v3_tiling_key_public.h"

namespace optiling {
namespace matmul_v3_advanced {
using StrideIndexPairs = std::vector<std::pair<int64_t, std::pair<int64_t, int64_t>>>;
bool IsMatMulTiling(const MatMulV3Args& args, NpuArch npuArch);
// 判断是否为DAV_3510系列(含9201/9202兼容版本)，语义与op_api层cube_util.h的IsNpuArch3510Series一致
inline bool IsNpuArch3510Series(NpuArch npuArch)
{
    return npuArch == NpuArch::DAV_3510 || npuArch == NpuArch::DAV_9201 || npuArch == NpuArch::DAV_9202;
}

class MatMulV3TilingHelper {
public:
    static void ResetBase(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args, MatMulV3RunInfo& runInfo);
    // cubeBoundEdge收缩系数：按芯片规格定义，3510为0.85，9201/9202为0.95
    static double GetCubeBoundRatio(const MatmulV3CompileInfo& compileInfo);
    // UB双缓冲判定：9201/9202按双缓冲空间(2倍)判断，3510按单缓冲判断
    static uint64_t CalcUbDB(const MatmulV3CompileInfo& compileInfo, uint64_t baseM, uint64_t baseN);
    // streamK的AIV/AIC核数配比：3510为2:1，9201/9202为1:1
    static uint64_t GetAivToAicRatio(const MatmulV3CompileInfo& compileInfo);
    static void CalL1Tiling(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args, MatMulV3RunInfo& runInfo);
    static uint64_t SelectSplitKBase(const gert::TilingContext* context, uint64_t mValue, uint64_t nValue,
                                     uint64_t kValue, uint64_t baseK);
    static MatMulV3L0C2Out GetL0C2Out(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args,
                                      MatMulV3RunInfo& runInfo, bool isSplitK = false,
                                      fe::PlatFormInfos* platformInfo = nullptr);
    static bool IsSplitSingleCoreK(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args, bool isTensorApi,
                                   uint64_t singleCoreK = 0);
    static uint64_t GetStepSmallK(bool isBL1FullLoad, const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args,
                                  MatMulV3RunInfo& runInfo);
    static void AdjustBL1TilingCommon(uint64_t aBatchDimAll, const MatmulV3CompileInfo& compileInfo,
                                      const MatMulV3Args& args, MatMulV3RunInfo& runInfo);
    static void AdjustAL1TilingCommon(uint64_t bBatchDimAll, const MatmulV3CompileInfo& compileInfo,
                                      const MatMulV3Args& args, MatMulV3RunInfo& runInfo);
    static void ResetFullLoadLoadBalance(MatMulV3RunInfo& runInfo);
    static bool IsSelfNonContiguous(const gert::TilingContext* context);
    static void GetRebalanceBlock(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args,
                                  MatMulV3RunInfo& runInfo, const gert::TilingContext* context);
    static double GetHbmBW(int64_t usedCoreNum, const MatmulV3CompileInfo& compileInfo,
                           fe::PlatFormInfos* platformInfo);
    static double GetL2BW(int64_t usedCoreNum, const MatmulV3CompileInfo& compileInfo, fe::PlatFormInfos* platformInfo);
    static double GetCoreFreq(fe::PlatFormInfos* platformInfo);
    static bool IsTransposeNonContiguous(const gert::TilingContext* context, uint64_t idx);
    static bool IsContiguousStride(StrideIndexPairs& strideIndexPairs);
    struct SliceInfo {
        uint32_t sliceM;
        uint32_t srcNdStride;
        uint32_t rowStride;
    };
    static SliceInfo GetSliceInfo(const gert::TilingContext* context, uint32_t baseM, bool isATrans);
};
} // namespace matmul_v3_advanced
} // namespace optiling
