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
 * \file tiling_cost_evaluator.h
 * \brief
 */
#ifndef TILING_COST_EVALUATOR_H
#define TILING_COST_EVALUATOR_H

#include <cstdint>
#include <string>
#include "../quant_batch_matmul_v3_compile_info.h"
#include "platform/platform_infos_def.h"
#include "../../../op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"

namespace optiling {

struct QuantBatchMatmulInfo;

struct AicTimeCost {
    double mmadTotalTime;
    double mmadLastRoundMaxtime;

    double mte1TotalTime;
    double mte1LastRoundMaxtime;

    double mte2TotalTime;
    double mte2LastRoundMaxtime;

    double fixpipeTotalTime;
    double fixpipeLastRoundMaxtime;

    double totalTime;
};

struct StreamKTimeCost {
    // AIV后处理耗时（仅streamK有效）
    AicTimeCost dpAicCost;
    AicTimeCost skAicCost;
    double aicCost = 0;

    double aivTotalTime = 0;
    double vectorHeadCost = 0;
    double workspace2UBCost = 0;
    double vecCalCost = 0;
    double ub2GMCost = 0;
    double totalTime;
};

struct CaseInfo {
    uint64_t mSize;
    uint64_t nSize;
    uint64_t kSize;
    uint64_t baseM;
    uint64_t baseN;
    uint64_t baseK;
    uint64_t usedCoreNum;
    uint64_t kL1;
    uint64_t outTypeSize;
};

class TilingCostEvaluator {
public:
    TilingCostEvaluator(const SocSpecifyParams& socParmas);
    ~TilingCostEvaluator() = default;

    AicTimeCost Evaluate(const DequantBmm::QuantBatchMatmulV3BasicAPITilingData& tilingData,
                         const QuantBatchMatmulInfo& inputInfo, const QuantBatchMatmulV3CompileInfo& compileInfo);

    void EvaluateAicTime(const CaseInfo& caseInfo, const QuantBatchMatmulInfo& inputInfo,
                         const QuantBatchMatmulV3CompileInfo& compileInfo, AicTimeCost& res);

    StreamKTimeCost EvaluateStreamkCost(const CaseInfo& caseInfo, uint64_t skSingleCoreK,
                                        const QuantBatchMatmulInfo& inputInfo,
                                        const QuantBatchMatmulV3CompileInfo& compileInfo);

private:
    double GetRealL2Bw(const CaseInfo& caseInfo, const QuantBatchMatmulInfo& inputInfo,
                       const QuantBatchMatmulV3CompileInfo& compileInfo);
    double GetRealHbmBw(const CaseInfo& caseInfo, const QuantBatchMatmulInfo& inputInfo,
                        const QuantBatchMatmulV3CompileInfo& compileInfo);
    void EsitmateMmadTime(const CaseInfo& caseInfo, const QuantBatchMatmulInfo& inputInfo, AicTimeCost& res);

    // 单轮MTE1耗时，主要B4时如果有MTE1指令拆分，需要看是增加了mte1的好时，还是由于issueque问题反压了mte2
    void EstimateMte1Time(const CaseInfo& caseInfo, const QuantBatchMatmulInfo& inputInfo, AicTimeCost& res);

    // 实际的l2带宽，hbm带宽由启动的总核数，busrtlen是否128B对齐，等因素的影响
    void EstimateMte2Time(const CaseInfo& caseInfo, const QuantBatchMatmulInfo& inputInfo,
                          const QuantBatchMatmulV3CompileInfo& compileInfo, AicTimeCost& res);

    // 实际的fix带宽，v200=128B/c，总带宽其实不到l2带宽；
    // 每轮所有核的fix 耗时最长的可能不是传输数据量最大的，可能主块fixN 128B对齐带宽高，耗时反而比尾块非对齐的耗时短
    // 需要考虑对齐非对齐对拆包的影响
    void EstimateFixpTime(const CaseInfo& caseInfo, const QuantBatchMatmulInfo& inputInfo,
                          const QuantBatchMatmulV3CompileInfo& compileInfo, AicTimeCost& res);

    void EstimateStreamkAivTime(const CaseInfo& caseInfo, const QuantBatchMatmulInfo& inputInfo,
                                const QuantBatchMatmulV3CompileInfo& compileInfo, uint64_t kCnt,
                                StreamKTimeCost& skAivTime);

    SocSpecifyParams socParams_;
};

} // namespace optiling

#endif // TILING_COST_EVALUATOR_H
