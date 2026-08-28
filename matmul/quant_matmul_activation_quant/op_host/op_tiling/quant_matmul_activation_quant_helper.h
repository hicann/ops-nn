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
 * \file quant_matmul_activation_quant_helper.h
 * \brief Common parser and validator for QuantMatmulActivationQuant basic-api tiling strategies.
 */
#ifndef QUANT_MATMUL_ACTIVATION_QUANT_HELPER_H
#define QUANT_MATMUL_ACTIVATION_QUANT_HELPER_H

#include <cstdint>
#include "matmul/common/op_host/op_tiling/tiling_type.h"
#include "error_util.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "matmul/common/op_host/log_format_util.h"
#include "../quant_matmul_activation_quant_host_utils.h"
#include "../../op_kernel/arch35/quant_matmul_activation_quant_tiling_data.h"
#include "matmul/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling_base.h"
#include "matmul/quant_batch_matmul_v3/op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"
#include "quant_base_block_calculator.h"
#include "matmul/quant_batch_matmul_v3/op_host/op_tiling/arch35/quant_batch_matmul_v3_tiling_util.h"

namespace optiling {
using namespace QuantMatmulActivationQuantTilingConstant;
using Ops::NN::FormatString;

template <typename BaseT>
class QuantMatmulActivationQuantHelper : public BaseT {
public:
    explicit QuantMatmulActivationQuantHelper(gert::TilingContext* context) : BaseT(context) {}
    ~QuantMatmulActivationQuantHelper() override = default;

protected:
    using BaseT::BaseT;

    ge::graphStatus GetShapeAttrsInfo() override;
    bool AnalyzeAttrs() override;
    bool AnalyzeDtype() override;
    bool AnalyzeInputs() override;
    bool CheckDtype() const override;
    const char* GetDefaultOpName() const override;
    bool CalcBasicBlock() override;
    bool IsFp8Dtype(const ge::DataType dtype) const;
    bool IsFp4Dtype(const ge::DataType dtype) const;

    bool IsMxQuant() const;

    bool CheckParamsForMxQuant(const gert::Shape& x1Shape, const gert::Shape& x1ScaleShape,
                               const gert::Shape& x2ScaleShape) const;
    bool CheckShapeValid(const gert::Shape& x1Shape, const gert::Shape& x2Shape) const;
    bool InitMatmulSize(const gert::Shape& x1Shape, const gert::Shape& x2Shape);
    bool ValidateQuantParams(const gert::Shape& x1Shape, const gert::Shape& x1ScaleShape,
                             const gert::Shape& scaleShape);
    uint64_t GetBatchCoreCnt() const override;
    void SetQuantParams(QMMAQ::QuantMatmulActivationQuantTilingData& tilingData);
    void ResetActivationQuantTilingData(QMMAQ::QuantMatmulActivationQuantTilingData& tilingData);
    void CopyV3BasicApiTilingData(const DequantBmm::QuantBatchMatmulV3BasicAPITilingData& src,
                                  DequantBmm::QuantBatchMatmulV3BasicAPITilingData& dst);
    uint64_t GetBaseNAlignSize(uint64_t innerAlignSize) const override;
    void CalcTailBasicBlockAfullLoad() override;
    uint64_t GetTailSplitState(bool isPreSplitM, bool isPreSplit, uint64_t split, uint64_t splitSize) const override;
    void GetOuterNAxisTailCnt(uint64_t& baseTailSplitCnt, uint64_t& tailMain) override;
    void CalculateCurrentPerf(uint64_t mergeLen, uint64_t nTail, uint64_t& newTailMain, uint64_t& curPerf) override;
    bool IsAligned32(uint64_t value) const;

private:
    QMMAQ::GeluAlg activationType_ = QMMAQ::GeluAlg::TANH;
    QMMAQ::QuantAlg scaleAlg_ = QMMAQ::QuantAlg::OCP;
    QMMAQ::MX_QUANT_ROUND_MODE roundMode_ = QMMAQ::MX_QUANT_ROUND_MODE::RINT;
    float dstTypeMax_ = 0.0;
};

} // namespace optiling

#endif
