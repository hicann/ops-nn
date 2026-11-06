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
 * \file gru_grad_tiling.h
 * \brief GRU反向算子Tiling数据结构定义
 */
#ifndef _GRU_GRAD_TILING_H_
#define _GRU_GRAD_TILING_H_

#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "error_util.h"
#include "platform/platform_infos_def.h"
#include "rnn/gru_grad/op_kernel/gru_grad_tiling_data.h"
#include "rnn/gru_grad/op_kernel/arch22/gru_grad_tiling_key.h"

namespace optiling {

struct GruGradCompileInfo {
    uint32_t aicCoreNum{0};
    uint32_t aivCoreNum{0};
    int64_t sysWorkspaceSize{0};
    int64_t ubSizePlatForm{0};
};

// CompileInfo 解析（平台核数/UB/系统 workspace）——arch22 与 arch35 两套 tiling 共用，
// 经本头文件 inline 提供（两 arch 目录互不链接，仅头文件共享）。
inline ge::graphStatus TilingPrepareForGruGrad(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareForGruGrad start.");
    auto ci = context->GetCompiledInfo<GruGradCompileInfo>();
    OP_CHECK_IF(ci == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{context->GetNodeName(),
                                              "compile info is unavailable; check operator registration and compile "
                                              "info configuration."})),
                return ge::GRAPH_FAILED);
    fe::PlatFormInfos* pi = context->GetPlatformInfo();
    OP_CHECK_IF(pi == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{context->GetNodeName(), "platform info is unavailable; check platform "
                                                                      "metadata configuration."})),
                return ge::GRAPH_FAILED);
    auto plat = platform_ascendc::PlatformAscendC(pi);
    ci->aicCoreNum = plat.GetCoreNumAic();
    OP_CHECK_IF(ci->aicCoreNum <= 0,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{context->GetNodeName(),
                                              (std::string("Failed to get aicCoreNum; actual=") +
                                               std::to_string(ci->aicCoreNum) + "; check platform resource metadata.")
                                                  .c_str()})),
                return ge::GRAPH_FAILED);
    ci->aivCoreNum = ci->aicCoreNum * 2;
    OP_CHECK_IF(ci->aivCoreNum <= 0,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{context->GetNodeName(),
                                              (std::string("Failed to get aivCoreNum; actual=") +
                                               std::to_string(ci->aivCoreNum) + "; check platform resource metadata.")
                                                  .c_str()})),
                return ge::GRAPH_FAILED);
    uint64_t ub = 0;
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    ci->ubSizePlatForm = static_cast<int64_t>(ub);
    ci->sysWorkspaceSize = static_cast<int64_t>(plat.GetLibApiWorkSpaceSize());
    return ge::GRAPH_SUCCESS;
}

class GruGradTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit GruGradTiling(gert::TilingContext* context) : TilingBaseClass(context), context_(context) {}
    ~GruGradTiling() override = default;

    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    [[nodiscard]] uint64_t GetTilingKey() const override
    {
        // dxMM 的 N 维(内轴)= inputSize,当 inputSize >= 65535 时单核内轴超过
        // 搬入上限,需使能 GM→L1 循环搬入(intrinsicsLimit=true)以保证精度。
        constexpr int64_t HUGE_I_THRESHOLD = 65535;
        return (tilingData_.inputSize >= HUGE_I_THRESHOLD) ? GRU_GRAD_TILING_KEY_HUGE : GRU_GRAD_TILING_KEY;
    }
    ge::graphStatus PostTiling() override;

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus GetWorkspaceSize() override;

private:
    bool CheckParamsShape();
    bool CheckAttr();

    void GetMatmulTiling();
    void GetDgateMMTiling(matmul_tiling::DataType mmDataType);
    void GetDwIhMMTiling(matmul_tiling::DataType mmDataType);
    void GetDwHhMMTiling(matmul_tiling::DataType mmDataType);
    void GetDxMMTiling(matmul_tiling::DataType mmDataType);
    void ReduceBlockCalculate();
    void SplitDxhBlockCalculate();
    void ConcatXhBlockCalculate();
    void SetTilingData();

    CutBatchTiling CalculateCutBatchTiling(int64_t ub, int64_t al, int64_t act, int64_t ml, int64_t b);

    bool ValidateInputShape(int idx, const std::vector<int64_t>& e);
    bool ValidateOutputShape(int idx, const std::vector<int64_t>& e);
    void LogShapeCheck(bool isInput, int index, const std::vector<int64_t>& expected, const char* name, bool ok);

private:
    GruGradTilingData tilingData_;
    gert::TilingContext* context_ = nullptr;
    const GruGradCompileInfo* compileInfo_ = nullptr;
    const char* nodeName_ = nullptr;
    int64_t sysAicCoreNum_{0};
    int64_t sysAivCoreNum_{0};
    int64_t alignedPara_{0};
    int64_t inputDSize_{0};
    std::string firstBadParam_;
    std::string firstBadParamShape_;

    CutBatchTiling dxhInputParam_;
    CutBatchTiling dxhHiddenParam_;
    CutBatchTiling xhInputParam_;
    CutBatchTiling xhHiddenParam_;
};

} // namespace optiling

#endif // _GRU_GRAD_TILING_H_
