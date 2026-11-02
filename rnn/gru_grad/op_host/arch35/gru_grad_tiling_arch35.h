/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gru_grad_tiling_arch35.h
 * \brief GRU 反向 Tiling（ascend950）声明：GruGradTilingArch35 + GruGradCompileInfo。
 *
 * IMPL_OP_OPTILING(GruGrad) 按 SoC 分目录注册（arch22/arch35 构建期隔离，本目录随
 * ascend950 编译）。四路 MM tiling 后 depthA1/depthB1 钳制为 2；GetWorkspaceSize
 * 在基线段之后追加扩展段（见 .cpp 文件头说明）。TilingData 用共享公用声明。
 */
#ifndef _GRU_GRAD_TILING_ARCH35_H_
#define _GRU_GRAD_TILING_ARCH35_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "rnn/gru_grad/op_kernel/gru_grad_tiling_data.h" // 公用 TilingData（struct GruGradTilingData）
#include "../arch22/gru_grad_tiling.h" // GruGradCompileInfo（单注册点统一 TilingParse 类型）

constexpr uint64_t GRU_GRAD_ARCH35_TILING_KEY = 0;

namespace optiling {

class GruGradTilingArch35 : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit GruGradTilingArch35(gert::TilingContext* context) : TilingBaseClass(context), context_(context) {}
    ~GruGradTilingArch35() override = default;

    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    [[nodiscard]] uint64_t GetTilingKey() const override { return GRU_GRAD_ARCH35_TILING_KEY; }
    ge::graphStatus PostTiling() override;

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus GetWorkspaceSize() override;

private:
    bool CheckParamsShape();
    bool CheckAttr();

    ge::graphStatus GetMatmulTiling();
    ge::graphStatus GetDgateMMTiling(matmul_tiling::DataType mmDataType);
    ge::graphStatus GetDwIhMMTiling(matmul_tiling::DataType mmDataType);
    ge::graphStatus GetDwHhMMTiling(matmul_tiling::DataType mmDataType);
    ge::graphStatus GetDxMMTiling(matmul_tiling::DataType mmDataType);
    void ClampDepth(AscendC::tiling::TCubeTiling& param);
    void ReduceBlockCalculate();
    void SplitDxhBlockCalculate();
    void ConcatXhBlockCalculate();
    void SetTilingData();

    CutBatchTiling CalculateCutBatchTiling(int64_t ub, int64_t al, int64_t act, int64_t ml, int64_t b);

    bool ValidateInputShape(int idx, const std::vector<int64_t>& e);
    bool ValidateOutputShape(int idx, const std::vector<int64_t>& e);
    bool ValidateAndLogShape(bool isInput, int idx, const std::vector<int64_t>& expected, const char* name);
    void LogTCubeTiling(const char* nm, const AscendC::tiling::TCubeTiling& q);
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

    CutBatchTiling dxhInputParam_;
    CutBatchTiling dxhHiddenParam_;
    CutBatchTiling xhInputParam_;
    CutBatchTiling xhHiddenParam_;
};

} // namespace optiling

#endif // _GRU_GRAD_TILING_ARCH35_H_
