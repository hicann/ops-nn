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
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "rnn/gru_grad/op_kernel/gru_grad_tiling_data.h"
#include "rnn/gru_grad/op_kernel/gru_grad_tiling_key.h"

namespace optiling {

struct GruGradCompileInfo {
    uint32_t aicCoreNum{0};
    uint32_t aivCoreNum{0};
    int64_t sysWorkspaceSize{0};
    int64_t ubSizePlatForm{0};
};

// ==== Tiling类 (继承自 TilingBaseClass) ====
class GruGradTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit GruGradTiling(gert::TilingContext* context) : TilingBaseClass(context), context_(context) {}
    ~GruGradTiling() override = default;

    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    // gru_grad 不按 dtype 区分 tilingKey, 统一返回 GRU_GRAD_TILING_KEY
    [[nodiscard]] uint64_t GetTilingKey() const override { return GRU_GRAD_TILING_KEY; }
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
    void ReduceBlockCalculate();
    void SplitDxhBlockCalculate();
    void ConcatXhBlockCalculate();
    void SetTilingData();

    CutBatchTiling CalculateCutBatchTiling(int64_t ubParaNum, int64_t alignedSize, int64_t actualSize,
                                           int64_t copyMLinesMax, int64_t batch);

    bool ValidateInputShape(int index, const std::vector<int64_t>& expected_dims);
    bool ValidateOutputShape(int index, const std::vector<int64_t>& expected_dims);

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

#endif // _GRU_GRAD_TILING_H_
