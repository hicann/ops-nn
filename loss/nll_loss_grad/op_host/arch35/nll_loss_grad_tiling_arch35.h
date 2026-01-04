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
 * \file nll_loss_grad_tiling_arch35.h
 * \brief nll_loss_grad_tiling_arch35
 */

#ifndef OPS_LOSS_NLL_LOSS_GRAD_OP_HOST_NLL_LOSS_GRAD_TILING_ARCH35_H_
#define OPS_LOSS_NLL_LOSS_GRAD_OP_HOST_NLL_LOSS_GRAD_TILING_ARCH35_H_

#include "platform/platform_ascendc.h"
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"

namespace optiling
{
using namespace Ops::NN::Optiling;

constexpr uint64_t MAX_THREAD = 1024;
// ///////////////////////////////////
// tilingdata define
// ///////////////////////////////////
BEGIN_TILING_DATA_DEF(NLLLossGradSimtTilingData)
TILING_DATA_FIELD_DEF(uint64_t, batchNum);
TILING_DATA_FIELD_DEF(uint64_t, classNum);
TILING_DATA_FIELD_DEF(uint64_t, usedThread);
TILING_DATA_FIELD_DEF(uint64_t, height);
TILING_DATA_FIELD_DEF(uint64_t, width);
TILING_DATA_FIELD_DEF(uint64_t, veryImportantProcessCoreNums);
TILING_DATA_FIELD_DEF(uint64_t, notVeryImportantProcessCoreNums);
TILING_DATA_FIELD_DEF(uint64_t, blockPerCore);
TILING_DATA_FIELD_DEF(uint64_t, blockTailCore);
TILING_DATA_FIELD_DEF(uint32_t, reductionMode);
TILING_DATA_FIELD_DEF(int32_t, ignoreIdx);
TILING_DATA_FIELD_DEF(uint32_t, xDims);
END_TILING_DATA_DEF;

// simt template ascendc tools
REGISTER_TILING_DATA_CLASS(NLLLossGrad, NLLLossGradSimtTilingData)

class NLLLossGradSimtTiling : public TilingBaseClass
{
public:
    explicit NLLLossGradSimtTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {
    }
    uint64_t maxThread_{MAX_THREAD};
    uint64_t coreNum_{1};

protected:
    bool IsCapable() override
    {
        return true;
    }
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    // customized functions
    ge::graphStatus GenerateTilingKey();

private:
    uint64_t batchNum_{1};
    uint64_t tilingKey_{0};
    uint64_t height_{1};
    uint64_t width_{1};
    uint64_t ubSizePlatForm{0};
    uint32_t xDims_{1};
    NLLLossGradSimtTilingData tilingData_;

    int64_t aivCoreNum_{1};
    int64_t notVeryImportantProcessCoreNums_{1};
    int64_t blockPerCore_{0};
    int64_t blockTailCore_{0};

    ge::graphStatus ProcessShapeInfo();
    ge::graphStatus ProcessAttributesInfo();
};

struct NLLLossGradCompileInfo {
  int64_t reduction = 2;
  int64_t ub_size = 1;
  int64_t block_dim = 1;
  int64_t core_num {1};
  uint32_t max_thread{1024};
};
}  // namespace optiling
#endif  // OPS_LOSS_NLL_LOSS_GRAD_OP_HOST_NLL_LOSS_GRAD_TILING_ARCH35_H_
