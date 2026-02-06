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
 * \file max_pool3d_with_argmax_v2_tiling_big_kernel_regbase.h
 * \brief big kernel imply for max_pool3d_with_argmax_v2
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MAX_POOL3D_WITH_AGRMAX_V2_TILING_BIG_KERNEL_REGBASE_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MAX_POOL3D_WITH_AGRMAX_V2_TILING_BIG_KERNEL_REGBASE_H_

#include "max_pool3d_with_argmax_v2_tiling_base.h"
#include "../op_kernel/arch35/max_pool3d_with_argmax_v2_tiling_struct.h"
#include "error_util.h"
#include "op_common/op_host/util/platform_util.h"

namespace optiling
{
using Ops::NN::Optiling::TilingBaseClass;

static constexpr uint64_t MAX_POOL_WITH_ARGMAX_V2_TILING_KEY_BIG_KERNEL_REGBASE_NCDHW = 611110;
const int64_t INPUT_IDX_X = 0;
const int64_t NCDHW_DIMS = 5;
const int64_t KERNEL_POS = 0;
const int64_t STRIDE_POS = 1;
const int64_t PADDING_POS = 2;
const int64_t DTYPE_POS = 6;
const int64_t DILATION_POS = 3;
const int64_t CEIL_POS = 4;
const int64_t FORMAT_POS = 5;
const int64_t WS_SYS_SIZE = 16 * 1024 * 1024;
static const int64_t MP_MAX_3D_DIM_ZERO = 0;
static const int64_t MP_MAX_3D_DIM_ONE = 1;
static const int64_t MP_MAX_3D_DIM_TWO = 2;
static const int64_t MP_MAX_3D_DIM_THREE = 3;
static const int64_t MP_MAX_3D_DIM_FOUR = 4;
static const int64_t MP_MAX_3D_TYPE_INT32 = 3;
static const int64_t MP_MAX_3D_TYPE_INT64 = 9;
static constexpr int64_t OUT_BUFFER_LEN = 1024;
static constexpr int64_t BUFFER_NUM = 2;
static constexpr int64_t MIN_COUNT = 1024;
static constexpr int64_t BYTES_FOUR = 4;
static constexpr int64_t BYTES_EIGHT = 8;
static constexpr int64_t KW_THRESHOLD = 128;
static constexpr int64_t THREE = 3;

struct BigKernelInputInfo {
    uint64_t batches;
    std::array<uint64_t, DHW_DIMS> inputShape;
    std::array<uint64_t, DHW_DIMS> outShape;
    std::array<uint64_t, DHW_DIMS> kernelSize;
    std::array<uint64_t, DHW_DIMS> stride;
    std::array<uint64_t, DHW_DIMS> pad;
    std::array<uint64_t, DHW_DIMS> dilation;
    bool ceilMode;
    ge::DataType indexDtype;
    ge::Format inputFormat;
    uint64_t nInput;
    uint64_t cInput;
};

class MaxPool3DWithArgmaxV2BigKernelRegbaseTiling : public MaxPool3DWithArgmaxV2BaseTiling {
public:
    explicit MaxPool3DWithArgmaxV2BigKernelRegbaseTiling(gert::TilingContext* context) : MaxPool3DWithArgmaxV2BaseTiling(context)
    {
    }
    ~MaxPool3DWithArgmaxV2BigKernelRegbaseTiling() override
    {
    }

private:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    
    void DoUBTiling();
    void SetTilingData();
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;

    MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2BigKernelRegbaseTilingData* tilingData_ = 
        context_->GetTilingData<MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2BigKernelRegbaseTilingData>();
    int64_t totalIdx_{0};
    int64_t blockFactor_{0};
    int64_t blockTail_{0};
    int64_t maxCount_{0};
    int64_t isSigOut_{0};
    int64_t coreNums_{0};

public:
    BigKernelInputInfo inputData;
    ge::DataType dtype = ge::DataType::DT_FLOAT;
    uint32_t coreNum = 1;
    uint32_t ubSize = 0;
};

}
#endif  // CANN_MAX_POOL_3D_WITH_ARGMAX_V2_TILING_BIG_KERNEL_REGBASE