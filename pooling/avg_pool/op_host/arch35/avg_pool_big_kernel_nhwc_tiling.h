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
 * \file avg_pool_big_kernel_nhwc_tiling.h
 * \brief big kernel imply for pool_3d ndhwc format
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_AVG_POOL_BIG_KERNEL_NHWC_TILING_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_AVG_POOL_BIG_KERNEL_NHWC_TILING_H_

#include <array>

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "pooling/avg_pool/op_host/avg_pool_tiling_common.h"
#include "pooling/avg_pool/op_kernel/arch35/avg_pool_struct.h"

namespace optiling
{
class AvgPoolCommonNHWCBigKernelTiling : public TilingBaseClass
{
public:
    explicit AvgPoolCommonNHWCBigKernelTiling(gert::TilingContext* context) : TilingBaseClass(context)
    {
    }

    ~AvgPoolCommonNHWCBigKernelTiling() override
    {
    }

protected:
    void DoUBTiling();
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    void SetTilingData();

public:
    uint64_t totalCoreNum_ = 1;
    uint64_t ubSize_ = 0;
    AvgPoolInputInfo inputData_;
    ge::DataType dtype_ = ge::DataType::DT_FLOAT;
    int64_t totalIdx_{0};
    int64_t blockFactor_{0};
    int64_t blockTail_{0};
    int64_t coreNums_{0};
    int64_t inUbSize_{0};
    int64_t outUbSize_{0};
    int64_t isSigOut_{0};
    int64_t tilingMode_{0};
    int64_t onceOutNum_{1};
};

class AvgPoolNHWCBigKernelTiling : public AvgPoolCommonNHWCBigKernelTiling
{
public:
    explicit AvgPoolNHWCBigKernelTiling(gert::TilingContext* context) : AvgPoolCommonNHWCBigKernelTiling(context)
    {
    }
    ~AvgPoolNHWCBigKernelTiling() override
    {
    }

private:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
};

class AvgPoolV2NHWCBigKernelTiling : public AvgPoolCommonNHWCBigKernelTiling
{
public:
    explicit AvgPoolV2NHWCBigKernelTiling(gert::TilingContext* context) : AvgPoolCommonNHWCBigKernelTiling(context)
    {
    }
    ~AvgPoolV2NHWCBigKernelTiling() override
    {
    }

private:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
};
}  // namespace optiling

#endif
