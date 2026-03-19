/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file concat_offset_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_CONCATOFFSET_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_CONCATOFFSET_H

#include "concat_offset_arch35.h"
#include "error_util.h"
#include "log/log.h"
#include "op_host/tiling_base.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_common/atvoss/broadcast/broadcast_tiling.h"
#include "index/concat_offset/op_kernel/arch35/concat_offset_struct.h"

namespace optiling 
{

ge::graphStatus ConcatOffsetTilingForAscendC(gert::TilingContext* context);

class ConcatOffsetTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit ConcatOffsetTiling(gert::TilingContext *context) : TilingBaseClass(context)
    {
        Reset();
    }
    ~ConcatOffsetTiling() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        return true;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;
    // 8、Dump Tiling数据
    void DumpTilingInfo() override;
    void Reset();

private:
    ge::graphStatus GetXInfoAndCheck();
    inline ge::graphStatus GetConcatDimInfoAndCheck();
    inline ge::graphStatus GetAttrInfoAndCheck();

    const char *opName_ = "";
#ifdef DAVID_FPGA
    int64_t threadNum_ = 512;
#else
    int64_t threadNum_ = 2048;
#endif
    int64_t concatDim_ = 0;
    int64_t sizeN_ = 0;
    int64_t ubSize_ = 0;
    int64_t totalCoreNum_ = 0;
    int64_t perTensorShapeSize_ = 0;
    int64_t needCalNum_ = 0;
};
}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_CONCATOFFSET_H