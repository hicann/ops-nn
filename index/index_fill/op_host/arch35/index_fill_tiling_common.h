/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file index_fill_tiling_common.h
 * \brief
 */

#ifndef INDEX_FILL_TILING_COMMON_H_
#define INDEX_FILL_TILING_COMMON_H_

#include "op_host/tiling_base.h"
#include "index/index_fill/op_kernel/arch35/index_fill_struct.h"
#include "index/index_fill/op_kernel/arch35/index_fill_tiling_key.h"

namespace optiling
{
constexpr int64_t INDEX_INPUT_X = 0;
constexpr int64_t INDEX_INPUT_INDICES = 1;
constexpr int64_t INDEX_INPUT_VALUE = 2;
constexpr int64_t INDEX_OUTPUT_Y = 0;
constexpr int64_t ROW_ELE_NUM_THRESHOLD = 128;
constexpr int64_t MAX_INT32 = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
constexpr uint64_t BLOCK_SPLIT_THREHOLD = 1024;
constexpr int64_t N_BUFFER = 2;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
constexpr uint64_t SIMT_DCACHE_SIZE = 32 * 1024;
constexpr uint64_t MAX_THREAD_NUM = 2048;
constexpr uint64_t MIN_THREAD_NUM = 128;
constexpr uint64_t TWO = 2;

struct IndexFillInputInfo {
    int64_t xDims;
    int64_t dim;
    uint64_t P = 0;
    uint64_t N = 0;
    uint64_t Q = 0;
    int64_t  numel = 0;             // numel表示number of element
    uint64_t indicesNum = 0;        // 索引tensor长度
    uint64_t tilingKey = 0;
    ge::DataType dtype;
    uint64_t dtypeSize = 0;
    int64_t frontCoreNum;           // 前frontCoreNum个核每个多处理一个block分片
    int64_t blockSize;              // 表示切分的一个block分片中有多少个元素
    int64_t tailBlockSize;          // 表示尾块中有多少个元素
    int64_t loopsPerFrontCore;      // 前frontCoreNum个核的单核循环次数
    int64_t loopsPerTailCore;       // 尾部这(usedCoreNum-frontCoreNum)个核的单核循环次数
};

class IndexFillCommonTiling : public Ops::NN::Optiling::TilingBaseClass
{
public:
    explicit IndexFillCommonTiling(gert::TilingContext* context) : Ops::NN::Optiling::TilingBaseClass(context)
    {
    }
    ~IndexFillCommonTiling() override
    {
    }

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    virtual void SetTilingData() = 0;

public:
    ge::DataType dtype_;
    uint64_t dtypeSize_ = 0;
    int64_t coreNum_ = 1;
    uint64_t ubSize_ = 0;
    int64_t availableUb_ = 0;
    uint32_t sysWorkspaceSize_ = 0;
    int64_t totalSliceNum;

protected:
    IndexFillInputInfo inputData;
    int64_t usedCoreNum_ = 0;
};

ge::graphStatus Tiling4IndexFillArch35(gert::TilingContext* context);
}  // namespace optiling
#endif  // INDEX_FILL_TILING_COMMON_H_
