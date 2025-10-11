/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the
 * License. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of
 * the License.
 */

/*!
 * \file edge_softmax_tiling.cpp
 * \brief
 */
#include "edge_softmax_tiling.h"

#include "tiling/platform/platform_ascendc.h"

namespace optiling {

constexpr int32_t DATA_BLOCK_SIZE = 32;  // 数据块大小: 32字节
constexpr int32_t DATA_BLOCK_LEN_32 =
    DATA_BLOCK_SIZE / sizeof(float);  // 32位数据块长度: 32字节/4字节=8个元素
constexpr int32_t coreNum = 40;

constexpr inline int32_t Min(int32_t a, int32_t b) { return a < b ? a : b; }

struct InputInfo {
    explicit InputInfo(gert::TilingContext* context) {
        shape = context->GetInputShape(0);
        const auto& storage = shape->GetStorageShape();
        auto attr = context->GetAttrs()->GetAttrPointer<int>(0);

        E = storage.GetDim(0);
        F = storage.GetDim(1);
        N = *attr;
    }

    const gert::StorageShape* shape = nullptr;
    int32_t E = 0;
    int32_t F = 0;
    int32_t N = 0;
};

static ge::graphStatus EdgeSoftmaxTilingFunc(gert::TilingContext* context) {
    // 获取输入信息
    InputInfo inputInfo{context};
    // 设置核数
    // int32_t blockNum = Min(inputInfo.N, coreNum);
    int32_t blockNum = (inputInfo.F % DATA_BLOCK_LEN_32 == 0) ? Min(inputInfo.N, coreNum) : 1;
    context->SetBlockDim(blockNum);

    // // 调试打印
    // std::cout << "Input Info: " << "\n"
    //           << "  E: " << inputInfo.E << "\n"
    //           << "  F: " << inputInfo.F << "\n"
    //           << "  N: " << inputInfo.N << "\n";

    // 设置tiling data
    EdgeSoftmaxTilingData tiling_data;
    tiling_data.set_E(inputInfo.E);
    tiling_data.set_F(static_cast<int16_t>(inputInfo.F));
    tiling_data.set_N(static_cast<int16_t>(inputInfo.N));
    tiling_data.set_blockNum(static_cast<int8_t>(blockNum));
    tiling_data.SaveToBuffer(context->GetRawTilingData()->GetData(),
                             context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling_data.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling