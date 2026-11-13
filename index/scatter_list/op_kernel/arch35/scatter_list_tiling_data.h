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
 * \file scatter_list_tiling_data.h
 * \brief tiling data struct for scatter_list arch35(A5) SIMT 实现
 *
 * ScatterList 语义是纯搬运(reduce 仅支持 update): 把 updates 按 indice 给出的轴上偏移
 * 搬进 var 张量列表。按归一化后的更新轴 axis 把任意 rank 的输入折成四段坐标:
 *
 *   updates : [tensorNum, A..., updateAxisLen, T...]
 *   var[i]  : [          A..., varAxisLen,     T...]      (var rank = updates rank - 1)
 *
 *   leadNum = prod(A...)   trailLen = prod(T...)
 *
 * 搬运公式(对所有合法 axis 统一成立, 含更新轴为末轴即 trailLen==1 的情形):
 *   var[t][(a * varAxisLen + off + u) * trailLen + e] =
 *       updates[((t * leadNum + a) * updateAxisLen + u) * trailLen + e]
 * 其中 off / 有效更新长度由 indice 给出(见 indiceDims)。
 */

#ifndef SCATTER_LIST_ARCH35_TILING_DATA_H
#define SCATTER_LIST_ARCH35_TILING_DATA_H

struct ScatterListArch35TilingData {
    int64_t totalElements = 0; // tensorNum * leadNum * updateAxisLen * trailLen, SIMT 线程网格总量
    int64_t tensorNum = 0;     // var 张量个数, 等于 updates 首维
    int64_t leadNum = 0;       // 更新轴之前各维的乘积(var 与 updates 必须一致)
    int64_t updateAxisLen = 0; // updates 在更新轴上的长度
    int64_t varAxisLen = 0;    // var 在更新轴上的长度, 写入边界
    int64_t trailLen = 0;      // 更新轴之后各维的乘积(var 与 updates 必须一致)
    int32_t indiceDims = 1;    // 1: indice[t] 为轴上偏移, 更新长度即 updateAxisLen
                               // 2: indice[t*2] 为轴上偏移, indice[t*2+1] 为该张量实际更新长度
    int32_t hasMask = 0;       // 0: mask 为空, 全部张量参与; 1: 仅 mask[t]==1 的张量参与
    int32_t needCoreNum = 0;   // SetBlockDim 下发的核数
    int32_t reserved = 0;      // 8 字节对齐占位
};

#endif // SCATTER_LIST_ARCH35_TILING_DATA_H
