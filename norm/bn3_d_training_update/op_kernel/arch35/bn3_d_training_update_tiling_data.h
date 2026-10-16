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
 * \file bn3_d_training_update_tiling_data.h
 * \brief BN3DTrainingUpdate arch35 TilingData(宿主/内核共享的纯 struct)+ UB 记账常量。
 */

#ifndef BN3_D_TRAINING_UPDATE_TILING_DATA_H
#define BN3_D_TRAINING_UPDATE_TILING_DATA_H

#include <cstddef>
#include <cstdint>

namespace Bn3dTrainingUpdateOps {

// 本文件的布局/反解函数被 host(tiling)与 device(kernel 的 InitBuffer)共用。设备侧编译时这些
// 纯函数必须带 __aicore__ 限定(否则被视作 host 函数、无法从 __aicore__ 上下文调用);
// host 侧没有该限定符。两侧共用**同一份函数体**,不存在第二套记账口径。
#if defined(__NPU_ARCH__) || defined(__CCE_AICORE__)
#define BN3D_TU_LAYOUT_FN __aicore__ inline constexpr
#else
#define BN3D_TU_LAYOUT_FN inline constexpr
#endif

// ======================= UB 记账常量(与具体形状无关) =======================
// TPipe/TQue/TBuf 元数据与事件表预留(同族 bn_* arch35 tiling 口径)。
constexpr int64_t kPipeMetaReserveBytes = 8512;
// 通道折叠一次处理的通道数上界 = kParamChunkVL * VL_FP32。
constexpr int64_t kParamChunkVL = 8;
// x / y 两条队列。
constexpr int64_t kXyQueueNum = 2;
// 每条队列的 UB 块数。
constexpr int64_t kXyTileBufferNum = 1;
// 系数数组三块(非折叠式):coeffMult / coeffBm / coeffOffset。
constexpr int64_t kCoeffTileBufferNum = 3;
// 相 S 的输入 staging 槽位:sum / square_sum / scale / offset / mean / variance。
constexpr int64_t kStatInputSlots = 6;
// 相 S 的结果槽位:batchMean / saveVar / meanOut / varOut(折叠的 mult / offset(直通)复用后两者作临时)。
constexpr int64_t kStatResultSlots = 4;
// 并行度目标:每核至少多少个 X 单元。块长 = "UB 上界" 与 "单元数目标" 取小。
constexpr int64_t kUnitsPerCoreTarget = 4;
// 系数折叠 staging:4 路输入(sum / square_sum / scale / offset)+ 折叠出参的临时槽(saveVar)。
// 非折叠式下折叠的 4 个出参是 mult / bm / offset(直通)/ saveVar;相 X 把前三个直接写进系数数组三块,
// 只有 saveVar 需要一个额外临时槽;相 S 的 4 个出参各落自己的结果槽(故相 S 不占此临时槽)。
constexpr int64_t kFoldInputSlots = 4;
constexpr int64_t kFoldScratchSlots = 1;
// 折叠槽下标(CF 与 CL 共用同一组折叠槽):4 路输入槽 + 1 个折叠出参槽(saveVar)。
constexpr int64_t kFoldInSum = 0;
constexpr int64_t kFoldInSquareSum = 1;
constexpr int64_t kFoldInScale = 2;
constexpr int64_t kFoldInOffset = 3;
constexpr int64_t kFoldSaveVar = 0;
// 系数数组三块(折叠结果本体;CF 逐平面 / CL 逐行直接读取):mult / bm / offset。
constexpr int64_t kCoeffMultSlot = 0;
constexpr int64_t kCoeffBmSlot = 1;
constexpr int64_t kCoeffOffsetSlot = 2;

// 通道折叠的分块长度(与"单元/瓦片"块长解耦)。
BN3D_TU_LAYOUT_FN int64_t kParamChunkLen(int64_t vlFp32) { return kParamChunkVL * vlFp32; }

// UB 对齐粒度补齐。
BN3D_TU_LAYOUT_FN int64_t AlignUpBlock(int64_t bytes, int64_t blockBytes)
{
    return (bytes + blockBytes - 1) / blockBytes * blockBytes;
}

// 按任意粒度补齐(用于把一个长度对齐到 VL 的整数倍)。
BN3D_TU_LAYOUT_FN int64_t AlignUpValue(int64_t value, int64_t align) { return (value + align - 1) / align * align; }

// 相 S / 折叠 staging 的单个槽位字节数:按一个折叠分块(kParamChunk 个 fp32)记账。
// 槽位是 VL 的整数倍,使 VF 内的整 VL 对齐 load 恒在界内。
BN3D_TU_LAYOUT_FN int64_t ChunkSlotBytes(int64_t vlFp32, int64_t blockBytes)
{
    return AlignUpBlock(kParamChunkLen(vlFp32) * static_cast<int64_t>(sizeof(float)), blockBytes);
}

// channel-last 系数数组单块的字节数(coeffCap = 系数长度上限)。
BN3D_TU_LAYOUT_FN int64_t CoeffTileBytes(int64_t coeffCap, int64_t blockBytes)
{
    return AlignUpBlock(coeffCap * static_cast<int64_t>(sizeof(float)), blockBytes);
}

// x/y 瓦片缓冲:VF 恒按整 VL 块读写(尾块只靠掩码屏蔽,不做部分宽度访存)⇒ 末块可能读到
// roundUp(元素数, VL) 处,故在精确尺寸之上预留一个 VL 的余量,保证越界访问始终落在本缓冲区内。
BN3D_TU_LAYOUT_FN int64_t XyTileBufferBytes(int64_t elements, int64_t elemBytes, int64_t vlFp32, int64_t blockBytes)
{
    return AlignUpBlock(elements * elemBytes, blockBytes) + AlignUpBlock(vlFp32 * elemBytes, blockBytes);
}

// channel-last 系数数组的长度上限(host 给系数缓冲定价用)。
// 一行内 VF 按 VL 整数倍偏移整 VL 对齐读系数,末块读到 roundUp(cLen, VL) 处 ⇒ 取 VL 的整数倍上界
// 即可覆盖所有 cLen ∈ [1, cb];host 用它定价后,kernel 侧的整 VL 读恒在界内,无反解失败分支。
BN3D_TU_LAYOUT_FN int64_t CoeffTileCap(int64_t cb, int64_t vlFp32) { return AlignUpValue(cb, vlFp32); }

// ======================= TilingData(key 2000 / 3000) =======================

// key 2000 —— channel-first(NCHW / NCDHW):x 视图 [N][C][S],S = 非通道轴之积。
struct Bn3dTrainingUpdateChannelFirstTilingData {
    int64_t c;             // C
    int64_t spatial;       // S = H*W(4D)或 D*H*W(5D)
    int64_t spatialTile;   // Ss(UB 反解出的瓦片段长)
    int64_t spatialTiles;  // ceil(S / Ss)
    int64_t channelWindow; // K:每单元的连续通道平面数
    int64_t baseTilesPerCore;
    int64_t extraCoreCount;
    int64_t statUnitCount; // 相 S 单元数
    int64_t statChunk;     // 相 S 单元的通道数上界
    int64_t baseStatPerCore;
    int64_t extraStatCount;
    float numRec;         // num>0 ? 1/num : 0(host 预算,先取倒数再乘)
    float besselScaler;   // num>1 ? num/(num-1) : 0
    float factor;         // attr factor
    float oneMinusFactor; // 1 - factor(host 预算)
    float epsilon;        // attr epsilon
    // 显式 ABI 对齐填充:int64 → float → uint32 的声明次序下自然对齐不插填充,
    // 尾部补 1 个 uint32 使总长落回 8 字节整数倍,令"紧凑布局 == 自然对齐布局"
    // (与宿主 TILING_DATA_DEF 的序列化布局逐字节一致)。
    uint32_t reserved;
};

// key 3000 —— channel-last(NHWC / NDHWC):x 视图 [num][C],num = N*S(拍平后的空间位置数)。
struct Bn3dTrainingUpdateChannelLastTilingData {
    int64_t c;             // C
    int64_t num;           // num = N*S
    int64_t channelBlock;  // Cb = min(C, CbCap)
    int64_t rowsPerTile;   // rows(UB 反解)
    int64_t tilesPerBlock; // ceil(num / rows)
    int64_t baseTilesPerCore;
    int64_t extraCoreCount;
    int64_t statUnitCount;
    int64_t statChunk;
    int64_t baseStatPerCore;
    int64_t extraStatCount;
    int64_t coeffCap; // 系数数组长度上限(UB 反解,= roundUp(Cb, VL))
    float numRec;
    float besselScaler;
    float factor;
    float oneMinusFactor;
    float epsilon;
    // 显式 ABI 对齐填充(同 channel-first 的口径)。
    uint32_t reserved;
};

// ABI 锁:两份结构体的字段序列与对齐与 host 侧 TILING_DATA_DEF 生成类逐字段一致
// (int64 → float → uint32 的声明次序保证自然对齐不插入填充)。
constexpr size_t BN3D_TRAINING_UPDATE_CF_TILING_DATA_BYTES = 112U;
constexpr size_t BN3D_TRAINING_UPDATE_CL_TILING_DATA_BYTES = 120U;
static_assert(sizeof(Bn3dTrainingUpdateChannelFirstTilingData) == BN3D_TRAINING_UPDATE_CF_TILING_DATA_BYTES,
              "Bn3dTrainingUpdate channel-first tiling ABI size changed unexpectedly");
static_assert(sizeof(Bn3dTrainingUpdateChannelLastTilingData) == BN3D_TRAINING_UPDATE_CL_TILING_DATA_BYTES,
              "Bn3dTrainingUpdate channel-last tiling ABI size changed unexpectedly");

} // namespace Bn3dTrainingUpdateOps

#endif // BN3_D_TRAINING_UPDATE_TILING_DATA_H
