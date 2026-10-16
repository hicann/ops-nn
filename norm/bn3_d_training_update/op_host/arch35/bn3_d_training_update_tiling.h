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
 * \file bn3_d_training_update_tiling.h
 * \brief BN3DTrainingUpdate arch35 tiling 内部声明与 UB 记账函数。
 */

#ifndef BN3_D_TRAINING_UPDATE_TILING_ARCH35_H
#define BN3_D_TRAINING_UPDATE_TILING_ARCH35_H

#include <cstdint>
#include "register/op_impl_registry.h"
#include "../../op_kernel/arch35/bn3_d_training_update_tiling_data.h"

namespace optiling {

using Bn3dTrainingUpdateOps::ChunkSlotBytes;
using Bn3dTrainingUpdateOps::CoeffTileBytes;
using Bn3dTrainingUpdateOps::CoeffTileCap;
using Bn3dTrainingUpdateOps::kCoeffTileBufferNum;
using Bn3dTrainingUpdateOps::kFoldInputSlots;
using Bn3dTrainingUpdateOps::kFoldScratchSlots;
using Bn3dTrainingUpdateOps::kParamChunkLen;
using Bn3dTrainingUpdateOps::kPipeMetaReserveBytes;
using Bn3dTrainingUpdateOps::kStatInputSlots;
using Bn3dTrainingUpdateOps::kStatResultSlots;
using Bn3dTrainingUpdateOps::kUnitsPerCoreTarget;
using Bn3dTrainingUpdateOps::kXyQueueNum;
using Bn3dTrainingUpdateOps::kXyTileBufferNum;
using Bn3dTrainingUpdateOps::XyTileBufferBytes;

// VF 循环计数器的 uint16_t 上界。依据(实测原始编译输出,本算子 kernel 编译,clang 15.0.5 / ccec):
//   <...>/bn3_d_training_update_common.h:180:14: error: Induction variable must have a type uint16_t. Example:
//    (uint16_t $var=0; $var< bound; $var++))
//   前置 warning: expected type of ops in vector loop cond is uint16_t ... [-Wcce-compat]
// ⇒ VF 循环的归纳变量**只能**是 uint16_t(编译器强制,非作者防御);且 kernel 侧写作
//   static_cast<uint16_t>(...),越界时是**静默截断回绕**(结果错、不报错) ⇒ 该上界承重,host 必须收敛。
// 口径:channel-last 的 VF 循环计数器是"行号 row"(上界 = rows)与"行内块号 i"(上界 = ceil(cLen/VL)),
//   二者都不超过瓦片元素数 rows*cLen ⇒ "瓦片元素数 ≤ UINT16_MAX"同时覆盖它们(充分、无 off-by-one);
// 只有 host 用它(kernel 侧不读)⇒ 定义在 host 头,不放 kernel 共享头。
constexpr int64_t kMaxTileElementsPerVf = static_cast<int64_t>(UINT16_MAX);

// 平台参数:一律从 PlatformAscendC / compileInfo 运行时读取,不硬编码。
struct Bn3dTrainingUpdatePlatform {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
    int64_t vlFp32 = 0;                // 一个 fp32 向量寄存器的车道数
    int64_t blockBytes = 0;            // UB 对齐粒度
    uint64_t libApiWorkspaceBytes = 0; // 框架自用 workspace 预留(平台值,不写死)
};

struct Bn3dTrainingUpdateCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
    int64_t vectorLength = 0;
    int64_t blockSize = 0;
    uint64_t libApiWorkspaceBytes = 0;
};

// channel-first 的相 X 占用:x/y 瓦片(K 个通道平面 × Ss 个空间元素)+ 折叠槽 + 系数窗
// coeffMult/coeffBm/coeffOffset[0..K)。 系数窗按 roundUp(K, VL) 定价 —— 折叠的整 VL 对齐写会写到 roundUp(K, VL) 处(与
// CL 对 Cb 的处理同口径)。
inline int64_t CfPhaseXBytes(int64_t spatialTile, int64_t channelWindow, int64_t elemBytes,
                             const Bn3dTrainingUpdatePlatform& p)
{
    const int64_t tileElements = channelWindow * spatialTile;
    return kXyQueueNum * kXyTileBufferNum * XyTileBufferBytes(tileElements, elemBytes, p.vlFp32, p.blockBytes) +
           (kFoldInputSlots + kFoldScratchSlots) * ChunkSlotBytes(p.vlFp32, p.blockBytes) +
           kCoeffTileBufferNum * CoeffTileBytes(CoeffTileCap(channelWindow, p.vlFp32), p.blockBytes);
}

// channel-last 的相 X 占用:x/y 瓦片(rows 行 × Cb 列)+ 折叠槽 + 3 路系数数组(coeffCap = roundUp(Cb,VL))。
inline int64_t ClPhaseXBytes(int64_t rowsPerTile, int64_t channelBlock, int64_t coeffCap, int64_t elemBytes,
                             const Bn3dTrainingUpdatePlatform& p)
{
    const int64_t tileElements = rowsPerTile * channelBlock;
    return kXyQueueNum * kXyTileBufferNum * XyTileBufferBytes(tileElements, elemBytes, p.vlFp32, p.blockBytes) +
           (kFoldInputSlots + kFoldScratchSlots) * ChunkSlotBytes(p.vlFp32, p.blockBytes) +
           kCoeffTileBufferNum * CoeffTileBytes(coeffCap, p.blockBytes);
}

// 相 S 的占用(与 C / 块长无关:折叠一律按 kParamChunk 分块,槽位尺寸恒定)。
inline int64_t StatPhaseBytes(const Bn3dTrainingUpdatePlatform& p)
{
    return (kStatInputSlots + kStatResultSlots) * ChunkSlotBytes(p.vlFp32, p.blockBytes);
}

// 整 kernel 占用:两相共用一个 TPipe ⇒ 只有**一份**元数据预留(两条布局函数各自记一次
// kPipeMetaReserveBytes 会重复计入,故这里显式合并为单一函数)。
inline int64_t CfKernelBytes(int64_t spatialTile, int64_t channelWindow, int64_t elemBytes,
                             const Bn3dTrainingUpdatePlatform& p)
{
    return CfPhaseXBytes(spatialTile, channelWindow, elemBytes, p) + StatPhaseBytes(p) + kPipeMetaReserveBytes;
}

inline int64_t ClKernelBytes(int64_t rowsPerTile, int64_t channelBlock, int64_t coeffCap, int64_t elemBytes,
                             const Bn3dTrainingUpdatePlatform& p)
{
    return ClPhaseXBytes(rowsPerTile, channelBlock, coeffCap, elemBytes, p) + StatPhaseBytes(p) + kPipeMetaReserveBytes;
}

// 相 S 单元切分:每个 S 单元的通道数 <= 一个折叠分块 ⇒ 折叠 staging 与 C 解耦。
// 目标"每核一段"(ceil(C / coreNum)),下界取 min(C, kParamChunk) 以保住分块摊销。
inline int64_t SolveStatChunk(int64_t channel, const Bn3dTrainingUpdatePlatform& p)
{
    const int64_t chunk = kParamChunkLen(p.vlFp32);
    const int64_t perCore = (channel + p.coreNum - 1) / p.coreNum;
    const int64_t lower = channel < chunk ? channel : chunk;
    const int64_t r = perCore > lower ? perCore : lower;
    return r < 1 ? 1 : r; // C == 0:无相 S 单元(unitCount 自然为 0),chunk 取 1 以保证后续不为 0
}

// 块长 = "UB 上界" 与 "单元数目标" 取小。
//   shapeExtent   = 被切的形状维长度(S 或 num)
//   parallelUnits = 与本维相乘构成单元数的其余因子(cWindows·N 或 channelBlocks)
//   floorLen      = 保证每单元至少一个整 VL 块的最小块长 = max(1, ceilDiv(VL, 另一维))
// 单调性:parallelUnits ≥ unitsTarget 时 perBlockTarget 退化为 1 ⇒ 目标块长 = shapeExtent ⇒ min() 取到 ubLen,
// 与"只按 UB 上界取"**逐字相同**;只有单元数不足时才压小块长、抬高单元数。全程只有除法取整与取小/取大。
inline int64_t TargetBlockLen(int64_t shapeExtent, int64_t ubLen, int64_t floorLen, int64_t parallelUnits,
                              int64_t unitsTarget)
{
    if (shapeExtent <= 0) {
        return ubLen < 1 ? 1 : ubLen; // S == 0 / num == 0:无 X 单元,块长只需 >= 1(不会被用到)
    }
    // parallelUnits == 0(C == 0,无并行单元)时目标块长取 1(无并行度约束可言),避免除零。
    const int64_t perBlockTarget = (parallelUnits < 1 || parallelUnits >= unitsTarget) ?
                                       1 :
                                       ((unitsTarget + parallelUnits - 1) / parallelUnits);
    const int64_t targetLen = (shapeExtent + perBlockTarget - 1) / perBlockTarget;
    const int64_t lo = floorLen < 1 ? 1 : floorLen;
    int64_t len = targetLen < lo ? lo : targetLen;
    if (len > ubLen) {
        len = ubLen;
    }
    return len < 1 ? 1 : len;
}

class Bn3dTrainingUpdateTiling {
public:
    explicit Bn3dTrainingUpdateTiling(gert::TilingContext* context) : context_(context) {}
    ge::graphStatus DoTiling();

private:
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus CheckInputs();
    ge::graphStatus CheckAttrs();
    ge::graphStatus TilingChannelFirst();
    ge::graphStatus TilingChannelLast();
    ge::graphStatus FillChannelFirst();
    ge::graphStatus FillChannelLast();

    gert::TilingContext* context_ = nullptr;
    Bn3dTrainingUpdatePlatform plat_{};
    ge::DataType xDtype_ = ge::DT_UNDEFINED;
    int64_t elemBytes_ = 0;
    bool channelLast_ = false;

    int64_t channel_ = 0; // C
    int64_t num_ = 0;     // num = N * S
    int64_t batch_ = 0;   // N
    int64_t spatial_ = 0; // S

    // channel-first
    int64_t spatialTile_ = 0;
    int64_t spatialTiles_ = 0;
    // channel-last
    int64_t channelWindow_ = 0; // CF: K
    int64_t cWindows_ = 0;      // CF: ceil(C/K)
    int64_t channelBlock_ = 0;
    int64_t channelBlocks_ = 0;
    int64_t rowsPerTile_ = 0;
    int64_t tilesPerBlock_ = 0;
    int64_t coeffCap_ = 0;

    int64_t xTileCount_ = 0;
    int64_t baseTilesPerCore_ = 0;
    int64_t extraCoreCount_ = 0;
    int64_t statUnitCount_ = 0;
    int64_t statChunk_ = 0;
    int64_t baseStatPerCore_ = 0;
    int64_t extraStatCount_ = 0;
    int64_t usedCoreNum_ = 1;

    float numRec_ = 0.0f;
    float besselScaler_ = 0.0f;
    float factor_ = 0.0f;
    float oneMinusFactor_ = 0.0f;
    float epsilon_ = 0.0f;
};

} // namespace optiling

#endif // BN3_D_TRAINING_UPDATE_TILING_ARCH35_H
