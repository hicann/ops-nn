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
 * \file scatter_reduce_common_atomic_common.h
 * \brief 原子/UB 常驻路径的共用件: 硬件原子 dtype 判据、归约单位元填充、uint8 保序翻转、命中压缩、UB 缓冲规格。
 *
 * FRAGMENT header: #include'd from the dispatcher INSIDE namespace ScatterReduceCommon, after the
 * `using namespace AscendC`. 与 scatter_reduce_common_scalar.h 同款约定, 不独立编译。
 * 由 scatter_reduce_common_atomic.h 按职责拆出(原单文件 1027 行, 超 CodeCheck 500 行阈值)。
 */
#ifndef SCATTER_REDUCE_COMMON_ATOMIC_COMMON_H
#define SCATTER_REDUCE_COMMON_ATOMIC_COMMON_H

// 硬件原子 Max/Min 的 dtype 支持面, 取自 SetAtomicMax/SetAtomicMin 的"数据类型"章节
// (docs/zh/api/SIMD-API/basic_api/atomic_operations/SetAtomicMax_ISASI.md: int8_t/int16_t/half/
// bfloat16_t/int32_t/float)。uint8/int64 不在其中, 继续走排序路径。
template <typename T>
__aicore__ inline constexpr bool IsAtomicMaxMinSupported()
{
    return AscendC::IsSameType<T, float>::value || AscendC::IsSameType<T, half>::value ||
           AscendC::IsSameType<T, bfloat16_t>::value || AscendC::IsSameType<T, int32_t>::value ||
           AscendC::IsSameType<T, int16_t>::value || AscendC::IsSameType<T, int8_t>::value;
}

// uint8 没有硬件原子, 但 x^0x80 是 uint8 -> int8 的**保序双射**(0->-128, 127->-1, 128->0, 255->127),
// 于是 max_uint8(a,b) == max_int8(a^0x80, b^0x80) ^ 0x80, 整型运算无精度损失, 可以借 int8 的原子跑。
// 变换在 ss==1(AtomicReduceRow1D)与 ss>=2(AtomicReduceSliceN)都已接完整: var 原地翻转 +
// updates 逐 tile 翻转 + 收尾 DCCI/SyncAll 后翻回。判据因此对两个入口通用。
template <typename T>
__aicore__ inline constexpr bool IsAtomicMaxMinUsableRow1D()
{
    return IsAtomicMaxMinSupported<T>() || AscendC::IsSameType<T, uint8_t>::value;
}

// 每轮搬入的 index 数与批内槽数都由 host 按实测 UB 反解后经 tiling 下发(ResolveAtomicUbSpec:
// idx + updates + 槽 + pos 四块真的放得下才算, 装不下就折半, 保底各留一份、绝不拒收)。
// 这里只保留**编译期上限** —— 批内元数据数组开在标量栈上, 尺寸必须是编译期常量;
// 运行时实际条数取 min(tiling 下发值, 上限)。
constexpr uint32_t ATOMIC_TILE_MAX = 4096U;
// 在飞 granule 槽数。**抬它无效, 已实测**: 2026-10-28 把 256->1024 在写主导档(含靶子 L1_086/
// L1_1185)实测 6/6 全 1.000x。结论: 原子写已经是流水化的, 瓶颈不在"在飞数量不够", 而在
// 每次 32B 事务的固定开销(42.8ns, 与载荷字节无关)。交接件把"批次放大"列为失败尝试但理由写错了
// (它那次测在零原子写的扫描主导档上, 等于没测); 这里补上正确的证据, 别再试第三次。
constexpr uint32_t SLOT_NUM_MAX = 256U;

// 用 Duplicate 把 dst 填满 MAX/MIN 的单位元。浮点按位模式填 ∓inf(仓内惯例见
// index/apply_top_k_top_p_with_sorted/op_kernel/arch20: FLOAT32_n=0xFF800000 等);
// int8 借 int16 位模式填(0x8080/0x7F7F), 避开窄 dtype 的 Duplicate 支持面。
//
// 🔴 U8_XFORMED = 累加器是否活在 x^0x80 的**翻转域**。单位元取值由"域"决定, 不只由 dtype 决定:
//   - true : uint8 走变换路径(AtomicReduceRow1D / AtomicReduceSliceN), acc 实为 int8 域,
//            单位元取 int8 的 0x80/0x7F —— 与 var/updates 的翻转值同域;
//   - false: 在**原生 uint8 域**折叠(UbResidentReduceRow1D 的 workspace 跨核合并, 它刻意不做变换),
//            单位元必须取 uint8 自己的 0x00/0xFF。
// 填错会被数据掩盖而非报错: MAX 填成 0x80 时, 所有 <=128 的数据都被压成 128, 而含 255 的用例
// (泛化集的 *_extreme 轴)却因 max(128,255)=255 恰好通过 —— 假阴性。
// 实证: 泛化 case00064_*(uint8, ss=48) bin_precision=0.0%, 全 48 个输出元素恒为 128。
// int8 不受本参数影响(它没有变换域, 0x80 就是 INT8_MIN)。
// 32/16 位 lane: Duplicate 直接按 dtype 位模式填满。
template <uint8_t Mode, typename T>
__aicore__ inline void FillIdentityWideLane(const LocalTensor<T>& dst, uint32_t count)
{
    constexpr bool IS_MAX = (Mode == MODE_MAX);
    if constexpr (AscendC::IsSameType<T, float>::value) {
        constexpr int32_t NEG_INF_F32 = static_cast<int32_t>(0xFF800000);
        constexpr int32_t POS_INF_F32 = static_cast<int32_t>(0x7F800000);
        Duplicate(dst.template ReinterpretCast<int32_t>(), IS_MAX ? NEG_INF_F32 : POS_INF_F32, count);
    } else if constexpr (AscendC::IsSameType<T, half>::value) {
        constexpr int16_t NEG_INF_F16 = static_cast<int16_t>(0xFC00);
        constexpr int16_t POS_INF_F16 = static_cast<int16_t>(0x7C00);
        Duplicate(dst.template ReinterpretCast<int16_t>(), IS_MAX ? NEG_INF_F16 : POS_INF_F16, count);
    } else if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        constexpr int16_t NEG_INF_BF16 = static_cast<int16_t>(0xFF80);
        constexpr int16_t POS_INF_BF16 = static_cast<int16_t>(0x7F80);
        Duplicate(dst.template ReinterpretCast<int16_t>(), IS_MAX ? NEG_INF_BF16 : POS_INF_BF16, count);
    } else if constexpr (AscendC::IsSameType<T, int32_t>::value) {
        constexpr int32_t I32_LOW = static_cast<int32_t>(0x80000000);
        constexpr int32_t I32_HIGH = static_cast<int32_t>(0x7FFFFFFF);
        Duplicate(dst.template ReinterpretCast<int32_t>(), IS_MAX ? I32_LOW : I32_HIGH, count);
    } else { // int16_t
        constexpr int16_t I16_LOW = static_cast<int16_t>(0x8000);
        constexpr int16_t I16_HIGH = static_cast<int16_t>(0x7FFF);
        Duplicate(dst.template ReinterpretCast<int16_t>(), IS_MAX ? I16_LOW : I16_HIGH, count);
    }
}

// 单字节 lane(int8/uint8): 两个 lane 一组按 int16 位模式填 —— 窄 dtype 的 Duplicate 支持面不含 int8。
// 取上整而非 count/2: count 为奇数时下取整会漏掉最后一个字节, count==1 时更会算出 0 ——
// 那等于"槽没被初始化", UB 残留会被当真值原子写出(静默写错, 不报错)。
// 多填的那一个字节落在 InitBuffer 的 32B 对齐余量内(int8 张量的实际分配恒为 32B 整数倍),
// 不会越界; 且它必然在真值写入范围之外, 不影响结果。
template <uint8_t Mode, typename T, bool U8_XFORMED>
__aicore__ inline void FillIdentityBytePair(const LocalTensor<T>& dst, uint32_t count)
{
    constexpr bool IS_MAX = (Mode == MODE_MAX);
    const uint32_t n16 = (count + 1U) / 2U;
    if constexpr (AscendC::IsSameType<T, uint8_t>::value && !U8_XFORMED) {
        // 原生 uint8 域: 单位元是 uint8 自己的下/上界, 不是 int8 的
        constexpr int16_t U8_LOW_PAIR = static_cast<int16_t>(0x0000);  // (uint8)0 x2
        constexpr int16_t U8_HIGH_PAIR = static_cast<int16_t>(0xFFFF); // (uint8)255 x2
        Duplicate(dst.template ReinterpretCast<int16_t>(), IS_MAX ? U8_LOW_PAIR : U8_HIGH_PAIR, n16);
    } else {                                                           // int8_t, 或 uint8 的翻转域
        constexpr int16_t I8_LOW_PAIR = static_cast<int16_t>(0x8080);  // (int8)-128 x2
        constexpr int16_t I8_HIGH_PAIR = static_cast<int16_t>(0x7F7F); // (int8)127 x2
        Duplicate(dst.template ReinterpretCast<int16_t>(), IS_MAX ? I8_LOW_PAIR : I8_HIGH_PAIR, n16);
    }
}

// 按 lane 宽度分发。原先七个 dtype 分支挤在一个函数里, 圈复杂度 16 超阈值。
template <uint8_t Mode, typename T, bool U8_XFORMED>
__aicore__ inline void FillReduceIdentity(const LocalTensor<T>& dst, uint32_t count)
{
    if (count == 0) {
        return; // 调用方给 0 时直接返回: 各 Duplicate 对 count=0 的行为不在文档保证内
    }
    if constexpr (sizeof(T) == 1) {
        FillIdentityBytePair<Mode, T, U8_XFORMED>(dst, count);
    } else {
        FillIdentityWideLane<Mode, T>(dst, count);
    }
}

// uint8 要走原子, 必须先把值域映射到有硬件原子的 int8 上。
// 映射取 x ^ 0x80: 它是 uint8 -> int8 的保序双射(0->-128, 127->-1, 128->0, 255->127),
// 于是 max_uint8(a,b) == (max_int8(a^0x80, b^0x80)) ^ 0x80, 整型运算、无精度损失。
// 实现不用 Xor: 950 的 Xor 只在 regbase VF(RegTensor/MaskReg)里有, 基础 API 无。改用三步合成
// x^m = (x & ~m) | (~x & m), 在 uint16 lane 上做(Ands 在 950 只支持 int16/uint16, 不支持 uint8;
// Not/Or 支持 uint8 但为统一 lane 宽一并按 uint16 处理)。count 必为偶数: tile 均为 32B 整数倍。
constexpr uint16_t U8_SIGN_PAIR = 0x8080U; // 每字节的符号位
constexpr uint16_t U8_LOW_PAIR = 0x7F7FU;  // 每字节的低 7 位

template <typename T>
__aicore__ inline void FlipSignBitPerByte(const LocalTensor<T>& inout, const LocalTensor<T>& tmpA,
                                          const LocalTensor<T>& tmpB, uint32_t bytes)
{
    const uint32_t n16 = bytes / static_cast<uint32_t>(sizeof(uint16_t));
    LocalTensor<uint16_t> x = inout.template ReinterpretCast<uint16_t>();
    LocalTensor<uint16_t> a = tmpA.template ReinterpretCast<uint16_t>();
    LocalTensor<uint16_t> b = tmpB.template ReinterpretCast<uint16_t>();
    Ands(a, x, U8_LOW_PAIR, n16);
    Not(b, x, n16);
    PipeBarrier<PIPE_V>();
    Ands(b, b, U8_SIGN_PAIR, n16);
    PipeBarrier<PIPE_V>();
    Or(x, a, b, n16);
    PipeBarrier<PIPE_V>();
}

// var 原地做保序映射: 各核分担一段, 搬入 -> 翻符号位 -> 搬回。
// 🔴 暂存三块(x/a/b)由调用方传入(借 atomic 路径的 updUb_/xfA_/xfB_), 本函数**不得**再 InitBuffer:
// Begin/EndAtomicMode 两个调用点若各自 InitBuffer 3*XF_FLIP_TILE_BYTES, 会在 AllocBuffers 已占的
// 账本之外再叠两份(实测 ScatterMax_L0_005: 201024 + 2*49152 = 299328B > UB 253952B), 翻回阶段
// 整卡打出 UB、设备报 aivec MTE 错。借用时刻这三块恰好空闲: Begin 在首个 LoadTile 前; End 在末次
// FlushBatch 的 MTE3_MTE2 等待之后, updUb_ 上一轮 V 写已由 LoadTile 尾部的 V_S 等待排空
// (V->S 之后, S 发射序里后发的 MTE2 指令必然晚于 V 完成), 块内次序由既有的 V_MTE3/MTE3_MTE2 链保证。
// ⚠️ 不能对空区间提前 return: 调用方随后有集合性 SyncAll, 早退的核会把其他核挂死
// (同 scatter_reduce_common_sort.h:550 的告警)。所以空区间也要走完、只是不搬运。
template <typename PARAMS_T, typename ADDR_T>
__aicore__ inline void FlipVarInPlace(TPipe& pipe, GlobalTensor<PARAMS_T>& varGm, ADDR_T varElems, ADDR_T blockIdx,
                                      ADDR_T blockNum, const LocalTensor<PARAMS_T>& x, const LocalTensor<PARAMS_T>& a,
                                      const LocalTensor<PARAMS_T>& b, uint32_t scratchBytes)
{
    // 分块取 XF_FLIP_TILE_BYTES 与调用方暂存宽度的较小者, 保证 >=2 的偶数字节
    // (FlipSignBitPerByte 按 uint16 lane 处理, 奇数尾多翻的 1 字节落在 32B 对齐分配余量内)。
    uint32_t XF_TILE = ScatterReduceCommon::XF_FLIP_TILE_BYTES;
    if (scratchBytes < XF_TILE) {
        XF_TILE = (scratchBytes < 2U) ? 2U : (scratchBytes & ~1U);
    }
    const event_t e2v = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));
    const event_t ev3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
    const event_t e32 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));

    const ADDR_T per = (varElems + blockNum - 1) / blockNum;
    const ADDR_T begin = per * blockIdx;
    const ADDR_T cnt = (begin >= varElems) ? 0 : ((per < varElems - begin) ? per : varElems - begin);
    for (ADDR_T done = 0; done < cnt; done += static_cast<ADDR_T>(XF_TILE)) {
        const ADDR_T rest = cnt - done;
        const uint32_t n = static_cast<uint32_t>((rest < static_cast<ADDR_T>(XF_TILE)) ? rest :
                                                                                         static_cast<ADDR_T>(XF_TILE));
        DataCopyExtParams cp{1, n, 0, 0, 0}; // uint8: 元素数 == 字节数
        DataCopyPad(x, varGm[begin + done], cp, DataCopyPadExtParams<PARAMS_T>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(e2v);
        WaitFlag<HardEvent::MTE2_V>(e2v);
        // 尾段可能是奇数字节: 向上取偶做变换, 多翻的那一字节落在搬出长度之外, 不会写回。
        FlipSignBitPerByte<PARAMS_T>(x, a, b, (n + 1U) / 2U * 2U);
        SetFlag<HardEvent::V_MTE3>(ev3);
        WaitFlag<HardEvent::V_MTE3>(ev3);
        DataCopyPad(varGm[begin + done], x, cp);
        SetFlag<HardEvent::MTE3_MTE2>(e32);
        WaitFlag<HardEvent::MTE3_MTE2>(e32);
    }
    // 其它核随后要原子读改写这些字节, 必须先把本核的写回刷出缓存(同 sort 路径 DCCI + SyncAll 的处置)
    AscendC::DataCacheCleanAndInvalid<PARAMS_T, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(
        varGm);
    AscendC::SyncAll();
}

// 向量化命中筛选: 把 tile 内落在 [0, varFirstDim) 的元素位置压缩到 posUb, 返回命中数。
//
// 为什么必须做(实测依据): 稀疏档的耗时 100% 花在"逐元素标量读索引判越界"上, 一次原子写都没发生。
//   ScatterMax_L0_004: M=3.48 亿、索引全是 INT64_MAX(全越界), 实测 95263us
//   ≈ 每核 6.2M 次 idxUb.GetValue(j) × ~15ns。批次放大/双缓冲对这一档实测 1.00x(无效),
//   因为瓶颈不在批末等待, 而在 O(M) 的标量读。筛选向量化后标量工作量从 O(M) 降到 O(命中数)。
//
// 实现照 index/non_zero_with_value/op_kernel/arch35/non_zero_with_value_general.h 的压缩范式
// (CompactIdx 的 4 字节分支与 CompactIdxB64): Arange 位置斜坡 + CompareScalar 出掩码 +
// (8 字节再 MaskPack 到 int32 粒度) + GatherMask 压缩 + DataCopyUnAlign 追加 + GetSpr<AR> 取个数。
//
// 越界判定用**无符号单比较**: (unsigned)idx < (unsigned)varFirstDim 等价于 0<=idx<varFirstDim,
// 省掉"两次比较 + 掩码求与"。负索引在无符号下变成极大值, 自然落在界外。
// 64 位索引的"选一次": 比较 + 掩码打包 + 压缩写出命中位置。
// 与 32 位那份拆开, 是为了让 CompactInRangePos 主体保持短小(CodeCheck 函数长度阈值 50 行),
// 同 index/arg_max_grad 的 SelectLane32/SelectLaneB16/SelectLaneB8 一款做法。
template <typename INDICES_T>
__aicore__ inline void CompactInRangeLane64(const LocalTensor<INDICES_T>& idxUb, __ubuf__ int32_t* posPtr,
                                            uint16_t repeatElm, uint16_t repeatTimes, uint32_t& sreg1, int32_t advance,
                                            int64_t varFirstDim)
{
    auto idxPtr = (__ubuf__ uint64_t*)idxUb.GetPhyAddr();
    const uint64_t dimU = static_cast<uint64_t>(varFirstDim);
    __VEC_SCOPE__
    {
        AscendC::Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();
        AscendC::Reg::UnalignReg ureg;
        AscendC::Reg::RegTensor<int32_t> idsReg, sqzIdx;
        AscendC::Reg::RegTensor<uint64_t> xReg;
        AscendC::Reg::MaskReg cmpReg, packedMask;
        AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::Arange(idsReg, (int32_t)0);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            AscendC::Reg::MaskReg pnum = AscendC::Reg::UpdateMask<uint64_t>(sreg1);
            AscendC::Reg::AddrReg off = AscendC::Reg::CreateAddrReg<uint64_t>(i, repeatElm);
            AscendC::Reg::DataCopy(xReg, idxPtr, off);
            AscendC::Reg::CompareScalar<uint64_t, CMPMODE::LT>(cmpReg, xReg, dimU, pnum);
            AscendC::Reg::MaskPack<AscendC::Reg::HighLowPart::LOWEST>(packedMask, cmpReg);
            AscendC::Reg::GatherMask<int32_t, AscendC::Reg::GatherMaskMode::STORE_REG>(sqzIdx, idsReg, packedMask);
            AscendC::Reg::DataCopyUnAlign<int32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(posPtr, sqzIdx, ureg);
            AscendC::Reg::Adds(idsReg, idsReg, advance, preg);
        }
        AscendC::Reg::DataCopyUnAlignPost(posPtr, ureg);
    }
}

// 32 位索引的"选一次"。比 64 位那份少一步 MaskPack: 32 位比较结果的掩码宽度与 int32 位置寄存器
// 本就对齐, 不需要先把高低半部打包。
template <typename INDICES_T>
__aicore__ inline void CompactInRangeLane32(const LocalTensor<INDICES_T>& idxUb, __ubuf__ int32_t* posPtr,
                                            uint16_t repeatElm, uint16_t repeatTimes, uint32_t& sreg1, int32_t advance,
                                            int64_t varFirstDim)
{
    auto idxPtr = (__ubuf__ uint32_t*)idxUb.GetPhyAddr();
    // int32 索引的合法值 < 2^31, 故上界钳到 2^31: varFirstDim 更大时非负索引全部在界内,
    // 负索引按无符号恒 >= 2^31 仍被拒(无符号单边比较的前提)。直接窄化会回绕或放行负索引。
    constexpr int64_t kIdx32Sup = static_cast<int64_t>(1) << 31;
    const uint32_t dimU = (varFirstDim > kIdx32Sup) ? static_cast<uint32_t>(kIdx32Sup) :
                                                      static_cast<uint32_t>(varFirstDim);
    __VEC_SCOPE__
    {
        AscendC::Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();
        AscendC::Reg::UnalignReg ureg;
        AscendC::Reg::RegTensor<int32_t> idsReg, sqzIdx;
        AscendC::Reg::RegTensor<uint32_t> xReg;
        AscendC::Reg::MaskReg cmpReg;
        AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::Arange(idsReg, (int32_t)0);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            AscendC::Reg::MaskReg pnum = AscendC::Reg::UpdateMask<uint32_t>(sreg1);
            AscendC::Reg::AddrReg off = AscendC::Reg::CreateAddrReg<uint32_t>(i, repeatElm);
            AscendC::Reg::DataCopy(xReg, idxPtr, off);
            AscendC::Reg::CompareScalar<uint32_t, CMPMODE::LT>(cmpReg, xReg, dimU, pnum);
            AscendC::Reg::GatherMask<int32_t, AscendC::Reg::GatherMaskMode::STORE_REG>(sqzIdx, idsReg, cmpReg);
            AscendC::Reg::DataCopyUnAlign<int32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(posPtr, sqzIdx, ureg);
            AscendC::Reg::Adds(idsReg, idsReg, advance, preg);
        }
        AscendC::Reg::DataCopyUnAlignPost(posPtr, ureg);
    }
}

template <typename INDICES_T>
__aicore__ inline uint32_t CompactInRangePos(const LocalTensor<INDICES_T>& idxUb, const LocalTensor<int32_t>& posUb,
                                             uint32_t n, int64_t varFirstDim)
{
    if (n == 0) {
        return 0;
    }
    auto posPtr = (__ubuf__ int32_t*)posUb.GetPhyAddr();
    const uint16_t repeatElm = static_cast<uint16_t>(platform::GetVRegSize() / sizeof(INDICES_T));
    const uint16_t repeatTimes = static_cast<uint16_t>((n + repeatElm - 1U) / repeatElm);
    uint32_t sreg1 = n;
    const int32_t advance = static_cast<int32_t>(repeatElm);
    if constexpr (sizeof(INDICES_T) == sizeof(int64_t)) {
        CompactInRangeLane64<INDICES_T>(idxUb, posPtr, repeatElm, repeatTimes, sreg1, advance, varFirstDim);
    } else {
        CompactInRangeLane32<INDICES_T>(idxUb, posPtr, repeatElm, repeatTimes, sreg1, advance, varFirstDim);
    }
    return static_cast<uint32_t>(AscendC::Reg::GetSpr<AscendC::SpecialPurposeReg::AR>() / sizeof(int32_t));
}

// sliceSize==1, MAX/MIN: 按 M 均分到各核, 命中元素按 32B granule 原子写回。
// host 下发的 UB 缓冲规格。内核一律透传给 InitBuffer, 不再自己做 tile*sizeof(...) 的乘法 ——
// 那会让 host 的预算与内核的分配成为两套算式, 少算一个通道就 UB 越界
// (G2-kernel-initbuffer-selfsized; L1_105 整卡 aivec UB 越界就是这么来的)。
// 字段由 host ResolveAtomicUbSpec 用与预算同一组中间量算出, 按构造一致。
struct UbBufSpec {
    uint32_t idxBytes; // 索引缓冲
    uint32_t updBytes; // updates 缓冲(uint8 的 xfA/xfB 与之等宽, 预算里已按 3 份计)
    uint32_t posBytes; // 向量化筛选的命中位置缓冲
    uint32_t slotSpan; // granule 槽宽(元素数)
};

#endif // SCATTER_REDUCE_COMMON_ATOMIC_COMMON_H
