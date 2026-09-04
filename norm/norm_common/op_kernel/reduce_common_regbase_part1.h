/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* Internal implementation section of reduce_common_regbase.h. Include only from reduce_common_regbase.h. */

template <typename U, bool HAS_XOUT = false, bool HAS_XOUT_FP32 = false, bool IS_RSTD = false>
__aicore__ inline void ReduceSumRstdMulti(LocalTensor<float>& rstdLocal, LocalTensor<U>& xOutLocal,
                                          LocalTensor<float>& xOutFp32Local, LocalTensor<U>& x1Local,
                                          LocalTensor<U>& x2Local, LocalTensor<float>& workLocal,
                                          uint32_t rstdOffsetStart, uint32_t count, uint32_t powerSplit,
                                          uint32_t repeatTimes, float avgFactor = 1.0f, float epsilon = 0.0f)
{
    uint32_t rstdOffset = rstdOffsetStart;
    uint32_t remainTile = count - powerSplit;
    uint16_t remainRepeats = remainTile / (2 * V_LENGTH);

    uint32_t masterTile = powerSplit - remainTile;
    uint16_t masterRepeats = masterTile / (2 * V_LENGTH);

    uint32_t mergeTile = powerSplit / (2 * V_LENGTH);
    uint16_t mergeRepeats = mergeTile / (2 * V_LENGTH);

    uint32_t meanTile = mergeRepeats == 0 ? mergeTile : mergeRepeats;

    __local_mem__ U* x1MainAddr = (__ubuf__ U*)x1Local.GetPhyAddr();
    __local_mem__ U* x1TailAddr = (__ubuf__ U*)x1Local.GetPhyAddr() + int64_t(powerSplit);
    __local_mem__ U* x1MasterAddr = (__ubuf__ U*)x1Local.GetPhyAddr() + int64_t(remainTile);
    __local_mem__ U* x2MainAddr = (__ubuf__ U*)x2Local.GetPhyAddr();
    __local_mem__ U* x2TailAddr = (__ubuf__ U*)x2Local.GetPhyAddr() + int64_t(powerSplit);
    __local_mem__ U* x2MasterAddr = (__ubuf__ U*)x2Local.GetPhyAddr() + int64_t(remainTile);
    __local_mem__ U *xOutMainAddr, *xOutTailAddr, *xOutMasterAddr;
    if constexpr (HAS_XOUT) {
        xOutMainAddr = (__ubuf__ U*)xOutLocal.GetPhyAddr();
        xOutTailAddr = (__ubuf__ U*)xOutLocal.GetPhyAddr() + int64_t(powerSplit);
        xOutMasterAddr = (__ubuf__ U*)xOutLocal.GetPhyAddr() + int64_t(remainTile);
    }
    __local_mem__ float *xOutFp32MainAddr, *xOutFp32TailAddr, *xOutFp32MasterAddr;
    if constexpr (HAS_XOUT_FP32) {
        xOutFp32MainAddr = (__ubuf__ float*)xOutFp32Local.GetPhyAddr();
        xOutFp32TailAddr = (__ubuf__ float*)xOutFp32Local.GetPhyAddr() + int64_t(powerSplit);
        xOutFp32MasterAddr = (__ubuf__ float*)xOutFp32Local.GetPhyAddr() + int64_t(remainTile);
    }
    __local_mem__ float* workAddr = (__ubuf__ float*)workLocal.GetPhyAddr();
    __local_mem__ float* rstdAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();

        for (uint16_t row = 0; row < (uint16_t)repeatTimes; row++) {
            uint32_t remainSreg = remainTile;
            uint32_t masterSreg = masterTile;
            uint32_t mergeSreg = mergeTile;
            uint32_t meanSreg = meanTile;
            RegTensor<U> x1MainA, x1MainB, x1TailA, x1TailB;
            RegTensor<U> x2MainA, x2MainB, x2TailA, x2TailB;
            RegTensor<float> mainA, mainB, tailA, tailB, vSum, vDupReg, rstdReg;
            MaskReg pregLoop;

            for (uint16_t i = 0; i < (uint16_t)remainRepeats; ++i) {
                pregLoop = UpdateMask<float>(remainSreg);
                uint16_t offset = i * 2 * V_LENGTH;
                // 1. Copy in reg
                LoadTwoCloseRegVF(x1MainA, x1MainB, x1MainAddr, offset);
                LoadTwoCloseRegVF(x1TailA, x1TailB, x1TailAddr, offset);
                LoadTwoCloseRegVF(x2MainA, x2MainB, x2MainAddr, offset);
                LoadTwoCloseRegVF(x2TailA, x2TailB, x2TailAddr, offset);
                // 2. Cast add
                CastAddVF(mainA, x1MainA, x2MainA, pregLoop);
                CastAddVF(tailA, x1TailA, x2TailA, pregLoop);
                CastAddVF(mainB, x1MainB, x2MainB, pregLoop);
                CastAddVF(tailB, x1TailB, x2TailB, pregLoop);
                if constexpr (HAS_XOUT) {
                    CastStoreTwoCloseRegVF(xOutMainAddr, mainA, mainB, offset, pregLoop);
                    CastStoreTwoCloseRegVF(xOutTailAddr, tailA, tailB, offset, pregLoop);
                }
                if constexpr (HAS_XOUT_FP32) {
                    CastStoreTwoCloseRegVF(xOutFp32MainAddr, mainA, mainB, offset, pregLoop);
                    CastStoreTwoCloseRegVF(xOutFp32TailAddr, tailA, tailB, offset, pregLoop);
                }
                // 3. Cal x^2
                Mul(mainA, mainA, mainA, pregLoop);
                Mul(tailA, tailA, tailA, pregLoop);
                Mul(mainB, mainB, mainB, pregLoop);
                Mul(tailB, tailB, tailB, pregLoop);
                Add(mainA, mainA, tailA, pregLoop);
                Add(mainB, mainB, tailB, pregLoop);
                Add(mainA, mainA, mainB, pregLoop);
                ReduceSum(vSum, mainA, pregLoop);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr + i, vSum, pregMerge);
            }
            for (uint16_t i = 0; i < (uint16_t)masterRepeats; ++i) {
                pregLoop = UpdateMask<float>(masterSreg);
                uint16_t offset = i * 2 * V_LENGTH;
                // 1. Copy in reg
                LoadTwoCloseRegVF(x1MainA, x1MainB, x1MasterAddr, offset);
                LoadTwoCloseRegVF(x2MainA, x2MainB, x2MasterAddr, offset);
                // 2. Cast add
                CastAddVF(mainA, x1MainA, x2MainA, pregLoop);
                CastAddVF(mainB, x1MainB, x2MainB, pregLoop);
                if constexpr (HAS_XOUT) {
                    CastStoreTwoCloseRegVF(xOutMasterAddr, mainA, mainB, offset, pregLoop);
                }
                if constexpr (HAS_XOUT_FP32) {
                    CastStoreTwoCloseRegVF(xOutFp32MasterAddr, mainA, mainB, offset, pregLoop);
                }
                // 3. Cal x^2
                Mul(mainA, mainA, mainA, pregLoop);
                Mul(mainB, mainB, mainB, pregLoop);
                Add(mainA, mainA, mainB, pregLoop);
                ReduceSum(vSum, mainA, pregLoop);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr + remainRepeats + i, vSum, pregMerge);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            for (uint16_t i = 0; i < (uint16_t)mergeRepeats; ++i) {
                pregLoop = UpdateMask<float>(mergeSreg);
                uint16_t offset = i * 2 * V_LENGTH;
                LoadTwoCloseRegVF(mainA, mainB, workAddr, offset);
                Add(mainA, mainA, mainB, pregLoop);
                ReduceSum(vSum, mainA, pregLoop);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(workAddr + i, vSum, pregMerge);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            pregLoop = UpdateMask<float>(meanSreg);
            DataCopy(mainA, workAddr);
            ReduceSum(vSum, mainA, pregLoop);
            if constexpr (IS_RSTD) {
                Muls(vSum, vSum, avgFactor, pregMerge);
                Adds(vSum, vSum, epsilon, pregMerge);
                Sqrt(vSum, vSum, pregMerge);
                Duplicate(vDupReg, float(1.0), pregMerge);
                Div(rstdReg, vDupReg, vSum, pregMerge);
            }
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdAddr + rstdOffset, rstdReg, pregMerge);

            rstdOffset++;
            x1MainAddr += int64_t(count);
            x1TailAddr += int64_t(count);
            x1MasterAddr += int64_t(count);
            x2MainAddr += int64_t(count);
            x2TailAddr += int64_t(count);
            x2MasterAddr += int64_t(count);
            if constexpr (HAS_XOUT) {
                xOutMainAddr += int64_t(count);
                xOutTailAddr += int64_t(count);
                xOutMasterAddr += int64_t(count);
            }
            if constexpr (HAS_XOUT_FP32) {
                xOutFp32MainAddr += int64_t(count);
                xOutFp32TailAddr += int64_t(count);
                xOutFp32MasterAddr += int64_t(count);
            }
        }
    }
}

template <bool NEED_MAX = true>
__aicore__ inline void ComputeRstdNewtonRaphsonReg(RegTensor<float>& var, RegTensor<float>& rstd, MaskReg& preg,
                                                   float epsilon)
{
    static constexpr float POS_INF = 3.40282366920938E+38;
    static constexpr float SCALAR1 = -0.5;
    static constexpr float SCALAR2 = 1.5;
    static constexpr float SCALAR3 = 0.5;
    static constexpr float SCALAR0 = -99.99;

    RegTensor<float> r;
    RegTensor<float> y;
    RegTensor<float> s;
    RegTensor<float> t;
    RegTensor<float> one;
    RegTensor<float> scalar1;
    RegTensor<float> t1;
    RegTensor<float> t3;
    RegTensor<float> t4;
    RegTensor<float> scalarInf;
    RegTensor<float> scalarZero;
    MaskReg cmpRegZero;
    MaskReg cmpRegInf;

    Duplicate(scalarInf, POS_INF, preg);
    Duplicate(scalarZero, float(0.0), preg);
    Duplicate(one, float(1.0), preg);
    Duplicate(scalar1, SCALAR3, preg);
    Duplicate(t1, SCALAR2, preg);
    Duplicate(s, float(1.0), preg);

    Adds(var, var, epsilon, preg);
    if constexpr (NEED_MAX) {
        Maxs(var, var, SCALAR0, preg);
    }
    Div(r, one, var, preg);
    Sqrt(y, r, preg);
    Muls(t, var, SCALAR1, preg);
    Mul(t, t, y, preg);
    Mula(t1, t, y, preg);
    Mul(rstd, y, t1, preg);
    Muls(t3, var, float(-1.0), preg);
    Mula(s, t3, r, preg);
    Muls(t4, rstd, float(-1.0), preg);
    Mula(r, t4, rstd, preg);
    Mula(s, var, r, preg);
    Mul(s, s, rstd, preg);
    Mula(rstd, s, scalar1, preg);
    CompareScalar(cmpRegZero, var, POS_INF, preg);
    Select(rstd, scalarZero, rstd, cmpRegZero);
    CompareScalar(cmpRegInf, var, float(0.0), preg);
    Select(rstd, scalarInf, rstd, cmpRegInf);
}

template <typename T>
__aicore__ inline void LoadTensorUnAlignForDtypeT(__local_mem__ T*& src, RegTensor<float>& dst,
                                                  AscendC::Reg::UnalignReg& uSrc, MaskReg& preg,
                                                  uint32_t postUpdateStride)
{
    if constexpr (IsSameType<T, float>::value) {
        AscendC::Reg::DataCopyUnAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(dst, uSrc, src,
                                                                                          postUpdateStride);
    } else {
        RegTensor<T> xB16;
        RegTensor<T> xB16Unpack;
        AscendC::Reg::DataCopyUnAlign<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(xB16, uSrc, src,
                                                                                      postUpdateStride);
        UnPack((RegTensor<uint32_t>&)xB16Unpack, (RegTensor<uint16_t>&)xB16);
        Cast<float, T, castTraitB162B32>(dst, xB16Unpack, preg);
    }
}

template <typename T>
__aicore__ inline void StoreTensorUnAlignForDtypeT(__local_mem__ T*& dst, RegTensor<float>& src,
                                                   AscendC::Reg::UnalignReg& uDst, MaskReg& preg,
                                                   uint32_t postUpdateStride)
{
    if constexpr (IsSameType<T, float>::value) {
        AscendC::Reg::DataCopyUnAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(dst, src, uDst,
                                                                                          postUpdateStride);
    } else {
        RegTensor<T> xB16;
        RegTensor<T> xB16Pack;
        Cast<T, float, castTraitB322B16>(xB16, src, preg);
        Pack((RegTensor<uint16_t>&)xB16Pack, (RegTensor<uint32_t>&)xB16);
        AscendC::Reg::DataCopyUnAlign<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(dst, xB16Pack, uDst,
                                                                                      postUpdateStride);
    }
}

template <typename T>
__aicore__ inline void LoadTensorUnAlignForDtypeT(__local_mem__ T* src, RegTensor<float>& dst, MaskReg& preg,
                                                  uint32_t postUpdateStride)
{
    AscendC::Reg::UnalignReg uSrc;
    __local_mem__ T* srcTmp = src;
    AscendC::Reg::DataCopyUnAlignPre(uSrc, srcTmp);
    LoadTensorUnAlignForDtypeT(srcTmp, dst, uSrc, preg, postUpdateStride);
}

template <typename T>
__aicore__ inline void StoreTensorUnAlignForDtypeT(__local_mem__ T* dst, RegTensor<float>& src, MaskReg& preg,
                                                   uint32_t postUpdateStride)
{
    AscendC::Reg::UnalignReg uDst;
    __local_mem__ T* dstTmp = dst;
    StoreTensorUnAlignForDtypeT(dstTmp, src, uDst, preg, postUpdateStride);
    AscendC::Reg::DataCopyUnAlignPost(dstTmp, uDst, 0);
}

// NOTE: x is overwritten in place (x = (x - mean) * scale * rstd); only y is the
// downstream-usable result. Callers must not rely on the original x after this call.
__aicore__ inline void NormalizeWithScaleBiasReg(RegTensor<float>& x, RegTensor<float>& scale, RegTensor<float>& bias,
                                                 RegTensor<float>& mean, RegTensor<float>& rstd, RegTensor<float>& y,
                                                 MaskReg& preg)
{
    Sub(x, x, mean, preg);
    Mul(x, x, scale, preg);
    Mul(x, x, rstd, preg);
    Add(y, x, bias, preg);
}

template <bool NEED_MAX = true, bool NEED_AVG_FACTOR = false>
__aicore__ inline void ComputeRstdNewtonRaphson(__local_mem__ float* src, __local_mem__ float* dst, uint32_t rowCount,
                                                float epsilon, float avgFactor = 1.0f, uint32_t vectorLen = V_LENGTH)
{
    uint16_t loopRows = static_cast<uint16_t>((rowCount + vectorLen - 1) / vectorLen);
    __VEC_SCOPE__
    {
        RegTensor<float> var;
        RegTensor<float> rstd;
        MaskReg pregLoop;

        uint32_t sreg = rowCount;
        for (uint16_t i = 0; i < loopRows; ++i) {
            pregLoop = UpdateMask<float>(sreg);
            DataCopy(var, src + i * vectorLen);
            if constexpr (NEED_AVG_FACTOR) {
                Muls(var, var, avgFactor, pregLoop);
            }
            ComputeRstdNewtonRaphsonReg<NEED_MAX>(var, rstd, pregLoop, epsilon);
            DataCopy(dst + i * vectorLen, rstd, pregLoop);
        }
    }
}

template <bool NEED_MAX = true, bool NEED_AVG_FACTOR = false>
__aicore__ inline void ComputeRstdNewtonRaphson(LocalTensor<float> srcLocal, LocalTensor<float> dstLocal,
                                                uint32_t rowCount, float epsilon, float avgFactor = 1.0f,
                                                uint32_t vectorLen = V_LENGTH)
{
    __local_mem__ float* src = (__local_mem__ float*)srcLocal.GetPhyAddr();
    __local_mem__ float* dst = (__local_mem__ float*)dstLocal.GetPhyAddr();
    ComputeRstdNewtonRaphson<NEED_MAX, NEED_AVG_FACTOR>(src, dst, rowCount, epsilon, avgFactor, vectorLen);
}

/*!
 * @brief Compute ReduceSum mean
 *        IS_RSTD: if True, will cal rstd otherwise sum.
 * @param dstLocal dst levelTensor
 * @param srcLocal src LevelTensor
 * @param offset dst offset
 * @param count src level size, must be ONCE_VECTOR_SIZE
 * @param avgFactor avgFactor for cal rstd
 * @param epsilon epsilon for cal rstd
 * @return
 */
template <bool IS_RSTD>
__aicore__ inline void LevelMergeRstd(LocalTensor<float>& dstLocal, LocalTensor<float> srcLocal, uint64_t offset,
                                      uint32_t count, float avgFactor = 1.0f, float epsilon = 0.0f)
{
    uint64_t calCount = count / 4; // Div 4 for VF parallel execution.
    uint32_t sreg = (uint32_t)(calCount);
    uint16_t repeatTimes = CeilDivision(calCount, V_LENGTH);
    uint32_t meanTile = repeatTimes;

    __local_mem__ float* src1Addr = (__ubuf__ float*)srcLocal.GetPhyAddr() + 0 * calCount;
    __local_mem__ float* src2Addr = (__ubuf__ float*)srcLocal.GetPhyAddr() + 1 * calCount;
    __local_mem__ float* src3Addr = (__ubuf__ float*)srcLocal.GetPhyAddr() + 2 * calCount;
    __local_mem__ float* src4Addr = (__ubuf__ float*)srcLocal.GetPhyAddr() + 3 * calCount;
    __local_mem__ float* dstAddr = (__ubuf__ float*)dstLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> vRegA, vRegB, vRegC, vRegD, dstReg, vSum, vDupReg;
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        MaskReg pregLoop;
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            pregLoop = UpdateMask<float>(sreg);
            DataCopy(vRegA, src1Addr + i * V_LENGTH);
            DataCopy(vRegB, src2Addr + i * V_LENGTH);
            DataCopy(vRegC, src3Addr + i * V_LENGTH);
            DataCopy(vRegD, src4Addr + i * V_LENGTH);
            Add(vRegA, vRegA, vRegB, pregLoop);
            Add(vRegC, vRegC, vRegD, pregLoop);
            Add(dstReg, vRegA, vRegC, pregLoop);
            ReduceSum(vSum, dstReg, pregLoop);
            if constexpr (IS_RSTD) {
                Muls(vSum, vSum, avgFactor, pregMerge);
                Adds(vSum, vSum, epsilon, pregMerge);
                Sqrt(vSum, vSum, pregMerge);
                Duplicate(vDupReg, float(1.0), pregMerge);
                Div(vSum, vDupReg, vSum, pregMerge);
            }
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstAddr + offset, vSum, pregMerge);
        }
    }
}

/*!
 * @brief compute final ReduceSum result
 *        IS_RSTD: if True, will cal rstd otherwise sum.
 * @param dstLocal dst Tensor
 * @param offset dst offset
 * @param level1Local level1 Tensor
 * @param level2Local level2 Tensor
 * @param level3Local level3 Tensor
 * @param level1 level1 elements
 * @param level2 level2 elements
 * @param level3 level3 elements
 * @param avgFactor avgFactor for cal rstd
 * @param epsilon epsilon for cal rstd
 * @return
 */
template <bool IS_RSTD>
__aicore__ inline void ComputeMultiLevelRstd(LocalTensor<float>& dstLocal, uint32_t offset,
                                             LocalTensor<float>& level1Local, LocalTensor<float>& level2Local,
                                             LocalTensor<float>& level3Local, uint32_t& level1, uint32_t& level2,
                                             float avgFactor = 1.0f, float epsilon = 0.0f)
{
    if (level1 > 0 && level1 < ONCE_VECTOR_SIZE) {
        LevelMergeRstd<IS_RSTD>(dstLocal, level1Local, offset, ONCE_VECTOR_SIZE, avgFactor, epsilon);
    } else if (level2 > 0 && level2 < ONCE_VECTOR_SIZE) {
        LevelMergeRstd<IS_RSTD>(dstLocal, level2Local, offset, ONCE_VECTOR_SIZE, avgFactor, epsilon);
    } else {
        LevelMergeRstd<IS_RSTD>(dstLocal, level3Local, offset, ONCE_VECTOR_SIZE, avgFactor, epsilon);
    }
}
