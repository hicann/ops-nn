/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file copy_out_with_params.h
 * \brief
 */
#ifndef MATMUL_TILE_COPY_OUT_COPY_OUT_WITH_PARAMS_H
#define MATMUL_TILE_COPY_OUT_COPY_OUT_WITH_PARAMS_H

#include "./tile_copy_policy.h"

namespace Act {
namespace Gemm {
namespace Tile {
/**
 * @struct Copy
 * @brief Copy struct for implementing data copy operations on the Ascend910B architecture
 * 
 * This struct is only valid under the following conditions: 
 * output position is GM, output format is either CubeFormat::ND or CubeFormat::ND_ALIGN, and non-quantization scenario
 * 
 * @param [in] OutputType: the type of the output tensor
 * @param [in] InputType: the type of the input tensor
 */
template <class OutputType, class InputType>
struct Copy<Arch::Ascend910B, CopyWithParams, void, OutputType, InputType,
            AscendC::Std::enable_if_t<OutputType::pos == AscendC::TPosition::GM // output to GM
                                      && (OutputType::format == CubeFormat::ND ||
                                          OutputType::format == CubeFormat::ND_ALIGN) // ND
                                      && !AscendC::Impl::Detail::IsQuantSenario<typename OutputType::T,
                                                                                typename InputType::T>()>> { // no quant
public:
    using DstT = typename OutputType::T;
    using SrcT = typename AscendC::GetMmDstType<typename InputType::T>::Type;

    __aicore__ Copy() = default;
    __aicore__ ~Copy() = default;

    /**
     * @brief Overloaded operator() for performing data copy operations
     * @param [in] gm: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgM: original M value
     * @param [in] orgN: original N value
     * @param [in] orgKc: original Kc value
     */
    __aicore__ inline void operator()(const AscendC::GlobalTensor<DstT>& gm, const AscendC::LocalTensor<SrcT>& co1Local,
                                      int32_t curRow, int32_t curCol, int32_t l0CTileHeight, int32_t l0CTileWidth,
                                      int32_t baseM, int32_t baseN, int32_t orgM, int32_t orgN, int32_t orgKc)
    {
#if __CCE_AICORE__ == 220
        uint32_t dimN = orgKc != 0 ? orgKc : orgN;
        constexpr uint32_t blockCount = AscendC::ONE_BLK_SIZE / sizeof(DstT);
        if constexpr (OutputType::format == CubeFormat::ND_ALIGN) {
            dimN = AscendC::Ceil(dimN, blockCount) * blockCount;
        }
        auto stride = dimN;
        int64_t dstOffset =
            static_cast<int64_t>(static_cast<int64_t>(curRow * baseM) * stride) + static_cast<int64_t>(curCol * baseN);

        AscendC::FixpipeParamsV220 params;
        params.nSize = static_cast<uint16_t>(l0CTileWidth);
        params.mSize = static_cast<uint16_t>(l0CTileHeight);
        params.srcStride = AscendC::CeilAlign(l0CTileHeight, AscendC::BLOCK_CUBE);
        params.dstStride = stride;
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_ROW_MAJOR>(gm[dstOffset], co1Local, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910B"); });
#endif
    }
};

/**
 * @struct Copy
 * @brief Copy struct for implementing data copy operations on the Ascend910B architecture
 * 
 * This struct is only valid under the following conditions: 
 * output position is GM, output format is CubeFormat::NZ, and non-quantization scenario
 * 
 * @param [in] OutputType: the type of the output tensor
 * @param [in] InputType: the type of the input tensor
 */
template <class OutputType, class InputType>
struct Copy<Arch::Ascend910B, CopyWithParams, void, OutputType, InputType,
            AscendC::Std::enable_if_t<OutputType::pos == AscendC::TPosition::GM // output to GM
                                      && OutputType::format == CubeFormat::NZ   // NZ
                                      && !AscendC::Impl::Detail::IsQuantSenario<typename OutputType::T,
                                                                                typename InputType::T>()>> { // no quant
public:
    using DstT = typename OutputType::T;
    using SrcT = typename AscendC::GetMmDstType<typename InputType::T>::Type;

    __aicore__ Copy() = default;
    __aicore__ ~Copy() = default;

    /**
     * @brief Overloaded operator() for copying data from CO1 to GM
     * @param [in] gm: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgM: original M value
     * @param [in] orgN: original N value
     * @param [in] orgKc: original Kc value
     */
    __aicore__ inline void operator()(const AscendC::GlobalTensor<DstT>& gm, const AscendC::LocalTensor<SrcT>& co1Local,
                                      int32_t curRow, int32_t curCol, int32_t l0CTileHeight, int32_t l0CTileWidth,
                                      int32_t baseM, int32_t baseN, int32_t orgM, int32_t orgN, int32_t orgKc)
    {
#if __CCE_AICORE__ == 220
        int64_t dstOffset =
            static_cast<int64_t>(curCol * baseN) * orgM + static_cast<int64_t>(curRow * baseM) * AscendC::BLOCK_CUBE;
        uint32_t stride =
            static_cast<uint32_t>((orgM - l0CTileHeight) * AscendC::BLOCK_CUBE * sizeof(DstT) / AscendC::ONE_BLK_SIZE);

        AscendC::FixpipeParamsV220 params;
        int32_t baseBlockWidth = AscendC::Ceil(l0CTileWidth, AscendC::BLOCK_CUBE);
        params.nSize = static_cast<uint16_t>(baseBlockWidth * AscendC::BLOCK_CUBE);
        stride =
            stride + static_cast<uint32_t>(l0CTileHeight * AscendC::BLOCK_CUBE * sizeof(SrcT) / AscendC::ONE_BLK_SIZE) *
                         sizeof(DstT) / sizeof(SrcT);
        params.mSize = static_cast<uint16_t>(l0CTileHeight);
        params.srcStride = AscendC::CeilAlign(l0CTileHeight, AscendC::BLOCK_CUBE);
        params.dstStride = stride;
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_NZ>(gm[dstOffset], co1Local, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910B"); });
#endif
    }
};

/**
 * @struct Copy
 * @brief Copy struct for implementing data copy operations on the Ascend910B architecture
 * 
 * This struct is only valid under the following conditions: output position is GM, 
 * output format is either CubeFormat::ND or CubeFormat::ND_ALIGN, 
 * and AscendC::Impl::Detail::IsQuantSenario<typename OutputType::T, typename InputType::T>() returns true
 * 
 * @param [in] OutputType: the type of the output tensor
 * @param [in] InputType: the type of the input tensor
 */
template <class OutputType, class InputType>
struct Copy<Arch::Ascend910B, CopyWithParams, void, OutputType, InputType,
            AscendC::Std::enable_if_t<
                OutputType::pos == AscendC::TPosition::GM                                                     // CO1->GM
                && (OutputType::format == CubeFormat::ND || OutputType::format == CubeFormat::ND_ALIGN)       // ND
                && AscendC::Impl::Detail::IsQuantSenario<typename OutputType::T, typename InputType::T>()>> { // quant
public:
    using DstT = typename OutputType::T;
    using SrcT = typename AscendC::GetMmDstType<typename InputType::T>::Type;

    __aicore__ Copy() = default;
    __aicore__ ~Copy() = default;

    /**
     * @brief Overloaded operator() for copying data from CO1 to GM
     * @param [in] gm: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgM: original M value
     * @param [in] orgN: original N value
     * @param [in] orgKc: original Kc value
     * @param [in] quantTensor: quantization tensor
     */
    __aicore__ inline void operator()(const AscendC::GlobalTensor<DstT>& gm, const AscendC::LocalTensor<SrcT>& co1Local,
                                      int32_t curRow, int32_t curCol, int32_t l0CTileHeight, int32_t l0CTileWidth,
                                      int32_t baseM, int32_t baseN, int32_t orgM, int32_t orgN, int32_t orgKc,
                                      const AscendC::LocalTensor<uint64_t>& quantTensor)
    {
#if __CCE_AICORE__ == 220
        CopyOutNZ2ND(gm, co1Local, curRow, curCol, l0CTileHeight, l0CTileWidth, baseM, baseN, orgKc != 0 ? orgKc : orgN,
                     quantTensor);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910B"); });
#endif
    }

    /**
     * @brief Overloaded operator() for copying data from CO1 to GM
     * @param [in] gm: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgM: original M value
     * @param [in] orgN: original N value
     * @param [in] orgKc: original Kc value
     * @param [in] quantScalar: quantization scalar tensor
     */
    __aicore__ inline void operator()(const AscendC::GlobalTensor<DstT>& gm, const AscendC::LocalTensor<SrcT>& co1Local,
                                      int32_t curRow, int32_t curCol, int32_t l0CTileHeight, int32_t l0CTileWidth,
                                      int32_t baseM, int32_t baseN, int32_t orgM, int32_t orgN, int32_t orgKc, uint64_t quantScalar)
    {
#if __CCE_AICORE__ == 220
        CopyOutNZ2ND(gm, co1Local, curRow, curCol, l0CTileHeight, l0CTileWidth, baseM, baseN, orgKc != 0 ? orgKc : orgN,
                     quantScalar);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910B"); });
#endif
    }

private:
#if __CCE_AICORE__ == 220
    /**
     * @brief Copy data from a local tensor to a global tensor in ND format
     * @param [in] T: quantization object type
     * @param [in] dst: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgN: original N value
     * @param [in] quantObject: the quantization object
     */
    template <typename T>
    __aicore__ inline void CopyOutNZ2ND(const AscendC::GlobalTensor<DstT>& dst,
                                        const AscendC::LocalTensor<SrcT>& co1Local, int32_t curRow, int32_t curCol,
                                        int32_t l0CTileHeight, int32_t l0CTileWidth, int32_t baseM, int32_t baseN,
                                        int32_t orgN, T quantObject)
    {
        auto stride = orgN;
        constexpr uint32_t blockCount = AscendC::ONE_BLK_SIZE / sizeof(DstT);
        if constexpr (OutputType::format == CubeFormat::ND_ALIGN) {
            stride = AscendC::Ceil(stride, blockCount) * blockCount;
        }
        int64_t dstOffset =
            static_cast<int64_t>(static_cast<int64_t>(curRow * baseM) * stride) + static_cast<int64_t>(curCol * baseN);

        AscendC::FixpipeParamsV220 params;
        params.nSize = static_cast<uint16_t>(l0CTileWidth);
        params.mSize = static_cast<uint16_t>(l0CTileHeight);
        params.srcStride = AscendC::CeilAlign(l0CTileHeight, AscendC::BLOCK_CUBE);
        params.dstStride = stride;
        CopyTensor(dst[dstOffset], co1Local, quantObject, params);
    }

    /**
     * @brief Copy data from a local tensor to a global tensor with quantization
     * @param [in] dst: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] quantTensor: quantization tensor
     * @param [in] params: the fixpipe parameters
     */
    __aicore__ inline void CopyTensor(const AscendC::GlobalTensor<DstT>& dst,
                                      const AscendC::LocalTensor<SrcT>& co1Local,
                                      const AscendC::LocalTensor<uint64_t>& quantTensor,
                                      AscendC::FixpipeParamsV220& params)
    {
        if constexpr (AscendC::IsSameType<SrcT, int32_t>::value && AscendC::IsSameType<DstT, half>::value) {
            params.quantPre = QuantMode_t::VDEQF16;
        } else if constexpr (AscendC::IsSameType<SrcT, int32_t>::value &&
                             (AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value)) {
            params.quantPre = QuantMode_t::VREQ8;
        } else if constexpr (AscendC::IsSameType<SrcT, float>::value &&
                             (AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value)) {
            params.quantPre = QuantMode_t::VQF322B8_PRE;
        }
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_ROW_MAJOR>(dst, co1Local, quantTensor, params);
    }

    /**
     * @brief Copy data from a local tensor to a global tensor with a scalar quantization value
     * @param [in] dst: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] quantScalar: quantization scalar
     * @param [in] params: the fixpipe parameters
     */
    __aicore__ inline static void CopyTensor(const AscendC::GlobalTensor<DstT>& dst,
                                             const AscendC::LocalTensor<SrcT>& co1Local, uint64_t quantScalar,
                                             AscendC::FixpipeParamsV220& params)
    {
        if constexpr (AscendC::IsSameType<SrcT, int32_t>::value && AscendC::IsSameType<DstT, half>::value) {
            params.quantPre = QuantMode_t::DEQF16;
        } else if constexpr (AscendC::IsSameType<SrcT, int32_t>::value &&
                             (AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value)) {
            params.quantPre = QuantMode_t::REQ8;
        } else if constexpr (AscendC::IsSameType<SrcT, float>::value &&
                             (AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value)) {
            params.quantPre = QuantMode_t::QF322B8_PRE;
        }
        params.deqScalar = quantScalar;
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_ROW_MAJOR>(dst, co1Local, params);
    }
#endif
};

/**
 * @struct Copy
 * @brief Template struct for data copy operations
 * 
 * This struct is only valid under the following conditions: output position is GM, 
 * output format is either CubeFormat::ND or CubeFormat::ND_ALIGN, 
 * and AscendC::Impl::Detail::IsQuantSenario<typename OutputType::T, typename InputType::T>() returns true
 * 
 * @param [in] OutputType: the type of the output tensor
 * @param [in] InputType: the type of the input tensor
 */
template <class OutputType, class InputType>
struct Copy<Arch::Ascend910B, CopyWithParams, void, OutputType, InputType,
            AscendC::Std::enable_if_t<
                OutputType::pos == AscendC::TPosition::GM                                                     // CO1->GM
                && OutputType::format == CubeFormat::NZ                                                       // NZ
                && AscendC::Impl::Detail::IsQuantSenario<typename OutputType::T, typename InputType::T>()>> { // quant
public:
    using DstT = typename OutputType::T;
    using SrcT = typename AscendC::GetMmDstType<typename InputType::T>::Type;

    __aicore__ Copy() = default;
    __aicore__ ~Copy() = default;

    /**
     * @brief Overloaded operator() for copying data from CO1 to GM
     * @param [in] gm: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgM: original M value
     * @param [in] orgN: original N value
     * @param [in] orgKc: original Kc value
     * @param [in] quantTensor: quantization tensor
     */
    __aicore__ inline void operator()(const AscendC::GlobalTensor<DstT>& gm, const AscendC::LocalTensor<SrcT>& co1Local,
                                      int32_t curRow, int32_t curCol, int32_t l0CTileHeight, int32_t l0CTileWidth,
                                      int32_t baseM, int32_t baseN, int32_t orgM, int32_t orgN, int32_t orgKc,
                                      const AscendC::LocalTensor<uint64_t>& quantTensor)
    {
#if __CCE_AICORE__ == 220
        CopyOutNZ2NZ(gm, co1Local, curRow, curCol, l0CTileHeight, l0CTileWidth, baseM, baseN, orgM, quantTensor);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910B"); });
#endif
    }

    /**
     * @brief Overloaded operator() for copying data from CO1 to GM
     * @param [in] gm: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgM: original M value
     * @param [in] orgN: original N value
     * @param [in] orgKc: original Kc value
     * @param [in] quantScalar: quantization scalar tensor
     */
    __aicore__ inline void operator()(const AscendC::GlobalTensor<DstT>& gm, const AscendC::LocalTensor<SrcT>& co1Local,
                                      int32_t curRow, int32_t curCol, int32_t l0CTileHeight, int32_t l0CTileWidth,
                                      int32_t baseM, int32_t baseN, int32_t orgM, int32_t orgN, int32_t orgKc, uint64_t quantScalar)
    {
#if __CCE_AICORE__ == 220
        CopyOutNZ2NZ(gm, co1Local, curRow, curCol, l0CTileHeight, l0CTileWidth, baseM, baseN, orgM, quantScalar);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910B"); });
#endif
    }

private:
#if __CCE_AICORE__ == 220
    /**
     * @brief Copy data from a local tensor to a global tensor with quantization support
     * @param [in] T: quantization object type
     * @param [in] dst: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgN: original N value
     * @param [in] quantObject: the quantization object
     */
    template <typename T>
    __aicore__ inline static void CopyOutNZ2NZ(const AscendC::GlobalTensor<DstT>& dst,
                                               const AscendC::LocalTensor<SrcT>& co1Local, int32_t curRow, int32_t curCol,
                                               int32_t l0CTileHeight, int32_t l0CTileWidth, int32_t baseM,
                                               int32_t baseN, int32_t orgM, T quantObject)
    {
        uint32_t stride =
            static_cast<uint32_t>((orgM - l0CTileHeight) * AscendC::BLOCK_CUBE * sizeof(DstT) / AscendC::ONE_BLK_SIZE) +
            static_cast<uint32_t>(l0CTileHeight * AscendC::BLOCK_CUBE * sizeof(SrcT) / AscendC::ONE_BLK_SIZE) *
                sizeof(DstT) / sizeof(SrcT);

        int32_t baseBlockWidth = AscendC::Ceil(l0CTileWidth, AscendC::BLOCK_CUBE);
        int64_t dstOffset =
            static_cast<int64_t>(curCol * baseN) * orgM + static_cast<int64_t>(curRow * baseM) * AscendC::BLOCK_CUBE;

        AscendC::FixpipeParamsV220 params;
        params.nSize = static_cast<uint16_t>(baseBlockWidth * AscendC::BLOCK_CUBE);
        params.mSize = static_cast<uint16_t>(l0CTileHeight);
        params.srcStride = AscendC::CeilAlign(l0CTileHeight, AscendC::BLOCK_CUBE);
        params.dstStride = stride;
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }
        CopyTensor(dst[dstOffset], co1Local, quantObject, params);
    }

    /**
     * @brief Copy data from a local tensor to a global tensor with quantization
     * @param [in] dst: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] quantTensor: quantization tensor
     * @param [in] params: the fixpipe parameters
     */
    __aicore__ inline static void CopyTensor(const AscendC::GlobalTensor<DstT>& dst,
                                             const AscendC::LocalTensor<SrcT>& co1Local,
                                             const AscendC::LocalTensor<uint64_t>& quantTensor,
                                             AscendC::FixpipeParamsV220& params)
    {
        if constexpr (AscendC::IsSameType<SrcT, int32_t>::value && AscendC::IsSameType<DstT, half>::value) {
            params.quantPre = QuantMode_t::VDEQF16;
        } else if constexpr (AscendC::IsSameType<SrcT, int32_t>::value &&
                             (AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value)) {
            params.quantPre = QuantMode_t::VREQ8;
        } else if constexpr (AscendC::IsSameType<SrcT, float>::value &&
                             (AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value)) {
            params.quantPre = QuantMode_t::VQF322B8_PRE;
        }
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_NZ>(dst, co1Local, quantTensor, params);
    }

    /**
     * @brief Copy data from a local tensor to a global tensor with a scalar quantization value
     * @param [in] dst: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] quantScalar: quantization scalar
     * @param [in] params: the fixpipe parameters
     */
    __aicore__ inline static void CopyTensor(const AscendC::GlobalTensor<DstT>& dst,
                                             const AscendC::LocalTensor<SrcT>& co1Local, uint64_t quantScalar,
                                             AscendC::FixpipeParamsV220& params)
    {
        if constexpr (AscendC::IsSameType<SrcT, int32_t>::value && AscendC::IsSameType<DstT, half>::value) {
            params.quantPre = QuantMode_t::DEQF16;
        } else if constexpr (AscendC::IsSameType<SrcT, int32_t>::value &&
                             (AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value)) {
            params.quantPre = QuantMode_t::REQ8;
        } else if constexpr (AscendC::IsSameType<SrcT, float>::value &&
                             (AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value)) {
            params.quantPre = QuantMode_t::QF322B8_PRE;
        }
        params.deqScalar = quantScalar;
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_NZ>(dst, co1Local, params);
    }
#endif
};

/**
 * @struct Copy
 * @brief Copy struct for Ascend910_95 architecture with specific parameters
 * @param [in] OutputType: the type of the output tensor
 * @param [in] InputType: the type of the input tensor
 */
template <class OutputType, class InputType>
struct Copy<Arch::Ascend910_95, CopyWithParams, void, OutputType, InputType,
            AscendC::Std::enable_if_t<OutputType::pos == AscendC::TPosition::GM // output to GM
                                      && (OutputType::format == CubeFormat::ND ||
                                          OutputType::format == CubeFormat::ND_ALIGN) // ND
                                      && !AscendC::Impl::Detail::IsQuantSenario<typename OutputType::T,
                                                                                typename InputType::T>()>> { // no quant
public:
    using DstT = typename OutputType::T;
    using SrcT = typename AscendC::GetMmDstType<typename InputType::T>::Type;

    __aicore__ Copy() = default;
    __aicore__ ~Copy() = default;

    /**
     * @brief Overloaded operator() for copying data from CO1 to GM
     * @param [in] gm: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgM: original M value
     * @param [in] orgN: original N value
     * @param [in] orgKc: original Kc value
     */
    __aicore__ inline void operator()(const AscendC::GlobalTensor<DstT>& gm, const AscendC::LocalTensor<SrcT>& co1Local,
                                      int32_t curRow, int32_t curCol, int32_t l0CTileHeight, int32_t l0CTileWidth,
                                      int32_t baseM, int32_t baseN, int32_t orgM, int32_t orgN, int32_t orgKc)
    {
#if defined(__DAV_C310__)
        uint32_t dimN = orgKc != 0 ? orgKc : orgN;
        constexpr uint32_t blockCount = AscendC::ONE_BLK_SIZE / sizeof(DstT);
        if constexpr (OutputType::format == CubeFormat::ND_ALIGN) {
            dimN = AscendC::CeilAlign(dimN, blockCount);
        }
        auto stride = dimN;
        int64_t dstOffset =
            static_cast<int64_t>(static_cast<int64_t>(curRow * baseM) * stride) + static_cast<int64_t>(curCol * baseN);

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> params;
        params.nSize = static_cast<uint16_t>(l0CTileWidth);
        if constexpr (OutputType::format == CubeFormat::ND_ALIGN) {
            params.nSize = AscendC::CeilAlign(params.nSize, blockCount);
            stride = AscendC::CeilAlign(stride, blockCount);
        }
        params.mSize = static_cast<uint16_t>(l0CTileHeight);
        params.srcStride = AscendC::CeilAlign(l0CTileHeight, AscendC::BLOCK_CUBE);
        params.dstStride = stride;
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_ROW_MAJOR>(gm[dstOffset], co1Local, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }
};

/**
 * @struct Copy
 * @brief Copy struct for Ascend910_95 architecture with specific parameters
 * @param [in] OutputType: the type of the output tensor
 * @param [in] InputType: the type of the input tensor
 */
template <class OutputType, class InputType>
struct Copy<Arch::Ascend910_95, CopyWithParams, void, OutputType, InputType,
            AscendC::Std::enable_if_t<OutputType::pos == AscendC::TPosition::GM // output to GM
                                      && OutputType::format == CubeFormat::NZ   // NZ
                                      && !AscendC::Impl::Detail::IsQuantSenario<typename OutputType::T,
                                                                                typename InputType::T>()>> { // no quant
public:
    using DstT = typename OutputType::T;
    using SrcT = typename AscendC::GetMmDstType<typename InputType::T>::Type;

    __aicore__ Copy() = default;
    __aicore__ ~Copy() = default;

    /**
     * @brief Overloaded operator() for copying data from CO1 to GM
     * @param [in] gm: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgM: original M value
     * @param [in] orgN: original N value
     * @param [in] orgKc: original Kc value
     */
    __aicore__ inline void operator()(const AscendC::GlobalTensor<DstT>& gm, const AscendC::LocalTensor<SrcT>& co1Local,
                                      int32_t curRow, int32_t curCol, int32_t l0CTileHeight, int32_t l0CTileWidth,
                                      int32_t baseM, int32_t baseN, int32_t orgM, int32_t orgN, int32_t orgKc)
    {
#if defined(__DAV_C310__)
        int64_t dstOffset =
            static_cast<int64_t>(curCol * baseN) * orgM + static_cast<int64_t>(curRow * baseM) * AscendC::BLOCK_CUBE;
        uint32_t stride =
            static_cast<uint32_t>((orgM - l0CTileHeight) * AscendC::BLOCK_CUBE * sizeof(DstT) / AscendC::ONE_BLK_SIZE);

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ> params;
        int32_t baseBlockWidth = AscendC::Ceil(l0CTileWidth, AscendC::BLOCK_CUBE);
        params.nSize = static_cast<uint16_t>(baseBlockWidth * AscendC::BLOCK_CUBE);
        stride =
            stride + static_cast<uint32_t>(l0CTileHeight * AscendC::BLOCK_CUBE * sizeof(SrcT) / AscendC::ONE_BLK_SIZE) *
                         sizeof(DstT) / sizeof(SrcT);
        params.mSize = static_cast<uint16_t>(l0CTileHeight);
        params.srcStride = AscendC::CeilAlign(l0CTileHeight, AscendC::BLOCK_CUBE);
        params.dstStride = stride;
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_NZ>(gm[dstOffset], co1Local, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }
};

/**
 * @struct Copy
 * @brief Copy struct for Ascend910_95 architecture with specific parameters
 * 
 * This struct is enabled only if the output is in UB, the format is ND or ND_ALIGN,
 * and it is not a quantization scenario
 * 
 * @param [in] OutputType: the type of the output tensor
 * @param [in] InputType: the type of the input tensor
 */
template <class OutputType, class InputType>
struct Copy<Arch::Ascend910_95, CopyWithParams, void, OutputType, InputType,
            AscendC::Std::enable_if_t<AscendC::PhyPosIsUB(OutputType::pos) // output to UB
                                      && (OutputType::format == CubeFormat::ND ||
                                          OutputType::format == CubeFormat::ND_ALIGN) // ND
                                      && !AscendC::Impl::Detail::IsQuantSenario<typename OutputType::T,
                                                                                typename InputType::T>()>> { // no quant
public:
    using DstT = typename OutputType::T;
    using SrcT = typename AscendC::GetMmDstType<typename InputType::T>::Type;

    __aicore__ Copy() = default;
    __aicore__ ~Copy() = default;

    /**
     * @brief Overloaded operator() for Ascend910_95 architecture
     * @param [in] co2Local: destination local tensor
     * @param [in] co1Local: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgM: original M value
     * @param [in] orgN: original N value
     * @param [in] orgKc: original Kc value
     * @param [in] id: sub-block id, default is 0
     */
    __aicore__ inline void operator()(const AscendC::LocalTensor<DstT>& co2Local,
                                      const AscendC::LocalTensor<SrcT>& co1Local, int32_t curRow, int32_t curCol,
                                      int32_t l0CTileHeight, int32_t l0CTileWidth, int32_t baseM, int32_t baseN,
                                      int32_t orgM, int32_t orgN, int32_t orgKc, uint8_t id = 0)
    {
#if defined(__DAV_C310__)
        uint32_t dimN = orgKc != 0 ? orgKc : orgN;
        constexpr uint32_t blockCount = AscendC::ONE_BLK_SIZE / sizeof(DstT);
        if constexpr (OutputType::format == CubeFormat::ND_ALIGN) {
            dimN = AscendC::Ceil(dimN, blockCount) * blockCount;
        }
        auto stride = dimN;
        int64_t dstOffset =
            static_cast<int64_t>(static_cast<int64_t>(curRow * baseM) * stride) + static_cast<int64_t>(curCol * baseN);

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> params;
        params.nSize = static_cast<uint16_t>(l0CTileWidth);
        if constexpr (OutputType::format == CubeFormat::ND_ALIGN) {
            params.nSize = AscendC::Ceil(params.nSize, blockCount) * blockCount;
            stride = AscendC::Ceil(stride, blockCount) * blockCount;
        }
        params.mSize = static_cast<uint16_t>(l0CTileHeight);
        params.srcStride = AscendC::CeilAlign(l0CTileHeight, AscendC::BLOCK_CUBE);
        params.dstStride = stride;
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }
        params.subBlockId = id;
        AscendC::Fixpipe<DstT, SrcT, AscendC::Impl::CFG_ROW_MAJOR_UB>(co2Local[dstOffset], co1Local, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }
};

/**
 * @struct Copy
 * @brief Copy struct for Ascend910_95 architecture with specific parameters
 * 
 * This struct is enabled only if the output is in UB, the format is NZ,
 * and it is not a quantization scenario
 * 
 * @param [in] OutputType: the type of the output tensor
 * @param [in] InputType: the type of the input tensor
 */
template <class OutputType, class InputType>
struct Copy<Arch::Ascend910_95, CopyWithParams, void, OutputType, InputType,
            AscendC::Std::enable_if_t<AscendC::PhyPosIsUB(OutputType::pos)    // output to UB
                                      && OutputType::format == CubeFormat::NZ // NZ
                                      && !AscendC::Impl::Detail::IsQuantSenario<typename OutputType::T,
                                                                                typename InputType::T>()>> { // no quant
public:
    using DstT = typename OutputType::T;
    using SrcT = typename AscendC::GetMmDstType<typename InputType::T>::Type;

    __aicore__ Copy() = default;
    __aicore__ ~Copy() = default;

    /**
     * @brief Overloaded operator() for Ascend910_95 architecture
     * @param [in] co2Local: destination local tensor
     * @param [in] co1Local: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgM: original M value
     * @param [in] orgN: original N value
     * @param [in] orgKc: original Kc value
     * @param [in] id: sub-block id, default is 0
     */
    __aicore__ inline void operator()(const AscendC::LocalTensor<DstT>& co2Local,
                                      const AscendC::LocalTensor<SrcT>& co1Local, int32_t curRow, int32_t curCol,
                                      int32_t l0CTileHeight, int32_t l0CTileWidth, int32_t baseM, int32_t baseN,
                                      int32_t orgM, int32_t orgN, int32_t orgKc, uint8_t id = 0)
    {
#if defined(__DAV_C310__)
        int64_t dstOffset =
            static_cast<int64_t>(curCol * baseN) * orgM + static_cast<int64_t>(curRow * baseM) * AscendC::BLOCK_CUBE;
        uint32_t stride =
            static_cast<uint32_t>((orgM - l0CTileHeight) * AscendC::BLOCK_CUBE * sizeof(DstT) / AscendC::ONE_BLK_SIZE);

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ> params;
        int32_t baseBlockWidth = AscendC::Ceil(l0CTileWidth, AscendC::BLOCK_CUBE);
        params.nSize = static_cast<uint16_t>(baseBlockWidth * AscendC::BLOCK_CUBE);
        stride =
            stride + static_cast<uint32_t>(l0CTileHeight * AscendC::BLOCK_CUBE * sizeof(SrcT) / AscendC::ONE_BLK_SIZE) *
                         sizeof(DstT) / sizeof(SrcT);
        params.mSize = static_cast<uint16_t>(l0CTileHeight);
        params.srcStride = AscendC::CeilAlign(l0CTileHeight, AscendC::BLOCK_CUBE);
        params.dstStride = stride;
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }
        params.subBlockId = id;
        AscendC::Fixpipe<DstT, SrcT, AscendC::Impl::CFG_NZ_UB>(co2Local[dstOffset], co1Local, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }
};

/**
 * @struct Copy
 * @brief Copy struct for Ascend910_95 architecture with specific parameters
 * 
 * This struct is only valid when the output is in GM, the format is either ND or ND_ALIGN, and
 * the AscendC::Impl::Detail::IsQuantSenario condition is satisfied
 * 
 * @param [in] OutputType: the type of the output tensor
 * @param [in] InputType: the type of the input tensor
 */
template <class OutputType, class InputType>
struct Copy<Arch::Ascend910_95, CopyWithParams, void, OutputType, InputType,
            AscendC::Std::enable_if_t<
                OutputType::pos == AscendC::TPosition::GM                                                     // CO1->GM
                && (OutputType::format == CubeFormat::ND || OutputType::format == CubeFormat::ND_ALIGN)       // ND
                && AscendC::Impl::Detail::IsQuantSenario<typename OutputType::T, typename InputType::T>()>> { // quant
public:
    using DstT = typename OutputType::T;
    using SrcT = typename AscendC::GetMmDstType<typename InputType::T>::Type;

    __aicore__ Copy() = default;
    __aicore__ ~Copy() = default;

    /**
     * @brief Overloaded operator() for Ascend910_95 architecture
     * @param [in] gm: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgM: original M value
     * @param [in] orgN: original N value
     * @param [in] orgKc: original Kc value
     * @param [in] quantTensor: quantization tensor
     */
    __aicore__ inline void operator()(const AscendC::GlobalTensor<DstT>& gm, const AscendC::LocalTensor<SrcT>& co1Local,
                                      int32_t curRow, int32_t curCol, int32_t l0CTileHeight, int32_t l0CTileWidth,
                                      int32_t baseM, int32_t baseN, int32_t orgM, int32_t orgN, int32_t orgKc,
                                      const AscendC::LocalTensor<uint64_t>& quantTensor)
    {
#if defined(__DAV_C310__)
        CopyOutNZ2ND(gm, co1Local, curRow, curCol, l0CTileHeight, l0CTileWidth, baseM, baseN, orgKc != 0 ? orgKc : orgN,
                     quantTensor);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }

    /**
     * @brief Overloaded operator() for Ascend910_95 architecture
     * @param [in] gm: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgM: original M value
     * @param [in] orgN: original N value
     * @param [in] orgKc: original Kc value
     * @param [in] quantScalar: quantization scalar tensor
     */
    __aicore__ inline void operator()(const AscendC::GlobalTensor<DstT>& gm, const AscendC::LocalTensor<SrcT>& co1Local,
                                      int32_t curRow, int32_t curCol, int32_t l0CTileHeight, int32_t l0CTileWidth,
                                      int32_t baseM, int32_t baseN, int32_t orgM, int32_t orgN, int32_t orgKc, uint64_t quantScalar)
    {
#if defined(__DAV_C310__)
        CopyOutNZ2ND(gm, co1Local, curRow, curCol, l0CTileHeight, l0CTileWidth, baseM, baseN, orgKc != 0 ? orgKc : orgN,
                     quantScalar);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }

private:
#if defined(__DAV_C310__)
    /**
     * @brief Copy data from a local tensor to a global tensor, supporting both quantized and non-quantized modes
     * @param [in] T: quantization object type
     * @param [in] dst: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgN: original N value
     * @param [in] quantObject: the quantization object
     */
    template <typename T>
    __aicore__ inline void CopyOutNZ2ND(const AscendC::GlobalTensor<DstT>& dst,
                                        const AscendC::LocalTensor<SrcT>& co1Local, int32_t curRow, int32_t curCol,
                                        int32_t l0CTileHeight, int32_t l0CTileWidth, int32_t baseM, int32_t baseN,
                                        int32_t orgN, T quantObject)
    {
        auto stride = orgN;
        constexpr uint32_t blockCount = AscendC::ONE_BLK_SIZE / sizeof(DstT);
        if constexpr (OutputType::format == CubeFormat::ND_ALIGN) {
            stride = AscendC::Ceil(stride, blockCount) * blockCount;
        }
        int64_t dstOffset =
            static_cast<int64_t>(static_cast<int64_t>(curRow * baseM) * stride) + static_cast<int64_t>(curCol * baseN);

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> params;
        params.nSize = static_cast<uint16_t>(l0CTileWidth);
        params.mSize = static_cast<uint16_t>(l0CTileHeight);
        params.srcStride = AscendC::CeilAlign(l0CTileHeight, AscendC::BLOCK_CUBE);
        params.dstStride = stride;
        CopyTensor(dst[dstOffset], co1Local, quantObject, params);
    }

    /**
     * @brief Copy a tensor, supporting both quantized and non-quantized modes
     * @param [in] dst: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] quantTensor: quantization tensor
     * @param [in] params: the fixpipe parameters
     */
    __aicore__ inline void CopyTensor(const AscendC::GlobalTensor<DstT>& dst,
                                      const AscendC::LocalTensor<SrcT>& co1Local,
                                      const AscendC::LocalTensor<uint64_t>& quantTensor,
                                      AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR>& params)
    {
        if constexpr (AscendC::IsSameTypeV<SrcT, int32_t> && AscendC::IsSameTypeV<DstT, bfloat16_t>) {
            params.quantPre = QuantMode_t::VQS322BF16_PRE;
        } else if constexpr (AscendC::IsSameTypeV<SrcT, float> &&
                             AscendC::IsTypeOneOfV<DstT, fp8_e4m3fn_t, fp8_e5m2_t>) {
            params.quantPre = QuantMode_t::VQF322FP8_PRE;
        } else if constexpr (AscendC::IsSameTypeV<SrcT, float> && AscendC::IsSameTypeV<DstT, hifloat8_t>) {
            params.quantPre = QuantMode_t::VQF322HIF8_PRE;
        } else if constexpr (AscendC::IsTypeOneOfV<SrcT, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t> &&
                             AscendC::IsSameTypeV<DstT, half>) {
            params.quantPre = QuantMode_t::VQF322F16_PRE;
        } else if constexpr (AscendC::IsTypeOneOfV<SrcT, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t> &&
                             AscendC::IsSameTypeV<DstT, bfloat16_t>) {
            params.quantPre = QuantMode_t::VQF322BF16_PRE;
        } else if (AscendC::IsTypeOneOfV<SrcT, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t> &&
                   AscendC::IsSameTypeV<DstT, float>) {
            params.quantPre = QuantMode_t::VQF322F32_PRE;
        }
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_ROW_MAJOR>(dst, co1Local, quantTensor, params);
    }

    /**
     * @brief Copy a tensor, supporting both quantized and non-quantized modes
     * @param [in] dst: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] quantScalar: quantization scalar
     * @param [in] params: the fixpipe parameters
     */
    __aicore__ inline static void CopyTensor(const AscendC::GlobalTensor<DstT>& dst,
                                             const AscendC::LocalTensor<SrcT>& co1Local, uint64_t quantScalar,
                                             AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR>& params)
    {
        if constexpr (AscendC::IsSameTypeV<SrcT, int32_t> && AscendC::IsSameTypeV<DstT, bfloat16_t>) {
            params.quantPre = QuantMode_t::QS322BF16_PRE;
        } else if constexpr (AscendC::IsSameTypeV<SrcT, float> &&
                             AscendC::IsTypeOneOfV<DstT, fp8_e4m3fn_t, fp8_e5m2_t>) {
            params.quantPre = QuantMode_t::QF322FP8_PRE;
        } else if constexpr (AscendC::IsSameTypeV<SrcT, float> && AscendC::IsSameTypeV<DstT, hifloat8_t>) {
            params.quantPre = QuantMode_t::QF322HIF8_PRE;
        } else if constexpr (AscendC::IsTypeOneOfV<SrcT, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t> &&
                             AscendC::IsSameTypeV<DstT, half>) {
            params.quantPre = QuantMode_t::QF322F16_PRE;
        } else if constexpr (AscendC::IsTypeOneOfV<SrcT, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t> &&
                             AscendC::IsSameTypeV<DstT, bfloat16_t>) {
            params.quantPre = QuantMode_t::QF322BF16_PRE;
        } else if (AscendC::IsTypeOneOfV<SrcT, fp8_e4m3fn_t, fp8_e5m2_t, hifloat8_t> &&
                   AscendC::IsSameTypeV<DstT, float>) {
            params.quantPre = QuantMode_t::QF322F32_PRE;
        }
        params.deqScalar = quantScalar;
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_ROW_MAJOR>(dst, co1Local, params);
    }
#endif
};

/**
 * @struct Copy
 * @brief Copy struct for Ascend910_95 architecture with specific parameters
 * 
 * This struct is only valid when the output is in GM, the format is NZ, and
 * the AscendC::Impl::Detail::IsQuantSenario condition is satisfied
 * 
 * @param [in] OutputType: the type of the output tensor
 * @param [in] InputType: the type of the input tensor
 */
template <class OutputType, class InputType>
struct Copy<Arch::Ascend910_95, CopyWithParams, void, OutputType, InputType,
            AscendC::Std::enable_if_t<
                OutputType::pos == AscendC::TPosition::GM                                                     // CO1->GM
                && OutputType::format == CubeFormat::NZ                                                       // NZ
                && AscendC::Impl::Detail::IsQuantSenario<typename OutputType::T, typename InputType::T>()>> { // quant
public:
    using DstT = typename OutputType::T;
    using SrcT = typename AscendC::GetMmDstType<typename InputType::T>::Type;

    __aicore__ Copy() = default;
    __aicore__ ~Copy() = default;

    /**
     * @brief Overloaded operator() for performing data copy operations
     * @param [in] gm: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgM: original M value
     * @param [in] orgN: original N value
     * @param [in] orgKc: original Kc value
     * @param [in] quantTensor: quantization tensor
     */
    __aicore__ inline void operator()(const AscendC::GlobalTensor<DstT>& gm, const AscendC::LocalTensor<SrcT>& co1Local,
                                      int32_t curRow, int32_t curCol, int32_t l0CTileHeight, int32_t l0CTileWidth,
                                      int32_t baseM, int32_t baseN, int32_t orgM, int32_t orgN, int32_t orgKc,
                                      const AscendC::LocalTensor<uint64_t>& quantTensor)
    {
#if defined(__DAV_C310__)
        CopyOutNZ2NZ(gm, co1Local, curRow, curCol, l0CTileHeight, l0CTileWidth, baseM, baseN, orgM, quantTensor);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }

    /**
     * @brief Overloaded operator() for performing data copy operations
     * @param [in] gm: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgM: original M value
     * @param [in] orgN: original N value
     * @param [in] orgKc: original Kc value
     * @param [in] quantScalar: quantization scalar tensor
     */
    __aicore__ inline void operator()(const AscendC::GlobalTensor<DstT>& gm, const AscendC::LocalTensor<SrcT>& co1Local,
                                      int32_t curRow, int32_t curCol, int32_t l0CTileHeight, int32_t l0CTileWidth,
                                      int32_t baseM, int32_t baseN, int32_t orgM, int32_t orgN, int32_t orgKc, uint64_t quantScalar)
    {
#if defined(__DAV_C310__)
        CopyOutNZ2NZ(gm, co1Local, curRow, curCol, l0CTileHeight, l0CTileWidth, baseM, baseN, orgM, quantScalar);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }

private:
#if defined(__DAV_C310__)
    /**
     * @brief Copy data from a local tensor to a global tensor, supporting both quantized and non-quantized modes
     * @param [in] T: quantization object type
     * @param [in] dst: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgN: original N value
     * @param [in] quantObject: the quantization object
     */
    template <typename T>
    __aicore__ inline static void CopyOutNZ2NZ(const AscendC::GlobalTensor<DstT>& dst,
                                               const AscendC::LocalTensor<SrcT>& co1Local, int32_t curRow, int32_t curCol,
                                               int32_t l0CTileHeight, int32_t l0CTileWidth, int32_t baseM,
                                               int32_t baseN, int32_t orgM, T quantObject)
    {
        uint32_t stride =
            static_cast<uint32_t>((orgM - l0CTileHeight) * AscendC::BLOCK_CUBE * sizeof(DstT) / AscendC::ONE_BLK_SIZE) +
            static_cast<uint32_t>(l0CTileHeight * AscendC::BLOCK_CUBE * sizeof(SrcT) / AscendC::ONE_BLK_SIZE) *
                sizeof(DstT) / sizeof(SrcT);

        int32_t baseBlockWidth = AscendC::Ceil(l0CTileWidth, AscendC::BLOCK_CUBE);
        int64_t dstOffset =
            static_cast<int64_t>(curCol * baseN) * orgM + static_cast<int64_t>(curRow * baseM) * AscendC::BLOCK_CUBE;

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ> params;
        params.nSize = static_cast<uint16_t>(baseBlockWidth * AscendC::BLOCK_CUBE);
        params.mSize = static_cast<uint16_t>(l0CTileHeight);
        params.srcStride = AscendC::CeilAlign(l0CTileHeight, AscendC::BLOCK_CUBE);
        params.dstStride = stride;
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }
        CopyTensor(dst[dstOffset], co1Local, quantObject, params);
    }

    /**
     * @brief Copy data from a local tensor to a global tensor with quantization
     * @param [in] dst: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] quantTensor: quantization tensor
     * @param [in] params: the fixpipe parameters
     */
    __aicore__ inline static void CopyTensor(const AscendC::GlobalTensor<DstT>& dst,
                                             const AscendC::LocalTensor<SrcT>& co1Local,
                                             const AscendC::LocalTensor<uint64_t>& quantTensor,
                                             AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ>& params)
    {
        if constexpr (AscendC::IsSameType<SrcT, int32_t>::value && AscendC::IsSameType<DstT, half>::value) {
            params.quantPre = QuantMode_t::VDEQF16;
        } else if constexpr (AscendC::IsSameType<SrcT, int32_t>::value &&
                             (AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value)) {
            params.quantPre = QuantMode_t::VREQ8;
        } else if constexpr (AscendC::IsSameType<SrcT, float>::value &&
                             (AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value)) {
            params.quantPre = QuantMode_t::VQF322B8_PRE;
        }
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_NZ>(dst, co1Local, quantTensor, params);
    }

    /**
     * @brief Copydata from a local tensor to a global tensor with quantization
     * @param [in] dst: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] quantScalar: quantization scalar
     * @param [in] params: the fixpipe parameters
     */
    __aicore__ inline static void CopyTensor(const AscendC::GlobalTensor<DstT>& dst,
                                             const AscendC::LocalTensor<SrcT>& co1Local, uint64_t quantScalar,
                                             AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ>& params)
    {
        if constexpr (AscendC::IsSameType<SrcT, int32_t>::value && AscendC::IsSameType<DstT, half>::value) {
            params.quantPre = QuantMode_t::DEQF16;
        } else if constexpr (AscendC::IsSameType<SrcT, int32_t>::value &&
                             (AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value)) {
            params.quantPre = QuantMode_t::REQ8;
        } else if constexpr (AscendC::IsSameType<SrcT, float>::value &&
                             (AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value)) {
            params.quantPre = QuantMode_t::QF322B8_PRE;
        }
        params.deqScalar = quantScalar;
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_NZ>(dst, co1Local, params);
    }
#endif
};

/**
 * @struct Copy
 * @brief Copy struct for Ascend910_95 architecture with specific parameters
 * 
 * This struct is only valid when the output is in UB, the format is either ND or ND_ALIGN, and
 * the AscendC::Impl::Detail::IsQuantSenario condition is satisfied
 * 
 * @param [in] OutputType: the type of the output tensor
 * @param [in] InputType: the type of the input tensor
 */
template <class OutputType, class InputType>
struct Copy<Arch::Ascend910_95, CopyWithParams, void, OutputType, InputType,
            AscendC::Std::enable_if_t<
                AscendC::PhyPosIsUB(OutputType::pos)                                                    // output to UB
                && (OutputType::format == CubeFormat::ND || OutputType::format == CubeFormat::ND_ALIGN) // ND
                && AscendC::Impl::Detail::IsQuantSenario<typename OutputType::T, typename InputType::T>()>> { // quant
public:
    using DstT = typename OutputType::T;
    using SrcT = typename AscendC::GetMmDstType<typename InputType::T>::Type;

    __aicore__ Copy() = default;
    __aicore__ ~Copy() = default;

    /**
     * @brief Overloaded operator() for performing data copy operations
     * @param [in] co2Local: destination local tensor
     * @param [in] co1Local: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgM: original M value
     * @param [in] orgN: original N value
     * @param [in] orgKc: original Kc value
     * @param [in] quantTensor: quantization tensor
     */
    __aicore__ inline void operator()(const AscendC::LocalTensor<DstT>& co2Local,
                                      const AscendC::LocalTensor<SrcT>& co1Local, int32_t curRow, int32_t curCol,
                                      int32_t l0CTileHeight, int32_t l0CTileWidth, int32_t baseM, int32_t baseN,
                                      int32_t orgM, int32_t orgN, int32_t orgKc, const AscendC::LocalTensor<uint64_t>& quantTensor)
    {
#if defined(__DAV_C310__)
        CopyOutNZ2ND(co2Local, co1Local, curRow, curCol, l0CTileHeight, l0CTileWidth, baseM, baseN,
                     orgKc != 0 ? orgKc : orgN, quantTensor);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }

    /**
     * @brief Overloaded operator() for performing data copy operations
     * @param [in] co2Local: destination local tensor
     * @param [in] co1Local: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgM: original M value
     * @param [in] orgN: original N value
     * @param [in] orgKc: original Kc value
     * @param [in] quantScalar: quantization scalar tensor
     */
    __aicore__ inline void operator()(const AscendC::LocalTensor<DstT>& co2Local,
                                      const AscendC::LocalTensor<SrcT>& co1Local, int32_t curRow, int32_t curCol,
                                      int32_t l0CTileHeight, int32_t l0CTileWidth, int32_t baseM, int32_t baseN,
                                      int32_t orgM, int32_t orgN, int32_t orgKc, uint64_t quantScalar)
    {
#if defined(__DAV_C310__)
        CopyOutNZ2ND(co2Local, co1Local, curRow, curCol, l0CTileHeight, l0CTileWidth, baseM, baseN,
                     orgKc != 0 ? orgKc : orgN, quantScalar);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }

private:
#if defined(__DAV_C310__)
    /**
     * @brief Copy data from a local tensor to a global tensor, supporting multiple data types and quantization modes
     * @param [in] T: quantization object type
     * @param [in] dst: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgN: original N value
     * @param [in] quantObject: the quantization object
     */
    template <typename T>
    __aicore__ inline void CopyOutNZ2ND(const AscendC::LocalTensor<DstT>& dst,
                                        const AscendC::LocalTensor<SrcT>& co1Local, int32_t curRow, int32_t curCol,
                                        int32_t l0CTileHeight, int32_t l0CTileWidth, int32_t baseM, int32_t baseN,
                                        int32_t orgN, T quantObject)
    {
        auto stride = orgN;
        constexpr uint32_t blockCount = AscendC::ONE_BLK_SIZE / sizeof(DstT);
        if constexpr (OutputType::format == CubeFormat::ND_ALIGN) {
            stride = AscendC::Ceil(stride, blockCount) * blockCount;
        }
        int64_t dstOffset =
            static_cast<int64_t>(static_cast<int64_t>(curRow * baseM) * stride) + static_cast<int64_t>(curCol * baseN);

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> params;
        params.nSize = static_cast<uint16_t>(l0CTileWidth);
        params.mSize = static_cast<uint16_t>(l0CTileHeight);
        params.srcStride = AscendC::CeilAlign(l0CTileHeight, AscendC::BLOCK_CUBE);
        params.dstStride = stride;
        CopyTensor(dst[dstOffset], co1Local, quantObject, params);
    }

    /**
     * @brief Copy data from a local tensor to a global tensor, supporting multiple data types and quantization modes
     * @param [in] dst: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] quantTensor: quantization tensor
     * @param [in] params: the fixpipe parameters
     */
    __aicore__ inline void CopyTensor(const AscendC::LocalTensor<DstT>& dst, const AscendC::LocalTensor<SrcT>& co1Local,
                                      const AscendC::LocalTensor<uint64_t>& quantTensor,
                                      AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR>& params)
    {
        if constexpr (AscendC::IsSameType<SrcT, int32_t>::value && AscendC::IsSameType<DstT, half>::value) {
            params.quantPre = QuantMode_t::VDEQF16;
        } else if constexpr (AscendC::IsSameType<SrcT, int32_t>::value &&
                             (AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value)) {
            params.quantPre = QuantMode_t::VREQ8;
        } else if constexpr (AscendC::IsSameType<SrcT, float>::value &&
                             (AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value)) {
            params.quantPre = QuantMode_t::VQF322B8_PRE;
        }
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_ROW_MAJOR>(dst, co1Local, quantTensor, params);
    }

    /**
     * @brief Copydata from a local tensor to a global tensor, supporting multiple data types and quantization modes
     * @param [in] dst: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] quantScalar: quantization scalar
     * @param [in] params: the fixpipe parameters
     */
    __aicore__ inline static void CopyTensor(const AscendC::LocalTensor<DstT>& dst,
                                             const AscendC::LocalTensor<SrcT>& co1Local, uint64_t quantScalar,
                                             AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR>& params)
    {
        if constexpr (AscendC::IsSameType<SrcT, int32_t>::value && AscendC::IsSameType<DstT, half>::value) {
            params.quantPre = QuantMode_t::DEQF16;
        } else if constexpr (AscendC::IsSameType<SrcT, int32_t>::value &&
                             (AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value)) {
            params.quantPre = QuantMode_t::REQ8;
        } else if constexpr (AscendC::IsSameType<SrcT, float>::value &&
                             (AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value)) {
            params.quantPre = QuantMode_t::QF322B8_PRE;
        }
        params.deqScalar = quantScalar;
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_ROW_MAJOR>(dst, co1Local, params);
    }
#endif
};

/**
 * @struct Copy
 * @brief Copy struct for Ascend910_95 architecture with specific parameters
 * 
 * This struct is only valid when the output is in UB, the format is either NZ, and
 * the AscendC::Impl::Detail::IsQuantSenario condition is satisfied
 * 
 * @param [in] OutputType: the type of the output tensor
 * @param [in] InputType: the type of the input tensor
 */
template <class OutputType, class InputType>
struct Copy<Arch::Ascend910_95, CopyWithParams, void, OutputType, InputType,
            AscendC::Std::enable_if_t<
                AscendC::PhyPosIsUB(OutputType::pos)    // output to UB
                && OutputType::format == CubeFormat::NZ // NZ
                && AscendC::Impl::Detail::IsQuantSenario<typename OutputType::T, typename InputType::T>()>> { // quant
public:
    using DstT = typename OutputType::T;
    using SrcT = typename AscendC::GetMmDstType<typename InputType::T>::Type;

    __aicore__ Copy() = default;
    __aicore__ ~Copy() = default;

    /**
     * @brief Overloaded operator() for performing data copy operations
     * @param [in] co2Local: destination local tensor
     * @param [in] co1Local: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgM: original M value
     * @param [in] orgN: original N value
     * @param [in] orgKc: original Kc value
     * @param [in] quantTensor: quantization tensor
     */
    __aicore__ inline void operator()(const AscendC::LocalTensor<DstT>& co2Local,
                                      const AscendC::LocalTensor<SrcT>& co1Local, int32_t curRow, int32_t curCol,
                                      int32_t l0CTileHeight, int32_t l0CTileWidth, int32_t baseM, int32_t baseN,
                                      int32_t orgM, int32_t orgN, int32_t orgKc, const AscendC::LocalTensor<uint64_t>& quantTensor)
    {
#if defined(__DAV_C310__)
        CopyOutNZ2NZ(co2Local, co1Local, curRow, curCol, l0CTileHeight, l0CTileWidth, baseM, baseN, orgM, quantTensor);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }

    /**
     * @brief Overloaded operator() for performing data copy operations
     * @param [in] co2Local: destination local tensor
     * @param [in] co1Local: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgM: original M value
     * @param [in] orgN: original N value
     * @param [in] orgKc: original Kc value
     * @param [in] quantScalar: quantization scalar tensor
     */
    __aicore__ inline void operator()(const AscendC::LocalTensor<DstT>& co2Local,
                                      const AscendC::LocalTensor<SrcT>& co1Local, int32_t curRow, int32_t curCol,
                                      int32_t l0CTileHeight, int32_t l0CTileWidth, int32_t baseM, int32_t baseN,
                                      int32_t orgM, int32_t orgN, int32_t orgKc, uint64_t quantScalar)
    {
#if defined(__DAV_C310__)
        CopyOutNZ2NZ(co2Local, co1Local, curRow, curCol, l0CTileHeight, l0CTileWidth, baseM, baseN, orgM, quantScalar);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Only support Ascend910_95"); });
#endif
    }

private:
#if defined(__DAV_C310__)
    /**
     * @brief Copy data from a local tensor to a global tensor, supporting both quantized and non-quantized modes
     * @param [in] T: quantization object type
     * @param [in] dst: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] curRow: current row index
     * @param [in] curCol: current column index
     * @param [in] l0CTileHeight: L0C tile height
     * @param [in] l0CTileWidth: L0C tile width
     * @param [in] baseM: base M value
     * @param [in] baseN: base N value
     * @param [in] orgN: original N value
     * @param [in] quantObject: the quantization object
     */
    template <typename T>
    __aicore__ inline static void CopyOutNZ2NZ(const AscendC::LocalTensor<DstT>& dst,
                                               const AscendC::LocalTensor<SrcT>& co1Local, int32_t curRow, int32_t curCol,
                                               int32_t l0CTileHeight, int32_t l0CTileWidth, int32_t baseM,
                                               int32_t baseN, int32_t orgM, T quantObject)
    {
        uint32_t stride =
            static_cast<uint32_t>((orgM - l0CTileHeight) * AscendC::BLOCK_CUBE * sizeof(DstT) / AscendC::ONE_BLK_SIZE) +
            static_cast<uint32_t>(l0CTileHeight * AscendC::BLOCK_CUBE * sizeof(SrcT) / AscendC::ONE_BLK_SIZE) *
                sizeof(DstT) / sizeof(SrcT);

        int32_t baseBlockWidth = AscendC::Ceil(l0CTileWidth, AscendC::BLOCK_CUBE);
        int64_t dstOffset =
            static_cast<int64_t>(curCol * baseN) * orgM + static_cast<int64_t>(curRow * baseM) * AscendC::BLOCK_CUBE;

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ> params;
        params.nSize = static_cast<uint16_t>(baseBlockWidth * AscendC::BLOCK_CUBE);
        params.mSize = static_cast<uint16_t>(l0CTileHeight);
        params.srcStride = AscendC::CeilAlign(l0CTileHeight, AscendC::BLOCK_CUBE);
        params.dstStride = stride;
        if constexpr (AscendC::IsSameType<DstT, half>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322F16;
        } else if constexpr (AscendC::IsSameType<DstT, bfloat16_t>::value && AscendC::IsSameType<SrcT, float>::value) {
            params.quantPre = QuantMode_t::F322BF16;
        }
        CopyTensor(dst[dstOffset], co1Local, quantObject, params);
    }

    /**
     * @brief Copy data from a local tensor to a global tensor, supporting both quantized and non-quantized modes
     * @param [in] dst: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] quantTensor: quantization tensor
     * @param [in] params: the fixpipe parameters
     */
    __aicore__ inline static void CopyTensor(const AscendC::LocalTensor<DstT>& dst,
                                             const AscendC::LocalTensor<SrcT>& co1Local,
                                             const AscendC::LocalTensor<uint64_t>& quantTensor,
                                             AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ>& params)
    {
        if constexpr (AscendC::IsSameType<SrcT, int32_t>::value && AscendC::IsSameType<DstT, half>::value) {
            params.quantPre = QuantMode_t::VDEQF16;
        } else if constexpr (AscendC::IsSameType<SrcT, int32_t>::value &&
                             (AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value)) {
            params.quantPre = QuantMode_t::VREQ8;
        } else if constexpr (AscendC::IsSameType<SrcT, float>::value &&
                             (AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value)) {
            params.quantPre = QuantMode_t::VQF322B8_PRE;
        }
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_NZ>(dst, co1Local, quantTensor, params);
    }

    /**
     * @brief Copy data from a local tensor to a global tensor, supporting both quantized and non-quantized modes
     * @param [in] dst: destination global tensor
     * @param [in] co1Local: source local tensor
     * @param [in] quantScalar: quantization scalar
     * @param [in] params: the fixpipe parameters
     */
    __aicore__ inline static void CopyTensor(const AscendC::LocalTensor<DstT>& dst,
                                             const AscendC::LocalTensor<SrcT>& co1Local, uint64_t quantScalar,
                                             AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ>& params)
    {
        if constexpr (AscendC::IsSameType<SrcT, int32_t>::value && AscendC::IsSameType<DstT, half>::value) {
            params.quantPre = QuantMode_t::DEQF16;
        } else if constexpr (AscendC::IsSameType<SrcT, int32_t>::value &&
                             (AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value)) {
            params.quantPre = QuantMode_t::REQ8;
        } else if constexpr (AscendC::IsSameType<SrcT, float>::value &&
                             (AscendC::IsSameType<DstT, int8_t>::value || AscendC::IsSameType<DstT, uint8_t>::value)) {
            params.quantPre = QuantMode_t::QF322B8_PRE;
        }
        params.deqScalar = quantScalar;
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_NZ>(dst, co1Local, params);
    }
#endif
};
} // namespace Tile
} // namespace Gemm
} // namespace Act
#endif
