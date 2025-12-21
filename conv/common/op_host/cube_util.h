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
 * \file cube_util.h
 * \brief
 */
#ifndef OPS_CONV_OP_PROTO_CUBE_UTIL_H_
#define OPS_CONV_OP_PROTO_CUBE_UTIL_H_

#include "error_util.h"
#include "runtime/infer_shape_context.h"

namespace Ops {
#define OP_CHECK(cond, log_func, return_expr) \
    do {                                   \
        if (cond) {                        \
            log_func;                      \
            return_expr;                   \
        }                                  \
    } while (0)

constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;
constexpr int64_t UNKNOWN_RANK_DIM_VALUE_ = -2LL;

namespace NN {
namespace Conv {

struct Conv3DInputShapes {
    int64_t in = 0;
    int64_t id = 0;
    int64_t ih = 0;
    int64_t iw = 0;
    int64_t ic = 0;

    int64_t kn = 0;
    int64_t kd = 0;
    int64_t kh = 0;
    int64_t kw = 0;
    int64_t kc = 0;
};

struct Conv3DAttrs {
    bool ceil_mode = false; // only for AvgPool3D

    int64_t strd = 0;
    int64_t strh = 0;
    int64_t strw = 0;

    int64_t dild = 1;
    int64_t dilh = 1;
    int64_t dilw = 1;

    int64_t groups = 1;

    int64_t padf = 0;
    int64_t padb = 0;
    int64_t padu = 0;
    int64_t padd = 0;
    int64_t padl = 0;
    int64_t padr = 0;
};

struct Conv2DInputShapes {
    int64_t in = 0;
    int64_t ih = 0;
    int64_t iw = 0;
    int64_t ic = 0;

    int64_t kn = 0;
    int64_t kh = 0;
    int64_t kw = 0;
    int64_t kc = 0;
};

struct Conv2DAttrs {
    int64_t str_h = 0;
    int64_t str_w = 0;

    int64_t dil_h = 1;
    int64_t dil_w = 1;

    int64_t groups = 1;

    int64_t pad_u = 0;
    int64_t pad_d = 0;
    int64_t pad_l = 0;
    int64_t pad_r = 0;
};

// NDHWC
constexpr size_t kNDimNDHWCIdx = 0;
constexpr size_t kDDimNDHWCIdx = 1;
constexpr size_t kHDimNDHWCIdx = 2;
constexpr size_t kWDimNDHWCIdx = 3;
constexpr size_t kCDimNDHWCIdx = 4;

// NCDHW
constexpr size_t kNDimNCDHWIdx = 0;
constexpr size_t kCDimNCDHWIdx = 1;
constexpr size_t kDDimNCDHWIdx = 2;
constexpr size_t kHDimNCDHWIdx = 3;
constexpr size_t kWDimNCDHWIdx = 4;

// DHWCN
constexpr size_t kDDimDHWCNIdx = 0;
constexpr size_t kHDimDHWCNIdx = 1;
constexpr size_t kWDimDHWCNIdx = 2;
constexpr size_t kCDimDHWCNIdx = 3;
constexpr size_t kNDimDHWCNIdx = 4;

// NCHW
constexpr size_t kNDimNCHWIdx = 0;
constexpr size_t kCDimNCHWIdx = 1;
constexpr size_t kHDimNCHWIdx = 2;
constexpr size_t kWDimNCHWIdx = 3;

// NHWC
constexpr size_t kNDimNHWCIdx = 0;
constexpr size_t kHDimNHWCIdx = 1;
constexpr size_t kWDimNHWCIdx = 2;
constexpr size_t kCDimNHWCIdx = 3;

// HWCN
constexpr size_t kHDimHWCNIdx = 0;
constexpr size_t kWDimHWCNIdx = 1;
constexpr size_t kCDimHWCNIdx = 2;
constexpr size_t kNDimHWCNIdx = 3;

// ND
constexpr size_t kDimNDIdx = 0;

constexpr size_t kDDimUnsupportIdx = -1;

// Deconv Strides IDX
constexpr size_t kHDimNCHWStridesIdx = 0;
constexpr size_t kWDimNCHWStridesIdx = 1;

/*
 * @brief: check whether the output shape is unknown rank
 * @param [out] output_shape: the output shape ptr
 * @return ge::graphStatus
 */
inline bool IsUnknownRank(const gert::Shape* check_shape) {
  return check_shape->GetDimNum() == 1 && check_shape->GetDim(0) == UNKNOWN_RANK_DIM_VALUE_;
}

}  // namespace Conv
}  // namespace NN
}  // namespace Ops

#endif // OPS_CONV_OP_PROTO_CUBE_UTIL_H_