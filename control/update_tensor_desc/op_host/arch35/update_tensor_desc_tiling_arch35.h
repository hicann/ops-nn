/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file update_tensor_desc_tiling_arch35.h
 * \brief UpdateTensorDesc 编译期信息（arch35 / Ascend 950）。
 *   kernel RMW 粒度（128×int64）与 shape 无关，无编译期形状/平台信息需缓存，
 *   故 CompileInfo 为空结构体。
 */

#ifndef OPS_NN_CONTROL_UPDATE_TENSOR_DESC_TILING_ARCH35_H
#define OPS_NN_CONTROL_UPDATE_TENSOR_DESC_TILING_ARCH35_H

namespace optiling {

struct UpdateTensorDescCompileInfo {};

} // namespace optiling

#endif // OPS_NN_CONTROL_UPDATE_TENSOR_DESC_TILING_ARCH35_H
