/*
 * Copyright (c) 2026 联通（广东）产业互联网有限公司.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_INTERN_VL_ADD_RMS_NORM_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_INTERN_VL_ADD_RMS_NORM_H_

#include "graph/operator_reg.h"

namespace ge {
REG_OP(InternVlAddRmsNorm)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(residual, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(residual_out, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(epsilon, Float, 1e-6)
    .OP_END_FACTORY_REG(InternVlAddRmsNorm)
} // namespace ge

#endif
