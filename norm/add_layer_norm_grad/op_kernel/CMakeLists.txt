# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.

# A2/A3 use the generic kernel. A5 uses the arch35 implementation.
add_kernel_sources(
    COMPUTE_UNITS ascend310p ascend910b ascend910_93
)
add_kernel_sources(
    KERNEL_SRC arch35/add_layer_norm_grad.cpp
    COMPUTE_UNITS ascend950
    AUTO_SYNC false
)
