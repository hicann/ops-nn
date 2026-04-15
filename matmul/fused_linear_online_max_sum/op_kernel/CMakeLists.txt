# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# 该文件主要影响：
# 1.调用asc_opc 工具 编译二进制kernel时， --simplified_key_mode/--impl_mode 选项中填写的值。
# 2.调用asc_opc工具 算子名.py 中auto_sync与compile_option中的配置项。
# 格式如下所示：
# [COMPUTE_UNITS ascendxx]          soc版本，支持配置多个unit，缺省为配置所有的soc（支持单独配置soc，单独配置优先级更高）
# [SIMPLIFIED_KEY 0/None]           缺省为0，最终的编译参数则是--simplified_key_mode=0，若设置为None，则不会携带该参数
# [AUTO_SYNC false]                 同步选项，缺省为true
# [IMPL_MODE high_performance]      高性能模式，缺省为high_performance[,optional] [optional]是可选的 
# [OPTIONS "option1" "option2"]     其他编译选项

add_kernel_sources(
    COMPUTE_UNITS ascend910b ascend910_93
    SIMPLIFIED_KEY 1
)