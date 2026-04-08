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
 * \file extension.cpp
 * \brief Python扩展模块初始化文件
 * 
 * 本文件实现了一个空的Python扩展模块_C，该模块用于从Python导入时加载共享库(.so)。
 * 当Python导入这个模块时，会自动执行TORCH_LIBRARY静态初始化器，从而注册所有自定义算子。
 * 
 * 这是PyTorch C++扩展的标准初始化模式，确保所有算子的schema和实现都被正确注册到PyTorch系统中。
 */

#include <Python.h>

extern "C"
{
    // Python C扩展模块初始化函数
    // 创建一个名为_C的模块，用于从Python导入时加载共享库
    // 该函数在Python执行"import _C"时被调用
    PyObject *PyInit__C(void)
    {
        static struct PyModuleDef module_def = {
            PyModuleDef_HEAD_INIT,
            "_C", /* name of module */  // 模块名称
            NULL, /* module documentation, may be NULL */  // 模块文档字符串(可选)
            -1,   /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */  // 模块状态大小
            NULL, /* methods */  // 模块方法表
        };
        return PyModule_Create(&module_def);
    }
}
