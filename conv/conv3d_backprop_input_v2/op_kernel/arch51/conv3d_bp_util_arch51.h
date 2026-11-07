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
 * \file conv3d_bp_util_arch51.h
 * \brief
 */

#ifndef CONV3D_BP_UTIL_ARCH51_H
#define CONV3D_BP_UTIL_ARCH51_H

#include "kernel_common.h"
#include "basic_api/kernel_basic_intf.h"
#include "../../../inc/macro.h"

static __aicore__ inline uint64_t GetAicBlockIdx() { return AscendC::GetBlockIdx(); }

// API类中声明call函数的默认重载，支持任意类型任意数量的参数
#define DECLARE_DEFAULT_OVERLOADING_FUN(T, NAMESPACE) \
    template <class... Ts>                            \
    static __aicore__ inline NAMESPACE::TypeFalse call(T* self, Ts... args)

// 检查类T中是否有call(...)成员函数
#define CHECK_FUN(T, NAMESPACE, ...) (!IsSameType<decltype(T::call(__VA_ARGS__)), NAMESPACE::TypeFalse>::value)

/*
定义一个校验性的模板类，用于判断类型T是否具有模板成员函数MEMBER<U>
和宏DECLARE_IMPL配套使用，调用方式_has_impl_MEMBER<T, U>::value
49行：decltype获取表达式的类型，declval是模板函数，获取模板参数T的右值引用，如果T没有MEMBER<U>，会报错，否则返回TrueType
*/
#define DECLARE_CHECK_IMPL(MEMBER, args...)                                                                     \
    namespace __AuxCheckImpl {                                                                                  \
    template <typename T, typename U>                                                                           \
    struct _has_impl_##MEMBER {                                                                                 \
        template <typename C>                                                                                   \
        static auto check(int) -> decltype(std::declval<typename C::template MEMBER<U, ##args>>(), TrueType()); \
        template <typename C>                                                                                   \
        static FalseType check(...);                                                                            \
        static constexpr bool value = IsSameType<decltype(check<T>(0)), TrueType>::value;                       \
    };                                                                                                          \
    }

// 定义一个模板类，用于判断类型T是否具有模板成员函数MEMBER<typename U, bool sync>
#define DECLARE_CHECK_SYNC_IMPL(MEMBER, args...)                                                                      \
    namespace __AuxCheckImpl {                                                                                        \
    template <typename T, typename U, bool sync>                                                                      \
    struct _has_impl_##MEMBER {                                                                                       \
        template <typename C>                                                                                         \
        static auto check(int) -> decltype(std::declval<typename C::template MEMBER<U, sync, ##args>>(), TrueType()); \
        template <typename C>                                                                                         \
        static FalseType check(...);                                                                                  \
        static constexpr bool value = IsSameType<decltype(check<T>(0)), TrueType>::value;                             \
    };                                                                                                                \
    }

// 定义成员函数MEMBER<U>, 如果Config中存在MEMBER成员，MEMBER函数指向Config成员，否者指向Current::Init
#define DECLARE_IMPL(Config, Current, MEMBER, U)     \
    template <bool Default, class T>                 \
    struct __##MEMBER {                              \
        using Type = typename Current::MEMBER<U>;    \
    };                                               \
    template <class T>                               \
    struct __##MEMBER<true, T> {                     \
        using Type = typename T::template MEMBER<U>; \
    };                                               \
    using MEMBER = typename __##MEMBER<__AuxCheckImpl::_has_impl_##MEMBER<Config, U>::value, Config>::Type

// 定义成员函数MEMBER<U, sync>, 如果Config中存在MEMBER成员，MEMBER函数指向Config成员，否者指向Current::Init
#define DECLARE_SYNC_IMPL(Config, Current, MEMBER, U)      \
    template <bool Default, class T, bool sync>            \
    struct __##MEMBER {                                    \
        using Type = typename Current::MEMBER<U, sync>;    \
    };                                                     \
    template <class T, bool sync>                          \
    struct __##MEMBER<true, T, sync> {                     \
        using Type = typename T::template MEMBER<U, sync>; \
    };                                                     \
    template <bool sync>                                   \
    using MEMBER = typename __##MEMBER<__AuxCheckImpl::_has_impl_##MEMBER<Config, U, sync>::value, Config, sync>::Type

#define DEFINE_STUCT_FIELD(T, FIELD) T FIELD

#define DEFINE_STUCT_TEMPLATE_FIELD(T, FIELD, C, ...) T<C, ##__VA_ARGS__> FIELD

#define DEFINE_STUCT_ARRAY_FIELD(T, FIELD, NUM) T FIELD[NUM]

#endif
