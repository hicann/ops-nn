/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_EMU_LAYOUT_TUPLE_HPP
#define MATMUL_EMU_LAYOUT_TUPLE_HPP

#include <tuple>

#include "../catlass/matmul_emu_catlass.hpp"
#include "../catlass/arch/matmul_emu_arch.hpp"
#include "../catlass/layout/matmul_emu_layout.hpp"

#define TLA_REQUIRES(...) typename std::enable_if<(__VA_ARGS__)>::type* = nullptr

#define TLA_REQUIRES_T(...) typename std::enable_if<(__VA_ARGS__)>::type

namespace tla {

// using std::remove_cvref;
template <class T>
struct remove_cvref {
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

// using std::remove_cvref_t;
template <class T>
using remove_cvref_t = typename remove_cvref<T>::type;

// tuple_size, tuple_element
template <class T, class = void>
struct tuple_size;

template <class T>
struct tuple_size<T, std::void_t<typename std::tuple_size<T>::type>>
    : std::integral_constant<size_t, std::tuple_size<T>::value> {};

template <class T>
constexpr size_t tuple_size_v = tuple_size<T>::value;

} // end namespace tla

namespace tla {
//
// Common Operations
//

template <class T, class U, TLA_REQUIRES(std::is_arithmetic<T>::value&& std::is_arithmetic<U>::value)>
CATLASS_HOST_DEVICE constexpr auto max(T const& t, U const& u)
{
    return t < u ? u : t;
}

template <class T, class U, TLA_REQUIRES(std::is_arithmetic<T>::value&& std::is_arithmetic<U>::value)>
CATLASS_HOST_DEVICE constexpr auto min(T const& t, U const& u)
{
    return t < u ? t : u;
}

template <class T, class U, TLA_REQUIRES(std::is_arithmetic<T>::value&& std::is_arithmetic<U>::value)>
CATLASS_HOST_DEVICE constexpr auto add(T const& t, U const& u)
{
    return t + u;
}

// A constant value: short name and type-deduction for fast compilation
template <auto v>
struct C {
    using type = C<v>;
    static constexpr auto value = v;
    using value_type = decltype(v);
    CATLASS_HOST_DEVICE constexpr operator value_type() const noexcept { return value; }
    CATLASS_HOST_DEVICE constexpr value_type operator()() const noexcept { return value; }
};

// Deprecate
template <class T, T v>
using constant = C<v>;

template <bool b>
using bool_constant = C<b>;

using true_type = bool_constant<true>;
using false_type = bool_constant<false>;

template <class T>
using is_std_integral = std::is_integral<T>;

// A more std:: conforming integral_constant that enforces type but interops with C<v>
template <class T, T v>
struct integral_constant : C<v> {
    using type = integral_constant<T, v>;
    static constexpr T value = v;
    using value_type = T;
    CATLASS_HOST_DEVICE constexpr value_type operator()() const noexcept { return value; }
};

// Use tla::is_std_integral<T> to match built-in integral types (int, int64_t, unsigned, etc)
// Use tla::is_integral<T> to match both built-in integral types AND static integral types.

template <class T>
struct is_integral : bool_constant<is_std_integral<T>::value> {};
template <auto v>
struct is_integral<C<v>> : true_type {};
template <class T, T v>
struct is_integral<integral_constant<T, v>> : true_type {};

// is_static detects if an (abstract) value is defined completely by its type (no members)
template <class T>
struct is_static : bool_constant<std::is_empty<remove_cvref_t<T>>::value> {};

// is_constant detects if a type is a static integral type and if v is equal to a value

template <auto n, class T>
struct is_constant : false_type {};
template <auto n, class T>
struct is_constant<n, T const> : is_constant<n, T> {};
template <auto n, class T>
struct is_constant<n, T const&> : is_constant<n, T> {};
template <auto n, class T>
struct is_constant<n, T&> : is_constant<n, T> {};
template <auto n, class T>
struct is_constant<n, T&&> : is_constant<n, T> {};
template <auto n, auto v>
struct is_constant<n, C<v>> : bool_constant<v == n> {};
template <auto n, class T, T v>
struct is_constant<n, integral_constant<T, v>> : bool_constant<v == n> {};

//
// Specializations
//

template <int v>
using Int = C<v>;
using _0 = Int<0>;
using _64 = Int<64>;
using _128 = Int<128>;
using _256 = Int<256>;
using _512 = Int<512>;

//
// Underscore placeholder (for slicing semantics)
//
// Usage:
// - `tla::_` is an empty tag value that can be used inside `tla::Coord` / tensor indexing
//   to indicate "take the whole dimension" (full slice).
struct Underscore {
    using type = Underscore;
};

constexpr Underscore _{};

template <class T>
struct is_underscore : false_type {};
template <>
struct is_underscore<Underscore> : true_type {};
template <class T>
struct is_underscore<T const> : is_underscore<T> {};
template <class T>
struct is_underscore<T const&> : is_underscore<T> {};
template <class T>
struct is_underscore<T&> : is_underscore<T> {};
template <class T>
struct is_underscore<T&&> : is_underscore<T> {};

/***************/
/** Operators **/
/***************/

#define TLA_LEFT_UNARY_OP(OP)                                 \
    template <auto t>                                         \
    CATLASS_HOST_DEVICE constexpr C<(OP t)> operator OP(C<t>) \
    {                                                         \
        return {};                                            \
    }
#define TLA_BINARY_OP(OP)                                             \
    template <auto t, auto u>                                         \
    CATLASS_HOST_DEVICE constexpr C<(t OP u)> operator OP(C<t>, C<u>) \
    {                                                                 \
        return {};                                                    \
    }

TLA_LEFT_UNARY_OP(+);
TLA_LEFT_UNARY_OP(-);
TLA_LEFT_UNARY_OP(~);
TLA_LEFT_UNARY_OP(!);
TLA_LEFT_UNARY_OP(*);

TLA_BINARY_OP(+);
TLA_BINARY_OP(-);
TLA_BINARY_OP(*);
TLA_BINARY_OP(/);
TLA_BINARY_OP(%);
TLA_BINARY_OP(&);
TLA_BINARY_OP(|);
TLA_BINARY_OP(^);
TLA_BINARY_OP(<<);
TLA_BINARY_OP(>>);

#undef TLA_BINARY_OP
#undef TLA_LEFT_UNARY_OP
#undef TLA_RIGHT_UNARY_OP

//
// Named functions from math.hpp
//

#define TLA_NAMED_UNARY_FN(OP)                  \
    template <auto t>                           \
    CATLASS_HOST_DEVICE constexpr auto OP(C<t>) \
    {                                           \
        return C<OP(t)>{};                      \
    }
#define TLA_NAMED_BINARY_FN(OP)                                         \
    template <auto t, auto u>                                           \
    CATLASS_HOST_DEVICE constexpr auto OP(C<t>, C<u>)                   \
    {                                                                   \
        return C<OP(t, u)>{};                                           \
    }                                                                   \
    template <auto t, class U, TLA_REQUIRES(is_std_integral<U>::value)> \
    CATLASS_HOST_DEVICE constexpr auto OP(C<t>, U u)                    \
    {                                                                   \
        return OP(t, u);                                                \
    }                                                                   \
    template <class T, auto u, TLA_REQUIRES(is_std_integral<T>::value)> \
    CATLASS_HOST_DEVICE constexpr auto OP(T t, C<u>)                    \
    {                                                                   \
        return OP(t, u);                                                \
    }

TLA_NAMED_BINARY_FN(max);
TLA_NAMED_BINARY_FN(min);
TLA_NAMED_BINARY_FN(add);

#undef TLA_NAMED_UNARY_FN
#undef TLA_NAMED_BINARY_FN

template <typename T, T... Ns>
struct IntegerSequence {
    using value_type = T;
    static constexpr size_t size() { return sizeof...(Ns); }
};

template <typename Sequence, typename T, size_t N, typename = void>
struct MakeIntegerSequenceImpl;

template <typename T, typename NS>
struct MakeIntegerSequenceImpl<NS, T, 0> {
    typedef NS type;
};

template <typename T, T... Ns, size_t N>
struct MakeIntegerSequenceImpl<IntegerSequence<T, Ns...>, T, N, TLA_REQUIRES_T(N > 0)> {
    typedef typename MakeIntegerSequenceImpl<IntegerSequence<T, N - 1, Ns...>, T, N - 1>::type type;
};

template <typename T, T N>
using MakeIntegerSequence = typename MakeIntegerSequenceImpl<IntegerSequence<T>, T, N>::type;

// index_sequence
template <size_t... Ints>
using index_sequence = IntegerSequence<size_t, Ints...>;

template <size_t N>
using make_index_sequence = MakeIntegerSequence<size_t, N>;

// int_sequence
template <int... Ints>
using int_sequence = IntegerSequence<int, Ints...>;

template <int N>
using make_int_sequence = MakeIntegerSequence<int, N>;

// Shortcuts
template <int... Ints>
using seq = int_sequence<Ints...>;

template <int N>
using make_seq = make_int_sequence<N>;

template <class Tuple>
using tuple_seq = make_seq<tuple_size<tla::remove_cvref_t<Tuple>>::value>;

} // namespace tla

namespace tla {

namespace detail {

// EBO stands for "empty base optimization."
template <size_t N, class T, bool IsEmpty = std::is_empty<T>::value>
struct EBO;

// Specialization for types T that are empty;
template <size_t N, class T>
struct EBO<N, T, true> {
    CATLASS_HOST_DEVICE constexpr EBO() {}

    CATLASS_HOST_DEVICE constexpr EBO(T const&) {}
};

template <size_t N, class T>
CATLASS_HOST_DEVICE constexpr T getv(EBO<N, T, true> const&)
{
    return {};
}

// Specialization for types T that are not empty;
template <size_t N, class T>
struct EBO<N, T, false> {
    CATLASS_HOST_DEVICE constexpr EBO() : t_{} {}

    CATLASS_HOST_DEVICE constexpr EBO(T const& t) : t_{t} {}

    T t_;
};

template <size_t N, class T>
CATLASS_HOST_DEVICE constexpr T const& getv(EBO<N, T, false> const& x)
{
    return x.t_;
}

template <size_t N, class T>
CATLASS_HOST_DEVICE constexpr T& getv(EBO<N, T, false>& x)
{
    return x.t_;
}

// TupleBase
template <class IdxSeq, class... T>
struct TupleBase;

template <size_t... I, class... T>
struct TupleBase<index_sequence<I...>, T...> : EBO<I, T>... {
    CATLASS_HOST_DEVICE constexpr TupleBase() {}

    CATLASS_HOST_DEVICE constexpr TupleBase(T const&... t) : EBO<I, T>(t)... {}
};

} // end namespace detail

// tla::tuple class.
template <class... T>
struct tuple : detail::TupleBase<make_index_sequence<sizeof...(T)>, T...> {
    CATLASS_HOST_DEVICE constexpr tuple() {}

    CATLASS_HOST_DEVICE constexpr tuple(T const&... t)
        : detail::TupleBase<make_index_sequence<sizeof...(T)>, T...>(t...)
    {}
};

template <>
struct tuple<> {};

// get for tla::tuple
template <size_t I, class... T>
CATLASS_HOST_DEVICE constexpr decltype(auto) get(tuple<T...> const& t) noexcept
{
    static_assert(I < sizeof...(T), "Index out of range");
    return detail::getv<I>(t);
}

template <size_t I, class... T>
CATLASS_HOST_DEVICE constexpr decltype(auto) get(tuple<T...>& t) noexcept
{
    static_assert(I < sizeof...(T), "Index out of range");
    return detail::getv<I>(t);
}

template <size_t I, class... T>
CATLASS_HOST_DEVICE constexpr decltype(auto) get(tuple<T...>&& t) noexcept
{
    static_assert(I < sizeof...(T), "Index out of range");
    return detail::getv<I>(static_cast<tuple<T...>&&>(t));
}

namespace detail {

template <class T>
auto has_tuple_size(T*) -> bool_constant<(0 <= tuple_size<T>::value)>;
auto has_tuple_size(...) -> false_type;

} // end namespace detail

template <class T>
struct is_tuple : decltype(detail::has_tuple_size((T*)0)){};

template <class... T>
struct tuple_size<tla::tuple<T...>> : std::integral_constant<size_t, sizeof...(T)> {};

template <class... T>
struct tuple_size<const tla::tuple<T...>> : std::integral_constant<size_t, sizeof...(T)> {};

// make_tuple
template <class... T>
CATLASS_HOST_DEVICE constexpr tuple<T...> MakeTuple(T const&... t)
{
    return {t...};
}

} // end namespace tla

namespace tla {
//
// Apply (Unpack)
// (t, f) => f(t_0,t_1,...,t_n)
//

namespace detail {

template <class T, class F, int... I>
CATLASS_HOST_DEVICE constexpr auto apply(T&& t, F&& f, seq<I...>)
{
    return f(get<I>(static_cast<T&&>(t))...);
}

template <class T, class F, class G, int... I>
CATLASS_HOST_DEVICE constexpr auto tapply(T&& t, F&& f, G&& g, seq<I...>)
{
    return g(f(get<I>(static_cast<T&&>(t)))...);
}

template <class T0, class T1, class F, class G, int... I>
CATLASS_HOST_DEVICE constexpr auto tapply(T0&& t0, T1&& t1, F&& f, G&& g, seq<I...>)
{
    return g(f(get<I>(static_cast<T0&&>(t0)), get<I>(static_cast<T1&&>(t1)))...);
}

} // end namespace detail

template <class T, class F>
CATLASS_HOST_DEVICE constexpr auto apply(T&& t, F&& f)
{
    return detail::apply(static_cast<T&&>(t), f, tuple_seq<T>{});
}

template <class T, class F, class G>
CATLASS_HOST_DEVICE constexpr auto transform_apply(T&& t, F&& f, G&& g)
{
    if constexpr (is_tuple<remove_cvref_t<T>>::value) {
        return detail::tapply(static_cast<T&&>(t), f, g, tuple_seq<T>{});
    } else {
        return g(f(static_cast<T&&>(t)));
    }
}

template <class T0, class T1, class F, class G>
CATLASS_HOST_DEVICE constexpr auto transform_apply(T0&& t0, T1&& t1, F&& f, G&& g)
{
    if constexpr (is_tuple<remove_cvref_t<T0>>::value) {
        return detail::tapply(static_cast<T0&&>(t0), static_cast<T1&&>(t1), f, g, tuple_seq<T0>{});
    } else {
        return g(f(static_cast<T0&&>(t0), static_cast<T1&&>(t1)));
    }
}

template <class T, class F>
CATLASS_HOST_DEVICE constexpr void for_each(T&& t, F&& f)
{
    if constexpr (is_tuple<remove_cvref_t<T>>::value) {
        return detail::apply(t, [&](auto&&... a) { (f(static_cast<decltype(a)&&>(a)), ...); }, tuple_seq<T>{});
    } else {
        return f(static_cast<T&&>(t));
    }
}

struct UnpackedMakeTuple {
    template <class... T>
    CATLASS_HOST_DEVICE constexpr auto operator()(T const&... a) const
    {
        return tla::MakeTuple(a...);
    }
};

template <class T0, class T1, class F>
CATLASS_HOST_DEVICE constexpr auto transform(T0 const& t0, T1 const& t1, F&& f)
{
    if constexpr (is_tuple<T0>::value) {
        static_assert(tuple_size<T0>::value == tuple_size<T1>::value, "Mismatched tuple_size");
        return detail::tapply(t0, t1, f, UnpackedMakeTuple{}, tuple_seq<T0>{});
    } else {
        return f(t0, t1);
    }
}

template <size_t I, class T, TLA_REQUIRES(tla::is_integral<tla::remove_cvref_t<T>>::value)>
CATLASS_HOST_DEVICE constexpr decltype(auto) get(T&& t) noexcept
{
    static_assert(I == 0, "Index out of range");
    return static_cast<T&&>(t);
}

template <size_t I0, size_t I1, size_t... Is, class T>
CATLASS_HOST_DEVICE constexpr decltype(auto) get(T&& t) noexcept
{
    return get<I1, Is...>(get<I0>(static_cast<T&&>(t)));
}

// max
template <class T0, class... Ts>
CATLASS_HOST_DEVICE constexpr auto max(T0 const& t0, Ts const&... ts);

struct UnpackedMax {
    template <class... T>
    CATLASS_HOST_DEVICE constexpr auto operator()(T const&... v) const
    {
        return tla::max(v...);
    }
};

template <class T0, class... Ts>
CATLASS_HOST_DEVICE constexpr auto max(T0 const& t0, Ts const&... ts)
{
    if constexpr (is_tuple<T0>::value) {
        return tla::max(tla::apply(t0, UnpackedMax{}), ts...);
    } else if constexpr (sizeof...(Ts) == 0) {
        return t0;
    } else {
        return tla::max(t0, tla::max(ts...));
    }
}

// rank
template <int... Is, class Tuple>
CATLASS_HOST_DEVICE constexpr auto rank(Tuple const& t)
{
    if constexpr (sizeof...(Is) == 0) {
        if constexpr (is_tuple<Tuple>::value) {
            return Int<tuple_size<Tuple>::value>{};
        } else {
            return Int<1>{};
        }
    } else {
        return rank(get<Is...>(t));
    }
}

template <class Tuple>
using rank_t = decltype(rank(std::declval<Tuple>()));

template <class Tuple>
constexpr auto rank_v = rank_t<Tuple>::value;

// depth
template <int... Is, class Tuple>
CATLASS_HOST_DEVICE constexpr auto depth(Tuple const& t);

struct UnpackedDepth {
    template <class... T>
    CATLASS_HOST_DEVICE constexpr auto operator()(T const&... v) const
    {
        return tla::max(depth(v)...);
    }
};

template <int... Is, class Tuple>
CATLASS_HOST_DEVICE constexpr auto depth(Tuple const& t)
{
    if constexpr (sizeof...(Is) == 0) {
        if constexpr (is_tuple<Tuple>::value) {
            return Int<1>{} + tla::apply(t, UnpackedDepth{});
        } else {
            return Int<0>{};
        }
    } else {
        return depth(get<Is...>(t));
    }
}

template <class Tuple>
using depth_t = decltype(depth(std::declval<Tuple>()));

template <class Tuple>
constexpr auto depth_v = depth_t<Tuple>::value;

struct MultipliesUnaryLfold {
    template <class... T>
    CATLASS_HOST_DEVICE constexpr auto operator()(T const&... v) const
    {
        return (... * v);
    }
};

// Implementation of product as a function object
struct Product {
    template <class IntTuple>
    CATLASS_HOST_DEVICE constexpr auto operator()(IntTuple const& a) const
    {
        if constexpr (is_tuple<IntTuple>::value) {
            if constexpr (tuple_size<IntTuple>::value == 0) {
                return Int<1>{};
            } else {
                return tla::transform_apply(a, Product{}, MultipliesUnaryLfold{});
            }
        } else if constexpr (tla::is_integral<IntTuple>::value) {
            return a;
        }
    }
};

namespace detail {

template <size_t N, typename Sequence>
struct MakeZeroTupleImpl;

template <size_t N, size_t... Is>
struct MakeZeroTupleImpl<N, tla::index_sequence<Is...>> {
    using type = tla::tuple<tla::Int<Is * 0>...>;
};

template <size_t N>
using MakeZeroTuple = typename MakeZeroTupleImpl<N, tla::make_index_sequence<N>>::type;

} // end namespace detail

// Add
template <class IntTupleA, class IntTupleB>
CATLASS_HOST_DEVICE constexpr auto Add(IntTupleA const& a, IntTupleB const& b);

struct UnpackedAdd {
    template <class IntTupleA, class IntTupleB>
    CATLASS_HOST_DEVICE constexpr auto operator()(IntTupleA const& x, IntTupleB const& y) const
    {
        return Add(x, y);
    }
};

template <class IntTupleA, class IntTupleB>
CATLASS_HOST_DEVICE constexpr auto Add(IntTupleA const& a, IntTupleB const& b)
{
    if constexpr (is_tuple<IntTupleA>::value && is_tuple<IntTupleB>::value) {
        static_assert(tuple_size<IntTupleA>::value == tuple_size<IntTupleB>::value, "Mismatched ranks");
        return transform(a, b, UnpackedAdd{});
    } else {
        return tla::add(a, b);
    }
}

} // end namespace tla

#endif // MATMUL_EMU_LAYOUT_TUPLE_HPP
