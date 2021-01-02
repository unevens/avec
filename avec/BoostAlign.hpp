// the code in this file is taken from boost::align and adapted into a single header.
// this was done to avoid having to add boost as a dependency.

/*
Copyright 2014 Glen Joseph Fernandes
(glenjofe@gmail.com)

Distributed under the Boost Software License, Version 1.0.
(http://www.boost.org/LICENSE_1_0.txt)
*/
#pragma once
#include <cstddef>
#include <cassert>
#include <memory>
#include <type_traits>
#include <new>
#include <utility>


#if defined(__APPLE__) || defined(__APPLE_CC__) || defined(macintosh)
#include <AvailabilityMacros.h>
#endif

namespace boost {
namespace alignment {
namespace detail {

constexpr inline bool
is_alignment(std::size_t value) noexcept
{
    return (value > 0) && ((value & (value - 1)) == 0);
}

template<std::size_t N>
struct is_alignment_constant
        : std::integral_constant<bool, (N > 0) && ((N & (N - 1)) == 0)> { };

template<std::size_t A, std::size_t B>
struct max_size
        : std::integral_constant<std::size_t, (A > B) ? A : B> { };

template<class T>
struct max_objects
        : std::integral_constant<std::size_t,
                ~static_cast<std::size_t>(0) / sizeof(T)> { };
} // namespace detail


#if (defined(_MSC_VER) && !defined(UNDER_CE)) || (defined(__MINGW32__) && (__MSVCRT_VERSION__ >= 0x0700))
#include <malloc.h>

inline void*
aligned_alloc(std::size_t alignment, std::size_t size) noexcept
{
    assert(detail::is_alignment(alignment));
    return ::_aligned_malloc(size, alignment);
}

inline void
aligned_free(void* ptr) noexcept
{
    ::_aligned_free(ptr);
}

#elif defined(__ANDROID__)
#include <malloc.h>

inline void*
aligned_alloc(std::size_t alignment, std::size_t size) noexcept
{
    assert(detail::is_alignment(alignment));
    return ::memalign(alignment, size);
}

inline void
aligned_free(void* ptr) noexcept
{
    ::free(ptr);
}

#elif MAC_OS_X_VERSION_MIN_REQUIRED >= 1090
#define USE_BOOST_ALIGN_POSIX

#elif MAC_OS_X_VERSION_MIN_REQUIRED >= 1060
#include <stdlib.h>

inline void*
aligned_alloc(std::size_t alignment, std::size_t size) noexcept
{
    assert(detail::is_alignment(alignment));
    if (size == 0) {
        return 0;
    }
    if (alignment < sizeof(void*)) {
        alignment = sizeof(void*);
    }
    void* p;
    if (::posix_memalign(&p, alignment, size) != 0) {
        p = 0;
    }
    return p;
}

inline void
aligned_free(void* ptr) noexcept
{
    ::free(ptr);
}

#elif defined(__SunOS_5_11) || defined(__SunOS_5_12)
#define USE_BOOST_ALIGN_POSIX

#elif defined(sun) || defined(__sun)
#include <stdlib.h>

inline void*
aligned_alloc(std::size_t alignment, std::size_t size) noexcept
{
    assert(detail::is_alignment(alignment));
    return ::memalign(alignment, size);
}

inline void
aligned_free(void* ptr) noexcept
{
    ::free(ptr);
}

#elif (_POSIX_C_SOURCE >= 200112L) || (_XOPEN_SOURCE >= 600)
#define USE_BOOST_ALIGN_POSIX

#else //generic implementation
#include <cstdlib>

inline void*
aligned_alloc(std::size_t alignment, std::size_t size) noexcept
{
    assert(detail::is_alignment(alignment));
    enum {
        N = std::alignment_of<void*>::value
    };
    if (alignment < N) {
        alignment = N;
    }
    std::size_t n = size + alignment - N;
    void* p = std::malloc(sizeof(void*) + n);
    if (p) {
        void* r = static_cast<char*>(p) + sizeof(void*);
        (void)std::align(alignment, size, r, n);
        *(static_cast<void**>(r) - 1) = p;
        p = r;
    }
    return p;
}

inline void
aligned_free(void* ptr) noexcept
{
    if (ptr) {
        std::free(*(static_cast<void**>(ptr) - 1));
    }
}

#endif

#ifdef USE_BOOST_ALIGN_POSIX
#include <stdlib.h>

inline void*
aligned_alloc(std::size_t alignment, std::size_t size) noexcept
{
    assert(detail::is_alignment(alignment));
    if (alignment < sizeof(void*)) {
        alignment = sizeof(void*);
    }
    void* p;
    if (::posix_memalign(&p, alignment, size) != 0) {
        p = 0;
    }
    return p;
}

inline void
aligned_free(void* ptr) noexcept
{
    ::free(ptr);
}

#endif

// assyme aligned macro

#if defined(_MSC_VER)
#include <cstddef>
#define BOOST_ALIGN_ASSUME_ALIGNED(ptr, alignment) __assume((reinterpret_cast<std::size_t>(ptr) & ((alignment) - 1)) == 0)
#elif defined(__clang__) && defined(__has_builtin)
#if __has_builtin(__builtin_assume_aligned)
#define BOOST_ALIGN_ASSUME_ALIGNED(p, n) \
(p) = static_cast<__typeof__(p)>(__builtin_assume_aligned((p), (n)))
#else
#define BOOST_ALIGN_ASSUME_ALIGNED(ptr, alignment)
#endif
#elif defined (__GNUC__)
#define BOOST_GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#if BOOST_GCC_VERSION >= 40700
#define BOOST_ALIGN_ASSUME_ALIGNED(p, n) \
(p) = static_cast<__typeof__(p)>(__builtin_assume_aligned((p), (n)))
#else
#define BOOST_ALIGN_ASSUME_ALIGNED(ptr, alignment)
#endif
#elif defined(__INTEL_COMPILER)
#define BOOST_ALIGN_ASSUME_ALIGNED(ptr, alignment) \
__assume_aligned((ptr), (alignment))
#else
#define BOOST_ALIGN_ASSUME_ALIGNED(ptr, alignment)
#endif

//allocator
#if __clang__
#if !__has_feature(cxx_exceptions) && !defined(BOOST_NO_EXCEPTIONS)
#  define BOOST_NO_EXCEPTIONS
#endif
#endif

template<class T, std::size_t Alignment>
class aligned_allocator {
    static_assert(detail::is_alignment_constant<Alignment>::value,"boost::alignment::is_alignment_constant failed");

public:
    typedef T value_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef void* void_pointer;
    typedef const void* const_void_pointer;
    typedef typename std::add_lvalue_reference<T>::type reference;
    typedef typename std::add_lvalue_reference<const T>::type const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;
    typedef std::true_type propagate_on_container_move_assignment;
    typedef std::true_type is_always_equal;

    template<class U>
    struct rebind {
        typedef aligned_allocator<U, Alignment> other;
    };

    aligned_allocator() = default;

    template<class U>
    aligned_allocator(const aligned_allocator<U, Alignment>&)
    noexcept { }

    pointer allocate(size_type size, const_void_pointer = 0) {
        enum {
            m = detail::max_size<Alignment,
                    std::alignment_of<value_type>::value>::value
        };
        if (size == 0) {
            return 0;
        }
        void* p = boost::alignment::aligned_alloc(m, sizeof(T) * size);
        if (!p) {
#ifndef BOOST_NO_EXCEPTIONS
            throw std::bad_alloc();
#endif
        }
        return static_cast<T*>(p);
    }

    void deallocate(pointer ptr, size_type) {
        boost::alignment::aligned_free(ptr);
    }

    constexpr size_type max_size() const noexcept {
        return detail::max_objects<T>::value;
    }

    template<class U, class... Args>
    void construct(U* ptr, Args&&... args) {
        ::new((void*)ptr) U(std::forward<Args>(args)...);
    }

    template<class U>
    void construct(U* ptr) {
        ::new((void*)ptr) U();
    }

    template<class U>
    void destroy(U* ptr) {
        (void)ptr;
        ptr->~U();
    }
};

template<class T, class U, std::size_t Alignment>
inline bool
operator==(const aligned_allocator<T, Alignment>&,
           const aligned_allocator<U, Alignment>&) noexcept
{
    return true;
}

template<class T, class U, std::size_t Alignment>
inline bool
operator!=(const aligned_allocator<T, Alignment>&,
           const aligned_allocator<U, Alignment>&) noexcept
{
    return false;
}


} //namespace alignment
} //namespace boost
