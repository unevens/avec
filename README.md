# [*avec*](https://github.com/unevens/avec)

*avec* is a little library for using SIMD instructions on both x86 and ARM. 

It features containers for aligned memory, with views, allocators and interleaving/deinterleaving functionality. 

The API is designed to work seamlessly with Agner Fog's [vectorclass](https://github.com/vectorclass/version2), which is included as a submodule. 

Since *vectorclass* only supports x86, *avec* reimplements a subset of its functionality for ARM using NEON. See the section *ARM support* for details.

## Containers and views

In vectorclass each SIMD type has its own class: `Vec4f` for `__m128`, `Vec8f` for `__m256`, `Vec4d` for `__m256d` and so on.

In *avec*, the template classes `VecBuffer<Vec>` and `VecView<Vec>` are used to manage blocks of aligned memory and convert it to and from the SIMD classes of vectorclass.

## Interleaving

The template class `InterleavedBuffer<Scalar>` (where `Scalar` can be either `float` or `double`) is used to interleave a buffer of any number of audio channels into a set of `VecBuffer<Vec8f>`, `VecBuffer<Vec4f>` and `VecBuffer<Vec2f>` (when `Scalar` is `float`), or of `VecBuffer<Vec8d>`, `VecBuffer<Vec4d>` and `VecBuffer<Vec2d>` (when `Scalar` is `double`). 

Only the `VecBuffers` whose underlying vectorclass type is supported by the hardware will be used, in order to easily abstract over the many SIMD instruction sets.


## ARM support

On ARM, `Vec4f` and `Vec2d` are implemented for `float32x4_t` and `float64x2_t`, with most of their member functions, all of their operators overloaded, and some math function overloads (`exp`, `log`, `sin`, `cos`, `sincos`, `tan`).

## Credits

*avec* includes code from [Boost.Align](https://www.boost.org/doc/libs/1_71_0/doc/html/align.html) by Joseph Fernandes, without depending on the whole Boost library. See the file `BoostAlign.hpp`.

The implementation of `exp`, `log`, `sin`, `cos`, `sincos`, for ARM NEON was written by Julien Pommier, and it is available at http://gruntthepeon.free.fr/ssemath/neon_mathfun.html.

## Documentation

The documentation, available at https://unevens.github.io/avec/, can be generated with [Doxygen](http://doxygen.nl/) running

```bash
$ doxygen doxyfile.txt
```
