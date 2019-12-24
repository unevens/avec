# [*avec*](https://github.com/unevens/avec)

*avec* is an header-only library for using SIMD instructions in audio applications. 
It features containers and views for aligned memory, with an API designed to work seamlessly with Agner Fog's [vectorclass](https://github.com/vectorclass/version2), which is included as a submodule.

In vectorclass each SIMD type has its own class: `Vec4f` for `__m128`, `Vec8f` for `__m256`, `Vec4d` for `__m256d` and so on.

In *avec* are implemented the template classes `VecBuffer<Vec>` and `VecView<Vec>` to manage aligned memory and convert it to and from the SIMD classes of vectorclass.

The template class `InterleavedBuffer<Scalar>` (where `Scalar` can be either `float` or `double`) can be used to interleave a buffer of any number of audio channels into a set of `VecBuffer<Vec8f>` and `VecBuffer<Vec4f>` (when `Scalar` is `float`), or of `VecBuffer<Vec4d>` and `VecBuffer<Vec2d>` (when `Scalar` is `double`). 

## Dependencies

*avec* uses [Boost.Align](https://www.boost.org/doc/libs/1_71_0/doc/html/align.html). 

## Documentation

The documentation can be generated with [Doxygen](http://doxygen.nl/) running

```bash
$ doxygen doxyfile.txt
```