# [*avec*](https://github.com/unevens/avec)

*avec* is a collection of classes for using SIMD instructions in audio applications. 
It features containers and views for aligned memory, with an API designed to work seamlessly with Agner Fog's [vectorclass](https://github.com/vectorclass/version2), which is included as a submodule.

## Containers and views

In vectorclass each SIMD type has its own class: `Vec4f` for `__m128`, `Vec8f` for `__m256`, `Vec4d` for `__m256d` and so on.

In *avec* are implemented the template classes `VecBuffer<Vec>` and `VecView<Vec>` to manage aligned memory and convert it to and from the SIMD classes of vectorclass.

The template class `InterleavedBuffer<Scalar>` (where `Scalar` can be either `float` or `double`) can be used to interleave a buffer of any number of audio channels into a set of `VecBuffer<Vec8f>` and `VecBuffer<Vec4f>` (when `Scalar` is `float`), or of `VecBuffer<Vec4d>` and `VecBuffer<Vec2d>` (when `Scalar` is `double`). 

## Noise Generator

The file `Noise.hpp` implements a (white) noise generator which can populate VecBuffers and InterleavedBuffers with noise, with a different seed for each channel.
It uses [a SIMD implementation](https://github.com/unevens/xorshift32_16bit_simd) of a [16 bit xorshift32 random number generator](https://b2d-f9r.blogspot.com/2010/08/16-bit-xorshift-rng-now-with-more.html
), to generate 4 samples of noise in parallel. 

It is the only part of *avec* which is not header only, as it needs to compile the file `xorshift32_16bit_simd.c`.

The header `Noise.hpp` is not included by `Avec.hpp`. Consider it an optional feature.

## Biquad Filters

The file `Biquad.hpp` implements simple biquad filters which take either VecBuffers or InterleavedBuffers as input and output. They use SIMD instructions to process all the channels interleaved in each VecBuffer at the same time.

The header `Biquad.hpp` is not included by `Avec.hpp`. Consider it an optional feature.

## Usage

Just add the folder `avec` to your project and `#include "avec/Avec.hpp` to get all the symbols for the containers and the views in the global namespace. Otherwise include the headers you need singularly, and they will define their symbols in the `avec` namespace.

## Dependencies

*avec* uses [Boost.Align](https://www.boost.org/doc/libs/1_71_0/doc/html/align.html). 

## Documentation

The documentation, available at https://unevens.github.io/avec/, can be generated with [Doxygen](http://doxygen.nl/) running

```bash
$ doxygen doxyfile.txt
```
