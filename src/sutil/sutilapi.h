//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#pragma once

#ifndef SUTILAPI
#if sutil_7_sdk_EXPORTS /* Set by CMAKE */
#if defined(_WIN32) || defined(_WIN64)
#define SUTILAPI __declspec(dllexport)
#define SUTILCLASSAPI
#elif defined(linux) || defined(__linux__) || defined(__CYGWIN__)
#define SUTILAPI __attribute__((visibility("default")))
#define SUTILCLASSAPI SUTILAPI
#elif defined(__APPLE__) && defined(__MACH__)
#define SUTILAPI __attribute__((visibility("default")))
#define SUTILCLASSAPI SUTILAPI
#else
#error "CODE FOR THIS OS HAS NOT YET BEEN DEFINED"
#endif

#else /* sutil_7_sdk_EXPORTS */

#if defined(_WIN32) || defined(_WIN64)
#define SUTILAPI __declspec(dllimport)
#define SUTILCLASSAPI
#elif defined(linux) || defined(__linux__) || defined(__CYGWIN__)
#define SUTILAPI __attribute__((visibility("default")))
#define SUTILCLASSAPI SUTILAPI
#elif defined(__APPLE__) && defined(__MACH__)
#define SUTILAPI __attribute__((visibility("default")))
#define SUTILCLASSAPI SUTILAPI
#elif defined(__CUDACC_RTC__)
#define SUTILAPI
#define SUTILCLASSAPI
#else
#error "CODE FOR THIS OS HAS NOT YET BEEN DEFINED"
#endif

#endif /* sutil_7_sdk_EXPORTS */
#endif

#if defined(__CUDACC__) || defined(__CUDABE__)
#define SUTIL_HOSTDEVICE __host__ __device__
#define SUTIL_INLINE __forceinline__
#define CONST_STATIC_INIT(...)

#define SUTILFN __device__ __inline__
#define closesthit_fn extern "C" __global__ void
#define raygen_fn extern "C" __global__ void
#define miss_fn extern "C" __global__ void

#else

#define SUTIL_HOSTDEVICE
#define SUTIL_INLINE inline
#define CONST_STATIC_INIT(...) = __VA_ARGS__

#define SUTILFN __host__ __inline__
#define closesthit_fn
#define raygen_fn
#define miss_fn
#endif

#if OPTIX_VERSION >= 70700
#define CREATEMODULE optixModuleCreate
#else
#define CREATEMODULE optixModuleCreateFromPTX
#endif
