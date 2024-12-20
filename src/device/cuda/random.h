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
// #include <sutil/Preprocessor.h>

// #define DRAND48
#define RND31

#ifdef DRAND48
typedef unsigned long long RNDTYPE;
#define MASK 0x0001000000000000
// Generate random unsigned int in [0, 2^48)
static __host__ __device__ __inline__ RNDTYPE lcg(RNDTYPE &prev) {
  const unsigned long long LCG_A = 0x5DEECE66D;
  const unsigned long long LCG_C = 0x00000000B;
  prev = (LCG_A * prev + LCG_C) & 0x0000FFFFFFFFFFFF;
  return prev;
}

template <unsigned int N>
static __host__ __device__ __inline__ RNDTYPE tea(unsigned int val0,
                                                  unsigned int val1) {
  unsigned long long v0 = val0;
  unsigned long long v1 = val1;
  unsigned long long s0 = 0;

  for (unsigned int n = 0; n < N; n++) {
    s0 += 0x9e3779b9;
    v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
    v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
  }
  return ((v0 << 16) & 0xFFFFFFFFFFFF0000) | 0x330E;
}


#elif defined(RND31)
typedef unsigned int RNDTYPE;
#define MASK 0x80000000

template <unsigned int N>
static __host__ __device__ __inline__ RNDTYPE tea(unsigned int val0,
                                                  unsigned int val1) {
  RNDTYPE v0 = val0;
  RNDTYPE v1 = val1;
  RNDTYPE s0 = 0;

  for (unsigned int n = 0; n < N; n++) {
    s0 += 0x9e3779b9;
    v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
    v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
  }

  return v0;
}

static __host__ __device__ __inline__ RNDTYPE lcg(unsigned int &prev) {
  const unsigned int LCG_A = 1977654935u;
  prev = (LCG_A * prev) & 0x7FFFFFFF;
  return prev;
}
#else
typedef unsigned int RNDTYPE;
#define MASK 0x01000000

template <unsigned int N>
static __host__ __device__ __inline__ RNDTYPE tea(unsigned int val0,
                                                  unsigned int val1) {
  RNDTYPE v0 = val0;
  RNDTYPE v1 = val1;
  RNDTYPE s0 = 0;

  for (unsigned int n = 0; n < N; n++) {
    s0 += 0x9e3779b9;
    v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
    v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
  }

  return v0;
}

// Generate random unsigned int in [0, 2^24)
static __host__ __device__ __inline__ RNDTYPE lcg(unsigned int &prev) {
  const unsigned int LCG_A = 1664525u;
  const unsigned int LCG_C = 1013904223u;
  prev = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}

static __host__ __device__ __inline__ RNDTYPE rot_seed(unsigned int seed,
                                                       unsigned int frame) {
  return seed ^ frame;
}

#endif

// Generate random double in [0, 1)
static __host__ __device__ __inline__ double drnd(RNDTYPE &prev) {
  return ((double)lcg(prev) / (double)MASK);
}

// Generate random float in [0, 1]
static __host__ __device__ __inline__ float rnd_closed(RNDTYPE &prev) {
  return static_cast<float>(lcg(prev)) / static_cast<float>(0x0000FFFFFFFFFFFF);
}

// Generate random float in [0, 1)
static __host__ __device__ __inline__ float rnd(RNDTYPE &prev) {
  lcg(prev);
  return static_cast<float>((prev & 0x0000FFFFFF000000) == 0x0000FFFFFF000000
                                ? prev & 0x0000FFFFFF7FFFFF
                                : prev) /
         static_cast<float>(MASK);
}

// Generate random integer in [0, 2^bitsize(RNDTYPE))
static __host__ __device__ __inline__ RNDTYPE irnd(RNDTYPE &prev) {
  return lcg(prev);
}

// Generate random integer in [0, mod)
static __host__ __device__ __inline__ RNDTYPE irnd_mod(RNDTYPE &prev,
                                                       RNDTYPE mod) {
  return lcg(prev) % mod;
}
