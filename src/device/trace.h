//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <optix.h>
#include "src/sutil/shared/structs.h"

SUTILFN void *unpackPointer(unsigned int i0, unsigned int i1) {
  const unsigned long long uptr =
      static_cast<unsigned long long>(i0) << 32 | i1;
  void *ptr = reinterpret_cast<void *>(uptr);
  return ptr;
}

SUTILFN void packPointer(void *ptr, unsigned int &i0, unsigned int &i1) {
  const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
  i0 = uptr >> 32;
  i1 = uptr & 0x00000000ffffffff;
}

static SUTILFN void traceRadiance(OptixTraversableHandle handle,
                                  const float3 &ray_origin,
                                  const float3 &ray_direction, float tmin,
                                  float tmax, PayloadRadiance *payload) {

  unsigned int u0 = __float_as_int(payload->origin.x),
               u1 = __float_as_int(payload->origin.y),
               u2 = __float_as_int(payload->origin.z);
  unsigned int u3 = __float_as_int(payload->direction.x),
               u4 = __float_as_int(payload->direction.y),
               u5 = __float_as_int(payload->direction.z);
  unsigned int u6 = 0u, u7 = 0u, u8 = 0u;
  unsigned int u9 = __float_as_int(payload->attenuation.x),
               u10 = __float_as_int(payload->attenuation.y),
               u11 = __float_as_int(payload->attenuation.z);
  unsigned int u12 = payload->depth;
  unsigned int u13 = payload->seed;
  unsigned int u14 = payload->emit;
  unsigned int u15 = payload->dead;
  unsigned int u16 = payload->intersect;

  optixTrace(handle, ray_origin, ray_direction, tmin, tmax,
             0.0f, // rayTime
             OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE,
             RAY_TYPE_RADIANCE, // SBT offset
             RAY_TYPE_COUNT,    // SBT stride
             RAY_TYPE_RADIANCE, // missSBTIndex
             u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14,
             u15, u16);

  payload->origin.x = __int_as_float(u0);
  payload->origin.y = __int_as_float(u1);
  payload->origin.z = __int_as_float(u2);
  payload->direction.x = __int_as_float(u3);
  payload->direction.y = __int_as_float(u4);
  payload->direction.z = __int_as_float(u5);
  payload->result.x = __int_as_float(u6);
  payload->result.y = __int_as_float(u7);
  payload->result.z = __int_as_float(u8);
  payload->attenuation.x = __int_as_float(u9);
  payload->attenuation.y = __int_as_float(u10);
  payload->attenuation.z = __int_as_float(u11);
  payload->depth = u12;
  payload->seed = u13;
  payload->emit = u14;
  payload->dead = u15;
  payload->intersect = u16;
}

//***************************/
/**** Radiance PAYLOAD ****/
/***************************/
SUTILFN void setPayloadOrigin(const float3 &p) {
  optixSetPayload_0(__float_as_int(p.x));
  optixSetPayload_1(__float_as_int(p.y));
  optixSetPayload_2(__float_as_int(p.z));
}

SUTILFN void setPayloadDirection(const float3 &w) {
  optixSetPayload_3(__float_as_int(w.x));
  optixSetPayload_4(__float_as_int(w.y));
  optixSetPayload_5(__float_as_int(w.z));
}

SUTILFN void setPayloadResult(const float3 &p) {
  optixSetPayload_6(__float_as_int(p.x));
  optixSetPayload_7(__float_as_int(p.y));
  optixSetPayload_8(__float_as_int(p.z));
}

SUTILFN void setPayloadAttenuation(const float3 &p) {
  optixSetPayload_9(__float_as_int(p.x));
  optixSetPayload_10(__float_as_int(p.y));
  optixSetPayload_11(__float_as_int(p.z));
}

SUTILFN void setPayloadDepth(unsigned int d) { optixSetPayload_12(d); }
SUTILFN void setPayloadSeed(unsigned int t) { optixSetPayload_13(t); }
SUTILFN void setPayloadEmit(unsigned int e) { optixSetPayload_14(e); }
SUTILFN void setPayloadDead(unsigned int d) { optixSetPayload_15(d); }
SUTILFN void setPayloadIntersect(unsigned int i) { optixSetPayload_16(i); }

SUTILFN float3 getPayloadOrigin() {
  float3 result;
  result.x = __int_as_float(optixGetPayload_0());
  result.y = __int_as_float(optixGetPayload_1());
  result.z = __int_as_float(optixGetPayload_2());
  return result;
}

SUTILFN float3 getPayloadDirection() {
  float3 result;
  result.x = __int_as_float(optixGetPayload_3());
  result.y = __int_as_float(optixGetPayload_4());
  result.z = __int_as_float(optixGetPayload_5());
  return result;
}

SUTILFN float3 getPayloadResult() {
  float3 result;
  result.x = __int_as_float(optixGetPayload_6());
  result.y = __int_as_float(optixGetPayload_7());
  result.z = __int_as_float(optixGetPayload_8());
  return result;
}

SUTILFN float3 getPayloadAttenuation() {
  float3 result;
  result.x = __int_as_float(optixGetPayload_9());
  result.y = __int_as_float(optixGetPayload_10());
  result.z = __int_as_float(optixGetPayload_11());
  return result;
}

SUTILFN unsigned int getPayloadDepth() { return optixGetPayload_12(); }
SUTILFN unsigned int getPayloadSeed() { return optixGetPayload_13(); }
SUTILFN unsigned int getPayloadEmit() { return optixGetPayload_14(); }
SUTILFN unsigned int getPayloadDead() { return optixGetPayload_15(); }
SUTILFN unsigned int getPayloadIntersect() { return optixGetPayload_16(); }

/***************************/
/**** OCCLUSION PAYLOAD ****/
/***************************/
static SUTILFN bool traceOcclusion(OptixTraversableHandle handle,
                                   const float3 &ray_origin,
                                   const float3 &ray_direction, float tmin,
                                   float tmax) {
  unsigned int occluded = 0u;
  optixTrace(handle, ray_origin, ray_direction, tmin, tmax,
             0.0f, // rayTime
             OptixVisibilityMask(1), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
             RAY_TYPE_OCCLUSION, // SBT offset
             RAY_TYPE_COUNT,     // SBT stride
             RAY_TYPE_OCCLUSION, // missSBTIndex
             occluded);
  return occluded;
}

/**************************/
/***** Feeler PAYLOAD *****/
/**************************/
static SUTILFN float traceFeeler(OptixTraversableHandle handle,
                                 const float3 &ray_origin,
                                 const float3 &ray_direction, float tmin,
                                 float tmax, PayloadFeeler *payload) {
  unsigned int occluded = 0u, u1 = 0u, u2 = 0u, u3 = 0u, u4 = 0u, u5 = 0u;
  optixTrace(handle, ray_origin, ray_direction, tmin, tmax,
             0.0f, // rayTime
             OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE,
             RAY_TYPE_OCCLUSION, // SBT offset
             RAY_TYPE_COUNT,     // SBT stride
             RAY_TYPE_OCCLUSION, // missSBTIndex
             occluded, u1, u2, u3, u4, u5);

  payload->dist = __int_as_float(u1);
  payload->normal.x = __int_as_float(u2);
  payload->normal.y = __int_as_float(u3);
  payload->normal.z = __int_as_float(u4);
  payload->n1_over_n2 = __int_as_float(u5);
  return occluded;
}

SUTILFN void setPayloadOcclusion(bool occluded) {
  optixSetPayload_0(static_cast<unsigned int>(occluded));
}

SUTILFN void setPayloadDistance(float dist) {
  optixSetPayload_1(__float_as_int(dist));
}

SUTILFN void setPayloadNormal(const float3 &n) {
  optixSetPayload_2(__float_as_int(n.x));
  optixSetPayload_3(__float_as_int(n.y));
  optixSetPayload_4(__float_as_int(n.z));
}

SUTILFN void setPayloadRelIOR(float n1_over_n2) {
  optixSetPayload_5(__float_as_int(n1_over_n2));
}
