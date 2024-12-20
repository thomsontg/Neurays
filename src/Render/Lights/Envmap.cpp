#include "Envmap.h"

#include <assert.h>
#include <iostream>
#include <memory>

#include <optix.h>

#include "sutil/shared/structs.h"
#include <sutil/sutil.h>

namespace
{
  void lookup(HDRLoader &hdr, float up, float vp, float3 &retval)
  {
    uint32_t w_offset = static_cast<uint32_t>(up * hdr.width());
    uint32_t h_offset = static_cast<uint32_t>(vp * hdr.height());

    uint32_t index = h_offset * hdr.width() * 4 + w_offset * 4;
    retval.x = hdr.raster()[index];
    retval.y = hdr.raster()[index + 1];
    retval.z = hdr.raster()[index + 2];
  }

  float luminance(const float3 &val)
  {
    return 0.299f * val.x + 0.587f * val.y + 0.114f * val.z;
  }

} // namespace

bool Envmap::load(sutil::BufferAlloc &b_alloc, sutil::SamplerAlloc &s_alloc,
                  LaunchParams *lp)
{
  if (!use_envmap_)
  {
    return false;
  }
  bool is_hdr = envfile.compare(envfile.length() - 3, 3, "hdr") == 0;
  if (is_hdr)
  {
    HDRLoader hdr(envfile);
    if (hdr.failed())
    {
      std::cerr << "Could not load HDR environment map called: " << envfile
                << std::endl;
      use_envmap_ = false;
    }
    s_alloc.addImage(hdr.width(), hdr.height(), 32, 4, hdr.raster());
    lp->env_height = hdr.height();
    lp->env_width = hdr.width();
    // createEnvMapBuffers(hdr, b_alloc, lp);
  }
  else
  {
    sutil::ImageBuffer img = sutil::load_Image(envfile.c_str());
    if (img.pixel_format != sutil::UNSIGNED_BYTE4)
    {
      std::cerr << "Environment map texture image with unknown pixel format: "
                << envfile << std::endl;
      use_envmap_ = false;
    }
    s_alloc.addImage(img.width, img.height, 8, 4, img.data);
  }
  if (use_envmap_)
  {
    s_alloc.addSampler(cudaAddressModeWrap, cudaAddressModeWrap,
                       cudaFilterModeLinear, 0, is_hdr);
  }
  return use_envmap_;
}

bool Envmap::replace(sutil::BufferAlloc &b_alloc, sutil::SamplerAlloc &s_alloc,
                     LaunchParams *lp)
{
  if (!use_envmap_)
  {
    return false;
  }
  bool is_hdr = envfile.compare(envfile.length() - 3, 3, "hdr") == 0;
  if (is_hdr)
  {
    HDRLoader hdr(envfile);
    if (hdr.failed())
    {
      std::cerr << "Could not load HDR environment map called: " << envfile
                << std::endl;
      use_envmap_ = false;
    }
    s_alloc.replaceImage(hdr.width(), hdr.height(), 32, 4, hdr.raster());
    lp->env_height = hdr.height();
    lp->env_width = hdr.width();
    // createEnvMapBuffers(hdr, b_alloc, lp);
  }
  else
  {
    sutil::ImageBuffer img = sutil::load_Image(envfile.c_str());
    if (img.pixel_format != sutil::UNSIGNED_BYTE4)
    {
      std::cerr << "Environment map texture image with unknown pixel format: "
                << envfile << std::endl;
      use_envmap_ = false;
    }
    s_alloc.replaceImage(img.width, img.height, 8, 4, img.data, 0);
  }
  if (use_envmap_)
  {
    s_alloc.replaceSampler(cudaAddressModeWrap, cudaAddressModeWrap,
                           cudaFilterModeLinear, 0, is_hdr, 0);
  }
  return use_envmap_;
}

// void Envmap::createEnvMapBuffers(HDRLoader &hdr, sutil::BufferAlloc &b_alloc,
//                                  LaunchParams *lp) {
//   const uint32_t h = hdr.height();
//   const uint32_t w = hdr.width();

//   // Create scalar valued image for sampling
//   auto marginal_pdf = std::make_unique<float[]>(h);
//   auto marginal_cdf = std::make_unique<float[]>(h);
//   auto conditional_pdf = std::make_unique<float[]>(h * w);
//   auto conditional_cdf = std::make_unique<float[]>(h * w);
//   float marginal_sum = 0.0f;
//   float rotation_phi = lp->envmap_phi / (2.0f * M_PIf) * static_cast<float>(w);
//   for (uint32_t v = 0; v < h; v++) {
//     float vp = (v + 0.5f) / static_cast<float>(h);
//     float sinTheta = std::sin(M_PIf * (v + 0.5f) / static_cast<float>(h));

//     float conditional_sum = 0.f;
//     for (uint32_t u = 0; u < w; u++) {
//       float up = std::fmod(u + 0.5f + rotation_phi, static_cast<float>(w)) /
//                  static_cast<float>(w);
//       float3 val;
//       lookup(hdr, up, vp, val);
//       conditional_pdf[u + v * w] = luminance(val) * sinTheta;
//       conditional_cdf[u + v * w] = conditional_sum + conditional_pdf[u + v * w];
//       conditional_sum += conditional_pdf[u + v * w];
//     }
//     marginal_pdf[v] = conditional_sum;
//     marginal_cdf[v] = marginal_sum + conditional_sum;
//     marginal_sum += conditional_sum;

//     // Normalize conditional pdf;
//     for (uint32_t u = 0; u < w; u++) {
//       conditional_pdf[u + v * w] /= conditional_sum;
//       conditional_cdf[u + v * w] /= conditional_sum;
//     }
//   }
//   // Normalize marginal pdf/cdf;
//   for (uint32_t v = 0; v < h; v++) {
//     marginal_pdf[v] /= marginal_sum;
//     marginal_cdf[v] /= marginal_sum;
//   }

//   // Upload buffers to launch params

//   { // Marginal pdf
//     BufferView<float> buffer_view;
//     unsigned int idx = b_alloc.addBuffer(
//         h * sizeof(float), reinterpret_cast<const void *>(&marginal_pdf[0]));
//     buffer_view.data = b_alloc.getBuffer(idx);
//     buffer_view.byte_stride = 0;
//     buffer_view.count = static_cast<uint32_t>(h);
//     buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float));
//     lp->marginal_pdf = buffer_view;
//   }

//   { // Marginal cdf
//     BufferView<float> buffer_view;
//     unsigned int idx = b_alloc.addBuffer(
//         h * sizeof(float), reinterpret_cast<const void *>(&marginal_cdf[0]));
//     buffer_view.data = b_alloc.getBuffer(idx);
//     buffer_view.byte_stride = 0;
//     buffer_view.count = static_cast<uint32_t>(h);
//     buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float));
//     lp->marginal_cdf = buffer_view;
//   }

//   { // Conditional pdf
//     BufferView<float> buffer_view;
//     unsigned int idx =
//         b_alloc.addBuffer(w * h * sizeof(float),
//                           reinterpret_cast<const void *>(&conditional_pdf[0]));
//     buffer_view.data = b_alloc.getBuffer(idx);
//     buffer_view.byte_stride = 0;
//     buffer_view.count = static_cast<uint32_t>(w * h);
//     buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float));
//     lp->conditional_pdf = buffer_view;
//   }

//   { // Conditional cdf
//     BufferView<float> buffer_view;
//     unsigned int idx =
//         b_alloc.addBuffer(w * h * sizeof(float),
//                           reinterpret_cast<const void *>(&conditional_cdf[0]));
//     buffer_view.data = b_alloc.getBuffer(idx);
//     buffer_view.byte_stride = 0;
//     buffer_view.count = static_cast<uint32_t>(w * h);
//     buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float));
//     lp->conditional_cdf = buffer_view;
//   }

//   lp->env_height = h;
//   lp->env_width = w;
// }
