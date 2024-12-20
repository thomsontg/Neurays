#include "Lights.h"

#include <vector>

// /********************************************/
// unsigned int Lights::extract_area_lights(
//     LaunchParams &launch_params,
//     std::vector<std::shared_ptr<sutil::MeshGroup>> &meshes, Materials &material,
//     const std::vector<Surface> &surfaces, sutil::BufferAlloc &alloc, bool first)
// /********************************************/
// {
//   const std::vector<MtlData> &materials = material.getMaterials();
//   std::vector<uint2> lights;
//   int mesh_idx = 0;
//   for (auto mesh : meshes) {
//     for (unsigned int j = 0; j < mesh->material_idx.size(); ++j) {
//       int mtl_idx = mesh->material_idx[j];
//       const MtlData &mtl = materials[mtl_idx];
//       bool emissive = false;
//       for (unsigned int k = 0; k < 3; ++k)
//         emissive = emissive || *(&mtl.emission.x + k) > 0.0f;
//       if (emissive)
//         lights.push_back(make_uint2(mesh_idx, mtl_idx));
//     }
//     ++mesh_idx;
//   }
//   if (lights.size() == 0) {
//     return 0;
//   }
//   Surface lightsurf;
//   std::vector<float3> emission;
//   for (unsigned int j = 0; j < lights.size(); ++j) {
//     uint2 light = lights[j];
//     auto mesh = meshes[light.x];
//     const Surface &surface = surfaces[light.x];
//     unsigned int no_of_verts =
//         static_cast<unsigned int>(lightsurf.positions.size());
//     lightsurf.positions.insert(lightsurf.positions.end(),
//                                surface.positions.begin(),
//                                surface.positions.end());
//     lightsurf.normals.insert(lightsurf.normals.end(), surface.positions.begin(),
//                              surface.positions.end());
//     lightsurf.indices.insert(lightsurf.indices.end(), surface.indices.begin(),
//                              surface.indices.end());
//     if (surface.normals.size() > 0)
//       for (unsigned int k = no_of_verts; k < lightsurf.normals.size(); ++k)
//         lightsurf.normals[k] = surface.normals[k - no_of_verts];
//     for (unsigned int k = lightsurf.no_of_faces; k < lightsurf.indices.size();
//          ++k) {
//       lightsurf.indices[k] += make_uint3(no_of_verts);
//       if (surface.normals.size() == 0) {
//         uint3 face = lightsurf.indices[k];
//         float3 p0 = lightsurf.positions[face.x];
//         float3 a = lightsurf.positions[face.y] - p0;
//         float3 b = lightsurf.positions[face.z] - p0;
//         lightsurf.normals[face.x] = lightsurf.normals[face.y] =
//             lightsurf.normals[face.z] = normalize(cross(a, b));
//       }
//     }
//     emission.insert(emission.end(), surface.no_of_faces,
//                     material.getMaterials()[light.y].emission);
//     lightsurf.no_of_faces += surface.no_of_faces;
//   }

//   {
//     BufferView<float3> buffer_view;
//     const uint64_t size = emission.size() * sizeof(float3);
//     if (first) {
//       unsigned int idx =
//           alloc.addBuffer(size, reinterpret_cast<const void *>(&emission[0]));
//       buffer_view.data = alloc.getBuffer(idx);
//       buffer_view.byte_stride = 0;
//       buffer_view.count = static_cast<uint32_t>(emission.size());
//       buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float3));
//       launch_params.light_emission = buffer_view;
//     } else {
//       CUdeviceptr buffer = launch_params.light_emission.data;
//       cudaMemcpy(reinterpret_cast<void *>(buffer),
//                  reinterpret_cast<const void *>(&emission[0]), size,
//                  cudaMemcpyHostToDevice);
//       return static_cast<unsigned int>(lights.size());
//     }
//   }

//   {

//     BufferView<uint3> buffer_view;
//     unsigned int idx =
//         alloc.addBuffer(lightsurf.indices.size() * sizeof(uint3),
//                         reinterpret_cast<const void *>(&lightsurf.indices[0]));
//     buffer_view.data = alloc.getBuffer(idx);
//     buffer_view.byte_stride = 0;
//     buffer_view.count = static_cast<uint32_t>(lightsurf.indices.size());
//     buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(uint3));
//     launch_params.light_idxs = buffer_view;
//   }
//   {
//     BufferView<float3> buffer_view;
//     unsigned int idx = alloc.addBuffer(
//         lightsurf.positions.size() * sizeof(float3),
//         reinterpret_cast<const void *>(&lightsurf.positions[0]));
//     buffer_view.data = alloc.getBuffer(idx);
//     buffer_view.byte_stride = 0;
//     buffer_view.count = static_cast<uint32_t>(lightsurf.positions.size());
//     buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float3));
//     launch_params.light_verts = buffer_view;
//   }
//   {
//     BufferView<float3> buffer_view;
//     unsigned int idx =
//         alloc.addBuffer(lightsurf.normals.size() * sizeof(float3),
//                         reinterpret_cast<const void *>(&lightsurf.normals[0]));
//     buffer_view.data = alloc.getBuffer(idx);
//     buffer_view.byte_stride = 0;
//     buffer_view.count = static_cast<uint32_t>(lightsurf.normals.size());
//     buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float3));
//     launch_params.light_norms = buffer_view;
//   }
//   float surface_area = 0.0f;
//   std::vector<float> face_areas(lightsurf.no_of_faces);
//   std::vector<float> face_area_cdf(lightsurf.no_of_faces);
//   for (unsigned int i = 0; i < lightsurf.no_of_faces; ++i) {
//     uint3 face = lightsurf.indices[i];
//     float3 p0 = lightsurf.positions[face.x];
//     float3 a = lightsurf.positions[face.y] - p0;
//     float3 b = lightsurf.positions[face.z] - p0;
//     face_areas[i] = 0.5f * length(cross(a, b));
//     face_area_cdf[i] = surface_area + face_areas[i];
//     surface_area += face_areas[i];
//   }
//   if (surface_area > 0.0f)
//     for (unsigned int i = 0; i < lightsurf.no_of_faces; ++i)
//       face_area_cdf[i] /= surface_area;
//   launch_params.light_area = surface_area;
//   {
//     BufferView<float> buffer_view;
//     unsigned int idx =
//         alloc.addBuffer(face_area_cdf.size() * sizeof(float),
//                         reinterpret_cast<const void *>(&face_area_cdf[0]));
//     buffer_view.data = alloc.getBuffer(idx);
//     buffer_view.byte_stride = 0;
//     buffer_view.count = static_cast<uint32_t>(face_area_cdf.size());
//     buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float));
//     launch_params.light_face_area_cdf = buffer_view;
//   }
//   return static_cast<unsigned int>(lights.size());
// }

/********************************************/
void Lights::add_default_light(LaunchParams &lp)
/********************************************/
{
  // The radiance of a directional source modeling the Sun should be equal
  // to the irradiance at the surface of the Earth.
  // We convert radiance to irradiance at the surface of the Earth using the
  // solid angle 6.74e-5 subtended by the solar disk as seen from Earth.

  // Default directional light
  lights.resize(1);
  lights[0].position = make_float3(0.0f, 1.0f, 0.0f);
  lights[0].emission =
      sunsky ? sun_sky.sunColor() * 6.74e-5f * (1.0f - sun_sky.getOvercast())
             : light_rad;
  lights[0].direction = sunsky ? -sun_sky.getSunDir() : normalize(light_dir);
  auto l = lights[0].direction;
  printf("%f %f %f\n", l.x, l.y, l.z);
  lights[0].type = DIRECTIONALLIGHT;
  update_light(lp);
}

/********************************************/
void Lights::update_light(LaunchParams &lp)
/********************************************/
{
  if (lp.lights.count == 0)
  {
    lp.lights.count = static_cast<uint32_t>(lights.size());
    (cudaMalloc(reinterpret_cast<void **>(&lp.lights.data),
                lights.size() * sizeof(SimpleLight)));
  }
  (cudaMemcpy(reinterpret_cast<void *>(lp.lights.data), lights.data(),
              lights.size() * sizeof(SimpleLight), cudaMemcpyHostToDevice));
}

/********************************************/
void Lights::initSunSky(float ordinal_day, float solar_time,
                        float globe_latitude, float turbidity, float overcast,
                        float sky_angle, const float3 &sky_up)
/********************************************/
{
  sun_sky.init(ordinal_day, solar_time, globe_latitude, turbidity, overcast,
               sky_angle, sky_up);
  sunsky = true;
}

/********************************************/
void Lights::handleSunSkyUpdate(LaunchParams &lp, float &solar_time,
                                float overcast)
/********************************************/
{
  sun_sky.updateParams(solar_time, overcast);
  lp.sunsky = sun_sky.getParams();
  add_default_light(lp);
}
