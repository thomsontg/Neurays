// Retrieve ray and hit info
#define SHADER_HEADER                                                          \
  const LaunchParams &lp = launch_params;                                      \
  const HitGroupData *hit_group_data =                                         \
      reinterpret_cast<HitGroupData *>(optixGetSbtDataPointer());              \
  const LocalGeometry geom = getLocalGeometry(hit_group_data->geometry);       \
  RNDTYPE t = getPayloadSeed();                                                \
  unsigned int depth = getPayloadDepth();                                      \
  unsigned int emit = getPayloadEmit();                                        \
  
#define SHADER_UNUSED(x) (void)(x)