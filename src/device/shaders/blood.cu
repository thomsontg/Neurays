float random_nn (const float3& i){
	return sin(dot(i,make_float3(41595.34636,861.15646,268.489)))*968423.156- floor(sin(dot(i,make_float3(41595.34636,861.15646,268.489)))*968423.156);
}

float sdSphere_blood( float3 p, float s )
{
    return (length(p)-s);
}


float sdspheres_blood(const float3& p)
{
  const unsigned int spheres = 1;
  const float3 pi0 = floor(p - 0.5f);
  float result = 1.0e10f;
  float d1 = 0.5f;
  float b1 = 0.01;
  float h1 =0.12;
  float t1 = (d1*d1)/(b1*b1);
  float t2 = (b1*b1)/(h1*h1);
  float P = (-(d1*d1)/2)+(h1*h1/2)*(t1-1)-(h1*h1/2)*(t1-1)*sqrt(1-t2);
  float Q = P*t1+(b1*b1/4)*(t1*t1-1);
  float R = -P*(d1*d1/4)-(d1*d1*d1*d1/16);

  for(unsigned int i = 0; i < 8; ++i)
  {
    const float3 corner = make_float3(i&1, (i>>1)&1, (i>>2)&1);
    float rn = random_nn(pi0 + corner);
    const float3 pi = pi0 + corner;//+make_float3(gaussian(p.x,0.1,0.3),gaussian(p.y,0.1,0.3),gaussian(p.z,0.1,0.5))+make_float3(gaussian(corner.x,0.1,0.3),gaussian(corner.y,0.1,0.3),gaussian(corner.z,0.1,0.5));
    
    const int3 pii = make_int3(pi.x, pi.y, pi.z);
    //unsigned int t = 4u*spheres*static_cast<unsigned int>(pii.x + pii.y + pii.z + pii.x*pii.y*pii.z*31+(pii.x*pii.y*11)+pii.x*pii.z*13+pii.y*pii.z*7+pii.x*pii.x);
    //unsigned int t = 4u*spheres*static_cast<unsigned int>(pii.x*17 + pii.y*13 + pii.z*15 +pii.x*pii.x*11+pii.y*pii.y*19+pii.z*pii.z*23+pii.x*pii.y*pii.z*13*17);
    unsigned int t = 4u*spheres*static_cast<unsigned int>(pii.x*29 + pii.y*23 + pii.z*19 +pii.x*pii.x*17+pii.y*pii.y*13+pii.z*pii.z*37+pii.x*pii.y*pii.z*19);
    //unsigned int t = 4u*spheres*static_cast<unsigned int>(pii.x + pii.y*1000 + pii.z*576);
   // unsigned int t1 = 4u*static_cast<unsigned int>(rn);
    for(unsigned int j = 0; j < spheres; ++j)
    {
      float r = rnd(t)*0.1f + 0.2f;
      float3 xi = make_float3(rnd(t), rnd(t), rnd(t));
      float3 x = pi+xi;
      result = fmin(result, ((p.x-x.x)*(p.x-x.x)+(p.y-x.y)*(p.y-x.y)+(p.z-x.z)*(p.z-x.z))*((p.x-x.x)*(p.x-x.x)+(p.y-x.y)*(p.y-x.y)+(p.z-x.z)*(p.z-x.z))+P*((p.x-x.x)*(p.x-x.x)+(p.y-x.y)*(p.y-x.y))+Q*(p.z-x.z)*(p.z-x.z)+R);
    }
  }

  return result;
}

float sdf_blood(float3 p)
{
   float d = 1.5;
   //float n = 0.0;
   
   float s = 0.1;
   //float th =0.25;
   //float a = 0.0;
   float n1 = s*sdSphere_blood(p,125.05);
   //float n1 = s*sdf_gyroid(p);
  //  float n2 = 0.0f;
   //unsigned int seed = 90.0;
   unsigned int seed = 4u*static_cast<unsigned int>(floor(p.x - 0.5f)+floor(p.y - 0.5f)+floor(p.y - 0.5f)/floor(p.x - 0.5f));
   float th = 45.0;
   //th = th+rnd(seed);
   

   for( int i=0; i<1; i++ ){
       
       //float n = s*sdspheres(p+3.14*sin(p.x)+3.14*sin(p.y)+3.14*sin(p.z));
      //  float f = 00004.2;
      //  float o1 = 4.5*(p.x+p.y+p.x);
      //  float o2 = 4.5*(p.y+p.x+p.y);
      //  float o3 = 0.5*(p.x+p.y+p.z);
       //float o4 = 0.03*o1+0.24*o2+0.13*o3;
       float phi = 1.0*max(0.0,15.5*3.14159265358979323846);
      //  float a = 0.5;

       //p.x=sin (2.0* M_PI * f  * (p.y*sin(o1+phi) + p.y*sin(o2+phi)+p.y*sin(o3+phi))+phi);
       //n2 = s*sdspheres(make_float3(gaussian(p.x,0.5,0.3),gaussian(p.y,0.5,0.3),gaussian(p.z,0.5,0.5)));
      //  n2 = s*sdspheres_blood(make_float3(p.x,p.y,p.z-0.1*sin(p.z)));
       float n22 = s*sdspheres_blood(make_float3(p.x,p.y,5.0*p.z)-(p.x*p.x+p.y*p.y+p.z*p.z-0.1));
       float n44 = s*sdspheres_blood(make_float3(0.5*p.x,0.7*p.y,0.9*p.z));
       float p1=p.x;//*sin(p.z); //*sin(3.14*p.y);
       float p2= p.y;//*sin(p.x);
       float p3 =p.z;
       p.x= p1;
       p.y = p2 * cos(th) - p3 * sin(th);
       p.z= p2 * sin(th) + p3 * cos(th);

       float n45 = sdspheres_blood(p);
       th = 30;
       p.x= p1;
       p.y = p2 * cos(th) - p3 * sin(th);
       p.z= p2 * sin(th) + p3 * cos(th);

       float n46 = sdspheres_blood(p);
       th = 60;
       p.x= p1;
       p.y = p2 * cos(th) - p3 * sin(th);
       p.z= p2 * sin(th) + p3 * cos(th);


       float n47 = sdspheres_blood(p);
       th = 75;
       p.x= p1;
       p.y = p2 * cos(th) - p3 * sin(th);
       p.z= p2 * sin(th) + p3 * cos(th);
       


       float n48 = sdspheres_blood(p);
       th = 90;
       p.x= p1;
       p.y = p2 * cos(th) - p3 * sin(th);
       p.z= p2 * sin(th) + p3 * cos(th);


       float n49 = sdspheres_blood(p);
       th = 15;
       p.x= p1;
       p.y = p2 * cos(th) - p3 * sin(th);
       p.z= p2 * sin(th) + p3 * cos(th);


       float n50 = sdspheres_blood(p);
       th = 105;
       p.x= p1;
       p.y = p2 * cos(th) - p3 * sin(th);
       p.z= p2 * sin(th) + p3 * cos(th);


       float n51 = sdspheres_blood(p);
        //n2= s*(sdspheres(p.z*p.x*p.y+sin(p.y)+sin(p.z)+sin(p.x)+make_float3(sin (2.0* M_PI * f  * (p.y*sin(o1+phi) + p.y*sin(o2+phi)+p.y*sin(o3+phi))+phi),sin (2.0* M_PI * f  * (p.z*sin(o2+phi) + p.y*sin(o2+phi)+p.x*sin(o2+phi))+phi),sin (2.0* M_PI * f  * (p.x*sin(o3+phi) + p.z*sin(o3+phi)+sin(o3+phi)*p.y)+phi))));             
       //float  n3= sdspheres(make_float3(sin(p.y)+sin(p.x) ,0.0f,0.0f));
       //float  n3= sdspheres(make_float3(0.0f,sin(p.x)+sin(p.y)+sin(p.z),0.0f));
       
       
       
       
       
       d = min(n45,n46);
       //d = smax( n1,min(d,n47), 0.1*s );
       d = min(d,n47);
       d = min(d,n48);
       d = min(d,n49);
       d = min(d,n50);
       //d = max(d,n50);
       n1= d;
       //n1 = s*sdSphere(p,0.5);
       s = 0.1*s;
       th =th+30;
       

   }
  
   return d;
}


float raycast_blood(const float3& ro, const float3& rd, float tmin, float tmax)
{
  // ray marching
  float t = tmin;
  float d = sdf_blood(ro + t*rd);
  const float sgn = copysignf(1.0f, d);
  for(unsigned int i = 0; i < 1500u; ++i)
  {
    if(fabsf(d) < precis*t || t > tmax) break;
    t += sgn*d*0.5f; // *1.2f;
    d = sdf_blood(ro + t*rd);
  }
  return t; //t < tmax ? t : 1.0e16f;
}

float3 calcNormal_blood(const float3& pos)
{
  float2 e = make_float2(1.0f, -1.0f)*0.5773f*precis;
  return normalize(
    make_float3(e.x, e.y, e.y)*sdf_blood(pos + make_float3(e.x, e.y, e.y)) +
    make_float3(e.y, e.y, e.x)*sdf_blood(pos + make_float3(e.y, e.y, e.x)) +
    make_float3(e.y, e.x, e.y)*sdf_blood(pos + make_float3(e.y, e.x, e.y)) +
    make_float3(e.x,e.x,e.x)*sdf_blood(pos + make_float3(e.x,e.x,e.x)));
}



closesthit_fn __closesthit__blood()
{
  SHADER_HEADER
  
  const unsigned int max_depth = 50;
  if(depth > lp.max_depth)
    return;
  const float tmin = 1.0e-3f;
  const float tmax = 1.0e16f;
  float3 x = geom.P;
  float3 w = optixGetWorldRayDirection();
  float3 n = normalize(geom.N);
  float dist = optixGetRayTmax();
  float3 result = make_float3(1.0f);
  const float noise_scale = hit_group_data->mtl_inside.noise_scale;
  // const float density = hit_group_data->mtl_inside.density;

  // Retrieve material data
  const float ior1 = hit_group_data->mtl_outside.ior;
  const float ior2 = hit_group_data->mtl_inside.ior;

  // Ray casting
  const bool inside_mesh = dot(w, n) > 0.0f;
  float sgn = 1.0f;
  if(inside_mesh)
  {
    PayloadFeeler feeler;
    float3 p = optixGetWorldRayOrigin();
    unsigned int i = 0;
    for(; i < max_depth; ++i)
    {
      float s = raycast_blood(p*noise_scale, w, 0.01f, dist*noise_scale)/noise_scale;
      if(s > 0.0f && s < dist)
      {
        p += s*w;
        n = calcNormal_blood(p*noise_scale);

        // Compute cosine of the angle of observation
        float cos_theta_o = dot(-w, n);

        // Check for absorption
        const bool inside = cos_theta_o < 0.0f;
        const float3& extinction = inside ? hit_group_data->mtl_inside.ext : hit_group_data->mtl_outside.ext;
        const float3 Tr = expf(-extinction*s);
        const float prob = (Tr.x + Tr.y + Tr.z)/3.0f;
        if(rnd(t) > prob)
          return;
        result *= Tr/prob;

        // Compute relative index of refraction
        float n1_over_n2 = ior1/ior2;
        if(inside)
        {
          n = -n;
          cos_theta_o = -cos_theta_o;
          n1_over_n2 = 1.0f/n1_over_n2;
        }

        // Compute Fresnel reflectance (R) and trace refracted ray if necessary
        float cos_theta_t;
        float R = 1.0f;
        const float sin_theta_t_sqr = n1_over_n2*n1_over_n2*(1.0f - cos_theta_o*cos_theta_o);
        if(sin_theta_t_sqr < 1.0f)
        {
          cos_theta_t = sqrtf(1.0f - sin_theta_t_sqr);
          R = fresnel_R(cos_theta_o, cos_theta_t, n1_over_n2);
        }

        // Russian roulette to select reflection or refraction
        float xi = rnd(t);
        w = xi < R ? reflect(w, n) : n1_over_n2*(cos_theta_o*n + w) - n*cos_theta_t;

        traceFeeler(lp.handle, p, w, tmin, tmax, &feeler);
        dist = feeler.dist;
      }
      else
        break;
    }
    if(i == max_depth)
      return;
    x = p + w*dist;
    sgn = copysignf(1.0f, sdspheres_blood(x*noise_scale));

    // Check for absorption
    const float3& extinction = sgn < 0.0f ? hit_group_data->mtl_inside.ext : hit_group_data->mtl_outside.ext;
    const float3 Tr = expf(-extinction*dist);
    const float prob = (Tr.x + Tr.y + Tr.z)/3.0f;
    if(rnd(t) > prob)
      return;
    result *= Tr/prob;

    n = feeler.normal;
  }
  else
    sgn = copysignf(1.0f, sdspheres_blood(x*noise_scale));
  
  // Compute cosine of the angle of observation
  float cos_theta_o = dot(-w, n);

  // Compute relative index of refraction
  float n1_over_n2 = sgn < 0.0f ? 1.0f/ior2 : 1.0f/ior1;
  if(inside_mesh)
  {
    n = -n;
    cos_theta_o = -cos_theta_o;
    n1_over_n2 = 1.0f/n1_over_n2;
  }

  // Compute Fresnel reflectance (R) and trace refracted ray if necessary
  float cos_theta_t;
  float R = 1.0f;
  const float sin_theta_t_sqr = n1_over_n2*n1_over_n2*(1.0f - cos_theta_o*cos_theta_o);
  if(sin_theta_t_sqr < 1.0f)
  {
    cos_theta_t = sqrtf(1.0f - sin_theta_t_sqr);
    R = fresnel_R(cos_theta_o, cos_theta_t, n1_over_n2);
  }

  // Russian roulette to select reflection or refraction
  float xi = rnd(t);
  w = xi < R ? reflect(w, n) : n1_over_n2*(cos_theta_o*n + w) - n*cos_theta_t;

  // Trace the new ray
  setPayloadSeed(t);
  setPayloadDirection(w);
  setPayloadOrigin(x);
  setPayloadAttenuation(getPayloadAttenuation());
  setPayloadDead(0);
  setPayloadEmit(1);
  setPayloadResult(make_float3(0.0f));
}