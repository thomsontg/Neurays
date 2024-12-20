#ifndef INTERFACE_H
#define INTERFACE_H

#include <string>
#include "Medium.h"

class Interface
{
public:
  Interface() : med_in(0), med_out(0) { }

  std::string name;
  Medium* med_in;
  Medium* med_out;
  //const Geometry::Material& mat_in;
  //const Geometry::Material& mat_out;
  //const Shader* shader_in;
  //const Shader* shader_out;
};

#endif
