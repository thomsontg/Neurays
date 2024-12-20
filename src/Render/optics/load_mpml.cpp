#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <complex>
#include <algorithm>
#include <tinyxml2/tinyxml2.h>
#include "Parse.h"
#include "Medium.h"
#include "Interface.h"
#include "load_mpml.h"

using namespace tinyxml2;

namespace
{
  ColorType parseColorAttrib(const std::string& s, const char* name)
  {
    if(s == "" || s == "rgb")
      return rgb;
    else if(s == "mono")
      return mono;
    else if(s == "xyz")
      return xyz;
    else if(s == "spectrum")
      return spectrum;

    std::cerr << "Warning loading " << name << ": Unknown color attribute" << std::endl;
    return rgb;
  }

  ScatteringUnit parseScatteringUnitAttrib(const std::string& s, const char* name)
  {
    if(s == "" || s == "1/m")
      return per_meter;
    else if(s == "1/cm")
      return per_cm;
    else if(s == "1/mm")
      return per_mm;

    std::cerr << "Warning loading " << name << ": Unknown scattering unit attribute" << std::endl;
    return per_meter;
  }

  SpectralUnit parseSpectralUnitAttrib(const std::string& s, const char* name)
  {
    if(s == "" || s == "nm")
      return nanometers;
    else if(s == "km")
      return kilometers;
    else if(s == "m")
      return meters;
    else if(s == "mm")
      return millimeters;
    else if(s == "um")
      return micrometers;
    else if(s == "�")
      return angstrom;
    else if(s == "eV")
      return electron_volt;

    std::cerr << "Warning loading " << name << ": Unknown spectral unit attribute" << std::endl;
    return nanometers;
  }

  template<class T> void checkColorSize(ColorType type, Color<T>& c, const char* name)
  {
    switch(type)
    {
    case mono:
      if(c.size() != 1)
          std::cerr << "Warning loading " << name << ": Incorrect number of data values" << std::endl;
      break;
    case rgb:
    case xyz:
      if(c.size() == 1)
      {
        T tmp = c[0];
        c.resize(3);
        for(int i = 0; i < 3; ++i)
          c[i] = tmp;
      }
      if(c.size() != 3)
          std::cerr << "Warning loading " << name << ": Incorrect number of data values" << std::endl;
      break;
    }
  }

  void parseColorData(XMLElement& elem, ColorType type, Color<double>& c)
  {
      std::vector<double> data;
    parse(elem.GetText(), data);
    c.resize(data.size());
    copy(data.begin(), data.end(), &c[0]);
    checkColorSize(type, c, elem.Attribute("name"));
    if(type == spectrum)
    {
      c.unit = parseSpectralUnitAttrib(elem.Attribute("unit"), elem.Attribute("name"));
      int values;
      parse(elem.Attribute("values"), values);
      std::vector<double> wavelengths;
      parse(elem.Attribute("wavelengths"), wavelengths);
      if(wavelengths.size() > 1)
      {
        c.wavelength = wavelengths[0];
        c.step_size = (wavelengths[1] - wavelengths[0])/static_cast<double>(values - 1);
      }
    }
  }
}

void handle_RefractiveIndex(XMLElement& elem, Medium *media)
{
  ColorType type = parseColorAttrib(elem.Attribute("color"), elem.Attribute("name"));
  Color< std::complex<double> >& c = media->get_ior(type);
  std::vector<double> data;
  parse(elem.GetText(), data);
  c.resize(data.size()/2);
  for(unsigned int i = 0; i < c.size(); ++i)
  {
    c[i] = std::complex<double>(data[i*2], data[i*2 + 1]);
  }
  checkColorSize(type, c, elem.Attribute("name"));
  if(type == spectrum)
  {
    c.unit = parseSpectralUnitAttrib(elem.Attribute("unit"), elem.Attribute("name"));
    int values;
    parse(elem.Attribute("values"), values);
    std::vector<double> wavelengths;
    parse(elem.Attribute("wavelengths"), wavelengths);
    if(wavelengths.size() > 1)
    {
      c.wavelength = wavelengths[0];
      c.step_size = (wavelengths[1] - wavelengths[0])/static_cast<double>(values - 1);
    }
  }
}

void process_Material(XMLElement& elem, Medium *media)
{
  for (XMLElement *child = elem.FirstChildElement(); child != NULL ; child = child->NextSiblingElement())
  {
    if(std::string(child->Name()) == "RefractiveIndex")
    {
      handle_RefractiveIndex(*child, media);
    }
    else if(std::string(child->Name()) == "Scattering")
    {
      media->turbid = true;
      media->scatter_unit = parseScatteringUnitAttrib(child->Attribute("unit"), elem.Attribute("name"));
      process_Material(*child, media);
    }
    else if(std::string(child->Name()) == "Coefficient")
    {
      ColorType type = parseColorAttrib(child->Attribute("color"), elem.Attribute("name"));
      parseColorData(*child, type, media->get_scattering(type));
    }
    else if(std::string(child->Name()) == "PhaseFunction")
    {
      process_Material(*child, media);
    }
    else if(std::string(child->Name()) == "Asymmetry")
    {
      ColorType type = parseColorAttrib(child->Attribute("color"), elem.Attribute("name"));
      parseColorData(*child, type, media->get_asymmetry(type));
    }
    else
    {
      std::cout << "<<<<< Undefined Attribute found while parsing Material. >>>>>" << std::endl; 
    }
  }
}

void fill_Interfaces(std::map<std::string, std::pair<std::string, std::string>> &interface_list, std::map<std::string, Interface> &interface_map, std::map<std::string, Medium> &medium_map)
{
  for (auto const &map : interface_list)
  {
    Interface iface;
    iface.name = map.first;
    iface.med_in = &(medium_map[map.second.first]);
    iface.med_out = &(medium_map[map.second.first]);
    interface_map[map.first] = iface;
  }
}

Medium get_material(XMLElement &material_element)
{
  Medium media;
  media.name = material_element.Attribute("name");
  process_Material(material_element, &media);
  return media;
}

void load_mpml(std::string filename, std::map<std::string, Medium>& media_map, std::map<std::string, Interface>& interface_map)
{
  std::cout << "Loading " << filename << std::endl;
  XMLDocument mpml;
  if(mpml.LoadFile(filename.c_str()) != XML_SUCCESS)
  {
    printf("Error loading MPML file.\n");
    return;
  }
  XMLElement* material_content =  mpml.FirstChildElement("MPML");

  int32_t inter = 0, mater = 0;
  std::map<std::string, std::pair<std::string, std::string>> interface_list;
  for (XMLElement *child = material_content->FirstChildElement(); child != NULL ; child = child->NextSiblingElement())
  {
    if(std::string(child->Name()) == std::string("Material"))
    {
      media_map[child->Attribute("name")] = get_material(*child);
      mater++;
    }
    else if(std::string(child->Name()) == std::string("Interface"))
    {
      interface_list[(*child).Attribute("name")] =  std::pair((*child).Attribute("inside"), (*child).Attribute("outside")) ;
      inter++;
    }
    else
    {
      std::cout << "Unhandled argument in MPML file.";
      return;
    }
  }
  fill_Interfaces(interface_list, interface_map, media_map);
  std::cout <<" Material Count : " << mater << std::endl <<
              " Interface Count : " << inter << std::endl;
}
