#pragma once

#include <iostream>
#include <filesystem>
#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <tinygltf/stb_image.h>
#include <tinyxml2/tinyxml2.h>
#include <sutil/shared/GLDisplay.h>
#include "sutil/sutil.h"
#include "Scene.h"

namespace fs = std::filesystem;


//------------------------------------------------------------------------------
//
// Utility functions
//
//------------------------------------------------------------------------------

using namespace tinyxml2;

namespace Utility{

inline void printUsageAndExit(const char *argv0)
{
  std::cerr << "Usage  : " << argv0 << " [options] any_object.obj [another.obj ...]" << std::endl
            << "Options: --help    | -h                 Print this usage message" << std::endl
            << "         --shader  | -sh <shader>       Specify the closest hit program to be used for shading" << std::endl
            << "                   | -env <filename>    Specify the environment map to be loaded in panoramic format" << std::endl
            << "                   | -bgc <r> <g> <b>   Specify RGB background color (not used if env is available)" << std::endl
            << "         --dim=<width>x<height>         Set image dimensions; defaults to 768x768" << std::endl
            << "         --no-gl-interop                Disable GL interop for display" << std::endl
            << "         --file    | -f <filename>      File for image output" << std::endl
            << "         --bsdf                         Render a BSDF slice in [-1,1]x[-1,1] of the xy-plane" << std::endl
            << "         --samples | -s                 Number of samples per pixel if rendering to file (default 16)" << std::endl
            << "                   | -sky <l> <t> <o>   Use the Preetham sun and sky model (latitude <l>, turbidity <t>, overcast <o>)" << std::endl
            << "                   | -t <d> <t>         Set the ordinal day <d> and solar time <t> for the sun and sky model" << std::endl
            << "                   | -r <a> <x> <y> <z> Angle and axis (up vector) for rotation of environment" << std::endl
            << "                   | -sc <s>            Scene scale <s> for scaling optical properties defined per meter (default 1e-2)" << std::endl
            << "                   | -dir <th> <ph>     Direction of default light in spherical coordinates (polar <th>, azimuth <ph>)" << std::endl
            << "                   | -rad <r> <g> <b>   Specify RGB radiance of default directional light (default PI)" << std::endl
            << "                   | -bf <s>            Beam factor in [0,1]: the ratio of beam to microgeometry width (default 1)" << std::endl;
  exit(0);
}


inline float smoothstep(float edge0, float edge1, float x)
{
  // Scale, bias and saturate x to 0..1 range
  x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
  // Evaluate polynomial
  return x * x * (3 - 2 * x);
}

inline float smootherstep(float edge0, float edge1, float x)
{
  // Scale, and clamp x to 0..1 range
  x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
  // Evaluate polynomial
  return x * x * x * (x * (x * 6 - 15) + 10);
}

inline void createsplines(SplineData& splinedata)
{
  tk::spline s_eye_dist(splinedata.X, splinedata.eye_dist);
  tk::spline s_qinc_qv_x(splinedata.X, splinedata.qinc_qv_x);
  tk::spline s_qinc_qv_y(splinedata.X, splinedata.qinc_qv_y);
  tk::spline s_qinc_qv_z(splinedata.X, splinedata.qinc_qv_z);
  tk::spline s_qinc_qw(splinedata.X, splinedata.qinc_qw);
  tk::spline s_qrot_qv_x(splinedata.X, splinedata.qrot_qv_x);
  tk::spline s_qrot_qv_y(splinedata.X, splinedata.qrot_qv_y);
  tk::spline s_qrot_qv_z(splinedata.X, splinedata.qrot_qv_z);
  tk::spline s_qrot_qw(splinedata.X, splinedata.qrot_qw);
}

inline QuatTrackBall& load_view_params(const std::string& filename, float& cam_const)
{
  QuatTrackBall* temp_trackball = new QuatTrackBall(make_float3(0.0f), 0.0f, 10, 10);
  std::ifstream ifs_view(filename.c_str(), std::ifstream::binary);
  if (ifs_view)
  {
      ifs_view.read(reinterpret_cast<char*>(temp_trackball), sizeof(QuatTrackBall));
      ifs_view.read(reinterpret_cast<char*>(&cam_const), sizeof(float));
  }
  ifs_view.close();
  return *temp_trackball;
}

inline void run_spline(int& keypoints, SplineData& splinedata, float& cam_const)
{
  for (int i = 0; i < keypoints; i++)
  {
    splinedata.append_data(load_view_params(std::to_string(i), cam_const), i);
  }
  createsplines(splinedata);
}

// Avoiding case sensitivity
inline void lower_case(char& x)
{
  x = tolower(x);
}

inline void lower_case_string(std::string& s)
{
  for_each(s.begin(), s.end(), lower_case);
}

inline void display_Frame(
    sutil::CUDAOutputBuffer<uchar4> &output_buffer,
    sutil::GLDisplay &gl_display,
    GLFWwindow *window)
{
  // Display
  int framebuf_res_x = 0; // The display's resolution (could be HDPI res)
  int framebuf_res_y = 0; //
  glfwGetFramebufferSize(window, &framebuf_res_x, &framebuf_res_y);
  gl_display.display( 0, output_buffer.width(),
                      output_buffer.height(), output_buffer.width(),
                      framebuf_res_y, output_buffer.getPBO());
}

inline void set_cursor(GLFWwindow *window)
{
  std::string f_name = std::string(SOURCE_DIR) + "/src/assets/cursor_mini.png";
  sutil::ImageBuffer stb_img = sutil::load_Image(f_name.c_str());

  GLFWimage img;
  img.height = stb_img.height;
  img.width = stb_img.width;
  img.pixels = (unsigned char *)stb_img.data;

  GLFWcursor *cursor = glfwCreateCursor(&img, 0, 0);
  glfwSetCursor(window, cursor);
}

inline void exportRawImage(Globals &gbx, std::string outfile)
{
  // Get image info
  size_t name_end = outfile.find_last_of('.');
  std::string name = outfile.substr(0, name_end);

  // Write image info in .txt-file
  std::ofstream ofs_data(name + ".txt");
  if (ofs_data.bad())
    return;
  ofs_data << gbx.launch_params.subframe_index << std::endl
           << gbx.rd_width << " " << gbx.rd_height << std::endl;
  ofs_data << gbx.theta_i << " " << gbx.phi_i;
  ofs_data.close();

  // Copy buffer data from device to host
  int size_buffer = gbx.rd_width * gbx.rd_height * 4;
  float *mapped = new float[size_buffer];
  CUDA_CHECK(cudaMemcpyAsync(mapped, gbx.launch_params.accum_buffer, size_buffer * sizeof(float), cudaMemcpyDeviceToHost, 0));

  // Export image data to binary .raw-file
  std::ofstream ofs_image;
  ofs_image.open(name + ".raw", std::ios::binary);
  if (ofs_image.bad())
  {
    std::cerr << "Error when exporting file" << std::endl;
    return;
  }

  int size_image = gbx.rd_width * gbx.rd_height * 3;
  float *converted = new float[size_image];
  float average = 0.0f;
  for (int i = 0; i < size_image / 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      float value = mapped[i * 4 + j] / gbx.launch_params.subframe_index;
      converted[i * 3 + j] = value;
      average += value;
    }
  }
  average /= size_image;
  delete[] mapped;
  ofs_image.write(reinterpret_cast<const char *>(converted), size_image * sizeof(float));
  ofs_image.close();
  delete[] converted;
  std::cout << "Exported buffer to " << name << ".raw (avg: " << average << ")" << std::endl;
}

inline void flip_image_vertically(float *buffer, int width, int height,
                                  int elem_size)
{
  // Flip vertical axis
  int buffer_size = width * height * elem_size;
  float *flipped = new float[buffer_size];
  for (int i = 0; i < height; i++)
  {
    int row = i * width * elem_size;
    int row_inv = (height - i - 1) * width * elem_size;
    memcpy(&flipped[row_inv], &buffer[row], sizeof(float) * width * elem_size);
  }
  memcpy(buffer, flipped, sizeof(float) * buffer_size);
}

inline float *map_image(float4 *cuda_buffer, size_t buffer_size)
{
  // Copy buffer data from device to host
  float *mapped = new float[buffer_size];
  CUDA_CHECK(cudaMemcpyAsync(mapped, cuda_buffer, buffer_size * sizeof(float),
                             cudaMemcpyDeviceToHost, 0));
  return mapped;
}

inline float *map_image(float3 *cuda_buffer, size_t buffer_size)
{
  // Copy buffer data from device to host
  float *mapped = new float[buffer_size];
  CUDA_CHECK(cudaMemcpyAsync(mapped, cuda_buffer, buffer_size * sizeof(float),
                             cudaMemcpyDeviceToHost, 0));
  return mapped;
}

inline void load_reference_img(LaunchParams &lp, const char *file_name)
{

  sutil::ImageBuffer img = sutil::load_Image(file_name);
  if (img.pixel_format != sutil::FLOAT4)
    std::cerr << "Texture image with unsupported pixel format: " << file_name << std::endl;
  else
  {
    lp.ref_width = img.width;
    lp.ref_height = img.height;

    Utility::flip_image_vertically((float*)img.data, img.width, img.height, 4);

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&lp.ref_img),
        img.width * img.height * sizeof(float4)));
    CUDA_CHECK(cudaMemcpyAsync(lp.ref_img, img.data, img.width * img.height * sizeof(float4), cudaMemcpyHostToDevice, 0));

  }

}

inline void save_Image(Globals &gbx, sutil::CUDAOutputBuffer<uchar4> &output_buffer, std::string outfile)
{
  if (gbx.file_type == PNG)
  {
    outfile += ".png";
    sutil::ImageBuffer buffer;
    buffer.data = output_buffer.getHostPointer();
    buffer.width = output_buffer.width();
    buffer.height = output_buffer.height();
    buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
    sutil::save_Image(outfile.c_str(), buffer, false);
    std::cout << "Rendered image stored in " << outfile << std::endl;
  }
  else if (gbx.file_type == PPM)
  {
    outfile += ".ppm";
    sutil::ImageBuffer buffer;
    buffer.data = output_buffer.getHostPointer();
    buffer.width = output_buffer.width();
    buffer.height = output_buffer.height();
    buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
    sutil::save_Image(outfile.c_str(), buffer, false);
    std::cout << "Rendered image stored in " << outfile << std::endl;
  }
  else if (gbx.file_type == EXR)
  {
    outfile += ".exr";
    int buffer_size = gbx.rd_width * gbx.rd_height * 4;
    float *mapped = map_image(gbx.launch_params.accum_buffer, buffer_size);
    flip_image_vertically(mapped, gbx.rd_width, gbx.rd_height, 4);

    sutil::ImageBuffer buffer;
    buffer.data = mapped;
    buffer.width = gbx.rd_width;
    buffer.height = gbx.rd_height;
    buffer.pixel_format = sutil::BufferImageFormat::FLOAT4;
    sutil::save_Image(outfile.c_str(), buffer, false);

    std::cout << "Rendered image stored in " << outfile << std::endl;
    delete[] mapped;
  }
  else if (gbx.file_type == RAW)
  {
    outfile += ".raw";
    exportRawImage(gbx, outfile);
  }
  else
  {
    std::cout << "Unsupported output file format" << std::endl;
    return;
  }
}

inline void save_Image(Globals &gbx, float3 *ray, std::string outfile)
{
  outfile += ".exr";
  sutil::ImageBuffer buffer;
  buffer.data = ray;
  buffer.width = gbx.rd_width;
  buffer.height = gbx.rd_height;
  buffer.pixel_format = sutil::BufferImageFormat::FLOAT3;
  sutil::save_Image(outfile.c_str(), buffer, false);
  std::cout << "Rendered image stored in " << outfile << std::endl;
}

inline void save_Image(Globals &gbx, float4 *img, std::string outfile)
{
  outfile += ".png";
  sutil::ImageBuffer buffer;
  buffer.data = img;
  buffer.width = gbx.rd_width;
  buffer.height = gbx.rd_height;
  buffer.pixel_format = sutil::BufferImageFormat::FLOAT4;
  sutil::save_Image(outfile.c_str(), buffer, false);
  std::cout << "Rendered image stored in " << outfile << std::endl;
}

inline void save_Image(float *hit, std::string outfile)
{
  outfile += ".png";
  sutil::ImageBuffer buffer;
  buffer.data = hit;
  buffer.width = 1280;
  buffer.height = 720;
  buffer.pixel_format = sutil::BufferImageFormat::FLOAT;
  sutil::save_Image(outfile.c_str(), buffer, false);
  std::cout << "Rendered image stored in " << outfile << std::endl;
}

inline void save_xml(std::string filename, XMLDocument& doc)
{
  // Save the XML document to a file
  if (doc.SaveFile(filename.c_str()) == XML_SUCCESS) {
      std::cout << "Scene exported to " << filename << " successfully." << std::endl;
  }
  else {
      std::cerr << "Error saving XML file." << std::endl;
  }
}

inline void insert_filenames(XMLDocument &doc, XMLElement &root, Globals &gbx)
{
  // Insert Filenames
  XMLElement* files = doc.NewElement("Files");
  files->SetAttribute("type", "obj");

  for(int i = 0; i < gbx.filenames.size(); i++)
  {
    // Add filename attributes inside the Object element
    XMLElement* filename = doc.NewElement("Filename");
    filename->SetAttribute("name", gbx.filenames[i].c_str());
    files->InsertEndChild(filename);
  }
  root.InsertEndChild(files);
}

inline void insert_camera_attributes(XMLDocument &doc, XMLElement &root, Globals &gbx, std::string &export_name)
{
  // Insert Camera Attributes
  XMLElement *camera = doc.NewElement("Camera");

  gbx.save_view(export_name);
  XMLElement *view = doc.NewElement("View");
  view->SetAttribute("filename", export_name.c_str());

  XMLElement *eye = doc.NewElement("Eye");
  eye->SetAttribute("x", gbx.camera->eye().x);
  eye->SetAttribute("y", gbx.camera->eye().y);
  eye->SetAttribute("z", gbx.camera->eye().z);

  XMLElement *lookat = doc.NewElement("LookAt");
  lookat->SetAttribute("x", gbx.camera->lookat().x);
  lookat->SetAttribute("y", gbx.camera->lookat().y);
  lookat->SetAttribute("z", gbx.camera->lookat().z);

  XMLElement *up = doc.NewElement("Up");
  up->SetAttribute("x", gbx.camera->up().x);
  up->SetAttribute("y", gbx.camera->up().y);
  up->SetAttribute("z", gbx.camera->up().z);

  XMLElement* constants = doc.NewElement("Constants");
  constants->SetAttribute("FOVY", gbx.camera->fovY());
  constants->SetAttribute("Aspect_Ratio", gbx.camera->aspectRatio());
  constants->SetAttribute("Cam_Constant", gbx.cam_const);

  XMLElement* frame = doc.NewElement("Frame");
  frame->SetAttribute("width", gbx.rd_width);
  frame->SetAttribute("height", gbx.rd_height);

  camera->InsertEndChild(view);
  camera->InsertEndChild(eye);
  camera->InsertEndChild(lookat);
  camera->InsertEndChild(up);
  camera->InsertEndChild(constants);
  camera->InsertEndChild(frame);
  
  root.InsertEndChild(camera);
}

inline void save_Scene(std::string export_name, Globals &gbx)
{
  std::string export_filename = export_name + ".xml";
  XMLDocument doc;
  // Create the root element for the scene
  XMLElement* root = doc.NewElement("Scene");
  doc.InsertFirstChild(root);

  // Insert Filenames
  insert_filenames(doc, *root, gbx);

  // Insert Environment File
  XMLElement *envfile = doc.NewElement("EnvironmentFile");
  envfile->SetAttribute("name", gbx.env_name.c_str());
  root->InsertEndChild(envfile);

  insert_camera_attributes(doc, *root, gbx, export_name);

  save_xml(export_filename, doc);
}

inline void parse_XML(XMLElement &element, Globals &gbx)
{
  if(std::string(element.Name()) == "Files")
  {
    for (XMLElement *child = element.FirstChildElement(); child != NULL ; child = child->NextSiblingElement())
    {
      parse_XML(*child, gbx);
    }
  }
  else if(std::string(element.Name()) == "Filename")
  {
    gbx.filenames.push_back(element.Attribute("name"));
  }
  else if(std::string(element.Name()) == "EnvironmentFile")
  {
    gbx.env_name = element.Attribute("name");
  }
  else if(std::string(element.Name()) == "Camera")
  {
    XMLElement *child = element.FirstChildElement("View");
    if(child != NULL)
    {
      gbx.trackball = new QuatTrackBall(make_float3(0.0f), 1.0f, 10, 10);
      gbx.load_view(child->Attribute("filename"));
      return;
    }

    child = element.FirstChildElement("Eye");
    float3 eye = make_float3( std::stof(child->Attribute("x")),
                              std::stof(child->Attribute("y")),
                              std::stof(child->Attribute("z")));

    child = element.FirstChildElement("LookAt");
    float3 lookat = make_float3(std::stof(child->Attribute("x")),
                                std::stof(child->Attribute("y")),
                                std::stof(child->Attribute("z")));

    child = element.FirstChildElement("Up");
    float3 up = make_float3(std::stof(child->Attribute("x")),
                            std::stof(child->Attribute("y")),
                            std::stof(child->Attribute("z")));

    child = element.FirstChildElement("Constants");
    gbx.cam_const = std::stof(child->Attribute("Cam_Constant"));
    float fovy = std::stof(child->Attribute("FOVY"));
    float aspect_ratio = std::stof(child->Attribute("Aspect_Ratio"));

    gbx.camera = new sutil::Camera(eye, lookat, up, fovy, aspect_ratio);
  }
  else
  {
    std::cout << "<<<<< Undefined Attribute found while parsing Scene file. >>>>>" << std::endl;
  }

}

inline void parse_scene_file(std::string scene_name, Globals &gbx)
{
  gbx.xml_loading = true;
  std::cout << "Loading Scene <<< " << scene_name << " >>>\n" << std::endl;
  XMLDocument doc;
  if(doc.LoadFile(scene_name.c_str()) != XML_SUCCESS)
  {
    throw std::runtime_error( "Failed to load scene file.\n");
  }

  XMLElement* material_content =  doc.FirstChildElement("Scene");

  for (XMLElement *child = material_content->FirstChildElement(); child != NULL ; child = child->NextSiblingElement())
  {
    parse_XML(*child, gbx);
  }

}

inline void fix_default_args(Globals& gbx)
{
  if (gbx.filenames.size() == 0)
  {  
    gbx.filenames.push_back(std::string(SOURCE_DIR) + "/models/geom/bunny.obj");
  }
  if (!gbx.outfile_selected)
  {
    size_t name_start = gbx.filenames.back().find_last_of("\\/") + 1;
    size_t name_end = gbx.filenames.back().find_last_of('.');
    gbx.outfile = gbx.filenames.back().substr(name_start < name_end ? name_start : 0, name_end - name_start);
  }

}

inline void parse_cmdline(int argc, char *argv[], Globals &gbx)
{
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h")
    {
      printUsageAndExit(argv[0]);
    }
    else if (arg == "-sh" || arg == "--shader")
    {
      if (i == argc - 1)
        printUsageAndExit(argv[0]);
      gbx.shadername = argv[++i];
      lower_case_string(gbx.shadername);
    }
    else if (arg == "-env")
    {
      if (i == argc - 1)
        printUsageAndExit(argv[0]);

      gbx.env_name = argv[++i];
      gbx.env_name = std::string(SOURCE_DIR) + "/models/env/" + gbx.env_name;
      std::string file_extension;
      size_t idx = gbx.env_name.find_last_of('.');
      if (idx < gbx.env_name.length())
      {
        file_extension = gbx.env_name.substr(idx, gbx.env_name.length() - idx);
        lower_case_string(file_extension);
      }
      if (file_extension == ".png" || file_extension == ".ppm" || file_extension == ".hdr")
      {
        lower_case_string(gbx.env_name);
      }
      else
      {
        std::cerr << "Please use environment maps in .png or .ppm  or .hdr format. Received: '" << gbx.env_name << "'" << std::endl;
        printUsageAndExit(argv[0]);
      }
    }
    else if (arg == "-env_phi")
    {
      if (i >= argc - 1)
        printUsageAndExit(argv[0]);
      gbx.launch_params.envmap_phi = static_cast<float>(atof(argv[++i])) * M_PIf / 180.f;
    }
    else if (arg == "-network")
    {
      gbx.networkdir = argv[++i];
    }
    else if (arg == "-bgc")
    {
      if (i >= argc - 3)
        printUsageAndExit(argv[0]);
      gbx.launch_params.miss_color.x = static_cast<float>(atof(argv[++i]));
      gbx.launch_params.miss_color.y = static_cast<float>(atof(argv[++i]));
      gbx.launch_params.miss_color.z = static_cast<float>(atof(argv[++i]));
    }
    else if (arg.substr(0, 6) == "--dim=")
    {
      const std::string dims_arg = arg.substr(6);
      sutil::parseDimensions(dims_arg.c_str(), gbx.rd_width, gbx.rd_height);
    }
    else if (arg == "--no-gl-interop")
    {
      gbx.output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
    }
    else if (arg == "--file" || arg == "-f")
    {
      if (i >= argc - 1)
        printUsageAndExit(argv[0]);
      gbx.outfile = argv[++i];
      gbx.outfile_selected = true;
    }
    else if (arg == "--samples" || arg == "-s")
    {
      if (i >= argc - 1)
        printUsageAndExit(argv[0]);
      gbx.samples = atoi(argv[++i]);
    }
    else if (arg == "--bsdf")
    {
      gbx.bsdf = true;
    }
    else if (arg == "-sky")
    {
      if (i >= argc - 3)
        printUsageAndExit(argv[0]);
      gbx.latitude = static_cast<float>(atof(argv[++i]));
      gbx.turbidity = static_cast<float>(atof(argv[++i]));
      gbx.clouds = static_cast<float>(atof(argv[++i]));
      gbx.use_sunsky = true;
    }
    else if (arg == "-t")
    {
      if (i >= argc - 2)
        printUsageAndExit(argv[0]);
      gbx.ordinal_day = static_cast<float>(atoi(argv[++i]));
      gbx.time_of_day = static_cast<float>(atof(argv[++i]));
      gbx.use_sunsky = true;
    }
    else if (arg == "-r")
    {
      if (i >= argc - 4)
        printUsageAndExit(argv[0]);
      gbx.sky_angle = static_cast<float>(atof(argv[++i]));
      gbx.sky_up.x = static_cast<float>(atof(argv[++i]));
      gbx.sky_up.y = static_cast<float>(atof(argv[++i]));
      gbx.sky_up.z = static_cast<float>(atof(argv[++i]));
      gbx.use_sunsky = true;
    }
    else if (arg == "-sc")
    {
      if (i >= argc - 1)
        printUsageAndExit(argv[0]);
      gbx.scene_scale = static_cast<float>(atof(argv[++i]));
    }
    else if (arg == "-dir")
    {
      if (i >= argc - 2)
        printUsageAndExit(argv[0]);
      gbx.theta_i = static_cast<float>(atof(argv[++i]));
      gbx.phi_i = static_cast<float>(atof(argv[++i]));
      gbx.light_dir = gbx.get_light_direction();
    }
    else if (arg == "-rad")
    {
      if (i >= argc - 3)
        printUsageAndExit(argv[0]);
      gbx.emission.x = static_cast<float>(atof(argv[++i]));
      gbx.emission.y = static_cast<float>(atof(argv[++i]));
      gbx.emission.z = static_cast<float>(atof(argv[++i]));
    }
    else if (arg == "-bf")
    {
      if (i >= argc - 1)
        printUsageAndExit(argv[0]);
      gbx.launch_params.beam_factor = static_cast<float>(atof(argv[++i]));
    }
    else
    {
      gbx.filename = argv[i];
      std::string file_extension;
      size_t idx = gbx.filename.find_last_of('.');
      if (idx < gbx.filename.length())
      {
        file_extension = gbx.filename.substr(idx, gbx.filename.length() - idx);
        lower_case_string(file_extension);
      }
      if (file_extension == ".obj")
      {
        gbx.filename = std::string(SOURCE_DIR) + "/models/geom/" + gbx.filename;
        gbx.filenames.push_back(gbx.filename);
        lower_case_string(gbx.filenames.back());
      }
      else if(file_extension == ".xml")
      {
        parse_scene_file(argv[i], gbx);
      }
      else
      {
        std::cerr << "Unknown option or not an obj file: '" << arg << "'" << std::endl;
        printUsageAndExit(argv[0]);
      }
    }
  }

  fix_default_args(gbx);
}

inline void construct_spline(Globals &gbx, Scene &scene)
{
  tk::spline s_eye_dist(gbx.splinedata.X, gbx.splinedata.eye_dist);
  tk::spline s_qinc_qv_x(gbx.splinedata.X, gbx.splinedata.qinc_qv_x);
  tk::spline s_qinc_qv_y(gbx.splinedata.X, gbx.splinedata.qinc_qv_y);
  tk::spline s_qinc_qv_z(gbx.splinedata.X, gbx.splinedata.qinc_qv_z);
  tk::spline s_qinc_qw(gbx.splinedata.X, gbx.splinedata.qinc_qw);
  tk::spline s_qrot_qv_x(gbx.splinedata.X, gbx.splinedata.qrot_qv_x);
  tk::spline s_qrot_qv_y(gbx.splinedata.X, gbx.splinedata.qrot_qv_y);
  tk::spline s_qrot_qv_z(gbx.splinedata.X, gbx.splinedata.qrot_qv_z);
  tk::spline s_qrot_qw(gbx.splinedata.X, gbx.splinedata.qrot_qw);
  gbx.x = smoothstep(0.0f, 1.0f, gbx.step);
  gbx.x = gbx.x * (gbx.keypoints - 1);
  gbx.step += 1 / (gbx.vid_len * 30); // 30 frames per second

  gbx.trackball->eye_dist = s_eye_dist(gbx.x);
  gbx.trackball->qinc.qv.x = s_qinc_qv_x(gbx.x);
  gbx.trackball->qinc.qv.y = s_qinc_qv_y(gbx.x);
  gbx.trackball->qinc.qv.z = s_qinc_qv_z(gbx.x);
  gbx.trackball->qinc.qw = s_qinc_qw(gbx.x);
  gbx.trackball->qrot.qv.x = s_qrot_qv_x(gbx.x);
  gbx.trackball->qrot.qv.y = s_qrot_qv_y(gbx.x);
  gbx.trackball->qrot.qv.z = s_qrot_qv_z(gbx.x);
  gbx.trackball->qrot.qw = s_qrot_qw(gbx.x);
  // cout << x << " " << step << " "<< 1/(vid_len * 30) << endl;
  if (gbx.video_save)
  {
    std::string Imagename = std::to_string(gbx.frame++) + ".png";
    save_Image(gbx, scene.get_output_buffer(), Imagename);
  }
  gbx.camera_changed = true;
}

inline void setup_docking(Globals &gbx)
{
  ImVec2 workPos = ImGui::GetMainViewport()->GetCenter();      // The coordinates of the top-left corner of the work area
  ImVec2 workSize = ImGui::GetMainViewport()->GetWorkCenter(); // The dimensions (size) of the work area

  ImVec2 workCenter{workPos.x + workSize.x * 0.5f, workPos.y + workSize.y * 0.5f};

  ImGuiID id = ImGui::GetID("MainWindowGroup"); // The string chosen here is arbitrary (it just gives us something to work with)
  ImGui::DockBuilderRemoveNode(id);             // Clear any preexisting layouts associated with the ID we just chose
  ImGui::DockBuilderAddNode(id);                // Create a new dock node to use

  ImVec2 size{(float)gbx.panel_width, (float)gbx.rd_height};

  ImVec2 nodePos{(float)gbx.rd_width, 0.0f};

  // Set the size and position:
  ImGui::DockBuilderSetNodeSize(id, size);
  ImGui::DockBuilderSetNodePos(id, nodePos);

  ImGuiID dock1 = ImGui::DockBuilderSplitNode(id, ImGuiDir_Left, 0.5f, nullptr, &id);

  ImGuiID dock2 = ImGui::DockBuilderSplitNode(id, ImGuiDir_Right, 0.5f, nullptr, &id);

  // Add windows to each docking space:
  ImGui::DockBuilderDockWindow("Settings", dock2);
  ImGui::DockBuilderDockWindow("Video", dock2);
  ImGui::DockBuilderDockWindow("Materials", dock2);
  ImGui::DockBuilderDockWindow("Transform", dock2);

  // Docking configuration Finihed:
  ImGui::DockBuilderFinish(id);
}

inline void load_window_logo(GLFWwindow *window)
{
  GLFWimage images[1];
  std::string path = std::string(SOURCE_DIR) + "/src/assets/logo.png";
  images[0].pixels = sutil::load_Image(path.c_str(), images[0].width, images[0].height, 0, 4); // rgba channels
  glfwSetWindowIcon(window, 1, images);
  free(images[0].pixels);
}

inline void list_devices()
{

  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++)
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
  }
}

inline void change_scene(Globals &gbx, Scene &scene)
{
  if(gbx.gen_type == GEN_DISTANCE)
  {
    gbx.load_view(std::to_string(++gbx.datapoint_num));
    return;
  }

  unsigned int x = scene.get_output_buffer().width();
  unsigned int y = scene.get_output_buffer().height();
  // unsigned int x = 400;
  // unsigned int y = 400;

  // gbx.theta_i = (rand() / static_cast<float>(RAND_MAX)) * 180;
  // gbx.phi_i = (rand() / static_cast<float>(RAND_MAX)) * 360;
  float a = 2 * (rand() / static_cast<float>(RAND_MAX)) -1;
  float b = 2 * (rand() / static_cast<float>(RAND_MAX)) -1;
  float c = 2 * (rand() / static_cast<float>(RAND_MAX)) -1;
  
  gbx.launch_params.light_dir = normalize(make_float3(a,b,c));

  // gbx.trackball->grab_ball(ORBIT_ACTION, make_float2((rand() / float(RAND_MAX)) * x, (rand() / float(RAND_MAX)) * y));
  gbx.trackball->grab_ball(ORBIT_ACTION, make_float2(x/2.0f, y/2.0f));

  gbx.trackball->roll_ball(make_float2((rand() / float(RAND_MAX)) * (x / 2.0f), (rand() / float(RAND_MAX)) * (y / 2.0f)));

  gbx.trackball->release_ball();

  gbx.light_changed = false;
  gbx.camera_changed = true;
}


inline void collect_data(Globals &gbx, Scene &scene)
{
//   if(gen_type == GEN_NERF)
//   {
//     if (launch_params.subframe_index >= 1000)
//     {
//       launch_params.buffer_width = output_buffer.width();
//       launch_params.buffer_height = output_buffer.height();
//       launch_params.update_networkdata(scene.light_dir, camera.eye, camera.m_lookat, camera.m_up);
//       std::string base = "nerf_dataset/";

//       float pr = (float)rand() / RAND_MAX;

//       if (pr < 0.7f)
//       {
//         std::string file = base + "train/" + std::to_string(x) + ".png";
//         std::string sub_dir = "train/" + std::to_string(x++) + ".png";
//         save_Image(launch_params.temp, file);
//         float3 up = scene.camera.up();
//         float3 forward = scene.camera.lookat();
//         float3 right = cross(forward, up);
//         float3 pos = scene.camera.eye();
//         train << "\"" << sub_dir << "\""
//               << " : { \n \"transform_matrix\" : [\n"
//               << "[" << right.x << ",\n"
//               << up.x << ",\n"
//               << forward.x << ",\n"
//               << pos.x << "],\n"
//               << "[" << right.y << ",\n"
//               << up.y << ",\n"
//               << forward.y << ",\n"
//               << pos.y << "],\n"
//               << "[" << right.z << ",\n"
//               << up.z << ",\n"
//               << forward.z << ",\n"
//               << pos.z << "],\n"
//               << "[ 0.0 ,\n 0.0 ,\n 0.0 ,\n 1.0 ]]\n"
//               << "},\n";
//       }
//       else
//       {
//         if (pr < 0.85f)
//         {
//           std::string file = base + "test/" + std::to_string(x) + ".png";
//           std::string sub_dir = "test/" + std::to_string(x++) + ".png";
//           save_Image(launch_params.temp, file);
//           float3 up = scene.camera.up();
//           float3 forward = scene.camera.lookat();
//           float3 right = cross(forward, up);
//           float3 pos = scene.camera.eye();
//           test << "\"" << sub_dir << "\""
//               << " : { \n \"transform_matrix\" : [\n"
//               << "[" << right.x << ",\n"
//               << up.x << ",\n"
//               << forward.x << ",\n"
//               << pos.x << "],\n"
//               << "[" << right.y << ",\n"
//               << up.y << ",\n"
//               << forward.y << ",\n"
//               << pos.y << "],\n"
//               << "[" << right.z << ",\n"
//               << up.z << ",\n"
//               << forward.z << ",\n"
//               << pos.z << "],\n"
//               << "[ 0.0 ,\n 0.0 ,\n 0.0 ,\n 1.0 ]]\n"
//               << "},\n";
//         }
//         else
//         {
//           std::string file = base + "val/" + std::to_string(x) + ".png";
//           std::string sub_dir = "val/" + std::to_string(x++) + ".png";
//           save_Image(launch_params.temp, file);
//           float3 up = scene.camera.up();
//           float3 forward = scene.camera.lookat();
//           float3 right = cross(forward, up);
//           float3 pos = scene.camera.eye();
//           val << "\"" << sub_dir << "\""
//               << " : { \n \"transform_matrix\" : [\n"
//               << "[" << right.x << ",\n"
//               << up.x << ",\n"
//               << forward.x << ",\n"
//               << pos.x << "],\n"
//               << "[" << right.y << ",\n"
//               << up.y << ",\n"
//               << forward.y << ",\n"
//               << pos.y << "],\n"
//               << "[" << right.z << ",\n"
//               << up.z << ",\n"
//               << forward.z << ",\n"
//               << pos.z << "],\n"
//               << "[ 0.0 ,\n 0.0 ,\n 0.0 ,\n 1.0 ]]\n"
//               << "},\n";
//         }
//       }
//       trackball->grab_ball(ORBIT_ACTION, make_float2(launch_params.buffer_width / 2, launch_params.buffer_height / 2));
//       // trackball->grab_ball(ORBIT_ACTION, make_float2(scene.width/2, scene.height/2));
//       trackball->roll_ball(make_float2(launch_params.buffer_width / 2 + 400.0f, launch_params.buffer_height / 2 + 0.0f));
//       // trackball->roll_ball(make_float2( (scene.width/2) + 100, scene.height/2));
//       trackball->release_ball();

//       camera_changed = true;
//     }

//   }

// #ifdef GEN_DATA
//   if (launch_params.subframe_index > 2000)
//   {
//     launch_params.buffer_width = scene.get_output_buffer().width();
//     launch_params.buffer_height = scene.get_output_buffer().height();

//     launch_params.update_networkdata(scene.light_dir, scene.camera.m_eye, scene.camera.m_lookat, scene.camera.m_up);
//     net.update_data(launch_params.result_pos, launch_params.result_ray_vec, launch_params.result_hit_data, launch_params.light_dir, launch_params.result_buffer, launch_params.buffer_width * launch_params.buffer_height);
//     append_data(file, net, theta_i, phi_i);

//     // Scene change params
//     theta_i = (rand() / static_cast<float>(RAND_MAX)) * 180;
//     phi_i = (rand() / static_cast<float>(RAND_MAX)) * 360;

//     trackball->grab_ball(ORBIT_ACTION, make_float2((rand() / float(RAND_MAX)) * 400, (rand() / float(RAND_MAX)) * 400));
//     trackball->roll_ball(make_float2((rand() / float(RAND_MAX)) * 800, (rand() / float(RAND_MAX)) * 800));
//     trackball->release_ball();

//     light_changed = true;
//     camera_changed = true;
//   }
// #endif
  if (gbx.gen_type == GEN_INVERSE)
  {
    if (gbx.launch_params.subframe_index > 100000)
    {
      save_Image(gbx, scene.get_output_buffer(), "candle_00" + std::to_string(gbx.datapoint_num));
      gbx.data_file << "candle_00" + std::to_string(gbx.datapoint_num) + ".exr" << std::endl;
      gbx.data_file << "\tCamera_eye " << scene.camera.eye().x << " " << scene.camera.eye().y 
                    << " " << scene.camera.eye().z << " " << std::endl;
      gbx.data_file << "\tCamera_lookat " << scene.camera.lookat().x << " " << scene.camera.lookat().y 
                    << " " << scene.camera.lookat().z << " " << std::endl;
      gbx.data_file << "\tCamera_up " << scene.camera.up().x << " " << scene.camera.up().y 
                    << " " << scene.camera.up().z << " " << std::endl;
      gbx.data_file << "\tLight_dir " << gbx.launch_params.light_dir.x << " " << gbx.launch_params.light_dir.y
                    << " " << gbx.launch_params.light_dir.z << std::endl << std::endl << std::endl;
      printf("Image saved %d\n", gbx.datapoint_num);
      
      change_scene(gbx, scene);
      gbx.datapoint_num++;
    }
  }

  if (gbx.gen_type == GEN_DISTANCE)
  {
    if (gbx.launch_params.subframe_index > 128)
    {
      // cudaMemcpyAsync(gbx.env, gbx.launch_params.buffer, gbx.rd_width * gbx.rd_height * sizeof(float3), cudaMemcpyDeviceToHost, 0);

      std::string dir = gbx.filenames[0].substr(0, gbx.filenames[0].size()-4);
      if (!fs::is_directory(dir) || !fs::exists(dir))
      {
        fs::create_directories(dir);
      }
      std::string save_filename = dir + "/teapot_00" + std::to_string(gbx.datapoint_num);
      save_Image(gbx, scene.get_output_buffer(), save_filename);
      // gbx.save_view(std::to_string(gbx.datapoint_num));

      // std::string target_dir = SOURCE_DIR;
      // target_dir += "/env_render/" + std::to_string(gbx.env_counter); 
      // save_Image(gbx, gbx.env, target_dir + "/teapot_00_env" + std::to_string(gbx.datapoint_num));

      gbx.data_file << "teapot_00" + std::to_string(gbx.datapoint_num) + ".exr" << std::endl;
      gbx.data_file << "\tCamera_eye " << scene.camera.eye().x << " " << scene.camera.eye().y 
                    << " " << scene.camera.eye().z << " " << std::endl;
      gbx.data_file << "\tCamera_lookat " << scene.camera.lookat().x << " " << scene.camera.lookat().y 
                    << " " << scene.camera.lookat().z << " " << std::endl;
      gbx.data_file << "\tCamera_up " << scene.camera.up().x << " " << scene.camera.up().y 
                    << " " << scene.camera.up().z << " " << std::endl;
      printf("Image saved %d\n", gbx.datapoint_num);
      
      change_scene(gbx, scene);
    }
  }

  if(gbx.gen_type == GEN_ERROR)
  {
    gbx.error_file << gbx.error[0] << std::endl;
  }
}

inline void start_data_collection(Globals &gbx)
{

  if (gbx.gen_type == GEN_NETWORK)
  {
    gbx.data_file.open("data.txt");
    gbx.data_file << "X_x,X_y,X_z,I_theta,I_phi,O_theta,O_phi,OUT_x,OUT_y,OUT_z\n";
  }
  else if(gbx.gen_type == GEN_INVERSE)
  {
    gbx.data_file.open("data.txt");
  }
  else if(gbx.gen_type == GEN_DISTANCE)
  {
    gbx.data_file.open("data.txt");
    gbx.load_view("0");
  }
  else if(gbx.gen_type == GEN_NERF)
  {
    gbx.train.open("nerf_dataset/transforms_train.txt");
    gbx.test.open("nerf_dataset/transforms_test.txt");
    gbx.val.open("nerf_dataset/transforms_val.txt");
    gbx.train << "{\n";
    gbx.test << "{\n";
    gbx.val << "{\n";
  }
  else if(gbx.gen_type == GEN_ERROR)
  {
    gbx.error_file.open("errors_list.txt");
  }

}

inline void end_data_collection(Globals &gbx)
{

  if(gbx.gen_type == GEN_NONE || gbx.datapoint_num < gbx.total_datapoints)
  {
    return;
  }

  if (gbx.gen_type == GEN_NETWORK)
  {
    gbx.data_file.close();
    printf("Network data file closed\n");
  }
  else if(gbx.gen_type == GEN_INVERSE)
  {
    gbx.data_file.close();
    printf("Inverse rendering data file closed\n");
  }
  else if(gbx.gen_type == GEN_DISTANCE)
  {
    gbx.data_file.close();
    printf("Distance rendering data file closed\n");
  }
  else if(gbx.gen_type == GEN_NERF)
  {
    gbx.train << "}";
    gbx.test << "}";
    gbx.val << "}";
    gbx.train.close();
    gbx.test.close();
    gbx.val.close();
    printf("Nerf data file closed\n");
  }
  else if(gbx.gen_type == GEN_ERROR)
  {
    gbx.error_file.close();
    printf("Error file produces");
  }

  if(gbx.datapoint_num >= gbx.total_datapoints)
  {
    if(gbx.env_counter < gbx.env_list.size())
    {
      gbx.env_name = gbx.env_list[gbx.env_counter++];
      gbx.env_dropped = true;
      gbx.datapoint_num = 0;
      return;
    }
    else
    {
      glfwSetWindowShouldClose(gbx.window, true);
      return;
    }
  }

}

inline void updateState(Globals &gbx, Scene &scene)
{
  bool reset = gbx.camera_changed || gbx.sky_changed || gbx.light_changed || gbx.resize_dirty ||
               gbx.settings_changed || gbx.material_changed || gbx.videoset_changed || gbx.obj_dropped ||
               gbx.env_dropped || gbx.instance_transformed || gbx.add_instance || gbx.delete_instance ||
               gbx.delete_all_instance || gbx.scene_dropped || gbx.reset;
  if (gbx.add_instance)
  {
    scene.add_Instance(gbx.mesh_idx);
    scene.update_accel_structure();
    gbx.add_instance = false;
  }
  if (gbx.delete_instance)
  {
    scene.delete_Instance(gbx.mesh_idx);
    gbx.delete_instance = false;
  }
  if (gbx.delete_all_instance)
  {
    scene.delete_all_Instances(gbx.mesh_idx);
    gbx.delete_all_instance = false;
  }
  if (gbx.instance_transformed)
  {
    scene.handleGeometryUpdate();
    gbx.instance_transformed = false;
    gbx.material_changed = true;
  }
  if(gbx.scene_dropped)
  {
    // ###### TODO : Fix error while loading new scene file
    gbx.filenames.clear();
    parse_scene_file(gbx.scene_name, gbx);  
    scene = Scene(&gbx);
    gbx.scene_dropped = false;
  }
  if (gbx.obj_dropped)
  {
    scene.append_files(gbx.dropped_files);
    scene.update_mesh_structures(gbx.update_bbox);
    gbx.dropped_files.clear();
    gbx.obj_dropped = false;
    gbx.camera_changed = true;
  }
  if (gbx.env_dropped)
  {
    scene.update_env(gbx.env_name);
    gbx.env_dropped = false;
    // gbx.first_loop = true;
  }
  if (gbx.resize_dirty)
  {
    gbx.trackball->set_screen_window(gbx.rd_width, gbx.rd_height);
    scene.handleResize(scene.get_output_buffer(), gbx.rd_width, gbx.rd_height);
    gbx.resize_dirty = false;
  }
  if (gbx.camera_changed)
  {
    float3 eye, lookat, up;
    gbx.trackball->get_view_param(eye, lookat, up);
    scene.camera.setEye(eye);
    scene.camera.setLookat(lookat);
    scene.camera.setUp(up);
    scene.camera.setFovY(atanf(gbx.cam_const) * 360.0f * M_1_PIf);
    scene.handleCameraUpdate();
    gbx.camera_changed = false;
  }
  // if(sky_changed)
  // {
  //   scene.handleSunSkyUpdate(time_of_day, clouds);
  //   sky_changed = false;
  // }
  if (gbx.light_changed)
  {
    scene.initLight(gbx.get_light_direction(), gbx.emission, gbx.env_name);
    scene.handleLightUpdate();
    gbx.material_changed = true;
    gbx.light_changed = false;
  }
  if (gbx.settings_changed)
  {
    gbx.settings_changed = false;
  }
  if (gbx.material_changed)
  {
    scene.set_materialdata(gbx.materialname, gbx.mtldata);
    gbx.material_changed = false;
  }
  if (gbx.videoset_changed)
  {
    std::cout << "<< VIDEO RENDER MODE >>" << std::endl;
    // SplineData splinedata;
    gbx.load_view("0");
    run_spline(gbx.keypoints, gbx.splinedata, gbx.cam_const);
    gbx.video = true;
    gbx.videoset_changed = false;
  }
  if(gbx.ref_dropped)
  {
    load_reference_img(gbx.launch_params, gbx.reference_name.c_str());
    gbx.ref_dropped = false;
  }

  // Update params on device
  if (reset)
  {
    gbx.launch_params.subframe_index = 0;
    gbx.reset = false;
    scene.dealloc_output_buffer();
  }
}

inline void print_error(const char* string, const char* include_string)
{
  const char* delimiter = "\n";  // Lines are separated by '\n'
  char* logCopy = new char[strlen(string) + 1];  // Create a writable copy of string
  strcpy(logCopy, string);  // Copy the contents of log to logCopy
  char* line = strtok(logCopy, delimiter);  // Start tokenizing the string

  while (line != nullptr) {
    if (strstr(line, include_string) != nullptr) // Check if "Error" is in the line
    {  
      std::cout << line << std::endl;
    }
    line = strtok(nullptr, delimiter);  // Get the next token (line)
  }
  delete[] logCopy; 
}

inline void list_files_in_directory(const std::string &path, std::vector<std::string> &env_list, std::string &extension) {
    try {
        // Iterate over the directory entries
        for (const auto &entry : fs::directory_iterator(path)) {
            // Check if the entry is a file (not a directory or other type)
            if (fs::is_regular_file(entry.status()) && (entry.path().extension() == extension || extension == "")) {
              env_list.push_back(entry.path().string());
            }
        }
    } catch (const fs::filesystem_error &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

} // namespace utility
