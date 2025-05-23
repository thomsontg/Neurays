#ifndef MEDIUM_H
#define MEDIUM_H

#include <string>
#include <valarray>
#include <complex>

enum ColorType { mono, rgb, xyz, spectrum };
enum ScatteringUnit { per_meter, per_cm, per_mm };
enum SpectralUnit { kilometers, meters, millimeters, micrometers, nanometers, angstrom, electron_volt };

double unit_to_iso(double value, SpectralUnit u);
double unit_to_iso(double value, ScatteringUnit u);

template<class T> 
class Color : public std::valarray<T>
{
public:
  Color() 
    : unit(nanometers), wavelength(380.0), step_size(5.0) 
  { }
  
  Color(size_t n) 
    : std::valarray<T>(n), unit(nanometers), wavelength(380.0), step_size(5.0) 
  { }

  Color(size_t n, SpectralUnit u) 
    : std::valarray<T>(n), unit(u), wavelength(380.0), step_size(5.0) 
  { }

  Color(const T* v, unsigned int n) 
    : std::valarray<T>(v, n), unit(nanometers), wavelength(380.0), step_size(5.0) 
  { }

  Color(const T* v, unsigned int n, SpectralUnit u) 
    : std::valarray<T>(v, n), unit(u), wavelength(380.0), step_size(5.0) 
  { }

  Color& operator=(const Color& c)
  {
    if(this->size() != c.size())
      this->resize(c.size());
    *dynamic_cast<std::valarray<T>*>(this) = std::valarray<T>(c);
    unit = c.unit;
    wavelength = c.wavelength;
    step_size = c.step_size;
    return *this;
  }

  SpectralUnit unit;
  double wavelength;
  double step_size;
  T get_nearest(double wavelength) const;
  T get_linear(double wavelength) const;
};

template<class T> 
T Color<T>::get_nearest(double lambda) const
{
  if(this->size() == 0)
    return T(0);

  int idx = static_cast<int>((lambda - wavelength)/step_size + 0.5);
  if(idx < 0)
    return (*this)[0];
  if(idx >= static_cast<int>(this->size()))
    return (*this)[this->size() - 1];

  return (*this)[idx];
}

template<class T> 
T Color<T>::get_linear(double lambda) const
{
  if(this->size() == 0)
    return T(0);

  int idx = static_cast<int>((lambda - wavelength)/step_size);
  if(idx < 0)
    return (*this)[0];
  if(idx >= static_cast<int>(this->size()) - 1)
    return (*this)[this->size() - 1];

  double v = lambda - (wavelength + step_size*idx);
  return static_cast<T>((*this)[idx]*(1.0 - v) + (*this)[idx]*v);
}

class Medium
{
public:
  Medium() : turbid(false), emissive(false), scatter_unit(per_meter) { }

  std::string name;
  bool turbid;
  bool emissive;
  ScatteringUnit scatter_unit;
  __declspec(dllexport) Color< std::complex<double> >& get_ior(ColorType type) { return ior[type]; }
  __declspec(dllexport) Color<double>& get_emission(ColorType type) { return emission[type]; }
  __declspec(dllexport) Color<double>& get_extinction(ColorType type) { return extinction[type]; }
  __declspec(dllexport) Color<double>& get_scattering(ColorType type) { return scattering[type]; }
  __declspec(dllexport) Color<double>& get_absorption(ColorType type) { return absorption[type]; }
  __declspec(dllexport) Color<double>& get_asymmetry(ColorType type) { return asymmetry[type]; }
  __declspec(dllexport) Color<double>& get_albedo(ColorType type) { return albedo[type]; }
  __declspec(dllexport) Color<double>& get_reduced_sca(ColorType type) { return reduced_sca[type]; }
  __declspec(dllexport) Color<double>& get_reduced_ext(ColorType type) { return reduced_ext[type]; }
  __declspec(dllexport) Color<double>& get_reduced_alb(ColorType type) { return reduced_alb[type]; }

  __declspec(dllexport) const Color< std::complex<double> >& get_ior(ColorType type) const { return ior[type]; }
  __declspec(dllexport) const Color<double>& get_emission(ColorType type) const { return emission[type]; }
  __declspec(dllexport) const Color<double>& get_extinction(ColorType type) const { return extinction[type]; }
  __declspec(dllexport) const Color<double>& get_scattering(ColorType type) const { return scattering[type]; }
  __declspec(dllexport) const Color<double>& get_absorption(ColorType type) const { return absorption[type]; }
  __declspec(dllexport) const Color<double>& get_asymmetry(ColorType type) const { return asymmetry[type]; }
  __declspec(dllexport) const Color<double>& get_albedo(ColorType type) const { return albedo[type]; }
  __declspec(dllexport) const Color<double>& get_reduced_sca(ColorType type) const { return reduced_sca[type]; }
  __declspec(dllexport) const Color<double>& get_reduced_ext(ColorType type) const { return reduced_ext[type]; }
  __declspec(dllexport) const Color<double>& get_reduced_alb(ColorType type) const { return reduced_alb[type]; }

  __declspec(dllexport) void fill_data(ColorType type);
  __declspec(dllexport) void fill_mono_data();
  __declspec(dllexport) void fill_rgb_data(bool verbose = false);
  __declspec(dllexport) void fill_xyz_data();
  __declspec(dllexport) void fill_spectral_data();

  void wavelength_to_rgb(double wavelength);

private:
  Color< std::complex<double> > ior[4];
  Color<double> emission[4];
  Color<double> extinction[4];    
  Color<double> scattering[4];
  Color<double> absorption[4];
  Color<double> asymmetry[4];
  Color<double> albedo[4];
  Color<double> reduced_sca[4];
  Color<double> reduced_ext[4];    
  Color<double> reduced_alb[4];
};

#endif // MEDIUM_H
