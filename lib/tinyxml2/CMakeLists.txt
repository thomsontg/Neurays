add_library(tinyxml2 STATIC
  tinyxml2.h
  tinyxml2.cpp
)

target_include_directories(tinyxml2 INTERFACE .. PRIVATE .)
