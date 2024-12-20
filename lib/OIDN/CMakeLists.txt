
set(OpenImageDenoise_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/cmake/OpenImageDenoise-2.2.2)

find_package(OpenImageDenoise REQUIRED GLOBAL)

install(IMPORTED_RUNTIME_ARTIFACTS OpenImageDenoise RUNTIME DESTINATION ./$<CONFIG>)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/bin/OpenImageDenoise_core.dll
    DESTINATION ./$<CONFIG>
)
