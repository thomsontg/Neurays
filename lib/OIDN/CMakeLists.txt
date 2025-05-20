
set(OpenImageDenoise_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/cmake/OpenImageDenoise-2.2.2)

find_package(OpenImageDenoise REQUIRED GLOBAL)

# set(DLLS
#     "${CMAKE_SOURCE_DIR}/lib/OIDN/bin/OpenImageDenoise.dll"
#     "${CMAKE_SOURCE_DIR}/lib/OIDN/bin/OpenImageDenoise_core.dll"
# )

# foreach(DLL ${DLLS})
#     add_custom_command(TARGET ${target_name} POST_BUILD
#         COMMAND ${CMAKE_COMMAND} -E copy_if_different
#             "${DLL}" "${OUTPUT_DIR}"
#     )
# endforeach()
