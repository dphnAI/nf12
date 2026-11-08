# SPDX-License-Identifier: Apache-2.0
# Apple GPU build: nf12.metallib (csrc/metal/*.metal, compiled with xcrun metal),
# libnf12_mps.dylib, which registers torch.ops.nf12 for the MPS device, and
# (when MLX and nanobind are importable) the nf12._mlx_ext module for MLX arrays.
# Both frameworks drive the same kernels through csrc/metal/launch.h.
# Requires Xcode with the Metal Toolchain component
# (xcodebuild -downloadComponent MetalToolchain).

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)  # torch >= 2.13 headers require C++20
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- nf12.metallib ----------------------------------------------------------
find_program(NF12_XCRUN xcrun REQUIRED)
execute_process(COMMAND "${NF12_XCRUN}" -sdk macosx metal --version
  RESULT_VARIABLE _nf12_metal_rc OUTPUT_VARIABLE _nf12_metal_version ERROR_VARIABLE _nf12_metal_err
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _nf12_metal_rc EQUAL 0)
  message(FATAL_ERROR "nf12: the Metal compiler is unavailable (${_nf12_metal_err}). Install full Xcode and run "
    "`xcodebuild -downloadComponent MetalToolchain`.")
endif()
message(STATUS "nf12: ${_nf12_metal_version}")

# Fast math would violate the numerical contract in SPEC.md section 6.
set(NF12_METAL_FLAGS -std=metal3.1 -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET} -fno-fast-math -O3 -Wall -Werror)
set(_nf12_metal_dir "${CMAKE_CURRENT_SOURCE_DIR}/csrc/metal")
file(GLOB NF12_METAL_SOURCES CONFIGURE_DEPENDS "${_nf12_metal_dir}/*.metal")
file(GLOB NF12_METAL_HEADERS CONFIGURE_DEPENDS "${_nf12_metal_dir}/*.h")
set(_nf12_airs "")
foreach(_src IN LISTS NF12_METAL_SOURCES)
  get_filename_component(_name "${_src}" NAME_WE)
  set(_air "${CMAKE_CURRENT_BINARY_DIR}/metal/${_name}.air")
  add_custom_command(OUTPUT "${_air}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/metal"
    COMMAND "${NF12_XCRUN}" -sdk macosx metal ${NF12_METAL_FLAGS} -I "${_nf12_metal_dir}" -c "${_src}" -o "${_air}"
    DEPENDS "${_src}" ${NF12_METAL_HEADERS}
    COMMENT "Compiling Metal ${_name}.metal"
    VERBATIM)
  list(APPEND _nf12_airs "${_air}")
endforeach()
set(NF12_METALLIB "${CMAKE_CURRENT_BINARY_DIR}/nf12.metallib")
add_custom_command(OUTPUT "${NF12_METALLIB}"
  COMMAND "${NF12_XCRUN}" -sdk macosx metallib ${_nf12_airs} -o "${NF12_METALLIB}"
  DEPENDS ${_nf12_airs}
  COMMENT "Linking nf12.metallib"
  VERBATIM)
add_custom_target(nf12_metallib ALL DEPENDS "${NF12_METALLIB}")
install(FILES "${NF12_METALLIB}" DESTINATION nf12)

find_package(Python COMPONENTS Interpreter REQUIRED)

# --- nf12._mlx_ext (MLX) --------------------------------------------------------
# AUTO builds it when the build interpreter can import mlx and nanobind.
set(NF12_BUILD_MLX AUTO CACHE STRING "Build the MLX extension (ON, OFF or AUTO)")
set_property(CACHE NF12_BUILD_MLX PROPERTY STRINGS ON OFF AUTO)
if(NOT NF12_BUILD_MLX STREQUAL "OFF")
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
      "import os, mlx.core, nanobind; print(os.path.dirname(mlx.core.__file__)); print(nanobind.cmake_dir()); print(mlx.core.__version__)"
    OUTPUT_VARIABLE _nf12_mlx_probe
    RESULT_VARIABLE _nf12_mlx_rc
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(_nf12_mlx_rc EQUAL 0)
    string(REPLACE "\n" ";" _nf12_mlx_probe "${_nf12_mlx_probe}")
    list(GET _nf12_mlx_probe 0 NF12_MLX_DIR)
    list(GET _nf12_mlx_probe 1 _nf12_nanobind_dir)
    list(GET _nf12_mlx_probe 2 NF12_MLX_VERSION)
    message(STATUS "nf12: MLX ${NF12_MLX_VERSION} at ${NF12_MLX_DIR}")
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(nanobind CONFIG REQUIRED PATHS "${_nf12_nanobind_dir}" NO_DEFAULT_PATH)
    unset(NF12_MLX_LIB CACHE)
    find_library(NF12_MLX_LIB mlx PATHS "${NF12_MLX_DIR}/lib" NO_DEFAULT_PATH REQUIRED)
    # MLX's nanobind bindings use the full (not stable) ABI in the "mlx" domain;
    # sharing both lets nf12 accept mx.array directly.
    nanobind_add_module(_mlx_ext NB_STATIC NB_DOMAIN mlx NOMINSIZE csrc/mlx/nf12_mlx.cpp)
    add_dependencies(_mlx_ext nf12_metallib)
    target_include_directories(_mlx_ext SYSTEM PRIVATE "${NF12_MLX_DIR}/include" "${NF12_MLX_DIR}/include/metal_cpp")
    target_include_directories(_mlx_ext PRIVATE "${_nf12_metal_dir}")
    # MLX has no stable C++ ABI; nf12.mlx rebuilds when the runtime version differs.
    target_compile_definitions(_mlx_ext PRIVATE _METAL_ ACCELERATE_NEW_LAPACK NF12_MLX_VERSION="${NF12_MLX_VERSION}")
    target_compile_options(_mlx_ext PRIVATE -Wall -Wextra)
    # MLX's exported CMake target hard-codes the frameworks of the SDK it was
    # built with, so link the library and system frameworks directly.
    target_link_libraries(_mlx_ext PRIVATE "${NF12_MLX_LIB}" "-framework Metal" "-framework Foundation")
    set_target_properties(_mlx_ext PROPERTIES
      CXX_STANDARD 20
      BUILD_RPATH "${NF12_MLX_DIR}/lib"
      INSTALL_RPATH "@loader_path/../mlx/lib;${NF12_MLX_DIR}/lib")
    install(TARGETS _mlx_ext LIBRARY DESTINATION nf12)
  elseif(NF12_BUILD_MLX STREQUAL "ON")
    message(FATAL_ERROR "nf12: NF12_BUILD_MLX=ON but ${Python_EXECUTABLE} cannot import mlx and nanobind")
  else()
    message(STATUS "nf12: mlx or nanobind not importable; skipping the MLX extension")
  endif()
endif()

# --- libnf12_mps.dylib --------------------------------------------------------
option(NF12_BUILD_TORCH "Build the torch extension (libnf12_mps.dylib)" ON)
if(NOT NF12_BUILD_TORCH)
  return()
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import os, torch; print(os.path.dirname(torch.__file__))"
  OUTPUT_VARIABLE NF12_TORCH_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
message(STATUS "nf12: torch at ${NF12_TORCH_DIR}")
# Isolated builds can locate torch in a different temporary environment each time.
foreach(_lib torch torch_cpu c10)
  unset(NF12_TORCH_${_lib}_LIB CACHE)
  find_library(NF12_TORCH_${_lib}_LIB ${_lib} PATHS "${NF12_TORCH_DIR}/lib" NO_DEFAULT_PATH REQUIRED)
endforeach()

# MPS stream access (ATen/mps/MPSStream.h) is not part of torch's stable ABI, so
# this library is built against the installed torch (see python/nf12/_jit.py).
add_library(nf12_mps SHARED csrc/metal/ops_mps.mm)
add_dependencies(nf12_mps nf12_metallib)
target_include_directories(nf12_mps SYSTEM PRIVATE
  "${NF12_TORCH_DIR}/include" "${NF12_TORCH_DIR}/include/torch/csrc/api/include")
target_compile_options(nf12_mps PRIVATE -Wall -Wextra -fno-objc-arc)
target_link_libraries(nf12_mps PRIVATE
  "${NF12_TORCH_torch_LIB}" "${NF12_TORCH_torch_cpu_LIB}" "${NF12_TORCH_c10_LIB}"
  "-framework Metal" "-framework Foundation")
set_target_properties(nf12_mps PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  OBJCXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  BUILD_RPATH "${NF12_TORCH_DIR}/lib"
  INSTALL_RPATH "@loader_path/../torch/lib;${NF12_TORCH_DIR}/lib")
install(TARGETS nf12_mps LIBRARY DESTINATION nf12)
