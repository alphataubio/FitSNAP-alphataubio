cmake_minimum_required(VERSION 3.15)

# ============================================================
#   Enforce Debug builds and full DWARF on macOS and Linux
# ============================================================

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_MACOSX_RPATH ON)

set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -fno-omit-frame-pointer")
set(CMAKE_C_FLAGS_DEBUG   "-g3 -O0 -fno-omit-frame-pointer")

# Disable stripping globally
set(CMAKE_STRIP ":")

add_compile_options("$<$<CONFIG:Debug>:-g3>")
add_compile_options("$<$<CONFIG:Debug>:-fno-omit-frame-pointer>")

# ============================================================
# Continue with project() and targets
# ============================================================

project(lammps_pyace_python LANGUAGES CXX)

# FIX: Force FetchContent to download to local build/_deps instead of /tmp
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build/_deps")

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# -------------------------------- lammps-user-pace --------------------------------

include(FetchContent)
FetchContent_Declare(
  lammps-user-pace
  GIT_REPOSITORY https://github.com/alphataubio/lammps-pyace.git
  GIT_TAG        alphataubio-fitsnap
)
FetchContent_MakeAvailable(lammps-user-pace)

set(PACE_ROOT ${lammps-user-pace_SOURCE_DIR})
message(STATUS "PACE_ROOT is set to: ${PACE_ROOT}")

# -------------------------------- yaml-cpp --------------------------------

# 1. Check if lammps-user-pace already defined the target
if(TARGET yaml-cpp-pace)
    message(STATUS "Target yaml-cpp-pace already defined by lammps-user-pace")
    set(YAML_CPP_TARGET yaml-cpp-pace)
    set(BUILD_YAML_CPP TRUE)
    
# 2. If not, check if we should build it ourselves
else()
    set(yaml-cpp_FOUND FALSE)
    
    if(NOT yaml-cpp_FOUND)
        # enforce building libyaml-cpp as static library and turn off optional features
        set(YAML_BUILD_SHARED_LIBS OFF CACHE INTERNAL "")
        set(YAML_CPP_BUILD_CONTRIB OFF CACHE INTERNAL "")
        set(YAML_CPP_BUILD_TOOLS OFF CACHE INTERNAL "")
        
        add_subdirectory(${PACE_ROOT}/yaml-cpp build-yaml-cpp)
        
        # yaml-cpp CMakeLists.txt creates target yaml-cpp-pace
        set(YAML_CPP_TARGET yaml-cpp-pace)

        install(DIRECTORY ${PACE_ROOT}/yaml-cpp/include/yaml-cpp
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
          FILES_MATCHING
          PATTERN *.h
        )
        set(BUILD_YAML_CPP TRUE)
    else()
        find_package(yaml-cpp REQUIRED)
        # System-installed yaml-cpp provides yaml-cpp::yaml-cpp target
        set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
        set(BUILD_YAML_CPP FALSE)
    endif()
endif()

# -------------------------------- cnpy --------------------------------

# 1. Check if lammps-user-pace already defined the target
if(TARGET cnpy-static)
    message(STATUS "Target cnpy-static already defined by lammps-user-pace")
    
    # Ensure the alias exists so we can link against cnpy::cnpy later
    if(NOT TARGET cnpy::cnpy)
        add_library(cnpy::cnpy ALIAS cnpy-static)
    endif()
    
    set(BUILD_CNPY TRUE)

# 2. If not, check if we should build it ourselves
else()
    find_package(cnpy QUIET)

    if(NOT cnpy_FOUND)
        add_library(cnpy-static STATIC ${PACE_ROOT}/cnpy/cnpy.cpp)
        target_include_directories(cnpy-static PUBLIC
            $<BUILD_INTERFACE:${PACE_ROOT}>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/pace>
        )
        set_target_properties(cnpy-static PROPERTIES LINKER_LANGUAGE CXX OUTPUT_NAME cnpy)
        add_library(cnpy::cnpy ALIAS cnpy-static)

        install(DIRECTORY ${PACE_ROOT}/cnpy
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pace
          FILES_MATCHING
          PATTERN *.h
        )
        set(BUILD_CNPY TRUE)
    else()
        find_package(cnpy REQUIRED)
        set(BUILD_CNPY FALSE)
    endif()
endif()

# -------------------------------- wigner-cpp --------------------------------
# this is header-only library

set(WIGNER_PATH ${PACE_ROOT}/wigner-cpp)
set(WIGNER_INCLUDE_PATH ${WIGNER_PATH}/include)

install(DIRECTORY ${WIGNER_INCLUDE_PATH}/wigner
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pace
  FILES_MATCHING
  PATTERN *.hpp
)

# -------------------------------- pybind --------------------------------

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.13.6
)
FetchContent_MakeAvailable(pybind11)

# -------------------------------- basis --------------------------------

message(STATUS "CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

set(ML_PACE "${PACE_ROOT}/ML-PACE")
set(ACE ${ML_PACE}/ace)
set(ACE_EVALUATOR ${ML_PACE}/ace-evaluator)
set(HELPERS "${CMAKE_CURRENT_SOURCE_DIR}/helpers")

set(SOURCES_BASIS "${ACE_EVALUATOR}/ace_c_basis.cpp"
        "${ACE_EVALUATOR}/ace_radial.cpp"
        "${ACE_EVALUATOR}/ace_spherical_cart.cpp"
        "${ACE_EVALUATOR}/ace_abstract_basis.cpp"
        "${ACE_EVALUATOR}/ace_flatten_basis.cpp"
        "${ACE_EVALUATOR}/ships_radial.cpp"
        "${ACE}/ace_b_basis.cpp"
        "${ACE}/ace_b_basisfunction.cpp"
        "${ACE}/ace_clebsch_gordan.cpp"
        "${ACE}/ace_couplings.cpp"
        "${ACE}/ace_yaml_input.cpp"
        "${HELPERS}/ace_c_basis_binding.cpp"
        "${HELPERS}/ace_bbasis_spec_helper.cpp"
        "${HELPERS}/ace_radial_helper.cpp"
        "${HELPERS}/ace_c_basisfunction_helper.cpp"
        "${HELPERS}/ace_c_basis_helper.cpp"
  )
  
pybind11_add_module(basis ${SOURCES_BASIS} )
target_include_directories(basis PRIVATE 
    ${WIGNER_INCLUDE_PATH}
    ${ML_PACE}
    ${HELPERS}
)
message(STATUS "YAML_CPP_TARGET = ${YAML_CPP_TARGET}")
message(STATUS "BUILD_YAML_CPP = ${BUILD_YAML_CPP}")
target_link_libraries(basis PRIVATE ${YAML_CPP_TARGET} cnpy::cnpy)
target_compile_options(basis PRIVATE -Wno-deprecated-declarations)
set(deps_targets "")

set_target_properties(basis PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

if(BUILD_YAML_CPP)
    list(APPEND deps_targets yaml-cpp-pace)
else()
    target_compile_definitions(basis PUBLIC YAML_PACE=YAML)
endif()

if(BUILD_CNPY)
    list(APPEND deps_targets cnpy-static)
else()
    # We only write the header redirection if we didn't build cnpy ourselves
    # AND if it wasn't already available in ML-PACE/cnpy
    if(NOT EXISTS ${ML_PACE}/cnpy/cnpy.h)
        file(MAKE_DIRECTORY ${ML_PACE}/cnpy)
        file(WRITE ${ML_PACE}/cnpy/cnpy.h "#include <cnpy.h>")
    endif()
endif()


# Honor the directory provided by setup.py
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/pyace")
endif()
