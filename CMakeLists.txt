cmake_minimum_required(VERSION 3.10)

project(visnav)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/" ${CMAKE_MODULE_PATH})

set(EIGEN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/eigen")
set(CERES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/build-ceres-solver")
set(PANGOLIN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/build-Pangolin/src")
set(OPENGV_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/build-opengv")
set(YAML_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/yaml-cpp/")

# Set default build type if not specified otherwise.
# See https://cmake.org/pipermail/cmake/2012-May/050243.html
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
  message(STATUS "Setting build type to '${CMAKE_BUILD_TYPE}' as none was specified.")
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

# Configure CCache if available
if (NOT CMAKE_C_COMPILER_LAUNCHER AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER   ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  else()
    message(STATUS "Dind't find ccache")
  endif()
else()
  message(STATUS "Compiler launcher already set. Not configuring ccache.")
  message(STATUS "CMAKE_C_COMPILER_LAUNCHER: ${CMAKE_C_COMPILER_LAUNCHER}")
  message(STATUS "CMAKE_CXX_COMPILER_LAUNCHER: ${CMAKE_CXX_COMPILER_LAUNCHER}")
endif()

# See comment in ./build_submodules.sh on the relation of arch=native
# and Eigen's alignment. TLDR: Passing arch=native for all build types
# is currently the only viable option to avoid suble bugs with Eigen.
SET(CXX_MARCH native)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# for some reason clang doesn't include them in Wall whereas gcc does, so include them explicitly.
SET(EXTRA_WARNING_FLAGS "-Wsign-compare")

# clang-specific compile flags to avoid warnings from thirdpary includes
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  set(EXTRA_WARNING_FLAGS "${EXTRA_WARNING_FLAGS} -Wno-exceptions")
endif()

set(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN")          # cmake default: "-g"
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DEIGEN_INITIALIZE_MATRICES_BY_NAN")  # cmake default: "-O2 -g -DNDEBUG"
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")                                      # cmake default: "-O3 -DNDEBUG"
SET(CMAKE_CXX_FLAGS " -ftemplate-backtrace-limit=0 -Wall -Wextra ${EXTRA_WARNING_FLAGS} -march=${CXX_MARCH} ${CMAKE_CXX_FLAGS}")


set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "disable yaml tests")
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "disable yaml tools")
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "disable yaml contrib")
add_subdirectory(${YAML_ROOT})

set(EIGEN_INCLUDE_DIR_HINTS ${EIGEN_ROOT})
find_package(Eigen3 3.3.7 EXACT REQUIRED MODULE)
set(EIGEN_INCLUDE_DIR ${EIGEN3_INCLUDE_DIR})  # for ceres / opengv
include_directories(${EIGEN3_INCLUDE_DIR})
message(STATUS "Found Eigen headers in: ${EIGEN3_INCLUDE_DIR}")


include_directories(thirdparty/Sophus)
include_directories(thirdparty/cereal/include)
include_directories(thirdparty/CLI11/include)
include_directories(thirdparty/yaml-cpp/include)
include_directories(include)

find_package(Ceres 1.14 REQUIRED CONFIG HINTS "${CERES_ROOT}" NO_DEFAULT_PATH)
include_directories(${CERES_INCLUDE_DIRS})

set(Pangolin_DIR ${PANGOLIN_ROOT})
find_package(Pangolin REQUIRED CONFIG NO_DEFAULT_PATH)
include_directories(${Pangolin_INCLUDE_DIR})
message(STATUS "Found Pangolin headers in: ${Pangolin_INCLUDE_DIR}")

find_package(opengv REQUIRED CONFIG HINTS ${OPENGV_ROOT} NO_DEFAULT_PATH)
message(STATUS "Found OpenGV headers in: ${opengv_DIR}")

find_package(TBB REQUIRED)
include_directories(${TBB_INCLUDE_DIR})
message(STATUS "Found TBB headers in: ${TBB_INCLUDE_DIR}")


# NOTE: not specifying version, since 2, 3 or 4 is fine
find_package(OpenCV REQUIRED COMPONENTS core features2d)
message(STATUS "Found OpenCV ${OpenCV_VERSION} headers in: ${OpenCV_INCLUDE_DIRS}")
add_library(OpenCV INTERFACE IMPORTED)
set_property(TARGET OpenCV PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${OpenCV_INCLUDE_DIRS})
set_property(TARGET OpenCV PROPERTY INTERFACE_LINK_LIBRARIES ${OpenCV_LIBS})
if(OpenCV_VERSION VERSION_LESS 3)
  set_property(TARGET OpenCV PROPERTY INTERFACE_COMPILE_DEFINITIONS DBATK_OPENCV2)
endif()



add_executable(test_ceres_se3 src/test_ceres_se3.cpp)
target_link_libraries(test_ceres_se3 ceres)

add_executable(calibration src/calibration.cpp)
target_link_libraries(calibration ceres ${Pangolin_LIBRARIES} ${TBB_LIBRARIES})

add_executable(sfm src/sfm.cpp)
target_link_libraries(sfm ceres ${Pangolin_LIBRARIES} ${TBB_LIBRARIES} ${OpenCV_LIBS} opengv)

add_executable(odometry src/odometry.cpp)
target_link_libraries(odometry ceres ${Pangolin_LIBRARIES} ${TBB_LIBRARIES} ${OpenCV_LIBS} opengv)

add_executable(orb_slam src/orb_slam.cpp)
target_link_libraries(orb_slam ceres ${Pangolin_LIBRARIES} ${TBB_LIBRARIES} ${OpenCV_LIBS} opengv yaml-cpp)



enable_testing()
add_subdirectory(test)
