cmake_minimum_required(VERSION 3.16)

if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    if (DEFINED ENV{VCPKG_ROOT} AND EXISTS "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE
            "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
            CACHE FILEPATH "Path to vcpkg toolchain file")
    elseif (WIN32 AND EXISTS "D:/vcpkg/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE
            "D:/vcpkg/scripts/buildsystems/vcpkg.cmake"
            CACHE FILEPATH "Path to vcpkg toolchain file")
    endif ()
endif ()

if (WIN32 AND NOT DEFINED VCPKG_TARGET_TRIPLET)
    set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "vcpkg target triplet")
endif ()

project(CT_FGO_SIM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (MSVC)
    add_compile_options(/permissive- /utf-8 /bigobj)
else ()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(Ceres REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(glog REQUIRED)

add_library(ct_fgo_sim
    src/core/system.cpp
    src/navigation/mechanization.cpp
)

target_include_directories(ct_fgo_sim
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(ct_fgo_sim
    PUBLIC
        Ceres::ceres
        Sophus::Sophus
        yaml-cpp::yaml-cpp
        glog::glog
        Eigen3::Eigen
)

add_executable(ct_fgo_sim_main
    apps/ct_fgo_sim_main.cpp
)

target_link_libraries(ct_fgo_sim_main
    PRIVATE
        ct_fgo_sim
)

add_executable(attitude_priority_main
    apps/attitude_priority_main.cpp
)

target_link_libraries(attitude_priority_main
    PRIVATE
        ct_fgo_sim
)
