cmake_minimum_required(VERSION 3.14)
project(main)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

# Check C++11 or C++0x support
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
list(APPEND CMAKE_PREFIX_PATH "/usr/local")

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(OpenCV 4.4 REQUIRED)
find_package(Torch REQUIRED)

include_directories(
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}/dbscan/nanoflann
)

add_library(dbscan
dbscan/dbscan.hpp
dbscan/dbscan.cpp
)

add_library(src
src/utils.h
src/utils.cc
src/cluster.h
src/cluster.cc
)

target_include_directories(src PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(main main.cpp)
add_executable(viz_cluster_crop viz_cluster_crop.cpp)
add_executable(viz_cluster_2d viz_cluster_2d.cpp)
add_executable(to_point_cloud to_point_cloud.cpp)
add_executable(to_fp_cloud to_fp_cloud.cpp)
add_executable(classify_clusters classify_clusters.cpp)

target_link_libraries(src
${Pangolin_LIBRARIES}
${OpenCV_LIBS}
${TORCH_LIBRARIES}
dbscan
)


target_link_libraries(main PRIVATE src)
target_link_libraries(viz_cluster_crop PRIVATE src)
target_link_libraries(to_point_cloud PRIVATE src)
target_link_libraries(to_fp_cloud PRIVATE src)
target_link_libraries(viz_cluster_2d PRIVATE src)
target_link_libraries(classify_clusters PRIVATE src)
