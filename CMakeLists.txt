cmake_minimum_required(VERSION 2.8)
project(viz_cluster)

add_executable(viz_cluster main.cpp)
add_executable(viz_cluster_2d viz_cluster_2d.cpp)
add_executable(to_point_cloud to_point_cloud.cpp)
add_executable(to_fp_cloud to_fp_cloud.cpp)

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

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(OpenCV 4.4 REQUIRED)

include_directories(
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}/dbscan/nanoflann
)

add_library(dbscan
dbscan/dbscan.hpp
dbscan/dbscan.cpp
)


target_link_libraries(${PROJECT_NAME}
${Pangolin_LIBRARIES}
${OpenCV_LIBS}
dbscan
)

target_link_libraries(to_point_cloud
${Pangolin_LIBRARIES}
${OpenCV_LIBS}
dbscan
)

target_link_libraries(to_fp_cloud
${Pangolin_LIBRARIES}
${OpenCV_LIBS}
dbscan
)

target_link_libraries(viz_cluster_2d
${Pangolin_LIBRARIES}
${OpenCV_LIBS}
dbscan
)