# cmake needs this line
cmake_minimum_required(VERSION 2.8)

project( visualOdometry )

set (CMAKE_CXX_STANDARD 11)

set(OpenCV_DIR /usr/local/share/OpenCV)

find_package( OpenCV 3.4.1 REQUIRED )
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
if(CERES_FOUND)
    message("CERES_FOUND is true")
else(CERES_FOUND)
    message("CERES_FOUND is false")
endif(CERES_FOUND)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${DBOW3_INCLUDE_DIRS}
)

file(GLOB source_files
    "src/*.h"
    "src/*.cpp"
)

set(LIB_SRC_FILES ${source_files})

add_executable( visualOdometry ${LIB_SRC_FILES} )

target_link_libraries( visualOdometry ${OpenCV_LIBS} )
