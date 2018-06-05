# cmake needs this line
cmake_minimum_required(VERSION 2.8)

project( visualOdometry )

set(OpenCV_DIR /usr/local/share/OpenCV)

find_package( OpenCV 3.4.1 REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

file(GLOB source_files
    "src/*.h"
    "src/*.cpp"
)

set(LIB_SRC_FILES ${source_files})

add_executable( visualOdometry ${LIB_SRC_FILES} )

target_link_libraries( visualOdometry ${OpenCV_LIBS} )
