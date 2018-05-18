cmake_minimum_required(VERSION 2.6)
project ( Tracking )

set(CMAKE_CXX_STANDARD 11)

SET(CMAKE_CXX_FLAGS "-std=c++0x")				# New C11
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)			# Binary file location
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})			# Exectuable file location for main and subprojects
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)		# Library file location for main and subprojects
set(PROJECT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)			# Source folder
set(PROJECT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/src)		# Header folder
file(GLOB SOURCES "src/*.cc")
#file(GLOB PROJECT_INCLUDE_DIR "src/*.h")



#set(CMAKE_BUILD_TYPE debug)					# choose if you want to compile with or without debugging flags,
set(CMAKE_BUILD_TYPE release)					# uncomment only one of the following two commands at a time

#To build the project openCV is required
find_package( OpenCV REQUIRED )

#Include Required Directories
include_directories(${PROJECT_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})


# Create a shared library (To use library creation uncomment the following)
#add_library(
#)



#target_include_directories(FOREST_LIB
#        PUBLIC
#        ${PROJECT_INCLUDE_DIR}
#        )


#Create an Executable for segment
add_executable( tracking ${SOURCES})
target_link_libraries( tracking ${OpenCV_LIBS})			#Link the executable with opencv library


#(If you want to add subprojects to your main the following is how you do)
#Continue with subdirectory
# add_subdirectory(SubFolder/That/Contains/Another/CMakeLists.txt)

