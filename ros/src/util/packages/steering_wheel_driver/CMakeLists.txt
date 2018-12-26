cmake_minimum_required(VERSION 2.8.3)
project(steering_wheel_driver)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  autoware_msgs
  roscpp
  pluginlib
)

#SDL2
set(SDL2_SEARCH_PATHS
	~/Library/Frameworks
	/Library/Frameworks
	/usr/local
	/usr
	/sw # Fink
	/opt/local # DarwinPorts
	/opt/csw # Blastwave
	/opt
)

find_path(SDL2_INCLUDE_DIR SDL.h
	HINTS
	$ENV{SDL2DIR}
	PATH_SUFFIXES include/SDL2 include
	PATHS ${SDL2_SEARCH_PATHS}
)

find_library(SDL2_LIBRARY_TEMP
	NAMES SDL2
	HINTS
	$ENV{SDL2DIR}
	PATH_SUFFIXES lib64 lib
	PATHS ${SDL2_SEARCH_PATHS}
)

if(NOT SDL2_BUILDING_LIBRARY)
	if(NOT ${SDL2_INCLUDE_DIR} MATCHES ".framework")
		# Non-OS X framework versions expect you to also dynamically link to
		# SDL2main. This is mainly for Windows and OS X. Other (Unix) platforms
		# seem to provide SDL2main for compatibility even though they don't
		# necessarily need it.
		find_library(SDL2MAIN_LIBRARY
			NAMES SDL2main
			HINTS
			$ENV{SDL2DIR}
			PATH_SUFFIXES lib64 lib
			PATHS ${SDL2_SEARCH_PATHS}
		)
	endif(NOT ${SDL2_INCLUDE_DIR} MATCHES ".framework")
endif(NOT SDL2_BUILDING_LIBRARY)
FIND_PACKAGE(Threads)
if(SDL2_LIBRARY_TEMP)
	# For SDL2main
	if(NOT SDL2_BUILDING_LIBRARY)
		if(SDL2MAIN_LIBRARY)
			set(SDL2_LIBRARY_TEMP ${SDL2MAIN_LIBRARY} ${SDL2_LIBRARY_TEMP})
		endif(SDL2MAIN_LIBRARY)
	endif(NOT SDL2_BUILDING_LIBRARY)
	set(SDL2_LIBRARY_TEMP ${SDL2_LIBRARY_TEMP} ${CMAKE_THREAD_LIBS_INIT})
	# set the final string here so the GUI reflects the final state.
	set(SDL2_LIBRARY ${SDL2_LIBRARY_TEMP} CACHE STRING "Where the SDL2 Library can be found")
	# set the temp variable to INTERNAL so it is NOT seen in the CMake GUI
	set(SDL2_LIBRARY_TEMP "${SDL2_LIBRARY_TEMP}" CACHE INTERNAL "")
endif(SDL2_LIBRARY_TEMP)

include(FindPackageHandleStandardArgs)

find_package(SDL2 REQUIRED)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(SDL2 REQUIRED_VARS SDL2_LIBRARY SDL2_INCLUDE_DIR)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS autoware_msgs roscpp
  DEPENDS SDL2
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${SDL2_INCLUDE_DIR}
)

add_executable(g29_driver_node src/g29_driver_node.cpp src/g29_driver.cpp)
target_link_libraries(g29_driver_node ${catkin_LIBRARIES} ${SDL2_LIBRARY})

# CPP Execution programs
set(CPP_EXEC_NAMES g29_driver_node)
foreach(cpp_exec_names ${CPP_EXEC_NAMES})
  install(TARGETS ${cpp_exec_names}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )
endforeach(cpp_exec_names)

# include header files
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

foreach(dir launch)
  install(DIRECTORY ${dir}/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)