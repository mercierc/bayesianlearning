# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cmercier/Documents/c++/taton_mlpack

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cmercier/Documents/c++/taton_mlpack

# Include any dependencies generated for this target.
include CMakeFiles/Ex1.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Ex1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Ex1.dir/flags.make

CMakeFiles/Ex1.dir/Test.cpp.o: CMakeFiles/Ex1.dir/flags.make
CMakeFiles/Ex1.dir/Test.cpp.o: Test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cmercier/Documents/c++/taton_mlpack/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Ex1.dir/Test.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Ex1.dir/Test.cpp.o -c /home/cmercier/Documents/c++/taton_mlpack/Test.cpp

CMakeFiles/Ex1.dir/Test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Ex1.dir/Test.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cmercier/Documents/c++/taton_mlpack/Test.cpp > CMakeFiles/Ex1.dir/Test.cpp.i

CMakeFiles/Ex1.dir/Test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Ex1.dir/Test.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cmercier/Documents/c++/taton_mlpack/Test.cpp -o CMakeFiles/Ex1.dir/Test.cpp.s

CMakeFiles/Ex1.dir/Test.cpp.o.requires:

.PHONY : CMakeFiles/Ex1.dir/Test.cpp.o.requires

CMakeFiles/Ex1.dir/Test.cpp.o.provides: CMakeFiles/Ex1.dir/Test.cpp.o.requires
	$(MAKE) -f CMakeFiles/Ex1.dir/build.make CMakeFiles/Ex1.dir/Test.cpp.o.provides.build
.PHONY : CMakeFiles/Ex1.dir/Test.cpp.o.provides

CMakeFiles/Ex1.dir/Test.cpp.o.provides.build: CMakeFiles/Ex1.dir/Test.cpp.o


# Object files for target Ex1
Ex1_OBJECTS = \
"CMakeFiles/Ex1.dir/Test.cpp.o"

# External object files for target Ex1
Ex1_EXTERNAL_OBJECTS =

Ex1: CMakeFiles/Ex1.dir/Test.cpp.o
Ex1: CMakeFiles/Ex1.dir/build.make
Ex1: src/libbayesregression.a
Ex1: /usr/lib/gcc/x86_64-linux-gnu/7/libgomp.so
Ex1: /usr/lib/x86_64-linux-gnu/libpthread.so
Ex1: CMakeFiles/Ex1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cmercier/Documents/c++/taton_mlpack/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Ex1"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Ex1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Ex1.dir/build: Ex1

.PHONY : CMakeFiles/Ex1.dir/build

CMakeFiles/Ex1.dir/requires: CMakeFiles/Ex1.dir/Test.cpp.o.requires

.PHONY : CMakeFiles/Ex1.dir/requires

CMakeFiles/Ex1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Ex1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Ex1.dir/clean

CMakeFiles/Ex1.dir/depend:
	cd /home/cmercier/Documents/c++/taton_mlpack && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cmercier/Documents/c++/taton_mlpack /home/cmercier/Documents/c++/taton_mlpack /home/cmercier/Documents/c++/taton_mlpack /home/cmercier/Documents/c++/taton_mlpack /home/cmercier/Documents/c++/taton_mlpack/CMakeFiles/Ex1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Ex1.dir/depend

