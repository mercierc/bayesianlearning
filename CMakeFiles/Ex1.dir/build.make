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
CMAKE_SOURCE_DIR = /home/clementm/Documents/Documents/c++/bayesianlearning

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/clementm/Documents/Documents/c++/bayesianlearning

# Include any dependencies generated for this target.
include CMakeFiles/Ex1.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Ex1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Ex1.dir/flags.make

CMakeFiles/Ex1.dir/src/ard_regression_test.cpp.o: CMakeFiles/Ex1.dir/flags.make
CMakeFiles/Ex1.dir/src/ard_regression_test.cpp.o: src/ard_regression_test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/clementm/Documents/Documents/c++/bayesianlearning/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Ex1.dir/src/ard_regression_test.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Ex1.dir/src/ard_regression_test.cpp.o -c /home/clementm/Documents/Documents/c++/bayesianlearning/src/ard_regression_test.cpp

CMakeFiles/Ex1.dir/src/ard_regression_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Ex1.dir/src/ard_regression_test.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/clementm/Documents/Documents/c++/bayesianlearning/src/ard_regression_test.cpp > CMakeFiles/Ex1.dir/src/ard_regression_test.cpp.i

CMakeFiles/Ex1.dir/src/ard_regression_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Ex1.dir/src/ard_regression_test.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/clementm/Documents/Documents/c++/bayesianlearning/src/ard_regression_test.cpp -o CMakeFiles/Ex1.dir/src/ard_regression_test.cpp.s

CMakeFiles/Ex1.dir/src/ard_regression_test.cpp.o.requires:

.PHONY : CMakeFiles/Ex1.dir/src/ard_regression_test.cpp.o.requires

CMakeFiles/Ex1.dir/src/ard_regression_test.cpp.o.provides: CMakeFiles/Ex1.dir/src/ard_regression_test.cpp.o.requires
	$(MAKE) -f CMakeFiles/Ex1.dir/build.make CMakeFiles/Ex1.dir/src/ard_regression_test.cpp.o.provides.build
.PHONY : CMakeFiles/Ex1.dir/src/ard_regression_test.cpp.o.provides

CMakeFiles/Ex1.dir/src/ard_regression_test.cpp.o.provides.build: CMakeFiles/Ex1.dir/src/ard_regression_test.cpp.o


# Object files for target Ex1
Ex1_OBJECTS = \
"CMakeFiles/Ex1.dir/src/ard_regression_test.cpp.o"

# External object files for target Ex1
Ex1_EXTERNAL_OBJECTS =

bin/Ex1: CMakeFiles/Ex1.dir/src/ard_regression_test.cpp.o
bin/Ex1: CMakeFiles/Ex1.dir/build.make
bin/Ex1: src/libbayesianlearning.a
bin/Ex1: /usr/lib/gcc/x86_64-linux-gnu/7/libgomp.so
bin/Ex1: /usr/lib/x86_64-linux-gnu/libpthread.so
bin/Ex1: CMakeFiles/Ex1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/clementm/Documents/Documents/c++/bayesianlearning/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bin/Ex1"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Ex1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Ex1.dir/build: bin/Ex1

.PHONY : CMakeFiles/Ex1.dir/build

CMakeFiles/Ex1.dir/requires: CMakeFiles/Ex1.dir/src/ard_regression_test.cpp.o.requires

.PHONY : CMakeFiles/Ex1.dir/requires

CMakeFiles/Ex1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Ex1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Ex1.dir/clean

CMakeFiles/Ex1.dir/depend:
	cd /home/clementm/Documents/Documents/c++/bayesianlearning && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/clementm/Documents/Documents/c++/bayesianlearning /home/clementm/Documents/Documents/c++/bayesianlearning /home/clementm/Documents/Documents/c++/bayesianlearning /home/clementm/Documents/Documents/c++/bayesianlearning /home/clementm/Documents/Documents/c++/bayesianlearning/CMakeFiles/Ex1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Ex1.dir/depend

