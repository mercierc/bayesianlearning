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
CMAKE_SOURCE_DIR = /home/clementm/Documents/Documents/c++/taton

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/clementm/Documents/Documents/c++/taton

# Include any dependencies generated for this target.
include src/CMakeFiles/bayesreg.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/bayesreg.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/bayesreg.dir/flags.make

src/CMakeFiles/bayesreg.dir/RegressionRidge.cpp.o: src/CMakeFiles/bayesreg.dir/flags.make
src/CMakeFiles/bayesreg.dir/RegressionRidge.cpp.o: src/RegressionRidge.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/clementm/Documents/Documents/c++/taton/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/bayesreg.dir/RegressionRidge.cpp.o"
	cd /home/clementm/Documents/Documents/c++/taton/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/bayesreg.dir/RegressionRidge.cpp.o -c /home/clementm/Documents/Documents/c++/taton/src/RegressionRidge.cpp

src/CMakeFiles/bayesreg.dir/RegressionRidge.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bayesreg.dir/RegressionRidge.cpp.i"
	cd /home/clementm/Documents/Documents/c++/taton/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/clementm/Documents/Documents/c++/taton/src/RegressionRidge.cpp > CMakeFiles/bayesreg.dir/RegressionRidge.cpp.i

src/CMakeFiles/bayesreg.dir/RegressionRidge.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bayesreg.dir/RegressionRidge.cpp.s"
	cd /home/clementm/Documents/Documents/c++/taton/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/clementm/Documents/Documents/c++/taton/src/RegressionRidge.cpp -o CMakeFiles/bayesreg.dir/RegressionRidge.cpp.s

src/CMakeFiles/bayesreg.dir/RegressionRidge.cpp.o.requires:

.PHONY : src/CMakeFiles/bayesreg.dir/RegressionRidge.cpp.o.requires

src/CMakeFiles/bayesreg.dir/RegressionRidge.cpp.o.provides: src/CMakeFiles/bayesreg.dir/RegressionRidge.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/bayesreg.dir/build.make src/CMakeFiles/bayesreg.dir/RegressionRidge.cpp.o.provides.build
.PHONY : src/CMakeFiles/bayesreg.dir/RegressionRidge.cpp.o.provides

src/CMakeFiles/bayesreg.dir/RegressionRidge.cpp.o.provides.build: src/CMakeFiles/bayesreg.dir/RegressionRidge.cpp.o


# Object files for target bayesreg
bayesreg_OBJECTS = \
"CMakeFiles/bayesreg.dir/RegressionRidge.cpp.o"

# External object files for target bayesreg
bayesreg_EXTERNAL_OBJECTS =

src/libbayesreg.a: src/CMakeFiles/bayesreg.dir/RegressionRidge.cpp.o
src/libbayesreg.a: src/CMakeFiles/bayesreg.dir/build.make
src/libbayesreg.a: src/CMakeFiles/bayesreg.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/clementm/Documents/Documents/c++/taton/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libbayesreg.a"
	cd /home/clementm/Documents/Documents/c++/taton/src && $(CMAKE_COMMAND) -P CMakeFiles/bayesreg.dir/cmake_clean_target.cmake
	cd /home/clementm/Documents/Documents/c++/taton/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bayesreg.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/bayesreg.dir/build: src/libbayesreg.a

.PHONY : src/CMakeFiles/bayesreg.dir/build

src/CMakeFiles/bayesreg.dir/requires: src/CMakeFiles/bayesreg.dir/RegressionRidge.cpp.o.requires

.PHONY : src/CMakeFiles/bayesreg.dir/requires

src/CMakeFiles/bayesreg.dir/clean:
	cd /home/clementm/Documents/Documents/c++/taton/src && $(CMAKE_COMMAND) -P CMakeFiles/bayesreg.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/bayesreg.dir/clean

src/CMakeFiles/bayesreg.dir/depend:
	cd /home/clementm/Documents/Documents/c++/taton && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/clementm/Documents/Documents/c++/taton /home/clementm/Documents/Documents/c++/taton/src /home/clementm/Documents/Documents/c++/taton /home/clementm/Documents/Documents/c++/taton/src /home/clementm/Documents/Documents/c++/taton/src/CMakeFiles/bayesreg.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/bayesreg.dir/depend

