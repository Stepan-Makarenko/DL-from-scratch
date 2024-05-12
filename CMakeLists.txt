cmake_minimum_required(VERSION 3.10)  # Minimum version of CMake required

# Set the project name and version
project(MyProject VERSION 1.0)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable testing with CMake
enable_testing()

# Fetch Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.11.0
)
FetchContent_MakeAvailable(googletest)

# Add your project files
add_executable(train train_linear_layer.cpp)
# target_link_libraries(MyExecutable PRIVATE someLibrary)

# Define a test executable
add_executable(MyTest test3d.cpp)
target_link_libraries(MyTest gtest_main)

# Custom target to run tests automatically
add_custom_target(
    run_tests ALL
    COMMAND MyTest
    DEPENDS MyTest
)

# Optionally discover tests for integration with CMake's testing infrastructure
include(GoogleTest)
gtest_discover_tests(MyTest)
