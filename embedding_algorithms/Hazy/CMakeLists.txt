cmake_minimum_required(VERSION 3.2)
project(hazytensor)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

SET(BASE_DIR "hazytensor")

set(HAZY_PYTHON_BUILD "${HAZY_PYTHON_BUILD}")
if(HAZY_PYTHON_BUILD)
  set(CMAKE_SKIP_BUILD_RPATH FALSE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE) 
else()
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif()

include_directories(hazytensor/src)

add_subdirectory(hazytensor)

###### FOR TESTING ######
set(TEST_DIR "tests")
set(TESTS ${SOURCES}
    "${TEST_DIR}/test_main.cc"
    "${TEST_DIR}/test_coo_shuffle_inplace.cc")

# Generate a test executable
include_directories(third_party/Catch2)
add_executable("${PROJECT_NAME}_test" ${TESTS})

target_link_libraries("${PROJECT_NAME}_test" PRIVATE
    cppembeddings
)
set_target_properties("${PROJECT_NAME}_test" PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

#find_package(PythonLibs REQUIRED)
#include_directories(${PYTHON_INCLUDE_DIRS})

###### COMPILES PYTHON PACKAGE ######
add_subdirectory(third_party/pybind11)

pybind11_add_module(hazytensor ${SOURCES} ${BASE_DIR}/bindings.cc)
target_link_libraries(hazytensor PRIVATE
  cppembeddings
)

set_target_properties(hazytensor PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
