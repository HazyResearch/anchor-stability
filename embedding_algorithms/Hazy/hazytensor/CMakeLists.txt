cmake_minimum_required(VERSION 3.2)
project(hazytensor)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake ${CMAKE_MODULE_PATH})

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

if(UNIX AND NOT APPLE)
  find_package(LAPACKE REQUIRED)
  find_package(OpenMP REQUIRED)
elseif(APPLE)
  set(OPEN_BLAS_PATH /usr/local/opt/openblas)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -stdlib=libc++")
  include_directories(${OPEN_BLAS_PATH}/include)
  find_library(OPEN_BLAS_LIBRARY openblas HINTS ${OPEN_BLAS_PATH}/lib)
endif()

include_directories(${BLAS_INCLUDE})
include_directories(${LAPACK_INCLUDE_DIR})
include_directories(src)
include_directories(../third_party/cxxopts/include)

message(STATUS ${PYTHON_INCLUDE_DIRS})

link_directories(${BLAS_LINK_DIR})
link_directories(${LAPACK_LINK_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -DDEBUG=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

add_library(cppembeddings SHARED
    src/datastructures/DenseVector.cc
    src/datastructures/DenseMatrix.cc
    src/datastructures/COO.cc
    src/datastructures/CSR.cc
    src/ops/multiply.cc
    src/utils/utils.cc
    src/utils/thread_pool.cc
    src/utils/parallel.cc
    src/glove/vocab_count.cc
    src/glove/cooccur.cc
    src/glove/shuffle.cc
    src/solver/common.cc
    src/solver/pi_solver.cc
    src/solver/gd_solver.cc
    src/solver/sgd_solver.cc
    src/solver/svrg_solver.cc
    src/solver/hybird_solver.cc
)

target_link_libraries(cppembeddings
  ${LAPACKE_LIBRARIES}
  ${LAPACK_LIBRARIES}
  ${BLAS_LIBRARIES}
  ${OPEN_BLAS_LIBRARY}
)

set_target_properties(cppembeddings PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

if(DEFINED ENV{HAZY_C_PREFIX})
  set(cprefix $ENV{HAZY_C_PREFIX})
elseif(UNIX AND NOT APPLE)
  set(cprefix "/usr/lib")
elseif(APPLE)
  set(cprefix "/usr/local/lib")
endif()

message(STATUS "CMAKE INSTALL PREFIX: ${cprefix}")
install(TARGETS cppembeddings DESTINATION ${cprefix})

add_executable(embedding
  apps/embedding.cc)

target_link_libraries(embedding
  cppembeddings
)

add_executable(sample
  apps/sample.cc)

target_link_libraries(sample
  cppembeddings
)

add_executable(ppmi
  apps/ppmi.cc)

target_link_libraries(ppmi
  cppembeddings
)

add_executable(procrustes
  apps/procrustes.cc)

target_link_libraries(procrustes
  cppembeddings
)

add_executable(vocab_count
  apps/glove_vocab_count.cc)

target_link_libraries(vocab_count
  cppembeddings
)

add_executable(cooccur
  apps/glove_cooccur.cc)

target_link_libraries(cooccur
  cppembeddings
)
