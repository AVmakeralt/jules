cmake_minimum_required(VERSION 3.20)
project(Jules LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(JULES_VERSION_MAJOR 0)
set(JULES_VERSION_MINOR 1)
set(JULES_VERSION_PATCH 0)
set(JULES_VERSION "${JULES_VERSION_MAJOR}.${JULES_VERSION_MINOR}.${JULES_VERSION_PATCH}")

# ── Find MLIR / LLVM ──────────────────────────────────────────────────────────
# Support both a pre-built LLVM/MLIR install and building from source.
# Users set -DLLVM_DIR and -DMLIR_DIR to their cmake config dirs.
option(JULES_ENABLE_MLIR "Enable MLIR dialect and lowering passes" ON)

if(JULES_ENABLE_MLIR)
  find_package(MLIR REQUIRED CONFIG)
  find_package(LLVM REQUIRED CONFIG)

  message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddMLIR)
  include(AddLLVM)

  # MLIR includes
  include_directories(${LLVM_INCLUDE_DIRS})
  include_directories(${MLIR_INCLUDE_DIRS})
  include_directories(${MLIR_GENERATED_INCLUDE_DIRS})

  separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
  add_definitions(${LLVM_DEFINITIONS_LIST})

  # StableHLO / HLO dialects – try to find them; they may live inside the
  # same MLIR install or in a separate openxla install.
  find_package(StableHLO QUIET CONFIG)
  if(StableHLO_FOUND)
    message(STATUS "Found StableHLO – enabling XLA lowering")
    include_directories(${StableHLO_INCLUDE_DIRS})
  else()
    message(STATUS "StableHLO not found – XLA lowering will be stub-compiled (link-time resolution)")
  endif()
endif()

# ── Project includes ──────────────────────────────────────────────────────────
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)  # for tablegen'd files

# ── Compiler flags ────────────────────────────────────────────────────────────
add_compile_options(-Wall -Wextra -Wno-unused-parameter -fno-exceptions)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_options(-g -O0 -DJULES_DEBUG)
else()
  add_compile_options(-O2)
endif()

# ── Sub-directories ───────────────────────────────────────────────────────────
add_subdirectory(lib)

# ── Tools ─────────────────────────────────────────────────────────────────────
add_subdirectory(tools/julesc)

# ── Testing ───────────────────────────────────────────────────────────────────
option(JULES_BUILD_TESTS "Build test suite" ON)
if(JULES_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

# ── Install rules ─────────────────────────────────────────────────────────────
install(TARGETS julesc DESTINATION bin)
install(DIRECTORY include/jules DESTINATION include)
