# CMakeLists.txt - Jules Library Build

# Collect all source files for the Jules library.
file(GLOB_RECURSE JULES_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

# Filter out the tools directory.
list(FILTER JULES_SOURCES EXCLUDE REGEX "tools/")

# Define the Jules library.
add_library(julesLib STATIC ${JULES_SOURCES})

# Threading support for JIT PGO background recompilation.
find_package(Threads REQUIRED)
target_link_libraries(julesLib PUBLIC Threads::Threads)

# BLAS / CBLAS for cblas_sgemm (used by FusedKernels.h and PJRT.cpp)
find_library(BLAS_LIB NAMES blas PATHS /home/z/my-project/mlir-prefix/usr/lib/x86_64-linux-gnu/blas /usr/lib/x86_64-linux-gnu/blas)
if(BLAS_LIB)
  target_link_libraries(julesLib PUBLIC ${BLAS_LIB})
else()
  message(WARNING "BLAS not found — cblas_sgemm symbols will be unresolved")
endif()

target_include_directories(julesLib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_BINARY_DIR}/../include
  ${CMAKE_CURRENT_BINARY_DIR}
)

# Link against MLIR/LLVM if enabled.
if(JULES_ENABLE_MLIR)
  target_link_libraries(julesLib PUBLIC
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRFuncDialect
    MLIRFuncTransforms
    MLIRSupport
    MLIRInferTypeOpInterface
    MLIRAffineDialect
    MLIRAffineTransforms
    MLIRMemRefDialect
    MLIRVectorDialect
    MLIRSCFDialect
    MLIRArithDialect
    MLIRBufferizationDialect
    MLIRMathDialect
    MLIROpenMPDialect
    MLIROpenMPToLLVM
    # LLVM Backend: conversion + JIT libraries
    MLIRArithToLLVM
    MLIRFuncToLLVM
    MLIRMemRefToLLVM
    MLIRSCFToControlFlow
    MLIRControlFlowToLLVM
    MLIRMathToLLVM
    MLIRIndexToLLVM
    MLIRVectorToLLVM
    MLIRLLVMCommonConversion
    MLIRTargetLLVMIRExport
    MLIRBuiltinToLLVMIRTranslation
    LLVMOrcJIT
    LLVMCore
    LLVMSupport
  )

  if(StableHLO_FOUND)
    target_link_libraries(julesLib PUBLIC StableHLODialect)
  endif()
endif()

# TableGen: generate op classes from the .td file.
if(JULES_ENABLE_MLIR)
  set(LLVM_TARGET_DEFINITIONS
    "${CMAKE_CURRENT_SOURCE_DIR}/../include/jules/Dialect/JulesOps.td"
  )

  mlir_tablegen(JulesOps.h.inc -gen-op-decls)
  mlir_tablegen(JulesOps.cpp.inc -gen-op-defs)
  mlir_tablegen(JulesOpsDialect.h.inc -gen-dialect-decls -dialect=jules)
  mlir_tablegen(JulesOpsTypes.h.inc -gen-typedef-decls)
  mlir_tablegen(JulesOpsTypes.cpp.inc -gen-typedef-defs)
  mlir_tablegen(JulesOpsDialect.cpp.inc -gen-dialect-defs -dialect=jules)

  add_public_tablegen_target(JulesOpsIncGen)
  add_dependencies(julesLib JulesOpsIncGen)

  # Copy generated .inc files to the include directory so they can be found
  # via the "jules/Dialect/*.inc" include path.
  set(GEN_INC_DIR ${CMAKE_CURRENT_BINARY_DIR}/../include/jules/Dialect)
  file(MAKE_DIRECTORY ${GEN_INC_DIR})
  foreach(INC_FILE JulesOps.h.inc JulesOps.cpp.inc JulesOpsDialect.h.inc JulesOpsDialect.cpp.inc JulesOpsTypes.h.inc JulesOpsTypes.cpp.inc)
    add_custom_command(TARGET JulesOpsIncGen POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_BINARY_DIR}/${INC_FILE}
        ${GEN_INC_DIR}/${INC_FILE}
    )
  endforeach()
endif()
