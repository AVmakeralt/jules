//===- CMakeLists.txt - Jules Library Build --------------------------------===//

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

target_include_directories(julesLib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_BINARY_DIR}/../include
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
    MLIROpenMP
    # LLVM Backend: conversion + JIT libraries
    MLIRArithToLLVM
    MLIRFuncToLLVM
    MLIRMemRefToLLVM
    MLIRSCFToControlFlow
    MLIRControlFlowToLLVM
    MLIRMathToLLVM
    MLIRIndexToLLVM
    MLIRVectorToLLVM
    MLIRLLVMCommon
    MLIRTargetLLVMIRExport
    MLIRBuiltinToLLVMIR
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

  add_public_tablegen_target(JulesOpsIncGen)
  add_dependencies(julesLib JulesOpsIncGen)
endif()
