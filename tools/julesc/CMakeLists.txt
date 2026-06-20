add_executable(julesc main.cpp)
target_link_libraries(julesc PRIVATE julesLib)

if(JULES_ENABLE_MLIR)
  # Register all dialects/passes requires linking the full MLIR set.
  target_link_libraries(julesc PRIVATE
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRFuncDialect
    MLIRFuncTransforms
    MLIRSupport
    MLIRArithDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRAffineDialect
    MLIRLinalgDialect
    MLIRTensorDialect
    MLIRVectorDialect
    MLIRBufferizationDialect
    MLIRControlFlowDialect
    MLIRIndexDialect
    MLIRComplexDialect
    MLIRLLVMDialect
    MLIRGPUDialect
    MLIRSPIRVDialect
    MLIRShapeDialect
    MLIRDLTIDialect
    MLIRTosaDialect
    MLIRNVVMDialect
    MLIRROCDLDialect
    MLIRNVGPUDialect
    MLIRAMDGPUDialect
    MLIRArmNeonDialect
    MLIRArmSVEDialect
    MLIRArmSMEDialect
    MLIRAMXDialect
    MLIRAsyncDialect
    MLIROpenACCDialect
    MLIROpenMPDialect
    MLIRPDLInterpDialect
    MLIRQuantDialect
    MLIRSparseTensorDialect
    MLIRTransformDialect
    MLIRUBDialect
    MLIRMLProgramDialect
    MLIRMeshDialect
    MLIREmitCDialect
    MLIRTensorTransforms
    MLIRTosaTransforms
    MLIRSparseTensorTransforms
    MLIRTensorTilingInterfaceImpl
    MLIRTensorInferTypeOpInterfaceImpl
    MLIRNVVMTarget
    MLIRROCDLTarget
    MLIRSPIRVTarget
    MLIRAMDGPUTransforms
    MLIRArithTransforms
    MLIRArithValueBoundsOpInterfaceImpl
    MLIRIRDL
    MLIRControlFlowTransforms
    MLIRGPUTransforms
    MLIRBufferizationTransforms
    MLIRLinalgTransforms
    MLIRMLProgramTransforms
    MLIRShapeOpsTransforms
    # MLIRMlirOptMain provides registerAllDialects/registerAllPasses
    # and pulls in all transitive dialect registrations.
    MLIRMlirOptMain
    MLIRLLVMCommonConversion
    MLIRBuiltinToLLVMIRTranslation
    MLIRSPIRVTarget
    MLIRTargetLLVMIRExport
    MLIRLLVMToLLVMIRTranslation
    MLIRGPUToLLVMIRTranslation
    LLVMOrcJIT
    LLVMCore
    LLVMSupport
  )
endif()

install(TARGETS julesc DESTINATION bin)
