add_executable(julesc main.cpp)
target_link_libraries(julesc PRIVATE julesLib)

if(JULES_ENABLE_MLIR)
  target_link_libraries(julesc PRIVATE
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRFuncDialect
    MLIRFuncTransforms
    MLIRSupport
  )
endif()

install(TARGETS julesc DESTINATION bin)
