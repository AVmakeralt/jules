# Jules Test CMakeLists
# Simple tests that don't require MLIR runtime.

function(add_jules_test name source)
  add_test(NAME ${name}
    COMMAND julesc -emit-ast ${CMAKE_CURRENT_SOURCE_DIR}/${source}
  )
endfunction()

# Add tests for each example file.
file(GLOB JULES_EXAMPLES "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.jules")
foreach(example ${JULES_EXAMPLES})
  get_filename_component(name ${example} NAME_WE)
  add_jules_test(${name} examples/${name}.jules)
endforeach()
