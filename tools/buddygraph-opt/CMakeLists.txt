get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

if(MLIR_LINK_MLIR_DYLIB)
  set(mlir_libs MLIR)
else()
  set(mlir_libs ${dialect_libs} ${conversion_libs} ${extension_libs})
endif()

add_llvm_executable(buddygraph-opt buddygraph-opt.cpp)

llvm_update_compile_flags(buddygraph-opt)

target_link_libraries(buddygraph-opt PRIVATE
  ${mlir_libs}
  BuddyGraphIR
  BuddyGraphTransforms
  BGraphToLinalg
  MLIRFuncInlinerExtension
  MLIROptLib
)

mlir_check_all_link_libraries(buddygraph-opt)
