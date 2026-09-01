get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_llvm_executable(buddygraph-opt buddygraph-opt.cpp)

llvm_update_compile_flags(buddygraph-opt)

target_link_libraries(buddygraph-opt PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  BuddyGraphIR
  BuddyGraphTransforms
  BGraphToLinalg
  MLIROptLib
)

mlir_check_all_link_libraries(buddygraph-opt)
