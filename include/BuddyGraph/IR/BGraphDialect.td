#ifndef BUDDYGRAPH_IR_BGRAPHDIALECT_TD
#define BUDDYGRAPH_IR_BGRAPHDIALECT_TD

include "mlir/IR/OpBase.td"

def BGraph_Dialect : Dialect {
  let name = "bgraph";
  let summary = "A small, verifiable neural-network graph dialect";
  let description = [{
    BGraph represents a deliberately small inference graph between ONNX import
    and structured MLIR.  It keeps graph-level legality explicit while reusing
    builtin tensors, functions, and constants.
  }];
  let cppNamespace = "::buddy::bgraph";
  let useDefaultAttributePrinterParser = 1;
  // Python Operation.create has no ODS properties argument in this MLIR build.
  // Attribute storage keeps the importer API-native without weakening ODS.
  let usePropertiesForAttributes = 0;
}

class BGraph_Op<string mnemonic, list<Trait> traits = []>
    : Op<BGraph_Dialect, mnemonic, traits>;

#endif // BUDDYGRAPH_IR_BGRAPHDIALECT_TD
