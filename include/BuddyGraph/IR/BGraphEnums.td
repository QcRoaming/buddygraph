#ifndef BUDDYGRAPH_IR_BGRAP_ENUMS_TD
#define BUDDYGRAPH_IR_BGRAP_ENUMS_TD

include "BuddyGraph/IR/BGraphDialect.td"
include "mlir/IR/EnumAttr.td"

def BGraph_LayoutNCHW : I32EnumAttrCase<"NCHW", 0, "nchw">;
def BGraph_LayoutNHWC : I32EnumAttrCase<"NHWC", 1, "nhwc">;

def BGraph_Layout : I32EnumAttr<"Layout", "Tensor and filter layout", [
    BGraph_LayoutNCHW,
    BGraph_LayoutNHWC
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::buddy::bgraph";
}

def BGraph_LayoutAttr
    : EnumAttr<BGraph_Dialect, BGraph_Layout, "layout"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // BUDDYGRAPH_IR_BGRAP_ENUMS_TD
