#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

using namespace tensorflow;

using shape_inference::InferenceContext;

REGISTER_OP("MatrixScale")
    .Input("matrix: T")
    .Input("alpha: T")
    .Output("output: T")
    .Attr("T: {double, float, complex64, complex128}")
    .SetShapeFn([](InferenceContext* c) {
		    return Status::OK();
    });

