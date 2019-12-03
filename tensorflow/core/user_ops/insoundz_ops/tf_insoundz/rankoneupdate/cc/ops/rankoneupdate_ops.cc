#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

using namespace tensorflow;

static const char kErrMsg[] =
    "Rank update was not successful. The input might not be valid.";

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

// Return in <out> the result of making the end of <s> a square matrix.
Status MakeBatchSquareMatrix(InferenceContext* c, ShapeHandle input,
                             ShapeHandle* out) {
  ShapeHandle s;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(input, 2, &s));

  DimensionHandle d;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(s, -2), c->Dim(s, -1), &d));

  ShapeHandle batch_shape;
  TF_RETURN_IF_ERROR(c->Subshape(s, 0, -2, &batch_shape));
  TF_RETURN_IF_ERROR(c->Concatenate(batch_shape, c->Matrix(d, d), out));
  return Status::OK();
}

//
// The first input is [...,M,N] and second input is either [...,M,K] or [...,M].
// Output is [...,N,K] or [...,N]. If <square>, then input is [...,M,M].
Status RankOneUpdateShapeFn(InferenceContext* c) {
  ShapeHandle lhs;
  ShapeHandle rhs;
  TF_RETURN_IF_ERROR(MakeBatchSquareMatrix(c, c->input(0), &lhs));
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 2, &rhs));

  ShapeHandle lhs_batch_shape;
  ShapeHandle rhs_batch_shape;
  // Make the common batch subshape.
  TF_RETURN_IF_ERROR(c->Subshape(lhs, 0, -2, &lhs_batch_shape));
  TF_RETURN_IF_ERROR(c->Subshape(rhs, 0, -2, &rhs_batch_shape));
  // Make sure the batch dimensions match between lhs and rhs.
  TF_RETURN_IF_ERROR(
      c->Merge(lhs_batch_shape, rhs_batch_shape, &lhs_batch_shape));

  DimensionHandle m;
  // lhs and rhs have the same value for m to be compatible.
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(lhs, -2), c->Dim(rhs, -2), &m));
  DimensionHandle n = c->Dim(lhs, -1);
  TF_RETURN_IF_ERROR(c->Merge(m, n, &n));

  ShapeHandle out;
  // Build final shape (batch_shape + n + k) in <out>.
  TF_RETURN_IF_ERROR(c->Concatenate(lhs_batch_shape, c->Vector(n), &out));
  TF_RETURN_IF_ERROR(c->Concatenate(out, c->Vector(n), &out));
  //TF_RETURN_IF_ERROR(c->Concatenate(out, c->Vector(c->Dim(rhs, -1)), &out));
  c->set_output(0, out);
  return Status::OK();
}

REGISTER_OP("RankOneUpdate")
    .Input("matrix: T")
    .Input("rhs: T")
    .Input("alpha: T")
    .Output("output: T")
    .Attr("T: {double, float, complex64, complex128}")
    .SetShapeFn([](InferenceContext* c) {
      return RankOneUpdateShapeFn(c);
    });

REGISTER_OP("RankOneUpdateWithScale")
    .Input("matrix: T")
    .Input("rhs: T")
    .Input("alpha: T")
    .Input("beta: T")
    .Output("output: T")
    .Attr("T: {double, float, complex64, complex128}")
    .SetShapeFn([](InferenceContext* c) {
      return RankOneUpdateShapeFn(c);
    });
