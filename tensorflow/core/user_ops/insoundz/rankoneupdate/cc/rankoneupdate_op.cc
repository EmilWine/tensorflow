//#include "third_party/eigen3/Eigen/Cholesky"
#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

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
Status RankOneUpdateShapeFn(InferenceContext* c, bool square) {
  ShapeHandle lhs;
  ShapeHandle rhs;
  if (square) {
    TF_RETURN_IF_ERROR(MakeBatchSquareMatrix(c, c->input(0), &lhs));
  } else {
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &lhs));
  }
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
  if (square) {
    TF_RETURN_IF_ERROR(c->Merge(m, n, &n));
  }

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
      return RankOneUpdateShapeFn(c, true /* square (*/);
    });

template <class Scalar>
class RankOneUpdateOp : public LinearAlgebraOp<Scalar> {
 public:
  INHERIT_LINALG_TYPEDEFS(Scalar);

  explicit RankOneUpdateOp(OpKernelConstruction* context) : Base(context) {}

  // Tell the base class to ignore the alpha parameter
  // in context->input(2).
  int NumMatrixInputs(const OpKernelContext* context) const final { return 2; }

  void ValidateInputMatrixShapes(
		  OpKernelContext* context,
		  const TensorShapes& input_matrix_shapes) const final {
  Base::ValidateSquareSolver(context, input_matrix_shapes);
  }

  //Output shape should be equal to input shape
  TensorShapes GetOutputMatrixShapes(
		  const TensorShapes& input_matrix_shapes) const final {
	  return TensorShapes({input_matrix_shapes[0]});
  }


  int64 GetCostPerUnit(const TensorShapes& input_matrix_shapes) const final {
	  double rows = static_cast<double>(input_matrix_shapes[0].dim_size(0));
	  double num_rhss = static_cast<double>(input_matrix_shapes[1].dim_size(1));
	  double cost = num_rhss* (rows + 0.5)* rows/ 2; //Rough estimate
	  return cost >= static_cast<double>(kint64max) ? kint64max
		  : static_cast<int64>(cost);
  }

  bool EnableInputForwarding() const final { return false; }

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
    const ConstMatrixMap& matrix = inputs[0];
    const ConstMatrixMap& rhs = inputs[1];
    const auto& alpha_in = context->input(2);
    
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(alpha_in.shape()),
        errors::InvalidArgument("alpha must be scalar, got shape ",
                                alpha_in.shape().DebugString()));
    const Scalar alpha = alpha_in.scalar<Scalar>()();
    if (matrix.rows() == 0 || matrix.cols() == 0  || rhs.cols() == 0) {
      // To be consistent with the MatrixInverse op, we define the solution for
      // an empty set of equation as the empty matrix.
      return;
    }

    // Perform the actual RankOneUpdate. This will only use
    // the lower triangular part of data_in by default. 
    //We can work directly on the output as it was forwarded directly from the input

    MatrixMap& output = outputs->at(0);
    
    //This is actually a copy operation of eigen, becaued its mapped. 
    //This is redundant if we could enforce direct update to the variable, like assign_add op.
    //Didn't find a way to do so yet. 
    output.noalias() = matrix; 
	
    //Take only the lower part to update. More efficient.
    auto triangle = output.template selfadjointView<Eigen::Lower>();

    //Perform the rank update on the output materix
    triangle.rankUpdate(rhs,alpha);
    //output.noalias() = triangle.rankUpdate(rhs,alpha);
    //outputs->at(0).template selfadjointView<Eigen::Lower>().rankUpdate(rhs,alpha);
  }
};


REGISTER_LINALG_OP("RankOneUpdate", (RankOneUpdateOp<float>), float);
REGISTER_LINALG_OP("RankOneUpdate", (RankOneUpdateOp<double>), double);
REGISTER_LINALG_OP("RankOneUpdate", (RankOneUpdateOp<complex64>), complex64);
REGISTER_LINALG_OP("RankOneUpdate", (RankOneUpdateOp<complex128>), complex128);
REGISTER_LINALG_OP("BatchRankOneUpdate", (RankOneUpdateOp<float>), float);
REGISTER_LINALG_OP("BatchRankOneUpdate", (RankOneUpdateOp<double>), double);
REGISTER_LINALG_OP("BatchRankOneUpdate", (RankOneUpdateOp<complex64>), complex64);
REGISTER_LINALG_OP("BatchRankOneUpdate", (RankOneUpdateOp<complex128>), complex128);

