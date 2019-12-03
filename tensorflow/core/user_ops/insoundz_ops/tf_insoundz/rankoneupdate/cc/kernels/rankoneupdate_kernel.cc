#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg_ops_common.h"
#include "tensorflow/core/user_ops/insoundz_ops/tf_insoundz/common_inz/linear_algebra_inplace.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

template <class Scalar>
class RankOneUpdateOp : public LinearAlgebraInPlaceOp<Scalar> {
 public:
  INHERIT_LINALG_INPLACE_TYPEDEFS(Scalar);

  explicit RankOneUpdateOp(OpKernelConstruction* context) : Base(context) { }

  int NumMatrixInputs(const OpKernelContext* context) const final { return 3; }

  void ValidateInputMatrixShapes( OpKernelContext* context,
		  const TensorShapes& input_matrix_shapes) const final {
	  OP_REQUIRES(context, input_matrix_shapes.size() == 3,
			  errors::InvalidArgument("Expected 3 input matrices, got %d.",
				  input_matrix_shapes.size()));
	  OP_REQUIRES(
			  context, TensorShapeUtils::IsSquareMatrix(input_matrix_shapes[0]),
			  errors::InvalidArgument("First input (lhs) must be a square matrix."));
	  OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_matrix_shapes[1]),
			  errors::InvalidArgument("Second input (rhs) must be a matrix."));
	  OP_REQUIRES(context, 
			  TensorShapeUtils::IsMatrix(input_matrix_shapes[2]) && 
			  (input_matrix_shapes[2].dim_size(0) == 1) &&
			  (input_matrix_shapes[2].dim_size(1) == 1),
			  errors::InvalidArgument("Third input (alpha) must be a matrix of size 1x1."));
	  OP_REQUIRES( context,
			  input_matrix_shapes[0].dim_size(0) == input_matrix_shapes[1].dim_size(0),
			  errors::InvalidArgument("Input matrix and rhs are incompatible."));
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

  bool EnableInputForwarding() const final { return false;}
  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs, MatrixMaps* outputs){
	  typedef Eigen::Map<const Eigen::Matrix<Scalar, 1,Eigen::Dynamic,  Eigen::RowMajor>> ConstVectorMap;

	  const ConstMatrixMap& matrix = inputs[0];
	  const ConstMatrixMap& rhs = inputs[1];
	  const ConstMatrixMap& alpha_in = inputs[2];
	  if (matrix.rows() == 0 || matrix.cols() == 0  || rhs.cols() == 0) {
		  return;
	  }

	  // Perform the actual RankOneUpdate. This will only use  the lower triangular part of data_in by default. 
	  //We can work directly on the output as it was forwarded directly from the input
	  MatrixMap& output = outputs->at(0);
	  output.template selfadjointView<Eigen::Lower>().rankUpdate(rhs,alpha_in(0,0));
  }
};

template <class Scalar>
class RankOneUpdateWithScaleOp : public LinearAlgebraInPlaceOp<Scalar> {
 public:
  INHERIT_LINALG_INPLACE_TYPEDEFS(Scalar);

  explicit RankOneUpdateWithScaleOp(OpKernelConstruction* context) : Base(context) { }


  int NumMatrixInputs(const OpKernelContext* context) const final { return 4; }

  void ValidateInputMatrixShapes( OpKernelContext* context,
		  const TensorShapes& input_matrix_shapes) const final {
	  OP_REQUIRES(context, input_matrix_shapes.size() == 4,
			  errors::InvalidArgument("Expected 3 input matrices, got %d.",
				  input_matrix_shapes.size()));
	  OP_REQUIRES(
			  context, TensorShapeUtils::IsSquareMatrix(input_matrix_shapes[0]),
			  errors::InvalidArgument("First input (lhs) must be a square matrix."));
	  OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_matrix_shapes[1]),
			  errors::InvalidArgument("Second input (rhs) must be a matrix."));
	  OP_REQUIRES(context, 
			  TensorShapeUtils::IsMatrix(input_matrix_shapes[2]) && 
			  (input_matrix_shapes[2].dim_size(0) == 1) &&
			  (input_matrix_shapes[2].dim_size(1) == 1),
			  errors::InvalidArgument("Third input (alpha) must be a matrix of size 1x1."));
	  OP_REQUIRES(context, 
			  TensorShapeUtils::IsMatrix(input_matrix_shapes[3]) && 
			  (input_matrix_shapes[3].dim_size(0) == 1) &&
			  (input_matrix_shapes[3].dim_size(1) == 1),
			  errors::InvalidArgument("Forth input (beta) must be a matrix of size 1x1."));
	  OP_REQUIRES( context,
			  input_matrix_shapes[0].dim_size(0) == input_matrix_shapes[1].dim_size(0),
			  errors::InvalidArgument("Input matrix and rhs are incompatible."));
  }

  //Output shape should be equal to input shape
  TensorShapes GetOutputMatrixShapes(
		  const TensorShapes& input_matrix_shapes) const final {
	  return TensorShapes({input_matrix_shapes[0]});
  }


  int64 GetCostPerUnit(const TensorShapes& input_matrix_shapes) const final {
	  double rows = static_cast<double>(input_matrix_shapes[0].dim_size(0));
	  double num_rhss = static_cast<double>(input_matrix_shapes[1].dim_size(1));
	  double cost = 3 * num_rhss* (rows + 0.5)* rows/ 2; //Rough estimate
	  return cost >= static_cast<double>(kint64max) ? kint64max
		  : static_cast<int64>(cost);
  }

  bool EnableInputForwarding() const final { return false;}
  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs, MatrixMaps* outputs){
	  typedef Eigen::Map<const Eigen::Matrix<Scalar, 1,Eigen::Dynamic,  Eigen::RowMajor>> ConstVectorMap;

	  const ConstMatrixMap& matrix = inputs[0];
	  const ConstMatrixMap& rhs = inputs[1];
	  const ConstMatrixMap& alpha_in = inputs[2];
	  const ConstMatrixMap& beta_in = inputs[3];
	  if (matrix.rows() == 0 || matrix.cols() == 0  || rhs.cols() == 0) {
		  return;
	  }

	  // Perform the actual RankOneUpdate. This will only use  the lower triangular part of data_in by default. 
	  //We can work directly on the output as it was forwarded directly from the input

	  MatrixMap& output = outputs->at(0);
	  output.template selfadjointView<Eigen::Lower>().rankUpdate(rhs,alpha_in(0,0),beta_in(0,0));
  }
};


REGISTER_LINALG_OP("RankOneUpdate", (RankOneUpdateOp<float>), float);
REGISTER_LINALG_OP("RankOneUpdate", (RankOneUpdateOp<double>), double);
REGISTER_LINALG_OP("RankOneUpdate", (RankOneUpdateOp<complex64>), complex64);
REGISTER_LINALG_OP("RankOneUpdate", (RankOneUpdateOp<complex128>), complex128);

REGISTER_LINALG_OP("RankOneUpdateWithScale", (RankOneUpdateWithScaleOp<float>), float);
REGISTER_LINALG_OP("RankOneUpdateWithScale", (RankOneUpdateWithScaleOp<double>), double);
REGISTER_LINALG_OP("RankOneUpdateWithScale", (RankOneUpdateWithScaleOp<complex64>), complex64);
REGISTER_LINALG_OP("RankOneUpdateWithScale", (RankOneUpdateWithScaleOp<complex128>), complex128);

