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

  bool EnableInputForwarding() const final { return false;}

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {

    typedef Eigen::Map<const Eigen::Matrix<Scalar, 1,Eigen::Dynamic,  Eigen::RowMajor>> ConstVectorMap;

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
    //
    //output.noalias() = matrix; 
	
    //Take only the lower part to update. More efficient.
    auto triangle = output.template selfadjointView<Eigen::Lower>();

    //Perform the rank update on the output materix
    
    //Convert rhs to vector type if is vector
    triangle.rankUpdate(rhs,alpha);
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

