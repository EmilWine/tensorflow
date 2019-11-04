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

template <class Scalar>
class HermitianMatmulOp : public LinearAlgebraOp<Scalar> {
 public:
  INHERIT_LINALG_TYPEDEFS(Scalar);

  explicit HermitianMatmulOp(OpKernelConstruction* context) : Base(context) {
	  OP_REQUIRES_OK(context, context->GetAttr("adjoint", &adjoint_));
  }

  void ValidateInputMatrixShapes(
		  OpKernelContext* context,
		  const TensorShapes& input_matrix_shapes) const final {
  Base::ValidateSquareSolver(context, input_matrix_shapes);
  }


  TensorShapes GetOutputMatrixShapes(
      const TensorShapes& input_matrix_shapes) const final {
    return TensorShapes({TensorShape({input_matrix_shapes[0].dim_size(1),
                                      input_matrix_shapes[1].dim_size(1)})});
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
    
    if (matrix.rows() == 0 || matrix.cols() == 0  || rhs.cols() == 0) {
      // To be consistent with the MatrixInverse op, we define the solution for
      // an empty set of equation as the empty matrix.
      return;
    }

    // Perform the actual HermitianMatmul. This will only use
    // the lower triangular part of data_in by default. 
    //We can work directly on the output as it was forwarded directly from the input

    MatrixMap& output = outputs->at(0);
    
    //This is actually a copy operation of eigen, becaued its mapped. 
    //This is redundant if we could enforce direct update to the variable, like assign_add op.
    //Didn't find a way to do so yet. 
    output.noalias() = matrix; 
	
    //Take only the lower part to update. More efficient.
    auto triangle = matrix.template selfadjointView<Eigen::Lower>();
    output = triangle * rhs;    
  }
 private:
  bool adjoint_;
};


REGISTER_LINALG_OP("HermitianMatmul", (HermitianMatmulOp<float>), float);
REGISTER_LINALG_OP("HermitianMatmul", (HermitianMatmulOp<double>), double);
REGISTER_LINALG_OP("HermitianMatmul", (HermitianMatmulOp<complex64>), complex64);
REGISTER_LINALG_OP("HermitianMatmul", (HermitianMatmulOp<complex128>), complex128);
REGISTER_LINALG_OP("BatchHermitianMatmul", (HermitianMatmulOp<float>), float);
REGISTER_LINALG_OP("BatchHermitianMatmul", (HermitianMatmulOp<double>), double);
REGISTER_LINALG_OP("BatchHermitianMatmul", (HermitianMatmulOp<complex64>), complex64);
REGISTER_LINALG_OP("BatchHermitianMatmul", (HermitianMatmulOp<complex128>), complex128);

