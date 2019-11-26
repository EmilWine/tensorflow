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
using namespace Eigen;

template <class Scalar>
class MatrixScaleOp : public LinearAlgebraOp<Scalar> {
 public:
  INHERIT_LINALG_TYPEDEFS(Scalar);

  explicit MatrixScaleOp(OpKernelConstruction* context) : Base(context) { }

  int NumMatrixInputs(const OpKernelContext* context) const final { return 1; }

  void ValidateInputMatrixShapes(
		  OpKernelContext* context,
		  const TensorShapes& input_matrix_shapes) const final {
  }


  TensorShapes GetOutputMatrixShapes(
      const TensorShapes& input_matrix_shapes) const final {
	  return TensorShapes({input_matrix_shapes[0]});
  }


  int64 GetCostPerUnit(const TensorShapes& input_matrix_shapes) const final {
	  double cost = static_cast<double>(input_matrix_shapes[0].num_elements());
	  return cost >= static_cast<double>(kint64max) ? kint64max
		  : static_cast<int64>(cost);
  }

  bool EnableInputForwarding() const final { return false; }

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {

    const ConstMatrixMap& matrix = inputs[0];
    const auto& alpha_in = context->input(1);

    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(alpha_in.shape()),
        errors::InvalidArgument("alpha must be scalar, got shape",
                                alpha_in.shape().DebugString()));
    const Scalar alpha = alpha_in.scalar<Scalar>()();
    
    if (matrix.rows() == 0 || matrix.cols() == 0) {
      return;
    }

    MatrixMap& output = outputs->at(0);
    matrix_scale_add<MatrixMap>::run(output,alpha);		
    //output = output*alpha;
  }
};


REGISTER_LINALG_OP("MatrixScale", (MatrixScaleOp<float>), float);
REGISTER_LINALG_OP("MatrixScale", (MatrixScaleOp<double>), double);
REGISTER_LINALG_OP("MatrixScale", (MatrixScaleOp<complex64>), complex64);
REGISTER_LINALG_OP("MatrixScale", (MatrixScaleOp<complex128>), complex128);
REGISTER_LINALG_OP("BatchMatrixScale", (MatrixScaleOp<float>), float);
REGISTER_LINALG_OP("BatchMatrixScale", (MatrixScaleOp<double>), double);
REGISTER_LINALG_OP("BatchMatrixScale", (MatrixScaleOp<complex64>), complex64);
REGISTER_LINALG_OP("BatchMatrixScale", (MatrixScaleOp<complex128>), complex128);

