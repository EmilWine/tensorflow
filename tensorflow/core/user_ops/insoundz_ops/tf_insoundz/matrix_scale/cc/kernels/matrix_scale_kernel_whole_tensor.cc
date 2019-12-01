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
class MatrixScaleOp : public OpKernel {
	public:
		explicit MatrixScaleOp(OpKernelConstruction* context) : OpKernel(context) { }
		typedef Eigen::Map<Eigen::Matrix<Scalar,Eigen::Dynamic,1> > ResMap;

		void Compute(OpKernelContext* context) override {
			// Grab the input tensor
			const Tensor& input_tensor = context->input(0);
			const auto& alpha_in = context->input(1);
			
			OP_REQUIRES(
					context, TensorShapeUtils::IsScalar(alpha_in.shape()),
					errors::InvalidArgument("alpha must be scalar, got shape",
						alpha_in.shape().DebugString()));
			const Scalar alpha = alpha_in.scalar<Scalar>()();

			// Create an output tensor
			Tensor* output_tensor = NULL;
			OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
						&output_tensor));
			CHECK(output_tensor->CopyFrom(input_tensor, output_tensor->shape()));

			auto output_flat = output_tensor->flat<Scalar>();
			ResMap output_map = ResMap(output_flat.data(),output_flat.size());
			Eigen::matrix_scale_add<ResMap>::run(output_map,alpha);		
		}
};

REGISTER_KERNEL_BUILDER(Name("MatrixScale").Device(DEVICE_CPU).TypeConstraint<float>("T"), MatrixScaleOp<float>);
REGISTER_KERNEL_BUILDER(Name("MatrixScale").Device(DEVICE_CPU).TypeConstraint<double>("T"), MatrixScaleOp<double>);
REGISTER_KERNEL_BUILDER(Name("MatrixScale").Device(DEVICE_CPU).TypeConstraint<complex64>("T"), MatrixScaleOp<complex64>);
REGISTER_KERNEL_BUILDER(Name("MatrixScale").Device(DEVICE_CPU).TypeConstraint<complex128>("T"), MatrixScaleOp<complex128>);
