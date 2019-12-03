#include "linear_algebra_inplace.h"

namespace tensorflow {

template <class Scalar>
void LinearAlgebraInPlaceOp<Scalar>::Compute(OpKernelContext* context) {
	TensorInputs inputs;
	TensorShapes input_matrix_shapes;
	TensorShape batch_shape;
	this->AnalyzeInputs(context, &inputs, &input_matrix_shapes, &batch_shape);

	TensorShapes output_matrix_shapes;
	TensorOutputs outputs;
	this->PrepareOutputs(context, input_matrix_shapes, batch_shape, &outputs,
			&output_matrix_shapes);

	//This is the main addition: in place operation
	//This basically copies the input tensor to output
	Tensor* output_tensor = context->mutable_output(0);
	const Tensor& input_tensor = context->input(0);
	//Note: CopyFrom doesnt actually copy the buffer. 
	//It sets the output_tensor buffer to point to input_tensor buffer.
	//This means operations occur in-place

	CHECK(output_tensor->CopyFrom(input_tensor, output_tensor->shape()));

	// Process the individual matrix problems in parallel using a threadpool.
	auto shard = [this, &inputs, &input_matrix_shapes, &outputs,
	     &output_matrix_shapes, context](int64 begin, int64 end) {
		     for (int64 i = begin; i < end; ++i) {
			     this->ComputeTensorSlice(context, i, inputs, input_matrix_shapes, outputs,
					     output_matrix_shapes);
		     }
	     };
	auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
	Shard(worker_threads.num_threads, worker_threads.workers,
			batch_shape.num_elements(), Base::GetCostPerUnit(input_matrix_shapes), shard);
}

// Explicitly instantiate LinearAlgebraInPlaceOp for the scalar types we expect to use.
// Otherwise this class is not linked because of bazel shit
template class LinearAlgebraInPlaceOp<float>;
template class LinearAlgebraInPlaceOp<double>;
template class LinearAlgebraInPlaceOp<complex64>;
template class LinearAlgebraInPlaceOp<complex128>;
}
