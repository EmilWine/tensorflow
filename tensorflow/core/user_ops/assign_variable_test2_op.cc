/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
//#include "tensorflow/core/kernels/gather_functor.h"
//#include "tensorflow/core/kernels/gather_nd_op.h"
//#include "tensorflow/core/kernels/resource_variable_ops.h"
//#include "tensorflow/core/kernels/scatter_functor.h"
//#include "tensorflow/core/kernels/training_op_helpers.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
using namespace tensorflow::shape_inference;
typedef Eigen::ThreadPoolDevice CPUDevice;

//REGISTER_OP("AssignVariableTest2")
//    .Input("input: T")
//    .Input("value: T")
//    .Output("output: T")
//    .Attr("T: type")
//    .SetAllowsUninitializedInput()
//    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
//      c->set_output(0, c->input(1));
//      return Status::OK();
//    });
                  
//REGISTER_OP("AssignVariableTest2")
//    .Input("resource: resource")
//    .Input("value: dtype")
//    .Attr("dtype: type")
//    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
//      c->set_output(0, c->input(0));
//      return Status::OK();
//    });

Status CreateAssignShapeFn(InferenceContext* c) {
  std::vector<ShapeAndType> handle_shape_and_type;
  TF_RETURN_IF_ERROR(shape_inference::ValidateVariableResourceHandle(
      c, &handle_shape_and_type));

  ShapeHandle value_shape = c->input(1);
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(
      c->Merge(handle_shape_and_type[0].shape, value_shape, &unused));

  if (handle_shape_and_type[0].dtype == DT_VARIANT &&
      handle_shape_and_type.size() > 1 &&
      c->input_handle_shapes_and_types(1) != nullptr) {
    auto* value_handle_shape_and_type = c->input_handle_shapes_and_types(1);
    if (value_handle_shape_and_type->size() !=
        handle_shape_and_type.size() - 1) {
      return errors::InvalidArgument(
          "Incompatible handle variant shape_and_type size and input "
          "shape_and_type size: ",
          handle_shape_and_type.size() - 1, " vs. ",
          value_handle_shape_and_type->size());
    }
  }
  return Status::OK();
}

REGISTER_OP("AssignVariableTest2")
    .Input("resource: float32")
    .Input("value: float32")
    .Attr("dtype: type")
    .SetShapeFn(CreateAssignShapeFn);

template <typename Device, typename T>
class AssignVariableTest2Op : public OpKernel {
 public:
  explicit AssignVariableTest2Op(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
    if (!c->GetAttr("_grappler_relax_allocator_constraints",
                    &relax_constraints_)
             .ok()) {
      relax_constraints_ = false;
    }
  } 

  void Compute(OpKernelContext* context) override {
    OP_REQUIRES(context, dtype_ == context->input(1).dtype(),
                errors::InvalidArgument(
                    "Variable and value dtypes don't match; respectively, ",
                    DataTypeString(dtype_), " and ",
                    DataTypeString(context->input(1).dtype())));
    core::RefCountPtr<Var> variable;
    const Tensor& value = context->input(1);

    OP_REQUIRES_OK(context, LookupOrCreateResource<Var>(
                                context, HandleFromInput(context, 0), &variable,
                                [this, &value](Var** ptr) {
                                  *ptr = new Var(dtype_);
                                  *(*ptr)->tensor() = value;
                                  (*ptr)->is_initialized = true;
                                  return Status::OK();
                                }));
    mutex_lock ml(*variable->mu());
    OP_REQUIRES(context, variable->tensor()->dtype() == dtype_,
                errors::InvalidArgument(
                    "Trying to assign variable with wrong dtype. Expected ",
                    DataTypeString(variable->tensor()->dtype()), " got ",
                    DataTypeString(dtype_)));
    if (variable->copy_on_read_mode.load()) {
      PersistentTensor unused;
      Tensor* tmp;
      AllocatorAttributes attr;
      attr.set_gpu_compatible(true);
      attr.set_nic_compatible(true);
      OP_REQUIRES_OK(context,
                     context->allocate_persistent(value.dtype(), value.shape(),
                                                  &unused, &tmp, attr));
      functor::DenseUpdate<Device, T, ASSIGN> copy_functor;
      copy_functor(context->eigen_device<Device>(), tmp->flat<T>(),
                   value.flat<T>());
      *variable->tensor() = *tmp;
    } else {
      *variable->tensor() = value;
    }
    variable->is_initialized = true;
  }
 private:
  DataType dtype_;
  bool relax_constraints_;
};


#define REGISTER_KERNELS(type)                                \
  REGISTER_KERNEL_BUILDER(Name("AssignVariableTest2")            \
                              .Device(DEVICE_CPU),             \
                          AssignVariableTest2Op<Eigen::ThreadPoolDevice, type>);
                              //.TypeConstraint<type>("dtype")

REGISTER_KERNEL_BUILDER(Name("AssignVariableTest2").Device(DEVICE_CPU), AssignVariableTest2Op<Eigen::ThreadPoolDevice, float>);

//TF_CALL_ALL_TYPES(REGISTER_KERNELS);
//TF_CALL_QUANTIZED_TYPES(REGISTER_KERNELS);
//REGISTER_KERNEL_BUILDER(Name("AssignVariableTest2").Device(DEVICE_CPU), AssignVariableTest2Op);

