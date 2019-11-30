#ifndef TENSORFLOW_CORE_KERNELS_LINALG_INPLACE_OPS_COMMON_H_
#define TENSORFLOW_CORE_KERNELS_LINALG_INPLACE_OPS_COMMON_H_

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

namespace tensorflow {

template <class Scalar>
class LinearAlgebraInPlaceOp : public LinearAlgebraOp<Scalar> {
 public:
  using TensorInputs = gtl::InlinedVector<const Tensor*, 4>;
  using TensorOutputs = gtl::InlinedVector<Tensor*, 4>;
  INHERIT_LINALG_TYPEDEFS(Scalar);
  explicit LinearAlgebraInPlaceOp(OpKernelConstruction* context) : LinearAlgebraOp<Scalar>(context) { }

  void Compute(OpKernelContext* context) override;

};
#define INHERIT_LINALG_INPLACE_TYPEDEFS(Scalar)               \
  typedef LinearAlgebraInPlaceOp<Scalar> Base;                \
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real; \
  using Matrix = typename Base::Matrix;                       \
  using MatrixMap = typename Base::MatrixMap;                 \
  using MatrixMaps = typename Base::MatrixMaps;               \
  using ConstMatrixMap = typename Base::ConstMatrixMap;       \
  using ConstMatrixMaps = typename Base::ConstMatrixMaps;     \
  using ConstVectorMap = typename Base::ConstVectorMap;       \
  using TensorShapes = typename Base::TensorShapes;

extern template class LinearAlgebraInPlaceOp<float>;
extern template class LinearAlgebraInPlaceOp<double>;
extern template class LinearAlgebraInPlaceOp<complex64>;
extern template class LinearAlgebraInPlaceOp<complex128>;

}
#endif
