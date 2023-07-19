#include "weighting_mod_cpu.h"

#include "utils.h"

torch::Tensor spline_weighting_fw_cpu(torch::Tensor x,
                                      torch::Tensor basis,
                                      torch::Tensor weight_index) {
  CHECK_CPU(x);
  CHECK_CPU(basis);
  CHECK_CPU(weight_index);


  auto E = x.size(0);
  auto M_out = x.size(1);
  auto S = x.size(2);

  auto out = at::empty({E, M_out}, x.options());

  auto weight_index_data = weight_index.data_ptr<int64_t>();

  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, x.scalar_type(), "weighting_fw", [&] {
    auto x_data = x.data_ptr<scalar_t>();
    auto basis_data = basis.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    scalar_t v;

    for (int64_t e = 0; e < E; e++) {
      for (int64_t m_out = 0; m_out < M_out; m_out++) {
        v = 0;
        for (int64_t s = 0; s < S; s++) {
          auto b = basis_data[e * S + s];
          auto wi = weight_index_data[e * S + s];
        //   for (int64_t m_in = 0; m_in < M_in; m_in++) {
            auto tmp =
                x_data[wi * x.stride(2) + m_out * x.stride(1) +
                            e * x.stride(0)];
            tmp *= b;
            v += tmp;
        //   }
        }
        out_data[e * M_out + m_out] = v;
      }
    }
  });

  return out;
}

torch::Tensor spline_weighting_bw_x_cpu(torch::Tensor grad_out,
                                        torch::Tensor basis,
                                        torch::Tensor weight_index, int64_t kernel_size) {
  CHECK_CPU(grad_out);
  CHECK_CPU(basis);
  CHECK_CPU(weight_index);


  auto E = grad_out.size(0);
  auto M_out = grad_out.size(1);
  auto S = grad_out.size(1);

  auto grad_x = at::zeros({E, M_out, S}, grad_out.options());

  auto weight_index_data = weight_index.data_ptr<int64_t>();

  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, grad_out.scalar_type(), "weighting_bw_x", [&] {
    auto grad_out_data = grad_out.data_ptr<scalar_t>();
    auto basis_data = basis.data_ptr<scalar_t>();
    auto grad_x_data = grad_x.data_ptr<scalar_t>();

    for (int64_t e = 0; e < E; e++) {
      for (int64_t m_out = 0; m_out < M_out; m_out++) {
        auto g =
            grad_out_data[e * grad_out.stride(0) + m_out * grad_out.stride(1)];
        for (int64_t s = 0; s < S; s++) {
          auto b = basis_data[e * S + s];
        //   for (int64_t m_in = 0; m_in < M_in; m_in++) {
            // auto w =
            //     grad_x_data[wi * grad_x_data.stride(2) + m_out * grad_x_data.stride(1) +
            //                 e * grad_x_data.stride(0)];
            //     weight_data[wi * weight.stride(0) + m_in * weight.stride(1) +
            //                 m_out * weight.stride(2)];
        grad_x_data[e* M_out* S +m_out * S + s] += g * b ;
        //   }
        }
      }
    }
  });

  return grad_x;
}



torch::Tensor spline_weighting_bw_basis_cpu(torch::Tensor grad_out,
                                            torch::Tensor x,
                                            torch::Tensor weight_index) {
  CHECK_CPU(grad_out);
  CHECK_CPU(x);
  CHECK_CPU(weight_index);

  

  auto E = x.size(0);
  auto M_out = x.size(1);
  auto S = x.size(2);

  auto grad_basis = at::zeros({E, S}, grad_out.options());

  auto weight_index_data = weight_index.data_ptr<int64_t>();

  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, x.scalar_type(), "weighting_bw_basis", [&] {
    auto grad_out_data = grad_out.data_ptr<scalar_t>();
    auto x_data = x.data_ptr<scalar_t>();
    auto grad_basis_data = grad_basis.data_ptr<scalar_t>();

    for (int64_t e = 0; e < E; e++) {
      for (int64_t m_out = 0; m_out < M_out; m_out++) {
        auto g =
            grad_out_data[e * grad_out.stride(0) + m_out * grad_out.stride(1)];
        for (int64_t s = 0; s < S; s++) {
          scalar_t b = 0;
          auto wi = weight_index_data[e * S + s];
        //   for (int64_t m_in = 0; m_in < M_in; m_in++) {
            // auto w =
            //     weight_data[wi * weight.stride(0) + m_in * weight.stride(1) +
            //                 m_out * weight.stride(2)];
        auto w = x_data[wi * x.stride(2) + m_out * x.stride(1) + e * x.stride(0)];
        b += w;
        //   }
          grad_basis_data[e * S + s] += g * b;
        }
      }
    }
  });

  return grad_basis;
}
