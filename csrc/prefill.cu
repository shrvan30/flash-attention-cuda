#include <ATen/ATen.h>

#include "flash_common.h"

void launch_prefill(const at::Tensor &q, const at::Tensor &k,
                    const at::Tensor &v, at::Tensor &o, bool causal,
                    float scale) {
  TORCH_CHECK(false, "prefill kernel is not implemented yet");
}
