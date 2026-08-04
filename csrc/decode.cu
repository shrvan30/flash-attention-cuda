#include <ATen/ATen.h>

#include "flash_common.h"

void launch_decode(const at::Tensor &q, const at::Tensor &k_cache,
                   const at::Tensor &v_cache, const at::Tensor &seq_lens,
                   at::Tensor &o, float scale) {
  TORCH_CHECK(false, "decode kernel is not implemented yet");
}
