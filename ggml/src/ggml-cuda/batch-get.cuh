#pragma once

#include "ggml-backend.h"

// Reads many ranges of tensors of a device with one call, faster than one copy each. False when it does not apply or
// the driver refuses a step of it: the caller then reads the ranges one by one.
bool ggml_cuda_get_tensor_n(int device,
        const ggml_tensor * const * tensors, void * const * data, const size_t * offsets,
        const size_t * sizes, size_t n_tensors);

// Makes what the batched reads of the device go through, where that is memory which is pinned once and kept
// (Windows, see batch-get.cu), so that no read has to pin it. Called when a backend of the device is made.
void ggml_cuda_batch_get_prepare(int device);
