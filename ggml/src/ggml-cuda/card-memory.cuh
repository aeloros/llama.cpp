#pragma once

#include <cstddef>

// The free memory of a whole card, counted over every program that uses it, for the card at the PCI address that
// cudaDeviceGetPCIBusId gives. Returns false where that reading is not needed or the driver does not give it.
//
// On Windows the figure of cudaMemGetInfo is what the calling program itself has not taken of the card: the memory
// that other programs hold there, a game for one, is not in it.
bool ggml_cuda_card_memory_free(const char * pci_bus_id, size_t * free);

// What the system can still commit: on Windows every program's memory counts against one limit, system memory and
// page file together, and what programs hold of the GPUs counts against it as well. At that limit the driver answers
// "out of memory" with memory of the device free. Returns false where there is no such figure.
bool ggml_cuda_host_commit_free(size_t * free);
