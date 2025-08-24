#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <type_traits>

// Template function to get appropriate epsilon for different floating point types
template<typename T>
__device__ __forceinline__ T get_epsilon() {
    if constexpr (std::is_same_v<T, float>) {
        return static_cast<T>(1e-8);
    } else if constexpr (std::is_same_v<T, double>) {
        return static_cast<T>(1e-12);
    } else {
        return static_cast<T>(1e-8); // fallback
    }
}

// Templated CUDA kernel for computing pairwise distances with PBC offsets
// This version avoids repeat_interleave by computing offsets directly in the kernel
template<typename T>
__global__ void pbc_distance_kernel_optimized(
    const T* pos1,
    const T* pos2,
    const T* pbc_offsets,           // [batch_size, 3]
    const int64_t* num_atoms_per_image_sqr,  // [batch_size]
    const int64_t* batch_offsets,   // [batch_size] - cumulative offsets for each batch
    T* distances_squared,
    bool* valid_mask,
    int num_pairs,
    T radius_squared
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_pairs) {
        // Find which batch this pair belongs to
        int batch_idx = 0;
        while (batch_idx < num_pairs && idx >= batch_offsets[batch_idx + 1]) {
            batch_idx++;
        }
        
        // Get PBC offset for this batch
        T offset_x = pbc_offsets[batch_idx * 3];
        T offset_y = pbc_offsets[batch_idx * 3 + 1];
        T offset_z = pbc_offsets[batch_idx * 3 + 2];
        
        // Get positions for this atom pair with PBC offset
        T dx = pos2[idx * 3] - pos1[idx * 3] + offset_x;
        T dy = pos2[idx * 3 + 1] - pos1[idx * 3 + 1] + offset_y;
        T dz = pos2[idx * 3 + 2] - pos1[idx * 3 + 2] + offset_z;
        
        // Compute squared distance
        T dist_sq = dx * dx + dy * dy + dz * dz;
        distances_squared[idx] = dist_sq;
        
        // Check if within radius
        valid_mask[idx] = (dist_sq <= radius_squared) && (dist_sq > get_epsilon<T>());
    }
}

// Original kernel for fallback
template<typename T>
__global__ void pbc_distance_kernel(
    const T* pos1,
    const T* pos2,
    const T* pbc_offsets,
    T* distances_squared,
    bool* valid_mask,
    int num_pairs,
    T radius_squared
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_pairs) {
        // Get positions for this atom pair
        T dx = pos2[idx * 3] - pos1[idx * 3] + pbc_offsets[idx * 3];
        T dy = pos2[idx * 3 + 1] - pos1[idx * 3 + 1] + pbc_offsets[idx * 3 + 1];
        T dz = pos2[idx * 3 + 2] - pos1[idx * 3 + 2] + pbc_offsets[idx * 3 + 2];
        
        // Compute squared distance
        T dist_sq = dx * dx + dy * dy + dz * dz;
        distances_squared[idx] = dist_sq;
        
        // Check if within radius
        valid_mask[idx] = (dist_sq <= radius_squared) && (dist_sq > get_epsilon<T>());
    }
}

// Template helper function to launch the appropriate optimized kernel
template<typename T>
inline void launch_pbc_distance_kernel_optimized(
    const T* pos1,
    const T* pos2,
    const T* pbc_offsets,
    const int64_t* num_atoms_per_image_sqr,
    const int64_t* batch_offsets,
    T* distances_squared,
    bool* valid_mask,
    int num_pairs,
    T radius_squared,
    int blocks,
    int threads_per_block
) {
    pbc_distance_kernel_optimized<T><<<blocks, threads_per_block>>>(
        pos1, pos2, pbc_offsets, num_atoms_per_image_sqr, batch_offsets,
        distances_squared, valid_mask, num_pairs, radius_squared
    );
}

// Template helper function to launch the appropriate kernel (fallback)
template<typename T>
void launch_pbc_distance_kernel(
    const T* pos1,
    const T* pos2,
    const T* pbc_offsets,
    T* distances_squared,
    bool* valid_mask,
    int num_pairs,
    T radius_squared,
    int blocks,
    int threads_per_block
) {
    pbc_distance_kernel<T><<<blocks, threads_per_block>>>(
        pos1, pos2, pbc_offsets, distances_squared, valid_mask, num_pairs, radius_squared
    );
}

// CUDA function to compute distances for all unit cell offsets
std::vector<torch::Tensor> pbc_distance_cuda(
    torch::Tensor pos1,
    torch::Tensor pos2, 
    torch::Tensor data_cell,
    torch::Tensor num_atoms_per_image_sqr,
    int batch_size,
    std::vector<int> max_rep,
    float radius,
    torch::Device device
) {
    // Convert tensors to CUDA if not already, but preserve original dtype
    pos1 = pos1.to(device).contiguous();
    pos2 = pos2.to(device).contiguous();
    data_cell = data_cell.to(device).contiguous();
    num_atoms_per_image_sqr = num_atoms_per_image_sqr.to(device);
    
    // Check that all position tensors have the same dtype
    TORCH_CHECK(pos1.dtype() == pos2.dtype(), "pos1 and pos2 must have the same dtype");
    TORCH_CHECK(pos1.dtype() == data_cell.dtype(), "pos1 and data_cell must have the same dtype");
    
    // Determine if we're working with float32 or float64
    bool is_float64 = pos1.dtype() == torch::kFloat64;
    
    int num_pairs = pos1.size(0);
    
    // Storage for all results across unit cells
    std::vector<torch::Tensor> all_index1, all_index2, all_unit_cell, all_distances_sq;
    
    // Create base indices for original atom pairs
    torch::Tensor base_indices = torch::arange(num_pairs, torch::dtype(torch::kLong).device(device));
    
    // Launch parameters
    int threads_per_block = 512;
    int blocks = (num_pairs + threads_per_block - 1) / threads_per_block;
    
    // Pre-allocate tensors outside the loop for reuse
    torch::Tensor distances_squared = torch::zeros({num_pairs}, 
        torch::dtype(pos1.dtype()).device(device));
    torch::Tensor valid_mask = torch::zeros({num_pairs}, 
        torch::dtype(torch::kBool).device(device));
    torch::Tensor unit_cell_offset = torch::zeros({3}, 
        torch::dtype(pos1.dtype()).device(device));
    torch::Tensor unit_cell_offset_batch = torch::zeros({batch_size, 3, 1}, 
        torch::dtype(pos1.dtype()).device(device));
    
    // Pre-compute batch offsets for optimized kernel
    torch::Tensor batch_offsets = torch::zeros({batch_size + 1}, 
        torch::dtype(torch::kLong).device(device));
    torch::Tensor cumsum = torch::cumsum(num_atoms_per_image_sqr, 0);
    batch_offsets.slice(0, 1, batch_size + 1) = cumsum;
    
    // Iterate over unit cell offsets (triple loop)
    // NOTE: for i, j, k loop can not be flatten, as we need to limit the device memory usage
    #pragma unroll
    for (int i = -max_rep[0]; i <= max_rep[0]; i++) {
        #pragma unroll
        for (int j = -max_rep[1]; j <= max_rep[1]; j++) {
            #pragma unroll
            for (int k = -max_rep[2]; k <= max_rep[2]; k++) {
                
                // Reuse pre-allocated unit cell offset tensor
                unit_cell_offset[0] = static_cast<float>(i);
                unit_cell_offset[1] = static_cast<float>(j);
                unit_cell_offset[2] = static_cast<float>(k);
                
                // Compute PBC offsets for this unit cell
                // unit_cell_offset_batch.fill_(0);
                unit_cell_offset_batch.select(2, 0) = unit_cell_offset.unsqueeze(0).expand({batch_size, -1});
                torch::Tensor pbc_offsets = torch::bmm(data_cell, unit_cell_offset_batch).squeeze(-1);
                
                // // Optimized: Use index_select instead of repeat_interleave
                // // Create index tensor for selecting pbc_offsets based on atom pairs
                // int64_t offset = 0;
                // for (int b = 0; b < batch_size; b++) {
                //     int64_t num_pairs_in_batch = num_atoms_per_image_sqr[b].item<int64_t>();
                //     auto batch_indices = torch::full({num_pairs_in_batch}, b, 
                //         torch::dtype(torch::kLong).device(device));
                //     pbc_offsets_per_atom.slice(0, offset, offset + num_pairs_in_batch) = 
                //         pbc_offsets.index_select(0, batch_indices);
                //     offset += num_pairs_in_batch;
                // }
                
                // Reset output tensors for reuse
                // distances_squared.fill_(0);
                // valid_mask.fill_(false);
                
                // Launch templated CUDA kernel
                if (is_float64) {
                    double radius_squared = static_cast<double>(radius) * static_cast<double>(radius);
                    launch_pbc_distance_kernel_optimized<double>(
                        pos1.data_ptr<double>(),
                        pos2.data_ptr<double>(),
                        // pbc_offsets_per_atom.data_ptr<double>(),
                        pbc_offsets.data_ptr<double>(),
                        num_atoms_per_image_sqr.data_ptr<int64_t>(),
                        batch_offsets.data_ptr<int64_t>(),
                        distances_squared.data_ptr<double>(),
                        valid_mask.data_ptr<bool>(),
                        num_pairs,
                        radius_squared,
                        blocks,
                        threads_per_block
                    );
                } else {
                    float radius_squared = radius * radius;
                    launch_pbc_distance_kernel_optimized<float>(
                        pos1.data_ptr<float>(),
                        pos2.data_ptr<float>(),
                        // pbc_offsets_per_atom.data_ptr<float>(),
                        pbc_offsets.data_ptr<float>(),
                        num_atoms_per_image_sqr.data_ptr<int64_t>(),
                        batch_offsets.data_ptr<int64_t>(),
                        distances_squared.data_ptr<float>(),
                        valid_mask.data_ptr<bool>(),
                        num_pairs,
                        radius_squared,
                        blocks,
                        threads_per_block
                    );
                }
                
                // Filter valid pairs
                torch::Tensor valid_indices = torch::nonzero(valid_mask).squeeze(-1);
                if (valid_indices.numel() > 0) {
                    torch::Tensor valid_base_indices = base_indices.index_select(0, valid_indices);
                    torch::Tensor valid_distances = distances_squared.index_select(0, valid_indices);
                    torch::Tensor valid_unit_cell = unit_cell_offset.unsqueeze(0).repeat({valid_indices.size(0), 1});
                    
                    all_index1.push_back(valid_base_indices);
                    all_unit_cell.push_back(valid_unit_cell);
                    all_distances_sq.push_back(valid_distances);
                }
            }
        }
    }
    
    // Single synchronization after all kernel launches
    cudaDeviceSynchronize();
    
    // Concatenate results
    torch::Tensor final_indices, final_unit_cell, final_distances;
    
    if (all_index1.size() > 0) {
        final_indices = torch::cat(all_index1);
        final_unit_cell = torch::cat(all_unit_cell);
        final_distances = torch::cat(all_distances_sq);
    } else {
        final_indices = torch::empty({0}, torch::dtype(torch::kLong).device(device));
        final_unit_cell = torch::empty({0, 3}, torch::dtype(pos1.dtype()).device(device));
        final_distances = torch::empty({0}, torch::dtype(pos1.dtype()).device(device));
    }
    
    return {final_indices, final_unit_cell, final_distances};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pbc_distance_cuda", &pbc_distance_cuda, "PBC distance computation with CUDA");
}
