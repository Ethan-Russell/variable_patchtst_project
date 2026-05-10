def get_perfect_patches(seq_len, num_patches, min_patch_size, num_scales):
    """
    Calculates patch boundaries so that the entire seq_len is covered
    using exactly num_patches, with resolution decaying into the past.
    """
    patch_sizes_per_scale = [min_patch_size * (2**i) for i in range(num_scales)]
    
    # Distribute tokens (num_patches) across the scales
    # We'll use a simple even split, adjusting the first one for the remainder
    tokens_per_scale = [num_patches // num_scales] * num_scales
    tokens_per_scale[0] += num_patches % num_scales

    biggest = patch_sizes_per_scale[-1]

    if biggest > seq_len:
        raise RuntimeError(f"Largest patch size ({biggest}) is bigger than the sequence length {seq_len}")

    # seq_len = p * (patch size for all patches except for the last) + (1-p) * (size of last patch)
    # seq_len = p * (patch_size * n_tokens for (patch_size, n_tokens) in zip(patch_sizes, tokens_per_scale) - patch_sizes[-1]) + (1-p)*(patch_sizes[-1])
    # seq_len - patch_sizes[-1] = p * (patch_size * n_tokens for (patch_size, n_tokens) in zip(patch_sizes, tokens_per_scale) - patch_sizes[-1]) - p * (patch_sizes[-1])
    # p = (seq_len - patch_sizes[-1]) / ((patch_size * n_tokens for (patch_size, n_tokens) in zip(patch_sizes, tokens_per_scale) - patch_sizes[-1]) - patch_sizes[-1])
    p = (seq_len - biggest) / ((sum(size * count for (size, count) in zip(patch_sizes_per_scale, tokens_per_scale))) - biggest)

    # p represents the fractional portion of each patch that does not overlap with the previous one.

    # This should only happen if biggest > seq_len, but just in case
    if p < 0:
        raise RuntimeError(f"Portion of each non-overlapping patch is too small ({p}), cannot patch.")

    # If non-overlapping, force patches to be overlapping.  This prioritizes the more recent tokens.
    if p > 1:
        p = 1

    # Make list of patch sizes
    patch_sizes = []
    for (size, count) in zip(patch_sizes_per_scale, tokens_per_scale):
        for _ in range(count):
            patch_sizes.append(size)
    
    # Create list of fractional end indices
    end_idxs_float = np.zeros(num_patches)
    for i in range(num_patches):
        if i == 0:
            end_idxs_float[i] = seq_len
        else:
            size = patch_sizes[i-1]
            prev_end_idx = end_idxs_float[i-1]
            cur_end_idx = prev_end_idx - p * size
            end_idxs_float[i] = cur_end_idx
    
    # Now round the end indices the nearest integer
    end_idxs = np.rint(end_idxs_float).astype(int)

    # Create start indices based on the end indices and patch sizes
    start_idxs = end_idxs - patch_sizes

    # Format the data in the expected return format
    patch_definitions = [(int(i1), int(i2), s) for (i1, i2, s) in zip(start_idxs, end_idxs, patch_sizes)]
    return patch_definitions

def get_uniform_patches(seq_len, patch_size, stride):
    seq_len_ext = seq_len + stride
    patch_definitions = []
    start = 0
    end = patch_size
    while end <= seq_len_ext:
        patch_definitions.append((start, end, patch_size))
        start += stride
        end += stride

    return patch_definitions

class VariablePatchAlgorithm:
    def __init__(self, min_patch_size:int, num_patch_sizes:int, num_patches:int):
        self.min_patch_size = min_patch_size
        self.num_patch_sizes = num_patch_sizes
        self.num_patches = num_patches
    
    def __call__(self, sequence_length):
        patch_definitions = get_perfect_patches(sequence_length, self.num_patches, self.min_patch_size, self.num_patch_sizes)
        assert len(patch_definitions) == self.num_patches
        
        return patch_definitions

class UniformPatchAlgorithm:
    def __init__(self, patch_size:int, patch_stride:int):
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def __call__(self, sequence_length):
        patch_definitions = get_uniform_patches(sequence_length, self.patch_size, self.patch_stride)
        return patch_definitions
