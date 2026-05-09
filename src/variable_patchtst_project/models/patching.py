def get_perfect_patches(seq_len, num_patches, min_patch_size, num_scales):
    """
    Calculates patch boundaries so that the entire seq_len is covered
    using exactly num_patches, with resolution decaying into the past.
    """
    patch_sizes = [min_patch_size * (2**i) for i in range(num_scales)]
    
    # Distribute tokens (num_patches) across the scales
    # We'll use a simple even split, adjusting the last one for the remainder
    tokens_per_scale = [num_patches // num_scales] * num_scales
    tokens_per_scale[-1] += num_patches % num_scales
    
    # Determine how much "territory" each scale covers.
    # We want smaller patches to cover less total time, but at higher density.
    # A simple heuristic: territory is proportional to (count * size) 
    # but we normalize so the sum of territories == seq_len.
    raw_weights = [count * size for count, size in zip(tokens_per_scale, patch_sizes)]
    total_weight = sum(raw_weights)
    territory_sizes = [int((w / total_weight) * seq_len) for w in raw_weights]
    
    # Adjust for rounding errors to ensure sum(territory_sizes) == seq_len
    territory_sizes[-1] += seq_len - sum(territory_sizes)

    patch_definitions = []
    current_offset = 0 # Starting from the past (index 0) toward the present
    
    # Calculate Strides for each territory
    # Territory = (Number of Patches - 1) * Stride + Patch_Size
    # Therefore: Stride = (Territory - Patch_Size) / (Number of Patches - 1)
    for i in range(num_scales):
        P = patch_sizes[i]
        N = tokens_per_scale[i]
        T = territory_sizes[i]

        if N > 1:
            stride = (T - P) / (N - 1)
        else:
            stride = 0  # Only one patch, it just sits in the territory

        for j in range(N):
            start = int(current_offset + j * stride)
            end = start + P
            # Keep [start, end) of length P inside [0, seq_len] so d0:d1 slices are valid
            if end > seq_len:
                end = seq_len
                start = end - P
            if start < 0:
                start = 0
                end = min(P, seq_len)
            d0 = seq_len - end
            d1 = seq_len - start
            # Third entry must match true slice length (may be < P if seq_len < P)
            patch_definitions.append((d0, d1, d1 - d0))

        current_offset += T
    
    patch_definitions.reverse()

    for start, end, P in patch_definitions:
        assert 0 <= start < end <= seq_len, (start, end, seq_len)
        assert end - start == P, (start, end, P)

    return patch_definitions

def get_uniform_patches(seq_len, patch_size, stride):
    patch_definitions = []
    start = 0
    end = patch_size
    while end <= seq_len:
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
