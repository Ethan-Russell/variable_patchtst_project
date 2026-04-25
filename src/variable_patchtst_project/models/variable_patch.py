import numpy as np

def make_variable_patch_matrix(
        sequence_length: int, 
        min_patch_size: int, 
        rate: float, 
        patch_overlap: float,
        alg = "linear"
    ):
    """
    Generates a matrix where each column represents a patch over a sequence.
    The patch size grows dynamically based on the chosen algorithm.
    """
    
    # -------------------------------------------------------------------------
    # Validate Inputs
    # -------------------------------------------------------------------------

    if not isinstance(sequence_length, int) or sequence_length <= 1:
        raise ValueError(f"sequence_length must be an integer > 1, got {sequence_length}")
    
    if min_patch_size <= 0 or min_patch_size > sequence_length:
        raise ValueError(f"min_patch_size must be > 0 and <= sequence_length, got {min_patch_size}")

    if rate <= 0:
        raise ValueError(f"rate must be > 0, got {rate}")

    if not (0 <= patch_overlap < 1):
        raise ValueError(f"patch_overlap must be in range [0, 1), got {patch_overlap}")

    valid_algs = ["linear", "lin", "exponential", "exp"]
    if alg not in valid_algs:
        raise ValueError(f"alg must be one of {valid_algs}, got '{alg}'")

    # -------------------------------------------------------------------------
    # Patch Generation Logic
    # -------------------------------------------------------------------------
    
    patch_list = []
    i_start = 0
    i_end = 0
    patch_length = -1
    t = 0
    while True:
        # Initialize empty patch
        patch = np.zeros(sequence_length)

        # Calculate the patch length based on the growth algorithm
        if alg == "linear" or alg == "lin":
            target_length = min_patch_size + t * rate
        elif alg == "exponential" or alg == "exp":
            target_length = min_patch_size * np.exp(rate*t)
            
        patch_length_new = int(np.floor(target_length))
        
        # Calculate the starting index using the overlap percentage.
        # (1-patch_overlap) determines how much "new" sequence space we jump.
        target_start = i_start + (i_end - i_start) * (1-patch_overlap)
        i_start_new = int(np.floor(target_start))
        
        # Check for edge case to prevent infinite loops or duplicate patches 
        # if the growth/overlap logic results in the same window.
        if i_start_new == i_start and patch_length_new == patch_length:
            i_start_new = i_start_new + 1
        
        # Assign the start and end index
        i_start = i_start_new
        patch_length = patch_length_new
        # Ensure we don't exceed the sequence boundaries
        i_end = min(i_start + patch_length, sequence_length)

        # Create patch with normalized weights (1/L)
        patch[i_start:i_end] = 1/patch_length
        patch_list.append(patch)

        # Exit if we've reached the end of the sequence or hit a safety limit
        if i_end == sequence_length or t > sequence_length:
            break

        t = t+1

    # Convert to 2D array and adjust axes for model compatibility
    patch_array = np.array(patch_list)[::-1, ::-1].T

    # print(f"Patch array for min_patch_size={min_patch_size}, sequence_length={sequence_length}, rate={rate}, patch_overlap={patch_overlap}, alg={alg}")
    # print(patch_array)
    return patch_array