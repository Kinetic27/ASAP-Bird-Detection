import math


def get_optimal_stride(dim_size, patch_size, min_overlap):
    if dim_size <= patch_size:
        return 0.0
    max_stride = patch_size - min_overlap
    patch_count = math.ceil((dim_size - patch_size) / max_stride) + 1
    return (dim_size - patch_size) / (patch_count - 1)


def build_patch_offsets(width, height, patch_size, min_overlap):
    stride_x = get_optimal_stride(width, patch_size, min_overlap)
    stride_y = get_optimal_stride(height, patch_size, min_overlap)

    y_coords = (
        [int(i * stride_y) for i in range(math.ceil((height - patch_size) / stride_y) + 1)]
        if stride_y > 0
        else [0]
    )
    x_coords = (
        [int(i * stride_x) for i in range(math.ceil((width - patch_size) / stride_x) + 1)]
        if stride_x > 0
        else [0]
    )

    offsets = []
    for y in y_coords:
        for x in x_coords:
            x_end = min(x + patch_size, width)
            y_end = min(y + patch_size, height)
            x_start = max(0, x_end - patch_size)
            y_start = max(0, y_end - patch_size)
            offsets.append((x_start, y_start))

    return offsets


def build_patch_windows(width, height, patch_size, min_overlap):
    return [
        (x_start, y_start, x_start + patch_size, y_start + patch_size)
        for x_start, y_start in build_patch_offsets(
            width, height, patch_size, min_overlap
        )
    ]
