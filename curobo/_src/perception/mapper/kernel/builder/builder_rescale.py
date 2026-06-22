# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Post-frame per-block accumulator rescale kernel, per-``block_size`` builder.

The fp16 per-block accumulators (``block_rgb``, ``block_features``,
``block_feature_weight``) grow monotonically across frames as atomic
adds accumulate weighted pixel contributions. Left unbounded, the
running sum eventually exceeds fp16's finite range (65504) and
saturates to ``inf``. This builder adds a post-integration pass that
caps each per-block weight at ``w_max`` and scales the weighted-sum
channels proportionally so the mean ``sum / weight`` is preserved
while magnitudes stay bounded.

The cap also gives the mapper EMA semantics: old observations decay at
a rate set by ``w_max / mean_per_frame_weight``. This is desirable for
dynamic scenes but should be called out in the caller's config
docstring so ``w_max`` can be picked deliberately.

The kernel is not BS-sensitive (one thread per (visible block, channel)
pair, independent of voxel count per block), but lives inside the
per-BS builder for launch-site locality with the integration builder; both
fire inside the per-frame integration pipeline.
"""

from __future__ import annotations

import warp as wp

from curobo._src.util.warp import warp_kernel


def make_rescale_kernels(
    block_size: int,
    *,
    feature_dim: int,
    use_color_grid: bool = False,
    color_grid_size: int = 1,
) -> dict[str, object]:
    """Build per-block accumulator rescale kernels."""
    FEATURE_DIM = wp.constant(wp.int32(feature_dim))
    USE_COLOR_GRID = wp.constant(bool(use_color_grid))
    color_grid_voxels = int(color_grid_size) ** 3
    COLOR_GRID_VOXELS = wp.constant(wp.int32(color_grid_voxels))
    COLOR_GRID_RGB_CELLS = wp.constant(wp.int32(color_grid_voxels * 3))

    @warp_kernel(f"rescale_block_accumulators_kernel_bs{block_size}_fd{feature_dim}")
    def rescale_block_accumulators_kernel(
        visible_pool_indices: wp.array(dtype=wp.int32),
        n_visible: wp.int32,
        w_max: wp.float32,
        block_features: wp.array2d(dtype=wp.float16),
        block_feature_weight: wp.array(dtype=wp.float16),
        block_rgb: wp.array2d(dtype=wp.float16),
    ):
        """Cap per-block weights at ``w_max``; scale sums proportionally.

        Launch with ``dim = (n_visible, n_channels)`` where
        ``n_channels = max(3, feature_dim)`` — one thread per
        ``(visible_block, channel)`` pair. Within a warp threads
        share ``pool_idx`` and stride consecutive ``ch`` slots, so
        ``block_features[pool_idx, ch]`` loads are coalesced and
        the per-block weights ``w_rgb`` / ``w_f`` broadcast.

        ``block_rgb`` and ``block_features`` track independent
        weights (the feature kernel aggregates over a footprint bbox
        while RGB uses per-voxel pixel coverage), so cap each
        independently.
        """
        vis_idx, ch = wp.tid()

        if vis_idx >= n_visible:
            return

        pool_idx = visible_pool_indices[vis_idx]
        if pool_idx < 0:
            return
        if FEATURE_DIM > 0 and ch < FEATURE_DIM:
            w_f = wp.float32(block_feature_weight[pool_idx])
            if w_f > w_max:
                s_f = w_max / w_f
                v_f = wp.float32(block_features[pool_idx, ch]) * s_f
                block_features[pool_idx, ch] = wp.float16(v_f)
                if ch == 0:
                    block_feature_weight[pool_idx] = wp.float16(w_max)

        if ch < 3:
            rgb_weight = wp.float32(block_rgb[pool_idx, 3])
            if rgb_weight > w_max:
                current_rgb = wp.float32(block_rgb[pool_idx, ch])
                s = w_max / rgb_weight
                scaled_rgb = current_rgb * s
                block_rgb[pool_idx, ch] = wp.float16(scaled_rgb)
                if ch == 0:
                    block_rgb[pool_idx, 3] = wp.float16(w_max)

    @warp_kernel(
        f"rescale_block_grid_rgb_kernel_bs{block_size}_cg{int(use_color_grid)}_gs{color_grid_size}"
    )
    def rescale_block_grid_rgb_kernel(
        visible_pool_indices: wp.array(dtype=wp.int32),
        n_visible: wp.int32,
        w_max: wp.float32,
        block_grid_rgb: wp.array3d(dtype=wp.float16),
    ):
        """Cap per-node RGB grid weights at ``w_max`` while preserving means."""
        vis_idx, cell_idx = wp.tid()
        if vis_idx >= n_visible or cell_idx >= COLOR_GRID_RGB_CELLS:
            return
        if not USE_COLOR_GRID:
            return

        pool_idx = visible_pool_indices[vis_idx]
        if pool_idx < 0:
            return

        node_idx = cell_idx // wp.int32(3)
        ch = cell_idx - node_idx * wp.int32(3)
        if node_idx >= COLOR_GRID_VOXELS:
            return

        rgb_weight = wp.float32(block_grid_rgb[pool_idx, node_idx, 3])
        if rgb_weight > w_max:
            current_rgb = wp.float32(block_grid_rgb[pool_idx, node_idx, ch])
            s = w_max / rgb_weight
            block_grid_rgb[pool_idx, node_idx, ch] = wp.float16(current_rgb * s)
            if ch == wp.int32(0):
                block_grid_rgb[pool_idx, node_idx, 3] = wp.float16(w_max)

    return {
        "rescale_block_accumulators_kernel": rescale_block_accumulators_kernel,
        "rescale_block_grid_rgb_kernel": rescale_block_grid_rgb_kernel,
    }
