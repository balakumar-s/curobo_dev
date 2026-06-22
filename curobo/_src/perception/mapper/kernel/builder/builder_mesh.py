# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Block-sparse marching-cubes mesh extraction, per-``block_size`` builder.

Moved from :mod:`curobo._src.perception.mapper.mesh_extractor` in the
block-size builder refactor.

All 7 kernels and 5 ``@wp.func`` helpers are BS-sensitive: cube and
edge indexing uses ``BS``-based packing (``local_idx = lz*BS^2 +
ly*BS + lx``), and block-boundary crossing logic tests against ``BS``
directly. Every kernel closure-captures ``BS = wp.constant(block_size)``.
"""

from __future__ import annotations

import warp as wp

from curobo._src.perception.mapper.kernel.warp_types import BlockSparseTSDFWarp
from curobo._src.perception.mapper.marching_cubes.kernel.wp_mc_common import (
    binary_search_int64,
    get_edge_vertex,
    interpolate_edge_vertex,
    local_edge_to_array_idx,
)
from curobo._src.util.warp import warp_func, warp_kernel


def make_mesh_kernels(
    block_size: int,
    *,
    num_cameras: int = 1,
    image_height: int = 1,
    image_width: int = 1,
    hash_lookup,
    sample_rgb,
    sample_voxel,
    sample_tsdf_trilinear,
    compute_gradient,
    compute_gradient_nearest,
    block_grid_to_key_coords,
    block_key_to_voxel_base,
) -> dict[str, object]:
    """Build block-sparse marching-cubes kernels."""
    BS = wp.constant(block_size)
    NUM_CAMERAS = wp.constant(num_cameras)
    IMAGE_HEIGHT = wp.constant(image_height)
    IMAGE_WIDTH = wp.constant(image_width)

    # Cross-domain helpers are explicit parameters so Warp sees them as
    # local closure bindings when compiling dependent functions.

    # =====================================================================
    # Vertex refinement
    # =====================================================================

    @warp_func(f"refine_vertex_mesh_bs{block_size}")
    def refine_vertex_mesh(
        tsdf: BlockSparseTSDFWarp,
        vertex: wp.vec3,
        level: wp.float32,
        iterations: wp.int32,
        minimum_tsdf_weight: wp.float32,
    ) -> wp.vec3:
        """Refine vertex to true SDF zero-crossing via Newton-Raphson."""
        pos = vertex
        for _ in range(iterations):
            result = sample_tsdf_trilinear(tsdf, pos, minimum_tsdf_weight)
            if result[1] < 0.5:
                break
            sdf_val = result[0] - level
            if wp.abs(sdf_val) < 1e-6 or sdf_val > 100.0:
                break

            grad = compute_gradient(tsdf, pos, minimum_tsdf_weight)
            grad_mag = wp.sqrt(wp.dot(grad, grad))
            if grad_mag < 1e-4:
                break

            step_size = wp.clamp(
                sdf_val / grad_mag,
                -tsdf.voxel_size * 0.5,
                tsdf.voxel_size * 0.5,
            )
            pos = pos - step_size * (grad / grad_mag)
        return pos

    # =====================================================================
    # SDF access (BS-sensitive: local_idx packing)
    # =====================================================================

    @warp_func(f"get_block_sdf_bs{block_size}")
    def get_block_sdf(
        tsdf: BlockSparseTSDFWarp,
        pool_idx: wp.int32,
        lx: wp.int32,
        ly: wp.int32,
        lz: wp.int32,
        level: float,
        minimum_tsdf_weight: float,
    ) -> wp.vec2:
        """Get combined SDF value at (pool_idx, lx, ly, lz)."""
        local_idx = lz * BS * BS + ly * BS + lx
        result = sample_voxel(tsdf, pool_idx, local_idx, minimum_tsdf_weight)
        if result[1] < 0.5:
            return wp.vec2(1e10, 0.0)
        return wp.vec2(result[0] - level, 1.0)

    @warp_func(f"sample_cube_corner_bs{block_size}")
    def sample_cube_corner(
        cx: wp.int32,
        cy: wp.int32,
        cz: wp.int32,
        bx: wp.int32,
        by: wp.int32,
        bz: wp.int32,
        pool_idx: wp.int32,
        tsdf: BlockSparseTSDFWarp,
        level: float,
        minimum_tsdf_weight: float,
    ) -> wp.vec2:
        """Sample combined SDF at cube corner, handling block boundary crossing."""
        if cx < BS and cy < BS and cz < BS:
            return get_block_sdf(tsdf, pool_idx, cx, cy, cz, level, minimum_tsdf_weight)

        nbx = bx
        nby = by
        nbz = bz
        nlx = cx
        nly = cy
        nlz = cz

        if cx >= BS:
            nbx = bx + 1
            nlx = 0
        if cy >= BS:
            nby = by + 1
            nly = 0
        if cz >= BS:
            nbz = bz + 1
            nlz = 0

        neighbor_idx = hash_lookup(tsdf.hash_table, nbx, nby, nbz, tsdf.hash_capacity)
        if neighbor_idx < 0:
            return wp.vec2(1e10, 0.0)

        return get_block_sdf(tsdf, neighbor_idx, nlx, nly, nlz, level, minimum_tsdf_weight)

    # =====================================================================
    # Surface-cube predicate
    # =====================================================================

    @warp_func(f"is_surface_cube_combined_bs{block_size}")
    def is_surface_cube_combined(
        cx: wp.int32,
        cy: wp.int32,
        cz: wp.int32,
        bx: wp.int32,
        by: wp.int32,
        bz: wp.int32,
        block_idx: wp.int32,
        tsdf: BlockSparseTSDFWarp,
        level: float,
        surface_band: float,
        minimum_tsdf_weight: float,
    ) -> wp.bool:
        """Check if a cube contains a surface (sign change across corners)."""
        s0 = sample_cube_corner(
            cx, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s1 = sample_cube_corner(
            cx + 1, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s2 = sample_cube_corner(
            cx + 1, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s3 = sample_cube_corner(
            cx, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s4 = sample_cube_corner(
            cx, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s5 = sample_cube_corner(
            cx + 1, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s6 = sample_cube_corner(
            cx + 1, cy + 1, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s7 = sample_cube_corner(
            cx, cy + 1, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )

        if s0[1] < 0.5 or s1[1] < 0.5 or s2[1] < 0.5 or s3[1] < 0.5:
            return False
        if s4[1] < 0.5 or s5[1] < 0.5 or s6[1] < 0.5 or s7[1] < 0.5:
            return False

        has_positive = (
            s0[0] > 0.0
            or s1[0] > 0.0
            or s2[0] > 0.0
            or s3[0] > 0.0
            or s4[0] > 0.0
            or s5[0] > 0.0
            or s6[0] > 0.0
            or s7[0] > 0.0
        )
        has_negative = (
            s0[0] < 0.0
            or s1[0] < 0.0
            or s2[0] < 0.0
            or s3[0] < 0.0
            or s4[0] < 0.0
            or s5[0] < 0.0
            or s6[0] < 0.0
            or s7[0] < 0.0
        )
        if not (has_positive and has_negative):
            return False

        if surface_band > 0.0:
            in_band = (
                wp.abs(s0[0]) < surface_band
                or wp.abs(s1[0]) < surface_band
                or wp.abs(s2[0]) < surface_band
                or wp.abs(s3[0]) < surface_band
                or wp.abs(s4[0]) < surface_band
                or wp.abs(s5[0]) < surface_band
                or wp.abs(s6[0]) < surface_band
                or wp.abs(s7[0]) < surface_band
            )
            if not in_band:
                return False

        return True

    # =====================================================================
    # Edge ownership
    # =====================================================================

    @warp_func(f"get_edge_owner_combined_bs{block_size}")
    def get_edge_owner_combined(
        edge: wp.int32,
        block_idx: wp.int32,
        cx: wp.int32,
        cy: wp.int32,
        cz: wp.int32,
        bx: wp.int32,
        by: wp.int32,
        bz: wp.int32,
        tsdf: BlockSparseTSDFWarp,
        edge_owner: wp.array(dtype=wp.int32),
    ) -> wp.vec3i:
        """Find which cube owns an edge, handling block boundaries."""
        owner_dx = edge_owner[edge * 4 + 2]
        owner_dy = edge_owner[edge * 4 + 1]
        owner_dz = edge_owner[edge * 4 + 0]
        local_edge = edge_owner[edge * 4 + 3]

        new_cx = cx + owner_dx
        new_cy = cy + owner_dy
        new_cz = cz + owner_dz

        if new_cx < BS and new_cy < BS and new_cz < BS:
            owner_cube = new_cz * BS * BS + new_cy * BS + new_cx
            return wp.vec3i(block_idx, owner_cube, local_edge)

        new_bx = bx
        new_by = by
        new_bz = bz

        if new_cx >= BS:
            new_bx = bx + 1
            new_cx = 0
        if new_cy >= BS:
            new_by = by + 1
            new_cy = 0
        if new_cz >= BS:
            new_bz = bz + 1
            new_cz = 0

        neighbor_idx = hash_lookup(tsdf.hash_table, new_bx, new_by, new_bz, tsdf.hash_capacity)
        if neighbor_idx < 0:
            return wp.vec3i(-1, -1, local_edge)

        owner_cube = new_cz * BS * BS + new_cy * BS + new_cx
        return wp.vec3i(neighbor_idx, owner_cube, local_edge)

    # =====================================================================
    # Surface detection kernels
    # =====================================================================

    @warp_kernel(f"count_surface_cubes_kernel_bs{block_size}", enable_backward=False)
    def count_surface_cubes_kernel(
        tsdf: BlockSparseTSDFWarp,
        level: float,
        surface_band: float,
        minimum_tsdf_weight: float,
        surface_count: wp.array(dtype=wp.int32),
    ):
        """Count cubes that contain a surface (pass 1).

        Launch with ``dim = (num_allocated, BS ** 3)``.
        """
        block_idx, cube_idx = wp.tid()

        if block_idx >= tsdf.num_allocated[0]:
            return
        if tsdf.block_to_hash_slot[block_idx] < 0:
            return

        cx = cube_idx % BS
        cy = (cube_idx // BS) % BS
        cz = cube_idx // (BS * BS)

        bx = tsdf.block_coords[block_idx * 3 + 0]
        by = tsdf.block_coords[block_idx * 3 + 1]
        bz = tsdf.block_coords[block_idx * 3 + 2]

        if is_surface_cube_combined(
            cx,
            cy,
            cz,
            bx,
            by,
            bz,
            block_idx,
            tsdf,
            level,
            surface_band,
            minimum_tsdf_weight,
        ):
            wp.atomic_add(surface_count, 0, wp.int32(1))

    @warp_kernel(f"append_active_blocks_kernel_bs{block_size}", enable_backward=False)
    def append_active_blocks_kernel(
        tsdf: BlockSparseTSDFWarp,
        active_count: wp.array(dtype=wp.int32),
        active_block_idx: wp.array(dtype=wp.int32),
    ):
        """Compact active pool indices into a dense block list."""
        block_idx = wp.tid()

        if block_idx >= tsdf.num_allocated[0]:
            return
        if tsdf.block_to_hash_slot[block_idx] < 0:
            return

        out_idx = wp.atomic_add(active_count, 0, wp.int32(1))
        active_block_idx[out_idx] = block_idx

    @warp_kernel(
        f"count_surface_cubes_from_blocks_kernel_bs{block_size}", enable_backward=False
    )
    def count_surface_cubes_from_blocks_kernel(
        tsdf: BlockSparseTSDFWarp,
        active_block_idx: wp.array(dtype=wp.int32),
        n_active_blocks: wp.int32,
        level: float,
        surface_band: float,
        minimum_tsdf_weight: float,
        surface_count: wp.array(dtype=wp.int32),
    ):
        """Count surface cubes from a compact active block list."""
        active_idx, cube_idx = wp.tid()

        if active_idx >= n_active_blocks:
            return

        block_idx = active_block_idx[active_idx]
        if block_idx < 0 or block_idx >= tsdf.num_allocated[0]:
            return
        if tsdf.block_to_hash_slot[block_idx] < 0:
            return

        cx = cube_idx % BS
        cy = (cube_idx // BS) % BS
        cz = cube_idx // (BS * BS)

        bx = tsdf.block_coords[block_idx * 3 + 0]
        by = tsdf.block_coords[block_idx * 3 + 1]
        bz = tsdf.block_coords[block_idx * 3 + 2]

        if is_surface_cube_combined(
            cx,
            cy,
            cz,
            bx,
            by,
            bz,
            block_idx,
            tsdf,
            level,
            surface_band,
            minimum_tsdf_weight,
        ):
            wp.atomic_add(surface_count, 0, wp.int32(1))

    @warp_kernel(f"append_surface_cubes_kernel_bs{block_size}", enable_backward=False)
    def append_surface_cubes_kernel(
        tsdf: BlockSparseTSDFWarp,
        level: float,
        surface_band: float,
        minimum_tsdf_weight: float,
        surface_count: wp.array(dtype=wp.int32),
        surface_block_idx: wp.array(dtype=wp.int32),
        surface_cube_idx: wp.array(dtype=wp.int32),
    ):
        """Append surface cubes to output arrays (pass 2)."""
        block_idx, cube_idx = wp.tid()

        if block_idx >= tsdf.num_allocated[0]:
            return
        if tsdf.block_to_hash_slot[block_idx] < 0:
            return

        cx = cube_idx % BS
        cy = (cube_idx // BS) % BS
        cz = cube_idx // (BS * BS)

        bx = tsdf.block_coords[block_idx * 3 + 0]
        by = tsdf.block_coords[block_idx * 3 + 1]
        bz = tsdf.block_coords[block_idx * 3 + 2]

        if is_surface_cube_combined(
            cx,
            cy,
            cz,
            bx,
            by,
            bz,
            block_idx,
            tsdf,
            level,
            surface_band,
            minimum_tsdf_weight,
        ):
            out_idx = wp.atomic_add(surface_count, 0, wp.int32(1))
            surface_block_idx[out_idx] = block_idx
            surface_cube_idx[out_idx] = cube_idx

    @warp_kernel(
        f"append_surface_cubes_from_blocks_kernel_bs{block_size}", enable_backward=False
    )
    def append_surface_cubes_from_blocks_kernel(
        tsdf: BlockSparseTSDFWarp,
        active_block_idx: wp.array(dtype=wp.int32),
        n_active_blocks: wp.int32,
        level: float,
        surface_band: float,
        minimum_tsdf_weight: float,
        surface_count: wp.array(dtype=wp.int32),
        surface_block_idx: wp.array(dtype=wp.int32),
        surface_cube_idx: wp.array(dtype=wp.int32),
    ):
        """Append surface cubes from a compact active block list."""
        active_idx, cube_idx = wp.tid()

        if active_idx >= n_active_blocks:
            return

        block_idx = active_block_idx[active_idx]
        if block_idx < 0 or block_idx >= tsdf.num_allocated[0]:
            return
        if tsdf.block_to_hash_slot[block_idx] < 0:
            return

        cx = cube_idx % BS
        cy = (cube_idx // BS) % BS
        cz = cube_idx // (BS * BS)

        bx = tsdf.block_coords[block_idx * 3 + 0]
        by = tsdf.block_coords[block_idx * 3 + 1]
        bz = tsdf.block_coords[block_idx * 3 + 2]

        if is_surface_cube_combined(
            cx,
            cy,
            cz,
            bx,
            by,
            bz,
            block_idx,
            tsdf,
            level,
            surface_band,
            minimum_tsdf_weight,
        ):
            out_idx = wp.atomic_add(surface_count, 0, wp.int32(1))
            surface_block_idx[out_idx] = block_idx
            surface_cube_idx[out_idx] = cube_idx

    # =====================================================================
    # Edge counting + vertex generation
    # =====================================================================

    @warp_kernel(f"count_edges_block_sparse_kernel_bs{block_size}", enable_backward=False)
    def count_edges_block_sparse_kernel(
        tsdf: BlockSparseTSDFWarp,
        level: float,
        minimum_tsdf_weight: float,
        surface_block_idx: wp.array(dtype=wp.int32),
        surface_cube_idx: wp.array(dtype=wp.int32),
        n_surfaces: wp.int32,
        edge_counts: wp.array(dtype=wp.int32),
    ):
        """Count owned edges (0, 3, 8) for each surface cube."""
        tid = wp.tid()
        if tid >= n_surfaces:
            return

        block_idx = surface_block_idx[tid]
        cube_idx = surface_cube_idx[tid]

        cx = cube_idx % BS
        cy = (cube_idx // BS) % BS
        cz = cube_idx // (BS * BS)

        bx = tsdf.block_coords[block_idx * 3 + 0]
        by = tsdf.block_coords[block_idx * 3 + 1]
        bz = tsdf.block_coords[block_idx * 3 + 2]

        s0 = sample_cube_corner(
            cx, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s1 = sample_cube_corner(
            cx + 1, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s3 = sample_cube_corner(
            cx, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s4 = sample_cube_corner(
            cx, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )

        count = wp.int32(0)
        if s0[0] * s1[0] < 0.0:
            count += wp.int32(1)
        if s0[0] * s3[0] < 0.0:
            count += wp.int32(1)
        if s0[0] * s4[0] < 0.0:
            count += wp.int32(1)

        edge_counts[tid] = count

    @warp_kernel(f"generate_vertices_block_sparse_kernel_bs{block_size}", enable_backward=False)
    def generate_vertices_block_sparse_kernel(
        tsdf: BlockSparseTSDFWarp,
        level: float,
        minimum_tsdf_weight: float,
        surface_block_idx: wp.array(dtype=wp.int32),
        surface_cube_idx: wp.array(dtype=wp.int32),
        edge_offsets: wp.array(dtype=wp.int32),
        n_surfaces: wp.int32,
        refine_iterations: wp.int32,
        vertices: wp.array(dtype=wp.vec3),
        normals: wp.array(dtype=wp.vec3),
        edge_vertex_indices: wp.array(dtype=wp.int32),
    ):
        """Generate vertices + normals for owned edges."""
        tid = wp.tid()
        if tid >= n_surfaces:
            return

        block_idx = surface_block_idx[tid]
        cube_idx = surface_cube_idx[tid]

        cx = cube_idx % BS
        cy = (cube_idx // BS) % BS
        cz = cube_idx // (BS * BS)

        bx = tsdf.block_coords[block_idx * 3 + 0]
        by = tsdf.block_coords[block_idx * 3 + 1]
        bz = tsdf.block_coords[block_idx * 3 + 2]

        s0 = sample_cube_corner(
            cx, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s1 = sample_cube_corner(
            cx + 1, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s3 = sample_cube_corner(
            cx, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s4 = sample_cube_corner(
            cx, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )

        base = block_key_to_voxel_base(bx, by, bz)
        gx = base[0] + cx
        gy = base[1] + cy
        gz = base[2] + cz

        center_offset_x = wp.float32(tsdf.grid_W) * 0.5
        center_offset_y = wp.float32(tsdf.grid_H) * 0.5
        center_offset_z = wp.float32(tsdf.grid_D) * 0.5

        p0 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx) - center_offset_x,
                wp.float32(gy) - center_offset_y,
                wp.float32(gz) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p1 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx + 1) - center_offset_x,
                wp.float32(gy) - center_offset_y,
                wp.float32(gz) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p3 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx) - center_offset_x,
                wp.float32(gy + 1) - center_offset_y,
                wp.float32(gz) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p4 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx) - center_offset_x,
                wp.float32(gy) - center_offset_y,
                wp.float32(gz + 1) - center_offset_z,
            )
            * tsdf.voxel_size
        )

        vertex_idx = edge_offsets[tid]

        edge_vertex_indices[tid * 3 + 0] = wp.int32(-1)
        edge_vertex_indices[tid * 3 + 1] = wp.int32(-1)
        edge_vertex_indices[tid * 3 + 2] = wp.int32(-1)

        if s0[0] * s1[0] < 0.0:
            v = interpolate_edge_vertex(p0, p1, s0[0], s1[0])
            if refine_iterations > 0:
                v = refine_vertex_mesh(tsdf, v, level, refine_iterations, minimum_tsdf_weight)
            n = compute_gradient_nearest(tsdf, v, minimum_tsdf_weight)
            vertices[vertex_idx] = v
            normals[vertex_idx] = wp.normalize(n)
            edge_vertex_indices[tid * 3 + 0] = vertex_idx
            vertex_idx += wp.int32(1)

        if s0[0] * s3[0] < 0.0:
            v = interpolate_edge_vertex(p0, p3, s0[0], s3[0])
            if refine_iterations > 0:
                v = refine_vertex_mesh(tsdf, v, level, refine_iterations, minimum_tsdf_weight)
            n = compute_gradient_nearest(tsdf, v, minimum_tsdf_weight)
            vertices[vertex_idx] = v
            normals[vertex_idx] = wp.normalize(n)
            edge_vertex_indices[tid * 3 + 1] = vertex_idx
            vertex_idx += wp.int32(1)

        if s0[0] * s4[0] < 0.0:
            v = interpolate_edge_vertex(p0, p4, s0[0], s4[0])
            if refine_iterations > 0:
                v = refine_vertex_mesh(tsdf, v, level, refine_iterations, minimum_tsdf_weight)
            n = compute_gradient_nearest(tsdf, v, minimum_tsdf_weight)
            vertices[vertex_idx] = v
            normals[vertex_idx] = wp.normalize(n)
            edge_vertex_indices[tid * 3 + 2] = vertex_idx

    # =====================================================================
    # Triangle generation / counting
    # =====================================================================

    @warp_kernel(f"generate_triangles_shared_kernel_bs{block_size}")
    def generate_triangles_shared_kernel(
        tsdf: BlockSparseTSDFWarp,
        level: float,
        minimum_tsdf_weight: float,
        surface_block_idx: wp.array(dtype=wp.int32),
        surface_cube_idx: wp.array(dtype=wp.int32),
        tri_offsets: wp.array(dtype=wp.int32),
        n_surfaces: wp.int32,
        tri_table: wp.array(dtype=wp.int32),
        edge_owner: wp.array(dtype=wp.int32),
        sorted_global_ids: wp.array(dtype=wp.int64),
        sparse_indices_sorted: wp.array(dtype=wp.int32),
        edge_vertex_indices: wp.array(dtype=wp.int32),
        triangles: wp.array(dtype=wp.int32),
    ):
        """Generate triangles with shared vertex lookup."""
        tid = wp.tid()
        if tid >= n_surfaces:
            return

        block_idx = surface_block_idx[tid]
        cube_idx = surface_cube_idx[tid]

        cx = cube_idx % BS
        cy = (cube_idx // BS) % BS
        cz = cube_idx // (BS * BS)

        bx = tsdf.block_coords[block_idx * 3 + 0]
        by = tsdf.block_coords[block_idx * 3 + 1]
        bz = tsdf.block_coords[block_idx * 3 + 2]

        s0 = sample_cube_corner(
            cx, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s1 = sample_cube_corner(
            cx + 1, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s2 = sample_cube_corner(
            cx + 1, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s3 = sample_cube_corner(
            cx, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s4 = sample_cube_corner(
            cx, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s5 = sample_cube_corner(
            cx + 1, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s6 = sample_cube_corner(
            cx + 1, cy + 1, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s7 = sample_cube_corner(
            cx, cy + 1, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )

        cube_config = wp.int32(0)
        if s0[0] < 0.0:
            cube_config = cube_config | wp.int32(1)
        if s1[0] < 0.0:
            cube_config = cube_config | wp.int32(2)
        if s2[0] < 0.0:
            cube_config = cube_config | wp.int32(4)
        if s3[0] < 0.0:
            cube_config = cube_config | wp.int32(8)
        if s4[0] < 0.0:
            cube_config = cube_config | wp.int32(16)
        if s5[0] < 0.0:
            cube_config = cube_config | wp.int32(32)
        if s6[0] < 0.0:
            cube_config = cube_config | wp.int32(64)
        if s7[0] < 0.0:
            cube_config = cube_config | wp.int32(128)

        tri_base = tri_offsets[tid]
        table_offset = cube_config * 16

        for t in range(5):
            e0 = tri_table[table_offset + t * 3]
            if e0 < 0:
                break

            e1 = tri_table[table_offset + t * 3 + 1]
            e2 = tri_table[table_offset + t * 3 + 2]

            for v_idx in range(3):
                edge = e0
                if v_idx == 1:
                    edge = e1
                elif v_idx == 2:
                    edge = e2

                owner = get_edge_owner_combined(
                    edge,
                    block_idx,
                    cx,
                    cy,
                    cz,
                    bx,
                    by,
                    bz,
                    tsdf,
                    edge_owner,
                )

                vertex_id = wp.int32(-1)

                if owner[0] >= 0:
                    owner_global_id = wp.int64(owner[0]) * wp.int64(BS * BS * BS) + wp.int64(
                        owner[1]
                    )
                    search_idx = binary_search_int64(
                        sorted_global_ids,
                        n_surfaces,
                        owner_global_id,
                    )
                    if search_idx >= 0:
                        owner_sparse = sparse_indices_sorted[search_idx]
                        edge_array_idx = local_edge_to_array_idx(owner[2])
                        vertex_id = edge_vertex_indices[owner_sparse * 3 + edge_array_idx]

                triangles[(tri_base + t) * 3 + v_idx] = vertex_id

    @warp_kernel(f"count_triangles_kernel_bs{block_size}", enable_backward=False)
    def count_triangles_kernel(
        tsdf: BlockSparseTSDFWarp,
        level: float,
        minimum_tsdf_weight: float,
        surface_block_idx: wp.array(dtype=wp.int32),
        surface_cube_idx: wp.array(dtype=wp.int32),
        n_surfaces: wp.int32,
        num_tris_table: wp.array(dtype=wp.int32),
        tri_counts: wp.array(dtype=wp.int32),
    ):
        """Count triangles for each surface cube (combined SDF)."""
        tid = wp.tid()
        if tid >= n_surfaces:
            return

        block_idx = surface_block_idx[tid]
        cube_idx = surface_cube_idx[tid]

        cx = cube_idx % BS
        cy = (cube_idx // BS) % BS
        cz = cube_idx // (BS * BS)

        bx = tsdf.block_coords[block_idx * 3 + 0]
        by = tsdf.block_coords[block_idx * 3 + 1]
        bz = tsdf.block_coords[block_idx * 3 + 2]

        s0 = sample_cube_corner(
            cx, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s1 = sample_cube_corner(
            cx + 1, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s2 = sample_cube_corner(
            cx + 1, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s3 = sample_cube_corner(
            cx, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s4 = sample_cube_corner(
            cx, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s5 = sample_cube_corner(
            cx + 1, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s6 = sample_cube_corner(
            cx + 1, cy + 1, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s7 = sample_cube_corner(
            cx, cy + 1, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )

        cube_config = wp.int32(0)
        if s0[0] < 0.0:
            cube_config = cube_config | wp.int32(1)
        if s1[0] < 0.0:
            cube_config = cube_config | wp.int32(2)
        if s2[0] < 0.0:
            cube_config = cube_config | wp.int32(4)
        if s3[0] < 0.0:
            cube_config = cube_config | wp.int32(8)
        if s4[0] < 0.0:
            cube_config = cube_config | wp.int32(16)
        if s5[0] < 0.0:
            cube_config = cube_config | wp.int32(32)
        if s6[0] < 0.0:
            cube_config = cube_config | wp.int32(64)
        if s7[0] < 0.0:
            cube_config = cube_config | wp.int32(128)

        tri_counts[tid] = num_tris_table[cube_config]

    @warp_kernel(f"count_total_triangles_kernel_bs{block_size}", enable_backward=False)
    def count_total_triangles_kernel(
        tsdf: BlockSparseTSDFWarp,
        level: float,
        minimum_tsdf_weight: float,
        surface_block_idx: wp.array(dtype=wp.int32),
        surface_cube_idx: wp.array(dtype=wp.int32),
        n_surfaces: wp.int32,
        num_tris_table: wp.array(dtype=wp.int32),
        triangle_count: wp.array(dtype=wp.int32),
    ):
        """Count approximate mesh triangles with one device-side total."""
        tid = wp.tid()
        if tid >= n_surfaces:
            return

        block_idx = surface_block_idx[tid]
        cube_idx = surface_cube_idx[tid]

        cx = cube_idx % BS
        cy = (cube_idx // BS) % BS
        cz = cube_idx // (BS * BS)

        bx = tsdf.block_coords[block_idx * 3 + 0]
        by = tsdf.block_coords[block_idx * 3 + 1]
        bz = tsdf.block_coords[block_idx * 3 + 2]

        s0 = sample_cube_corner(
            cx, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s1 = sample_cube_corner(
            cx + 1, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s2 = sample_cube_corner(
            cx + 1, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s3 = sample_cube_corner(
            cx, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s4 = sample_cube_corner(
            cx, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s5 = sample_cube_corner(
            cx + 1, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s6 = sample_cube_corner(
            cx + 1, cy + 1, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s7 = sample_cube_corner(
            cx, cy + 1, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )

        cube_config = wp.int32(0)
        if s0[0] < 0.0:
            cube_config = cube_config | wp.int32(1)
        if s1[0] < 0.0:
            cube_config = cube_config | wp.int32(2)
        if s2[0] < 0.0:
            cube_config = cube_config | wp.int32(4)
        if s3[0] < 0.0:
            cube_config = cube_config | wp.int32(8)
        if s4[0] < 0.0:
            cube_config = cube_config | wp.int32(16)
        if s5[0] < 0.0:
            cube_config = cube_config | wp.int32(32)
        if s6[0] < 0.0:
            cube_config = cube_config | wp.int32(64)
        if s7[0] < 0.0:
            cube_config = cube_config | wp.int32(128)

        wp.atomic_add(triangle_count, 0, num_tris_table[cube_config])

    @warp_kernel(f"generate_approximate_mesh_block_sparse_kernel_bs{block_size}")
    def generate_approximate_mesh_block_sparse_kernel(
        tsdf: BlockSparseTSDFWarp,
        level: float,
        minimum_tsdf_weight: float,
        surface_block_idx: wp.array(dtype=wp.int32),
        surface_cube_idx: wp.array(dtype=wp.int32),
        tri_offsets: wp.array(dtype=wp.int32),
        n_surfaces: wp.int32,
        tri_table: wp.array(dtype=wp.int32),
        vertices: wp.array(dtype=wp.vec3),
        triangles: wp.array(dtype=wp.int32),
        normals: wp.array(dtype=wp.vec3),
        colors: wp.array(dtype=wp.vec3ub),
    ):
        """Generate a triangle-soup mesh without shared vertex ownership."""
        tid = wp.tid()
        if tid >= n_surfaces:
            return

        block_idx = surface_block_idx[tid]
        cube_idx = surface_cube_idx[tid]

        cx = cube_idx % BS
        cy = (cube_idx // BS) % BS
        cz = cube_idx // (BS * BS)

        bx = tsdf.block_coords[block_idx * 3 + 0]
        by = tsdf.block_coords[block_idx * 3 + 1]
        bz = tsdf.block_coords[block_idx * 3 + 2]

        s0 = sample_cube_corner(
            cx, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s1 = sample_cube_corner(
            cx + 1, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s2 = sample_cube_corner(
            cx + 1, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s3 = sample_cube_corner(
            cx, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s4 = sample_cube_corner(
            cx, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s5 = sample_cube_corner(
            cx + 1, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s6 = sample_cube_corner(
            cx + 1, cy + 1, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s7 = sample_cube_corner(
            cx, cy + 1, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )

        cube_config = wp.int32(0)
        if s0[0] < 0.0:
            cube_config = cube_config | wp.int32(1)
        if s1[0] < 0.0:
            cube_config = cube_config | wp.int32(2)
        if s2[0] < 0.0:
            cube_config = cube_config | wp.int32(4)
        if s3[0] < 0.0:
            cube_config = cube_config | wp.int32(8)
        if s4[0] < 0.0:
            cube_config = cube_config | wp.int32(16)
        if s5[0] < 0.0:
            cube_config = cube_config | wp.int32(32)
        if s6[0] < 0.0:
            cube_config = cube_config | wp.int32(64)
        if s7[0] < 0.0:
            cube_config = cube_config | wp.int32(128)

        base = block_key_to_voxel_base(bx, by, bz)
        gx = base[0] + cx
        gy = base[1] + cy
        gz = base[2] + cz

        center_offset_x = wp.float32(tsdf.grid_W) * 0.5
        center_offset_y = wp.float32(tsdf.grid_H) * 0.5
        center_offset_z = wp.float32(tsdf.grid_D) * 0.5

        p0 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx) - center_offset_x,
                wp.float32(gy) - center_offset_y,
                wp.float32(gz) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p1 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx + 1) - center_offset_x,
                wp.float32(gy) - center_offset_y,
                wp.float32(gz) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p2 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx + 1) - center_offset_x,
                wp.float32(gy + 1) - center_offset_y,
                wp.float32(gz) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p3 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx) - center_offset_x,
                wp.float32(gy + 1) - center_offset_y,
                wp.float32(gz) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p4 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx) - center_offset_x,
                wp.float32(gy) - center_offset_y,
                wp.float32(gz + 1) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p5 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx + 1) - center_offset_x,
                wp.float32(gy) - center_offset_y,
                wp.float32(gz + 1) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p6 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx + 1) - center_offset_x,
                wp.float32(gy + 1) - center_offset_y,
                wp.float32(gz + 1) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p7 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx) - center_offset_x,
                wp.float32(gy + 1) - center_offset_y,
                wp.float32(gz + 1) - center_offset_z,
            )
            * tsdf.voxel_size
        )

        rgb = sample_rgb(tsdf, (p0 + p6) * wp.float32(0.5))
        color = wp.vec3ub(wp.uint8(rgb[0]), wp.uint8(rgb[1]), wp.uint8(rgb[2]))

        tri_base = tri_offsets[tid]
        table_offset = cube_config * 16

        for t in range(5):
            e0 = tri_table[table_offset + t * 3]
            if e0 < 0:
                break

            e1 = tri_table[table_offset + t * 3 + 1]
            e2 = tri_table[table_offset + t * 3 + 2]

            out_base = (tri_base + t) * 3
            for v_idx in range(3):
                edge = e0
                if v_idx == 1:
                    edge = e1
                elif v_idx == 2:
                    edge = e2

                vertex_id = out_base + v_idx
                v = get_edge_vertex(
                    edge,
                    p0,
                    p1,
                    p2,
                    p3,
                    p4,
                    p5,
                    p6,
                    p7,
                    s0[0],
                    s1[0],
                    s2[0],
                    s3[0],
                    s4[0],
                    s5[0],
                    s6[0],
                    s7[0],
                )
                n = compute_gradient_nearest(tsdf, v, minimum_tsdf_weight)
                vertices[vertex_id] = v
                normals[vertex_id] = wp.normalize(n)
                colors[vertex_id] = color
                triangles[vertex_id] = vertex_id

    @warp_kernel(f"generate_approximate_mesh_atomic_kernel_bs{block_size}")
    def generate_approximate_mesh_atomic_kernel(
        tsdf: BlockSparseTSDFWarp,
        level: float,
        minimum_tsdf_weight: float,
        surface_block_idx: wp.array(dtype=wp.int32),
        surface_cube_idx: wp.array(dtype=wp.int32),
        n_surfaces: wp.int32,
        tri_table: wp.array(dtype=wp.int32),
        vertices: wp.array(dtype=wp.vec3),
        triangles: wp.array(dtype=wp.int32),
        normals: wp.array(dtype=wp.vec3),
        colors: wp.array(dtype=wp.vec3ub),
        triangle_count: wp.array(dtype=wp.int32),
        triangle_capacity: wp.int32,
    ):
        """Generate approximate triangle soup with atomic output allocation."""
        tid = wp.tid()
        if tid >= n_surfaces:
            return

        block_idx = surface_block_idx[tid]
        cube_idx = surface_cube_idx[tid]

        cx = cube_idx % BS
        cy = (cube_idx // BS) % BS
        cz = cube_idx // (BS * BS)

        bx = tsdf.block_coords[block_idx * 3 + 0]
        by = tsdf.block_coords[block_idx * 3 + 1]
        bz = tsdf.block_coords[block_idx * 3 + 2]

        s0 = sample_cube_corner(
            cx, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s1 = sample_cube_corner(
            cx + 1, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s2 = sample_cube_corner(
            cx + 1, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s3 = sample_cube_corner(
            cx, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s4 = sample_cube_corner(
            cx, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s5 = sample_cube_corner(
            cx + 1, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s6 = sample_cube_corner(
            cx + 1, cy + 1, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s7 = sample_cube_corner(
            cx, cy + 1, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )

        cube_config = wp.int32(0)
        if s0[0] < 0.0:
            cube_config = cube_config | wp.int32(1)
        if s1[0] < 0.0:
            cube_config = cube_config | wp.int32(2)
        if s2[0] < 0.0:
            cube_config = cube_config | wp.int32(4)
        if s3[0] < 0.0:
            cube_config = cube_config | wp.int32(8)
        if s4[0] < 0.0:
            cube_config = cube_config | wp.int32(16)
        if s5[0] < 0.0:
            cube_config = cube_config | wp.int32(32)
        if s6[0] < 0.0:
            cube_config = cube_config | wp.int32(64)
        if s7[0] < 0.0:
            cube_config = cube_config | wp.int32(128)

        base = block_key_to_voxel_base(bx, by, bz)
        gx = base[0] + cx
        gy = base[1] + cy
        gz = base[2] + cz

        center_offset_x = wp.float32(tsdf.grid_W) * 0.5
        center_offset_y = wp.float32(tsdf.grid_H) * 0.5
        center_offset_z = wp.float32(tsdf.grid_D) * 0.5

        p0 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx) - center_offset_x,
                wp.float32(gy) - center_offset_y,
                wp.float32(gz) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p1 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx + 1) - center_offset_x,
                wp.float32(gy) - center_offset_y,
                wp.float32(gz) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p2 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx + 1) - center_offset_x,
                wp.float32(gy + 1) - center_offset_y,
                wp.float32(gz) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p3 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx) - center_offset_x,
                wp.float32(gy + 1) - center_offset_y,
                wp.float32(gz) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p4 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx) - center_offset_x,
                wp.float32(gy) - center_offset_y,
                wp.float32(gz + 1) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p5 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx + 1) - center_offset_x,
                wp.float32(gy) - center_offset_y,
                wp.float32(gz + 1) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p6 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx + 1) - center_offset_x,
                wp.float32(gy + 1) - center_offset_y,
                wp.float32(gz + 1) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p7 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx) - center_offset_x,
                wp.float32(gy + 1) - center_offset_y,
                wp.float32(gz + 1) - center_offset_z,
            )
            * tsdf.voxel_size
        )

        rgb = sample_rgb(tsdf, (p0 + p6) * wp.float32(0.5))
        color = wp.vec3ub(wp.uint8(rgb[0]), wp.uint8(rgb[1]), wp.uint8(rgb[2]))

        table_offset = cube_config * 16

        for t in range(5):
            e0 = tri_table[table_offset + t * 3]
            if e0 < 0:
                break

            tri_idx = wp.atomic_add(triangle_count, 0, wp.int32(1))
            if tri_idx >= triangle_capacity:
                continue

            e1 = tri_table[table_offset + t * 3 + 1]
            e2 = tri_table[table_offset + t * 3 + 2]

            out_base = tri_idx * 3
            for v_idx in range(3):
                edge = e0
                if v_idx == 1:
                    edge = e1
                elif v_idx == 2:
                    edge = e2

                vertex_id = out_base + v_idx
                v = get_edge_vertex(
                    edge,
                    p0,
                    p1,
                    p2,
                    p3,
                    p4,
                    p5,
                    p6,
                    p7,
                    s0[0],
                    s1[0],
                    s2[0],
                    s3[0],
                    s4[0],
                    s5[0],
                    s6[0],
                    s7[0],
                )
                n = compute_gradient_nearest(tsdf, v, minimum_tsdf_weight)
                vertices[vertex_id] = v
                normals[vertex_id] = wp.normalize(n)
                colors[vertex_id] = color
                triangles[vertex_id] = vertex_id

    @warp_kernel(f"generate_approximate_mesh_current_state_kernel_bs{block_size}")
    def generate_approximate_mesh_current_state_kernel(
        tsdf: BlockSparseTSDFWarp,
        level: float,
        surface_band: float,
        minimum_tsdf_weight: float,
        n_blocks: wp.int32,
        tri_table: wp.array(dtype=wp.int32),
        vertices: wp.array(dtype=wp.vec3),
        triangles: wp.array(dtype=wp.int32),
        normals: wp.array(dtype=wp.vec3),
        colors: wp.array(dtype=wp.vec3ub),
        mesh_counts: wp.array(dtype=wp.int32),
        output_flags: wp.int32,
        triangle_capacity: wp.int32,
    ):
        """Generate approximate mesh directly from current active TSDF blocks.

        ``mesh_counts`` layout:
          0 required vertices, 1 required triangles, 2 written vertices,
          3 written triangles, 4 surface cubes, 5 overflow triangles.
        """
        block_idx, cube_idx = wp.tid()

        if block_idx >= n_blocks:
            return
        if block_idx >= tsdf.num_allocated[0]:
            return
        if tsdf.block_to_hash_slot[block_idx] < 0:
            return

        if BS == wp.int32(1) and (
            (output_flags & wp.int32(4)) == wp.int32(0)
            or (output_flags & wp.int32(16)) != wp.int32(0)
        ):
            bx1 = tsdf.block_coords[block_idx * 3 + 0]
            by1 = tsdf.block_coords[block_idx * 3 + 1]
            bz1 = tsdf.block_coords[block_idx * 3 + 2]

            n100_idx = hash_lookup(tsdf.hash_table, bx1 + 1, by1, bz1, tsdf.hash_capacity)
            n010_idx = hash_lookup(tsdf.hash_table, bx1, by1 + 1, bz1, tsdf.hash_capacity)
            n110_idx = hash_lookup(
                tsdf.hash_table, bx1 + 1, by1 + 1, bz1, tsdf.hash_capacity
            )
            n001_idx = hash_lookup(tsdf.hash_table, bx1, by1, bz1 + 1, tsdf.hash_capacity)
            n101_idx = hash_lookup(
                tsdf.hash_table, bx1 + 1, by1, bz1 + 1, tsdf.hash_capacity
            )
            n011_idx = hash_lookup(
                tsdf.hash_table, bx1, by1 + 1, bz1 + 1, tsdf.hash_capacity
            )
            n111_idx = hash_lookup(
                tsdf.hash_table, bx1 + 1, by1 + 1, bz1 + 1, tsdf.hash_capacity
            )
            if (
                n100_idx < 0
                or n010_idx < 0
                or n110_idx < 0
                or n001_idx < 0
                or n101_idx < 0
                or n011_idx < 0
                or n111_idx < 0
            ):
                return

            s0b = sample_voxel(tsdf, block_idx, wp.int32(0), minimum_tsdf_weight)
            s1b = sample_voxel(tsdf, n100_idx, wp.int32(0), minimum_tsdf_weight)
            s2b = sample_voxel(tsdf, n110_idx, wp.int32(0), minimum_tsdf_weight)
            s3b = sample_voxel(tsdf, n010_idx, wp.int32(0), minimum_tsdf_weight)
            s4b = sample_voxel(tsdf, n001_idx, wp.int32(0), minimum_tsdf_weight)
            s5b = sample_voxel(tsdf, n101_idx, wp.int32(0), minimum_tsdf_weight)
            s6b = sample_voxel(tsdf, n111_idx, wp.int32(0), minimum_tsdf_weight)
            s7b = sample_voxel(tsdf, n011_idx, wp.int32(0), minimum_tsdf_weight)
            if s0b[1] < 0.5 or s1b[1] < 0.5 or s2b[1] < 0.5 or s3b[1] < 0.5:
                return
            if s4b[1] < 0.5 or s5b[1] < 0.5 or s6b[1] < 0.5 or s7b[1] < 0.5:
                return

            d0 = s0b[0] - level
            d1 = s1b[0] - level
            d2 = s2b[0] - level
            d3 = s3b[0] - level
            d4 = s4b[0] - level
            d5 = s5b[0] - level
            d6 = s6b[0] - level
            d7 = s7b[0] - level

            has_positive_bs1 = (
                d0 > 0.0
                or d1 > 0.0
                or d2 > 0.0
                or d3 > 0.0
                or d4 > 0.0
                or d5 > 0.0
                or d6 > 0.0
                or d7 > 0.0
            )
            has_negative_bs1 = (
                d0 < 0.0
                or d1 < 0.0
                or d2 < 0.0
                or d3 < 0.0
                or d4 < 0.0
                or d5 < 0.0
                or d6 < 0.0
                or d7 < 0.0
            )
            if not (has_positive_bs1 and has_negative_bs1):
                return

            if surface_band > 0.0:
                in_band_bs1 = (
                    wp.abs(d0) < surface_band
                    or wp.abs(d1) < surface_band
                    or wp.abs(d2) < surface_band
                    or wp.abs(d3) < surface_band
                    or wp.abs(d4) < surface_band
                    or wp.abs(d5) < surface_band
                    or wp.abs(d6) < surface_band
                    or wp.abs(d7) < surface_band
                )
                if not in_band_bs1:
                    return

            wp.atomic_add(mesh_counts, 4, wp.int32(1))

            cube_config_bs1 = wp.int32(0)
            if d0 < 0.0:
                cube_config_bs1 = cube_config_bs1 | wp.int32(1)
            if d1 < 0.0:
                cube_config_bs1 = cube_config_bs1 | wp.int32(2)
            if d2 < 0.0:
                cube_config_bs1 = cube_config_bs1 | wp.int32(4)
            if d3 < 0.0:
                cube_config_bs1 = cube_config_bs1 | wp.int32(8)
            if d4 < 0.0:
                cube_config_bs1 = cube_config_bs1 | wp.int32(16)
            if d5 < 0.0:
                cube_config_bs1 = cube_config_bs1 | wp.int32(32)
            if d6 < 0.0:
                cube_config_bs1 = cube_config_bs1 | wp.int32(64)
            if d7 < 0.0:
                cube_config_bs1 = cube_config_bs1 | wp.int32(128)

            table_offset_bs1 = cube_config_bs1 * 16
            num_tris_bs1 = wp.int32(0)
            for t in range(5):
                e = tri_table[table_offset_bs1 + t * 3]
                if e < 0:
                    break
                num_tris_bs1 = num_tris_bs1 + wp.int32(1)

            if num_tris_bs1 <= wp.int32(0):
                return

            tri_base_bs1 = wp.atomic_add(mesh_counts, 1, num_tris_bs1)
            wp.atomic_add(mesh_counts, 0, num_tris_bs1 * wp.int32(3))

            write_tris_bs1 = num_tris_bs1
            if tri_base_bs1 >= triangle_capacity:
                wp.atomic_add(mesh_counts, 5, num_tris_bs1)
                return
            if tri_base_bs1 + write_tris_bs1 > triangle_capacity:
                write_tris_bs1 = triangle_capacity - tri_base_bs1
                wp.atomic_add(mesh_counts, 5, num_tris_bs1 - write_tris_bs1)

            write_vertices_bs1 = (output_flags & wp.int32(1)) != wp.int32(0)
            write_triangles_bs1 = (output_flags & wp.int32(2)) != wp.int32(0)
            write_normals_bs1 = (output_flags & wp.int32(4)) != wp.int32(0)
            write_colors_bs1 = (output_flags & wp.int32(8)) != wp.int32(0)

            base_bs1 = block_key_to_voxel_base(bx1, by1, bz1)
            gx1 = base_bs1[0]
            gy1 = base_bs1[1]
            gz1 = base_bs1[2]

            center_offset_x_bs1 = wp.float32(tsdf.grid_W) * 0.5
            center_offset_y_bs1 = wp.float32(tsdf.grid_H) * 0.5
            center_offset_z_bs1 = wp.float32(tsdf.grid_D) * 0.5

            p0_bs1 = (
                tsdf.origin
                + wp.vec3(
                    wp.float32(gx1) - center_offset_x_bs1,
                    wp.float32(gy1) - center_offset_y_bs1,
                    wp.float32(gz1) - center_offset_z_bs1,
                )
                * tsdf.voxel_size
            )
            p1_bs1 = (
                tsdf.origin
                + wp.vec3(
                    wp.float32(gx1 + 1) - center_offset_x_bs1,
                    wp.float32(gy1) - center_offset_y_bs1,
                    wp.float32(gz1) - center_offset_z_bs1,
                )
                * tsdf.voxel_size
            )
            p2_bs1 = (
                tsdf.origin
                + wp.vec3(
                    wp.float32(gx1 + 1) - center_offset_x_bs1,
                    wp.float32(gy1 + 1) - center_offset_y_bs1,
                    wp.float32(gz1) - center_offset_z_bs1,
                )
                * tsdf.voxel_size
            )
            p3_bs1 = (
                tsdf.origin
                + wp.vec3(
                    wp.float32(gx1) - center_offset_x_bs1,
                    wp.float32(gy1 + 1) - center_offset_y_bs1,
                    wp.float32(gz1) - center_offset_z_bs1,
                )
                * tsdf.voxel_size
            )
            p4_bs1 = (
                tsdf.origin
                + wp.vec3(
                    wp.float32(gx1) - center_offset_x_bs1,
                    wp.float32(gy1) - center_offset_y_bs1,
                    wp.float32(gz1 + 1) - center_offset_z_bs1,
                )
                * tsdf.voxel_size
            )
            p5_bs1 = (
                tsdf.origin
                + wp.vec3(
                    wp.float32(gx1 + 1) - center_offset_x_bs1,
                    wp.float32(gy1) - center_offset_y_bs1,
                    wp.float32(gz1 + 1) - center_offset_z_bs1,
                )
                * tsdf.voxel_size
            )
            p6_bs1 = (
                tsdf.origin
                + wp.vec3(
                    wp.float32(gx1 + 1) - center_offset_x_bs1,
                    wp.float32(gy1 + 1) - center_offset_y_bs1,
                    wp.float32(gz1 + 1) - center_offset_z_bs1,
                )
                * tsdf.voxel_size
            )
            p7_bs1 = (
                tsdf.origin
                + wp.vec3(
                    wp.float32(gx1) - center_offset_x_bs1,
                    wp.float32(gy1 + 1) - center_offset_y_bs1,
                    wp.float32(gz1 + 1) - center_offset_z_bs1,
                )
                * tsdf.voxel_size
            )

            color_bs1 = wp.vec3ub(wp.uint8(0), wp.uint8(0), wp.uint8(0))
            if write_colors_bs1:
                rgb_bs1 = sample_rgb(tsdf, (p0_bs1 + p6_bs1) * wp.float32(0.5))
                color_bs1 = wp.vec3ub(
                    wp.uint8(rgb_bs1[0]), wp.uint8(rgb_bs1[1]), wp.uint8(rgb_bs1[2])
                )

            for t in range(5):
                if wp.int32(t) >= write_tris_bs1:
                    break

                e0_bs1 = tri_table[table_offset_bs1 + t * 3]
                if e0_bs1 < 0:
                    break

                e1_bs1 = tri_table[table_offset_bs1 + t * 3 + 1]
                e2_bs1 = tri_table[table_offset_bs1 + t * 3 + 2]

                tri_idx_bs1 = tri_base_bs1 + wp.int32(t)
                out_base_bs1 = tri_idx_bs1 * 3
                v0_bs1 = get_edge_vertex(
                    e0_bs1,
                    p0_bs1,
                    p1_bs1,
                    p2_bs1,
                    p3_bs1,
                    p4_bs1,
                    p5_bs1,
                    p6_bs1,
                    p7_bs1,
                    d0,
                    d1,
                    d2,
                    d3,
                    d4,
                    d5,
                    d6,
                    d7,
                )
                v1_bs1 = get_edge_vertex(
                    e1_bs1,
                    p0_bs1,
                    p1_bs1,
                    p2_bs1,
                    p3_bs1,
                    p4_bs1,
                    p5_bs1,
                    p6_bs1,
                    p7_bs1,
                    d0,
                    d1,
                    d2,
                    d3,
                    d4,
                    d5,
                    d6,
                    d7,
                )
                v2_bs1 = get_edge_vertex(
                    e2_bs1,
                    p0_bs1,
                    p1_bs1,
                    p2_bs1,
                    p3_bs1,
                    p4_bs1,
                    p5_bs1,
                    p6_bs1,
                    p7_bs1,
                    d0,
                    d1,
                    d2,
                    d3,
                    d4,
                    d5,
                    d6,
                    d7,
                )

                vertex_id0_bs1 = out_base_bs1
                vertex_id1_bs1 = out_base_bs1 + wp.int32(1)
                vertex_id2_bs1 = out_base_bs1 + wp.int32(2)
                if write_vertices_bs1:
                    vertices[vertex_id0_bs1] = v0_bs1
                    vertices[vertex_id1_bs1] = v1_bs1
                    vertices[vertex_id2_bs1] = v2_bs1
                if write_normals_bs1:
                    face_normal_bs1 = wp.cross(v1_bs1 - v0_bs1, v2_bs1 - v0_bs1)
                    face_normal_len_sq_bs1 = wp.dot(face_normal_bs1, face_normal_bs1)
                    if face_normal_len_sq_bs1 > 1.0e-20:
                        face_normal_bs1 = face_normal_bs1 / wp.sqrt(
                            face_normal_len_sq_bs1
                        )
                    normals[vertex_id0_bs1] = face_normal_bs1
                    normals[vertex_id1_bs1] = face_normal_bs1
                    normals[vertex_id2_bs1] = face_normal_bs1
                if write_colors_bs1:
                    colors[vertex_id0_bs1] = color_bs1
                    colors[vertex_id1_bs1] = color_bs1
                    colors[vertex_id2_bs1] = color_bs1
                if write_triangles_bs1:
                    triangles[vertex_id0_bs1] = vertex_id0_bs1
                    triangles[vertex_id1_bs1] = vertex_id1_bs1
                    triangles[vertex_id2_bs1] = vertex_id2_bs1

            wp.atomic_add(mesh_counts, 3, write_tris_bs1)
            wp.atomic_add(mesh_counts, 2, write_tris_bs1 * wp.int32(3))
            return

        cx = cube_idx % BS
        cy = (cube_idx // BS) % BS
        cz = cube_idx // (BS * BS)

        bx = tsdf.block_coords[block_idx * 3 + 0]
        by = tsdf.block_coords[block_idx * 3 + 1]
        bz = tsdf.block_coords[block_idx * 3 + 2]

        s0 = sample_cube_corner(
            cx, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s1 = sample_cube_corner(
            cx + 1, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s2 = sample_cube_corner(
            cx + 1, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s3 = sample_cube_corner(
            cx, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s4 = sample_cube_corner(
            cx, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s5 = sample_cube_corner(
            cx + 1, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s6 = sample_cube_corner(
            cx + 1, cy + 1, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s7 = sample_cube_corner(
            cx, cy + 1, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )

        if s0[1] < 0.5 or s1[1] < 0.5 or s2[1] < 0.5 or s3[1] < 0.5:
            return
        if s4[1] < 0.5 or s5[1] < 0.5 or s6[1] < 0.5 or s7[1] < 0.5:
            return

        has_positive = (
            s0[0] > 0.0
            or s1[0] > 0.0
            or s2[0] > 0.0
            or s3[0] > 0.0
            or s4[0] > 0.0
            or s5[0] > 0.0
            or s6[0] > 0.0
            or s7[0] > 0.0
        )
        has_negative = (
            s0[0] < 0.0
            or s1[0] < 0.0
            or s2[0] < 0.0
            or s3[0] < 0.0
            or s4[0] < 0.0
            or s5[0] < 0.0
            or s6[0] < 0.0
            or s7[0] < 0.0
        )
        if not (has_positive and has_negative):
            return

        if surface_band > 0.0:
            in_band = (
                wp.abs(s0[0]) < surface_band
                or wp.abs(s1[0]) < surface_band
                or wp.abs(s2[0]) < surface_band
                or wp.abs(s3[0]) < surface_band
                or wp.abs(s4[0]) < surface_band
                or wp.abs(s5[0]) < surface_band
                or wp.abs(s6[0]) < surface_band
                or wp.abs(s7[0]) < surface_band
            )
            if not in_band:
                return

        wp.atomic_add(mesh_counts, 4, wp.int32(1))

        cube_config = wp.int32(0)
        if s0[0] < 0.0:
            cube_config = cube_config | wp.int32(1)
        if s1[0] < 0.0:
            cube_config = cube_config | wp.int32(2)
        if s2[0] < 0.0:
            cube_config = cube_config | wp.int32(4)
        if s3[0] < 0.0:
            cube_config = cube_config | wp.int32(8)
        if s4[0] < 0.0:
            cube_config = cube_config | wp.int32(16)
        if s5[0] < 0.0:
            cube_config = cube_config | wp.int32(32)
        if s6[0] < 0.0:
            cube_config = cube_config | wp.int32(64)
        if s7[0] < 0.0:
            cube_config = cube_config | wp.int32(128)

        table_offset = cube_config * 16
        num_tris = wp.int32(0)

        for t in range(5):
            e0 = tri_table[table_offset + t * 3]
            if e0 < 0:
                break
            num_tris = num_tris + wp.int32(1)

        if num_tris <= wp.int32(0):
            return

        tri_base = wp.atomic_add(mesh_counts, 1, num_tris)
        wp.atomic_add(mesh_counts, 0, num_tris * wp.int32(3))

        write_tris = num_tris
        if tri_base >= triangle_capacity:
            wp.atomic_add(mesh_counts, 5, num_tris)
            return
        if tri_base + write_tris > triangle_capacity:
            write_tris = triangle_capacity - tri_base
            wp.atomic_add(mesh_counts, 5, num_tris - write_tris)

        write_vertices = (output_flags & wp.int32(1)) != wp.int32(0)
        write_triangles = (output_flags & wp.int32(2)) != wp.int32(0)
        write_normals = (output_flags & wp.int32(4)) != wp.int32(0)
        write_colors = (output_flags & wp.int32(8)) != wp.int32(0)
        use_face_normals = (output_flags & wp.int32(16)) != wp.int32(0)

        base = block_key_to_voxel_base(bx, by, bz)
        gx = base[0] + cx
        gy = base[1] + cy
        gz = base[2] + cz

        center_offset_x = wp.float32(tsdf.grid_W) * 0.5
        center_offset_y = wp.float32(tsdf.grid_H) * 0.5
        center_offset_z = wp.float32(tsdf.grid_D) * 0.5

        p0 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx) - center_offset_x,
                wp.float32(gy) - center_offset_y,
                wp.float32(gz) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p1 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx + 1) - center_offset_x,
                wp.float32(gy) - center_offset_y,
                wp.float32(gz) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p2 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx + 1) - center_offset_x,
                wp.float32(gy + 1) - center_offset_y,
                wp.float32(gz) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p3 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx) - center_offset_x,
                wp.float32(gy + 1) - center_offset_y,
                wp.float32(gz) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p4 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx) - center_offset_x,
                wp.float32(gy) - center_offset_y,
                wp.float32(gz + 1) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p5 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx + 1) - center_offset_x,
                wp.float32(gy) - center_offset_y,
                wp.float32(gz + 1) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p6 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx + 1) - center_offset_x,
                wp.float32(gy + 1) - center_offset_y,
                wp.float32(gz + 1) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p7 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx) - center_offset_x,
                wp.float32(gy + 1) - center_offset_y,
                wp.float32(gz + 1) - center_offset_z,
            )
            * tsdf.voxel_size
        )

        color = wp.vec3ub(wp.uint8(0), wp.uint8(0), wp.uint8(0))
        if write_colors:
            rgb = sample_rgb(tsdf, (p0 + p6) * wp.float32(0.5))
            color = wp.vec3ub(wp.uint8(rgb[0]), wp.uint8(rgb[1]), wp.uint8(rgb[2]))

        for t in range(5):
            if wp.int32(t) >= write_tris:
                break

            e0 = tri_table[table_offset + t * 3]
            if e0 < 0:
                break

            e1 = tri_table[table_offset + t * 3 + 1]
            e2 = tri_table[table_offset + t * 3 + 2]

            tri_idx = tri_base + wp.int32(t)
            out_base = tri_idx * 3
            v0 = get_edge_vertex(
                e0,
                p0,
                p1,
                p2,
                p3,
                p4,
                p5,
                p6,
                p7,
                s0[0],
                s1[0],
                s2[0],
                s3[0],
                s4[0],
                s5[0],
                s6[0],
                s7[0],
            )
            v1 = get_edge_vertex(
                e1,
                p0,
                p1,
                p2,
                p3,
                p4,
                p5,
                p6,
                p7,
                s0[0],
                s1[0],
                s2[0],
                s3[0],
                s4[0],
                s5[0],
                s6[0],
                s7[0],
            )
            v2 = get_edge_vertex(
                e2,
                p0,
                p1,
                p2,
                p3,
                p4,
                p5,
                p6,
                p7,
                s0[0],
                s1[0],
                s2[0],
                s3[0],
                s4[0],
                s5[0],
                s6[0],
                s7[0],
            )

            vertex_id0 = out_base
            vertex_id1 = out_base + wp.int32(1)
            vertex_id2 = out_base + wp.int32(2)
            if write_vertices:
                vertices[vertex_id0] = v0
                vertices[vertex_id1] = v1
                vertices[vertex_id2] = v2
            if write_normals:
                if use_face_normals:
                    face_normal = wp.cross(v1 - v0, v2 - v0)
                    face_normal_len_sq = wp.dot(face_normal, face_normal)
                    if face_normal_len_sq > 1.0e-20:
                        face_normal = face_normal / wp.sqrt(face_normal_len_sq)
                    normals[vertex_id0] = face_normal
                    normals[vertex_id1] = face_normal
                    normals[vertex_id2] = face_normal
                else:
                    n0 = compute_gradient_nearest(tsdf, v0, minimum_tsdf_weight)
                    n1 = compute_gradient_nearest(tsdf, v1, minimum_tsdf_weight)
                    n2 = compute_gradient_nearest(tsdf, v2, minimum_tsdf_weight)
                    normals[vertex_id0] = wp.normalize(n0)
                    normals[vertex_id1] = wp.normalize(n1)
                    normals[vertex_id2] = wp.normalize(n2)
            if write_colors:
                colors[vertex_id0] = color
                colors[vertex_id1] = color
                colors[vertex_id2] = color
            if write_triangles:
                triangles[vertex_id0] = vertex_id0
                triangles[vertex_id1] = vertex_id1
                triangles[vertex_id2] = vertex_id2

        wp.atomic_add(mesh_counts, 3, write_tris)
        wp.atomic_add(mesh_counts, 2, write_tris * wp.int32(3))

    @warp_kernel(f"count_fast_mesh_block_triangles_kernel_bs{block_size}", enable_backward=False)
    def count_fast_mesh_block_triangles_kernel(
        tsdf: BlockSparseTSDFWarp,
        level: float,
        surface_band: float,
        minimum_tsdf_weight: float,
        n_blocks: wp.int32,
        num_tris_table: wp.array(dtype=wp.int32),
        block_triangle_counts: wp.array(dtype=wp.int32),
        block_surface_counts: wp.array(dtype=wp.int32),
    ):
        """Count full-scene fast mesh triangles per TSDF block."""
        block_idx, cube_idx = wp.tid()

        if block_idx >= n_blocks:
            return
        if block_idx >= tsdf.num_allocated[0]:
            return
        if tsdf.block_to_hash_slot[block_idx] < 0:
            return

        if BS == wp.int32(1):
            return

        cx = cube_idx % BS
        cy = (cube_idx // BS) % BS
        cz = cube_idx // (BS * BS)

        bx = tsdf.block_coords[block_idx * 3 + 0]
        by = tsdf.block_coords[block_idx * 3 + 1]
        bz = tsdf.block_coords[block_idx * 3 + 2]

        s0 = sample_cube_corner(
            cx, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s1 = sample_cube_corner(
            cx + 1, cy, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s2 = sample_cube_corner(
            cx + 1, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s3 = sample_cube_corner(
            cx, cy + 1, cz, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s4 = sample_cube_corner(
            cx, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s5 = sample_cube_corner(
            cx + 1, cy, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s6 = sample_cube_corner(
            cx + 1, cy + 1, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )
        s7 = sample_cube_corner(
            cx, cy + 1, cz + 1, bx, by, bz, block_idx, tsdf, level, minimum_tsdf_weight
        )

        if s0[1] < 0.5 or s1[1] < 0.5 or s2[1] < 0.5 or s3[1] < 0.5:
            return
        if s4[1] < 0.5 or s5[1] < 0.5 or s6[1] < 0.5 or s7[1] < 0.5:
            return

        has_positive = (
            s0[0] > 0.0
            or s1[0] > 0.0
            or s2[0] > 0.0
            or s3[0] > 0.0
            or s4[0] > 0.0
            or s5[0] > 0.0
            or s6[0] > 0.0
            or s7[0] > 0.0
        )
        has_negative = (
            s0[0] < 0.0
            or s1[0] < 0.0
            or s2[0] < 0.0
            or s3[0] < 0.0
            or s4[0] < 0.0
            or s5[0] < 0.0
            or s6[0] < 0.0
            or s7[0] < 0.0
        )
        if not (has_positive and has_negative):
            return

        if surface_band > 0.0:
            in_band = (
                wp.abs(s0[0]) < surface_band
                or wp.abs(s1[0]) < surface_band
                or wp.abs(s2[0]) < surface_band
                or wp.abs(s3[0]) < surface_band
                or wp.abs(s4[0]) < surface_band
                or wp.abs(s5[0]) < surface_band
                or wp.abs(s6[0]) < surface_band
                or wp.abs(s7[0]) < surface_band
            )
            if not in_band:
                return

        cube_config = wp.int32(0)
        if s0[0] < 0.0:
            cube_config = cube_config | wp.int32(1)
        if s1[0] < 0.0:
            cube_config = cube_config | wp.int32(2)
        if s2[0] < 0.0:
            cube_config = cube_config | wp.int32(4)
        if s3[0] < 0.0:
            cube_config = cube_config | wp.int32(8)
        if s4[0] < 0.0:
            cube_config = cube_config | wp.int32(16)
        if s5[0] < 0.0:
            cube_config = cube_config | wp.int32(32)
        if s6[0] < 0.0:
            cube_config = cube_config | wp.int32(64)
        if s7[0] < 0.0:
            cube_config = cube_config | wp.int32(128)

        num_tris = num_tris_table[cube_config]
        if num_tris <= wp.int32(0):
            return

        wp.atomic_add(block_surface_counts, block_idx, wp.int32(1))
        wp.atomic_add(block_triangle_counts, block_idx, num_tris)

    @warp_kernel(f"generate_fast_mesh_current_state_kernel_bs{block_size}")
    def generate_fast_mesh_current_state_kernel(
        tsdf: BlockSparseTSDFWarp,
        level: float,
        surface_band: float,
        minimum_tsdf_weight: float,
        n_blocks: wp.int32,
        tri_table: wp.array(dtype=wp.int32),
        vertices: wp.array(dtype=wp.vec3),
        triangles: wp.array(dtype=wp.int32),
        colors: wp.array(dtype=wp.vec3ub),
        mesh_counts: wp.array(dtype=wp.int32),
        block_triangle_offsets: wp.array(dtype=wp.int32),
        block_write_counts: wp.array(dtype=wp.int32),
        triangle_capacity: wp.int32,
        write_triangles: wp.int32,
        use_voxel_colors: wp.int32,
        atomic_triangle_list: wp.int32,
    ):
        """Fast current-state mesh: vertices, triangles, colors; no normals."""
        block_idx, cube_idx = wp.tid()

        if block_idx >= n_blocks:
            return
        if block_idx >= tsdf.num_allocated[0]:
            return
        if tsdf.block_to_hash_slot[block_idx] < 0:
            return

        if BS != wp.int32(1):
            cx_g = cube_idx % BS
            cy_g = (cube_idx // BS) % BS
            cz_g = cube_idx // (BS * BS)

            bx_g = tsdf.block_coords[block_idx * 3 + 0]
            by_g = tsdf.block_coords[block_idx * 3 + 1]
            bz_g = tsdf.block_coords[block_idx * 3 + 2]

            s0_g = sample_cube_corner(
                cx_g, cy_g, cz_g, bx_g, by_g, bz_g, block_idx, tsdf, level, minimum_tsdf_weight
            )
            s1_g = sample_cube_corner(
                cx_g + 1, cy_g, cz_g, bx_g, by_g, bz_g, block_idx, tsdf, level, minimum_tsdf_weight
            )
            s2_g = sample_cube_corner(
                cx_g + 1, cy_g + 1, cz_g, bx_g, by_g, bz_g, block_idx, tsdf, level, minimum_tsdf_weight
            )
            s3_g = sample_cube_corner(
                cx_g, cy_g + 1, cz_g, bx_g, by_g, bz_g, block_idx, tsdf, level, minimum_tsdf_weight
            )
            s4_g = sample_cube_corner(
                cx_g, cy_g, cz_g + 1, bx_g, by_g, bz_g, block_idx, tsdf, level, minimum_tsdf_weight
            )
            s5_g = sample_cube_corner(
                cx_g + 1, cy_g, cz_g + 1, bx_g, by_g, bz_g, block_idx, tsdf, level, minimum_tsdf_weight
            )
            s6_g = sample_cube_corner(
                cx_g + 1, cy_g + 1, cz_g + 1, bx_g, by_g, bz_g, block_idx, tsdf, level, minimum_tsdf_weight
            )
            s7_g = sample_cube_corner(
                cx_g, cy_g + 1, cz_g + 1, bx_g, by_g, bz_g, block_idx, tsdf, level, minimum_tsdf_weight
            )

            if s0_g[1] < 0.5 or s1_g[1] < 0.5 or s2_g[1] < 0.5 or s3_g[1] < 0.5:
                return
            if s4_g[1] < 0.5 or s5_g[1] < 0.5 or s6_g[1] < 0.5 or s7_g[1] < 0.5:
                return

            has_positive_g = (
                s0_g[0] > 0.0
                or s1_g[0] > 0.0
                or s2_g[0] > 0.0
                or s3_g[0] > 0.0
                or s4_g[0] > 0.0
                or s5_g[0] > 0.0
                or s6_g[0] > 0.0
                or s7_g[0] > 0.0
            )
            has_negative_g = (
                s0_g[0] < 0.0
                or s1_g[0] < 0.0
                or s2_g[0] < 0.0
                or s3_g[0] < 0.0
                or s4_g[0] < 0.0
                or s5_g[0] < 0.0
                or s6_g[0] < 0.0
                or s7_g[0] < 0.0
            )
            if not (has_positive_g and has_negative_g):
                return

            if surface_band > 0.0:
                in_band_g = (
                    wp.abs(s0_g[0]) < surface_band
                    or wp.abs(s1_g[0]) < surface_band
                    or wp.abs(s2_g[0]) < surface_band
                    or wp.abs(s3_g[0]) < surface_band
                    or wp.abs(s4_g[0]) < surface_band
                    or wp.abs(s5_g[0]) < surface_band
                    or wp.abs(s6_g[0]) < surface_band
                    or wp.abs(s7_g[0]) < surface_band
                )
                if not in_band_g:
                    return

            cube_config_g = wp.int32(0)
            if s0_g[0] < 0.0:
                cube_config_g = cube_config_g | wp.int32(1)
            if s1_g[0] < 0.0:
                cube_config_g = cube_config_g | wp.int32(2)
            if s2_g[0] < 0.0:
                cube_config_g = cube_config_g | wp.int32(4)
            if s3_g[0] < 0.0:
                cube_config_g = cube_config_g | wp.int32(8)
            if s4_g[0] < 0.0:
                cube_config_g = cube_config_g | wp.int32(16)
            if s5_g[0] < 0.0:
                cube_config_g = cube_config_g | wp.int32(32)
            if s6_g[0] < 0.0:
                cube_config_g = cube_config_g | wp.int32(64)
            if s7_g[0] < 0.0:
                cube_config_g = cube_config_g | wp.int32(128)

            table_offset_g = cube_config_g * 16
            num_tris_g = wp.int32(0)
            for t in range(5):
                e_g = tri_table[table_offset_g + t * 3]
                if e_g < 0:
                    break
                num_tris_g = num_tris_g + wp.int32(1)

            if num_tris_g <= wp.int32(0):
                return

            if atomic_triangle_list != wp.int32(0):
                tri_base_g = wp.atomic_add(mesh_counts, wp.int32(1), num_tris_g)
                wp.atomic_add(mesh_counts, wp.int32(0), num_tris_g * wp.int32(3))
            else:
                local_tri_base_g = wp.atomic_add(block_write_counts, block_idx, num_tris_g)
                tri_base_g = block_triangle_offsets[block_idx] + local_tri_base_g

            write_tris_g = num_tris_g
            if tri_base_g >= triangle_capacity:
                if atomic_triangle_list != wp.int32(0):
                    wp.atomic_add(mesh_counts, wp.int32(5), num_tris_g)
                return
            if tri_base_g + write_tris_g > triangle_capacity:
                write_tris_g = triangle_capacity - tri_base_g
                if atomic_triangle_list != wp.int32(0):
                    wp.atomic_add(mesh_counts, wp.int32(5), num_tris_g - write_tris_g)

            if atomic_triangle_list != wp.int32(0):
                wp.atomic_add(mesh_counts, wp.int32(2), write_tris_g * wp.int32(3))
                wp.atomic_add(mesh_counts, wp.int32(3), write_tris_g)

            base_g = block_key_to_voxel_base(bx_g, by_g, bz_g)
            gx_g = base_g[0] + cx_g
            gy_g = base_g[1] + cy_g
            gz_g = base_g[2] + cz_g

            center_offset_x_g = wp.float32(tsdf.grid_W) * 0.5
            center_offset_y_g = wp.float32(tsdf.grid_H) * 0.5
            center_offset_z_g = wp.float32(tsdf.grid_D) * 0.5

            p0_g = (
                tsdf.origin
                + wp.vec3(
                    wp.float32(gx_g) - center_offset_x_g,
                    wp.float32(gy_g) - center_offset_y_g,
                    wp.float32(gz_g) - center_offset_z_g,
                )
                * tsdf.voxel_size
            )
            p1_g = (
                tsdf.origin
                + wp.vec3(
                    wp.float32(gx_g + 1) - center_offset_x_g,
                    wp.float32(gy_g) - center_offset_y_g,
                    wp.float32(gz_g) - center_offset_z_g,
                )
                * tsdf.voxel_size
            )
            p2_g = (
                tsdf.origin
                + wp.vec3(
                    wp.float32(gx_g + 1) - center_offset_x_g,
                    wp.float32(gy_g + 1) - center_offset_y_g,
                    wp.float32(gz_g) - center_offset_z_g,
                )
                * tsdf.voxel_size
            )
            p3_g = (
                tsdf.origin
                + wp.vec3(
                    wp.float32(gx_g) - center_offset_x_g,
                    wp.float32(gy_g + 1) - center_offset_y_g,
                    wp.float32(gz_g) - center_offset_z_g,
                )
                * tsdf.voxel_size
            )
            p4_g = (
                tsdf.origin
                + wp.vec3(
                    wp.float32(gx_g) - center_offset_x_g,
                    wp.float32(gy_g) - center_offset_y_g,
                    wp.float32(gz_g + 1) - center_offset_z_g,
                )
                * tsdf.voxel_size
            )
            p5_g = (
                tsdf.origin
                + wp.vec3(
                    wp.float32(gx_g + 1) - center_offset_x_g,
                    wp.float32(gy_g) - center_offset_y_g,
                    wp.float32(gz_g + 1) - center_offset_z_g,
                )
                * tsdf.voxel_size
            )
            p6_g = (
                tsdf.origin
                + wp.vec3(
                    wp.float32(gx_g + 1) - center_offset_x_g,
                    wp.float32(gy_g + 1) - center_offset_y_g,
                    wp.float32(gz_g + 1) - center_offset_z_g,
                )
                * tsdf.voxel_size
            )
            p7_g = (
                tsdf.origin
                + wp.vec3(
                    wp.float32(gx_g) - center_offset_x_g,
                    wp.float32(gy_g + 1) - center_offset_y_g,
                    wp.float32(gz_g + 1) - center_offset_z_g,
                )
                * tsdf.voxel_size
            )

            color_g = wp.vec3ub(wp.uint8(128), wp.uint8(128), wp.uint8(128))
            if use_voxel_colors != wp.int32(0):
                rgb_g = sample_rgb(tsdf, (p0_g + p6_g) * wp.float32(0.5))
                color_g = wp.vec3ub(
                    wp.uint8(rgb_g[0]), wp.uint8(rgb_g[1]), wp.uint8(rgb_g[2])
                )

            for t in range(5):
                if wp.int32(t) >= write_tris_g:
                    break

                e0_g = tri_table[table_offset_g + t * 3]
                if e0_g < 0:
                    break

                e1_g = tri_table[table_offset_g + t * 3 + 1]
                e2_g = tri_table[table_offset_g + t * 3 + 2]

                tri_idx_g = tri_base_g + wp.int32(t)
                out_base_g = tri_idx_g * 3
                v0_g = get_edge_vertex(
                    e0_g,
                    p0_g,
                    p1_g,
                    p2_g,
                    p3_g,
                    p4_g,
                    p5_g,
                    p6_g,
                    p7_g,
                    s0_g[0],
                    s1_g[0],
                    s2_g[0],
                    s3_g[0],
                    s4_g[0],
                    s5_g[0],
                    s6_g[0],
                    s7_g[0],
                )
                v1_g = get_edge_vertex(
                    e1_g,
                    p0_g,
                    p1_g,
                    p2_g,
                    p3_g,
                    p4_g,
                    p5_g,
                    p6_g,
                    p7_g,
                    s0_g[0],
                    s1_g[0],
                    s2_g[0],
                    s3_g[0],
                    s4_g[0],
                    s5_g[0],
                    s6_g[0],
                    s7_g[0],
                )
                v2_g = get_edge_vertex(
                    e2_g,
                    p0_g,
                    p1_g,
                    p2_g,
                    p3_g,
                    p4_g,
                    p5_g,
                    p6_g,
                    p7_g,
                    s0_g[0],
                    s1_g[0],
                    s2_g[0],
                    s3_g[0],
                    s4_g[0],
                    s5_g[0],
                    s6_g[0],
                    s7_g[0],
                )

                vertex_id0_g = out_base_g
                vertex_id1_g = out_base_g + wp.int32(1)
                vertex_id2_g = out_base_g + wp.int32(2)
                vertices[vertex_id0_g] = v0_g
                vertices[vertex_id1_g] = v1_g
                vertices[vertex_id2_g] = v2_g
                if write_triangles != wp.int32(0):
                    triangles[vertex_id0_g] = vertex_id0_g
                    triangles[vertex_id1_g] = vertex_id1_g
                    triangles[vertex_id2_g] = vertex_id2_g
                if use_voxel_colors == wp.int32(0):
                    colors[vertex_id0_g] = color_g
                    colors[vertex_id1_g] = color_g
                    colors[vertex_id2_g] = color_g
                    continue
                c0_g = sample_rgb(tsdf, v0_g)
                c1_g = sample_rgb(tsdf, v1_g)
                c2_g = sample_rgb(tsdf, v2_g)
                colors[vertex_id0_g] = wp.vec3ub(
                    wp.uint8(c0_g[0]), wp.uint8(c0_g[1]), wp.uint8(c0_g[2])
                )
                colors[vertex_id1_g] = wp.vec3ub(
                    wp.uint8(c1_g[0]), wp.uint8(c1_g[1]), wp.uint8(c1_g[2])
                )
                colors[vertex_id2_g] = wp.vec3ub(
                    wp.uint8(c2_g[0]), wp.uint8(c2_g[1]), wp.uint8(c2_g[2])
                )

            return

        if cube_idx != wp.int32(0):
            return

        bx = tsdf.block_coords[block_idx * 3 + 0]
        by = tsdf.block_coords[block_idx * 3 + 1]
        bz = tsdf.block_coords[block_idx * 3 + 2]

        n100_idx = hash_lookup(tsdf.hash_table, bx + 1, by, bz, tsdf.hash_capacity)
        n010_idx = hash_lookup(tsdf.hash_table, bx, by + 1, bz, tsdf.hash_capacity)
        n110_idx = hash_lookup(tsdf.hash_table, bx + 1, by + 1, bz, tsdf.hash_capacity)
        n001_idx = hash_lookup(tsdf.hash_table, bx, by, bz + 1, tsdf.hash_capacity)
        n101_idx = hash_lookup(tsdf.hash_table, bx + 1, by, bz + 1, tsdf.hash_capacity)
        n011_idx = hash_lookup(tsdf.hash_table, bx, by + 1, bz + 1, tsdf.hash_capacity)
        n111_idx = hash_lookup(
            tsdf.hash_table, bx + 1, by + 1, bz + 1, tsdf.hash_capacity
        )
        if (
            n100_idx < 0
            or n010_idx < 0
            or n110_idx < 0
            or n001_idx < 0
            or n101_idx < 0
            or n011_idx < 0
            or n111_idx < 0
        ):
            return

        s0 = sample_voxel(tsdf, block_idx, wp.int32(0), minimum_tsdf_weight)
        s1 = sample_voxel(tsdf, n100_idx, wp.int32(0), minimum_tsdf_weight)
        s2 = sample_voxel(tsdf, n110_idx, wp.int32(0), minimum_tsdf_weight)
        s3 = sample_voxel(tsdf, n010_idx, wp.int32(0), minimum_tsdf_weight)
        s4 = sample_voxel(tsdf, n001_idx, wp.int32(0), minimum_tsdf_weight)
        s5 = sample_voxel(tsdf, n101_idx, wp.int32(0), minimum_tsdf_weight)
        s6 = sample_voxel(tsdf, n111_idx, wp.int32(0), minimum_tsdf_weight)
        s7 = sample_voxel(tsdf, n011_idx, wp.int32(0), minimum_tsdf_weight)
        if s0[1] < 0.5 or s1[1] < 0.5 or s2[1] < 0.5 or s3[1] < 0.5:
            return
        if s4[1] < 0.5 or s5[1] < 0.5 or s6[1] < 0.5 or s7[1] < 0.5:
            return

        d0 = s0[0] - level
        d1 = s1[0] - level
        d2 = s2[0] - level
        d3 = s3[0] - level
        d4 = s4[0] - level
        d5 = s5[0] - level
        d6 = s6[0] - level
        d7 = s7[0] - level

        has_positive = (
            d0 > 0.0
            or d1 > 0.0
            or d2 > 0.0
            or d3 > 0.0
            or d4 > 0.0
            or d5 > 0.0
            or d6 > 0.0
            or d7 > 0.0
        )
        has_negative = (
            d0 < 0.0
            or d1 < 0.0
            or d2 < 0.0
            or d3 < 0.0
            or d4 < 0.0
            or d5 < 0.0
            or d6 < 0.0
            or d7 < 0.0
        )
        if not (has_positive and has_negative):
            return

        if surface_band > 0.0:
            in_band = (
                wp.abs(d0) < surface_band
                or wp.abs(d1) < surface_band
                or wp.abs(d2) < surface_band
                or wp.abs(d3) < surface_band
                or wp.abs(d4) < surface_band
                or wp.abs(d5) < surface_band
                or wp.abs(d6) < surface_band
                or wp.abs(d7) < surface_band
            )
            if not in_band:
                return

        cube_config = wp.int32(0)
        if d0 < 0.0:
            cube_config = cube_config | wp.int32(1)
        if d1 < 0.0:
            cube_config = cube_config | wp.int32(2)
        if d2 < 0.0:
            cube_config = cube_config | wp.int32(4)
        if d3 < 0.0:
            cube_config = cube_config | wp.int32(8)
        if d4 < 0.0:
            cube_config = cube_config | wp.int32(16)
        if d5 < 0.0:
            cube_config = cube_config | wp.int32(32)
        if d6 < 0.0:
            cube_config = cube_config | wp.int32(64)
        if d7 < 0.0:
            cube_config = cube_config | wp.int32(128)

        table_offset = cube_config * 16
        num_tris = wp.int32(0)
        for t in range(5):
            e = tri_table[table_offset + t * 3]
            if e < 0:
                break
            num_tris = num_tris + wp.int32(1)

        if num_tris <= wp.int32(0):
            return

        if atomic_triangle_list != wp.int32(0):
            tri_base = wp.atomic_add(mesh_counts, wp.int32(1), num_tris)
            wp.atomic_add(mesh_counts, wp.int32(0), num_tris * wp.int32(3))
        else:
            local_tri_base = wp.atomic_add(block_write_counts, block_idx, num_tris)
            tri_base = block_triangle_offsets[block_idx] + local_tri_base

        write_tris = num_tris
        if tri_base >= triangle_capacity:
            if atomic_triangle_list != wp.int32(0):
                wp.atomic_add(mesh_counts, wp.int32(5), num_tris)
            return
        if tri_base + write_tris > triangle_capacity:
            write_tris = triangle_capacity - tri_base
            if atomic_triangle_list != wp.int32(0):
                wp.atomic_add(mesh_counts, wp.int32(5), num_tris - write_tris)

        if atomic_triangle_list != wp.int32(0):
            wp.atomic_add(mesh_counts, wp.int32(2), write_tris * wp.int32(3))
            wp.atomic_add(mesh_counts, wp.int32(3), write_tris)

        base = block_key_to_voxel_base(bx, by, bz)
        gx = base[0]
        gy = base[1]
        gz = base[2]

        center_offset_x = wp.float32(tsdf.grid_W) * 0.5
        center_offset_y = wp.float32(tsdf.grid_H) * 0.5
        center_offset_z = wp.float32(tsdf.grid_D) * 0.5

        p0 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx) - center_offset_x,
                wp.float32(gy) - center_offset_y,
                wp.float32(gz) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p1 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx + 1) - center_offset_x,
                wp.float32(gy) - center_offset_y,
                wp.float32(gz) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p2 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx + 1) - center_offset_x,
                wp.float32(gy + 1) - center_offset_y,
                wp.float32(gz) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p3 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx) - center_offset_x,
                wp.float32(gy + 1) - center_offset_y,
                wp.float32(gz) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p4 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx) - center_offset_x,
                wp.float32(gy) - center_offset_y,
                wp.float32(gz + 1) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p5 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx + 1) - center_offset_x,
                wp.float32(gy) - center_offset_y,
                wp.float32(gz + 1) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p6 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx + 1) - center_offset_x,
                wp.float32(gy + 1) - center_offset_y,
                wp.float32(gz + 1) - center_offset_z,
            )
            * tsdf.voxel_size
        )
        p7 = (
            tsdf.origin
            + wp.vec3(
                wp.float32(gx) - center_offset_x,
                wp.float32(gy + 1) - center_offset_y,
                wp.float32(gz + 1) - center_offset_z,
            )
            * tsdf.voxel_size
        )

        color = wp.vec3ub(wp.uint8(128), wp.uint8(128), wp.uint8(128))
        if use_voxel_colors != wp.int32(0):
            rgb = sample_rgb(tsdf, (p0 + p6) * wp.float32(0.5))
            color = wp.vec3ub(wp.uint8(rgb[0]), wp.uint8(rgb[1]), wp.uint8(rgb[2]))

        for t in range(5):
            if wp.int32(t) >= write_tris:
                break

            e0 = tri_table[table_offset + t * 3]
            if e0 < 0:
                break

            e1 = tri_table[table_offset + t * 3 + 1]
            e2 = tri_table[table_offset + t * 3 + 2]

            tri_idx = tri_base + wp.int32(t)
            out_base = tri_idx * 3
            v0 = get_edge_vertex(
                e0,
                p0,
                p1,
                p2,
                p3,
                p4,
                p5,
                p6,
                p7,
                d0,
                d1,
                d2,
                d3,
                d4,
                d5,
                d6,
                d7,
            )
            v1 = get_edge_vertex(
                e1,
                p0,
                p1,
                p2,
                p3,
                p4,
                p5,
                p6,
                p7,
                d0,
                d1,
                d2,
                d3,
                d4,
                d5,
                d6,
                d7,
            )
            v2 = get_edge_vertex(
                e2,
                p0,
                p1,
                p2,
                p3,
                p4,
                p5,
                p6,
                p7,
                d0,
                d1,
                d2,
                d3,
                d4,
                d5,
                d6,
                d7,
            )

            vertex_id0 = out_base
            vertex_id1 = out_base + wp.int32(1)
            vertex_id2 = out_base + wp.int32(2)
            vertices[vertex_id0] = v0
            vertices[vertex_id1] = v1
            vertices[vertex_id2] = v2
            if write_triangles != wp.int32(0):
                triangles[vertex_id0] = vertex_id0
                triangles[vertex_id1] = vertex_id1
                triangles[vertex_id2] = vertex_id2
            if use_voxel_colors == wp.int32(0):
                colors[vertex_id0] = color
                colors[vertex_id1] = color
                colors[vertex_id2] = color
                continue
            c0 = sample_rgb(tsdf, v0)
            c1 = sample_rgb(tsdf, v1)
            c2 = sample_rgb(tsdf, v2)
            colors[vertex_id0] = wp.vec3ub(
                wp.uint8(c0[0]), wp.uint8(c0[1]), wp.uint8(c0[2])
            )
            colors[vertex_id1] = wp.vec3ub(
                wp.uint8(c1[0]), wp.uint8(c1[1]), wp.uint8(c1[2])
            )
            colors[vertex_id2] = wp.vec3ub(
                wp.uint8(c2[0]), wp.uint8(c2[1]), wp.uint8(c2[2])
            )

    # =====================================================================
    # Projective texture mapping for fast triangle-list meshes
    # =====================================================================

    @warp_kernel(f"copy_rgb_images_to_texture_atlas_kernel_bs{block_size}", enable_backward=False)
    def copy_rgb_images_to_texture_atlas_kernel(
        rgb_images: wp.array3d(dtype=wp.uint8),
        texture_atlas: wp.array3d(dtype=wp.uint8),
    ):
        """Copy flattened camera RGB images into a horizontal texture atlas."""
        tid = wp.tid()
        image_pixels = IMAGE_HEIGHT * IMAGE_WIDTH
        total_pixels = NUM_CAMERAS * image_pixels
        if tid >= total_pixels:
            return

        cam_idx = tid // image_pixels
        rem = tid - cam_idx * image_pixels
        py = rem // IMAGE_WIDTH
        px = rem - py * IMAGE_WIDTH

        src_y = cam_idx * IMAGE_HEIGHT + py
        atlas_x = cam_idx * IMAGE_WIDTH + px

        texture_atlas[py, atlas_x, 0] = rgb_images[src_y, px, 0]
        texture_atlas[py, atlas_x, 1] = rgb_images[src_y, px, 1]
        texture_atlas[py, atlas_x, 2] = rgb_images[src_y, px, 2]

    @warp_kernel(f"project_fast_mesh_uvs_kernel_bs{block_size}", enable_backward=False)
    def project_fast_mesh_uvs_kernel(
        vertices: wp.array(dtype=wp.vec3),
        vertex_count: wp.int32,
        intrinsics: wp.array3d(dtype=wp.float32),
        cam_positions: wp.array2d(dtype=wp.float32),
        cam_quaternions: wp.array2d(dtype=wp.float32),
        depth_images: wp.array3d(dtype=wp.float32),
        depth_min: float,
        depth_max: float,
        occlusion_tolerance_m: float,
        vertex_uvs: wp.array(dtype=wp.vec2),
        colors: wp.array(dtype=wp.vec3ub),
        fill_missing_colors: wp.int32,
        tsdf: BlockSparseTSDFWarp,
    ):
        """Project fast-mesh triangle vertices into the current camera atlas."""
        tri_idx = wp.tid()
        base = tri_idx * wp.int32(3)
        if base + wp.int32(2) >= vertex_count:
            return

        invalid_uv = wp.vec2(-1.0, -1.0)
        v0 = vertices[base]
        v1 = vertices[base + wp.int32(1)]
        v2 = vertices[base + wp.int32(2)]

        center = (v0 + v1 + v2) / wp.float32(3.0)
        face_normal = wp.cross(v1 - v0, v2 - v0)

        uv0 = invalid_uv
        uv1 = invalid_uv
        uv2 = invalid_uv
        projected = bool(False)
        score_limit = wp.float32(1.0e20)

        for _attempt in range(num_cameras):
            best_cam = wp.int32(-1)
            best_score = wp.float32(-1.0e20)

            for cam_i in range(num_cameras):
                cam_pos_i = wp.vec3(
                    cam_positions[cam_i, 0],
                    cam_positions[cam_i, 1],
                    cam_positions[cam_i, 2],
                )
                cam_quat_i = wp.quaternion(
                    cam_quaternions[cam_i, 1],
                    cam_quaternions[cam_i, 2],
                    cam_quaternions[cam_i, 3],
                    cam_quaternions[cam_i, 0],
                )
                center_cam_i = wp.quat_rotate(wp.quat_inverse(cam_quat_i), center - cam_pos_i)
                z_center_i = center_cam_i[2]

                if z_center_i > depth_min and z_center_i <= depth_max:
                    fx_i = intrinsics[cam_i, 0, 0]
                    fy_i = intrinsics[cam_i, 1, 1]
                    cx_i = intrinsics[cam_i, 0, 2]
                    cy_i = intrinsics[cam_i, 1, 2]

                    u_center_i = fx_i * center_cam_i[0] / z_center_i + cx_i
                    v_center_i = fy_i * center_cam_i[1] / z_center_i + cy_i
                    if (
                        u_center_i >= 0.0
                        and u_center_i < wp.float32(IMAGE_WIDTH)
                        and v_center_i >= 0.0
                        and v_center_i < wp.float32(IMAGE_HEIGHT)
                    ):
                        viewing_dir = cam_pos_i - center
                        viewing_len_sq = wp.dot(viewing_dir, viewing_dir)
                        if viewing_len_sq > 1.0e-20:
                            viewing_dir = viewing_dir / wp.sqrt(viewing_len_sq)
                            score = wp.dot(face_normal, viewing_dir)
                            if score < score_limit and score > best_score:
                                best_score = score
                                best_cam = wp.int32(cam_i)

            if best_cam < 0:
                break

            cam_pos = wp.vec3(
                cam_positions[best_cam, 0],
                cam_positions[best_cam, 1],
                cam_positions[best_cam, 2],
            )
            cam_quat = wp.quaternion(
                cam_quaternions[best_cam, 1],
                cam_quaternions[best_cam, 2],
                cam_quaternions[best_cam, 3],
                cam_quaternions[best_cam, 0],
            )
            cam_quat_inv = wp.quat_inverse(cam_quat)
            fx = intrinsics[best_cam, 0, 0]
            fy = intrinsics[best_cam, 1, 1]
            cx = intrinsics[best_cam, 0, 2]
            cy = intrinsics[best_cam, 1, 2]
            atlas_width = wp.float32(NUM_CAMERAS * IMAGE_WIDTH)
            atlas_x_offset = wp.float32(best_cam * IMAGE_WIDTH)

            all_valid = bool(True)

            p0_cam = wp.quat_rotate(cam_quat_inv, v0 - cam_pos)
            z0 = p0_cam[2]
            if z0 > depth_min and z0 <= depth_max:
                u0 = fx * p0_cam[0] / z0 + cx
                vv0 = fy * p0_cam[1] / z0 + cy
                if (
                    u0 >= 0.0
                    and u0 < wp.float32(IMAGE_WIDTH)
                    and vv0 >= 0.0
                    and vv0 < wp.float32(IMAGE_HEIGHT)
                ):
                    px0 = wp.int32(u0 + 0.5)
                    py0 = wp.int32(vv0 + 0.5)
                    if px0 >= IMAGE_WIDTH:
                        px0 = IMAGE_WIDTH - wp.int32(1)
                    if py0 >= IMAGE_HEIGHT:
                        py0 = IMAGE_HEIGHT - wp.int32(1)
                    observed0 = depth_images[best_cam, py0, px0]
                    if observed0 > 0.0 and z0 > observed0 + occlusion_tolerance_m:
                        all_valid = False
                    else:
                        uv0 = wp.vec2(
                            (atlas_x_offset + u0) / atlas_width,
                            vv0 / wp.float32(IMAGE_HEIGHT),
                        )
                else:
                    all_valid = False
            else:
                all_valid = False

            p1_cam = wp.quat_rotate(cam_quat_inv, v1 - cam_pos)
            z1 = p1_cam[2]
            if all_valid and z1 > depth_min and z1 <= depth_max:
                u1 = fx * p1_cam[0] / z1 + cx
                vv1 = fy * p1_cam[1] / z1 + cy
                if (
                    u1 >= 0.0
                    and u1 < wp.float32(IMAGE_WIDTH)
                    and vv1 >= 0.0
                    and vv1 < wp.float32(IMAGE_HEIGHT)
                ):
                    px1 = wp.int32(u1 + 0.5)
                    py1 = wp.int32(vv1 + 0.5)
                    if px1 >= IMAGE_WIDTH:
                        px1 = IMAGE_WIDTH - wp.int32(1)
                    if py1 >= IMAGE_HEIGHT:
                        py1 = IMAGE_HEIGHT - wp.int32(1)
                    observed1 = depth_images[best_cam, py1, px1]
                    if observed1 > 0.0 and z1 > observed1 + occlusion_tolerance_m:
                        all_valid = False
                    else:
                        uv1 = wp.vec2(
                            (atlas_x_offset + u1) / atlas_width,
                            vv1 / wp.float32(IMAGE_HEIGHT),
                        )
                else:
                    all_valid = False
            else:
                all_valid = False

            p2_cam = wp.quat_rotate(cam_quat_inv, v2 - cam_pos)
            z2 = p2_cam[2]
            if all_valid and z2 > depth_min and z2 <= depth_max:
                u2 = fx * p2_cam[0] / z2 + cx
                vv2 = fy * p2_cam[1] / z2 + cy
                if (
                    u2 >= 0.0
                    and u2 < wp.float32(IMAGE_WIDTH)
                    and vv2 >= 0.0
                    and vv2 < wp.float32(IMAGE_HEIGHT)
                ):
                    px2 = wp.int32(u2 + 0.5)
                    py2 = wp.int32(vv2 + 0.5)
                    if px2 >= IMAGE_WIDTH:
                        px2 = IMAGE_WIDTH - wp.int32(1)
                    if py2 >= IMAGE_HEIGHT:
                        py2 = IMAGE_HEIGHT - wp.int32(1)
                    observed2 = depth_images[best_cam, py2, px2]
                    if observed2 > 0.0 and z2 > observed2 + occlusion_tolerance_m:
                        all_valid = False
                    else:
                        uv2 = wp.vec2(
                            (atlas_x_offset + u2) / atlas_width,
                            vv2 / wp.float32(IMAGE_HEIGHT),
                        )
                else:
                    all_valid = False
            else:
                all_valid = False

            if all_valid:
                vertex_uvs[base] = uv0
                vertex_uvs[base + wp.int32(1)] = uv1
                vertex_uvs[base + wp.int32(2)] = uv2
                projected = True
                break

            score_limit = best_score - wp.float32(1.0e-6)

        if not projected:
            vertex_uvs[base] = invalid_uv
            vertex_uvs[base + wp.int32(1)] = invalid_uv
            vertex_uvs[base + wp.int32(2)] = invalid_uv
            if fill_missing_colors != wp.int32(0):
                c0 = sample_rgb(tsdf, v0)
                c1 = sample_rgb(tsdf, v1)
                c2 = sample_rgb(tsdf, v2)
                colors[base] = wp.vec3ub(wp.uint8(c0[0]), wp.uint8(c0[1]), wp.uint8(c0[2]))
                colors[base + wp.int32(1)] = wp.vec3ub(
                    wp.uint8(c1[0]), wp.uint8(c1[1]), wp.uint8(c1[2])
                )
                colors[base + wp.int32(2)] = wp.vec3ub(
                    wp.uint8(c2[0]), wp.uint8(c2[1]), wp.uint8(c2[2])
                )

    # =====================================================================
    # Color sampling
    # =====================================================================

    @warp_kernel(f"sample_vertex_colors_kernel_bs{block_size}", enable_backward=False)
    def sample_vertex_colors_kernel(
        vertices: wp.array(dtype=wp.vec3),
        n_vertices: wp.int32,
        tsdf: BlockSparseTSDFWarp,
        colors: wp.array(dtype=wp.vec3ub),
    ):
        """Sample colors for mesh vertices from weighted RGB sums."""
        tid = wp.tid()
        if tid >= n_vertices:
            return

        pos = vertices[tid]

        center_offset_x = wp.float32(tsdf.grid_W) * 0.5
        center_offset_y = wp.float32(tsdf.grid_H) * 0.5
        center_offset_z = wp.float32(tsdf.grid_D) * 0.5

        vx = (pos[0] - tsdf.origin[0]) / tsdf.voxel_size + center_offset_x
        vy = (pos[1] - tsdf.origin[1]) / tsdf.voxel_size + center_offset_y
        vz = (pos[2] - tsdf.origin[2]) / tsdf.voxel_size + center_offset_z

        block_size_f = wp.float32(tsdf.block_size)
        bx = wp.int32(wp.floor(vx / block_size_f))
        by = wp.int32(wp.floor(vy / block_size_f))
        bz = wp.int32(wp.floor(vz / block_size_f))

        key = block_grid_to_key_coords(bx, by, bz)
        pool_idx = hash_lookup(tsdf.hash_table, key[0], key[1], key[2], tsdf.hash_capacity)

        if pool_idx < 0:
            colors[tid] = wp.vec3ub(wp.uint8(128), wp.uint8(128), wp.uint8(128))
            return

        rgb_grid = sample_rgb(tsdf, pos)
        colors[tid] = wp.vec3ub(
            wp.uint8(rgb_grid[0]), wp.uint8(rgb_grid[1]), wp.uint8(rgb_grid[2])
        )

    # Expose kernels on the instance.
    return {
        "count_surface_cubes_kernel": count_surface_cubes_kernel,
        "append_active_blocks_kernel": append_active_blocks_kernel,
        "count_surface_cubes_from_blocks_kernel": count_surface_cubes_from_blocks_kernel,
        "append_surface_cubes_kernel": append_surface_cubes_kernel,
        "append_surface_cubes_from_blocks_kernel": append_surface_cubes_from_blocks_kernel,
        "count_edges_block_sparse_kernel": count_edges_block_sparse_kernel,
        "generate_vertices_block_sparse_kernel": generate_vertices_block_sparse_kernel,
        "generate_triangles_shared_kernel": generate_triangles_shared_kernel,
        "count_triangles_kernel": count_triangles_kernel,
        "count_total_triangles_kernel": count_total_triangles_kernel,
        "generate_approximate_mesh_block_sparse_kernel": generate_approximate_mesh_block_sparse_kernel,
        "generate_approximate_mesh_atomic_kernel": generate_approximate_mesh_atomic_kernel,
        "generate_approximate_mesh_current_state_kernel": (
            generate_approximate_mesh_current_state_kernel
        ),
        "count_fast_mesh_block_triangles_kernel": count_fast_mesh_block_triangles_kernel,
        "generate_fast_mesh_current_state_kernel": generate_fast_mesh_current_state_kernel,
        "copy_rgb_images_to_texture_atlas_kernel": copy_rgb_images_to_texture_atlas_kernel,
        "project_fast_mesh_uvs_kernel": project_fast_mesh_uvs_kernel,
        "sample_vertex_colors_kernel": sample_vertex_colors_kernel,
    }
