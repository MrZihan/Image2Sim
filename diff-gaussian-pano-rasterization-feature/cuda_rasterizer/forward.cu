/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];


	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// 在 forward.cu 中替换原来的 computeCov2D
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
    // 1. 将 World Space 转换到 View Space
    float3 t = transformPoint4x3(mean, viewmatrix);

    // 2. 360全景投影的雅可比矩阵 (Jacobian)
    // t.x, t.y, t.z 是 View Space 坐标
    float x = t.x;
    float y = t.y;
    float z = t.z;

    float x2 = x * x;
    float y2 = y * y;
    float z2 = z * z;
    float xz = x2 + z2;
    float r = sqrt(xz); // 水平距离
    float xyz = xz + y2; // 总距离的平方

    // 防止除以0的奇异点 (在极点附近)
    if (xyz < 1e-6f) return { 0.0f, 0.0f, 0.0f };
    
	float r_safe = r + 1e-6f;
	float xyz_safe = xyz + 1e-6f;

    float factor_u = focal_x; 
    float factor_v = focal_y; 

	// Recalculate derivatives with safe denominators
	float du_dx = factor_u * (z / (xz + 1e-7f)); // Protect xz
	float du_dy = 0.0f;
	float du_dz = factor_u * (-x / (xz + 1e-7f));

	float dv_dx = factor_v * (-x * y) / (r_safe * xyz_safe);
	float dv_dy = factor_v * r / xyz_safe;
	float dv_dz = factor_v * (-z * y) / (r_safe * xyz_safe);

    glm::mat3 J = glm::mat3(
    du_dx, dv_dx, 0.0f, // Column 0
    du_dy, dv_dy, 0.0f, // Column 1
    du_dz, dv_dz, 0.0f  // Column 2
	);

	glm::mat3 W = glm::mat3(
    viewmatrix[0], viewmatrix[1], viewmatrix[2],
    viewmatrix[4], viewmatrix[5], viewmatrix[6],
    viewmatrix[8], viewmatrix[9], viewmatrix[10]
	);

	// 3. 计算总变换矩阵 T = J * W
	// 在 GLM 中，matrix * matrix 是标准的矩阵乘法
	glm::mat3 T = J * W; 

	// 4. 构造 World Space Covariance Vrk
	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]
	);

	// 5. 计算 Screen Space Covariance
	// cov = T * Vrk * T^T
	glm::mat3 cov = T * Vrk * glm::transpose(T);

    // Apply low-pass filter
    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;
    return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// 在 forward.cu 中替换 preprocessCUDA
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
    const float* orig_points,
    const glm::vec3* scales,
    const float scale_modifier,
    const glm::vec4* rotations,
    const float* opacities,
    const float* shs,
    bool* clamped,
    const float* cov3D_precomp,
    const float* colors_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const glm::vec3* cam_pos,
    const int W, int H,
    const float tan_fovx, float tan_fovy,
    const float focal_x, float focal_y,
    int* radii,
    float2* points_xy_image,
    float* depths,
    float* cov3Ds,
    float* rgb,
    float4* conic_opacity,
    const dim3 grid,
    uint32_t* tiles_touched,
    bool prefiltered)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    radii[idx] = 0;
    tiles_touched[idx] = 0;

    // 1. 坐标变换 World -> View
    float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
    float3 p_view = transformPoint4x3(p_orig, viewmatrix);

    // 2. 360 Frustum Culling (简单的距离剔除，因为360度视野下没有视锥体边界)
    // 只要不在极近距离（避免奇异点），都应该渲染
    float dist_sq = p_view.x * p_view.x + p_view.y * p_view.y + p_view.z * p_view.z;
    if (dist_sq < 0.01f) return; 

    // 3. Equirectangular Projection (View Space -> Screen Space)
    // 假设 p_view 坐标系: +Z forward, +X right, +Y down (根据具体数据集可能需要调整 atan2 的参数顺序)
    // 标准惯例: atan2(x, z) 得到 yaw
    
    float r = sqrt(p_view.x * p_view.x + p_view.z * p_view.z);
    float phi = atan2(p_view.x, p_view.z); // Longitude: [-PI, PI]
    float theta = atan2(p_view.y, r);      // Latitude: [-PI/2, PI/2]

    // Normalize to [0, 1] then scale to [0, W]
    // u: (-PI, PI) -> (0, 1) => (phi + PI) / (2*PI)
    // v: (-PI/2, PI/2) -> (0, 1) => (theta + PI/2) / PI
    
    float u_ndc = (phi + 3.14159265f) / (2.0f * 3.14159265f);
    float v_ndc = (theta + 1.57079632f) / 3.14159265f;

    // 简单的 clamp，防止刚好在边缘溢出
    u_ndc = min(max(u_ndc, 0.0f), 1.0f);
    v_ndc = min(max(v_ndc, 0.0f), 1.0f);

    float2 point_image = { u_ndc * W, v_ndc * H };


    // 4. 计算协方差 (传入修改后的 factor)
    // 如果 cov3D 已经预计算，则使用；否则计算。
    const float* cov3D;
    if (cov3D_precomp != nullptr)
    {
        cov3D = cov3D_precomp + idx * 6;
    }
    else
    {
        computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
        cov3D = cov3Ds + idx * 6;
    }

    float factor_u = projmatrix[0];
	float factor_v = projmatrix[5];
    float3 cov = computeCov2D(p_orig, factor_u, factor_v, tan_fovx, tan_fovy, cov3D, viewmatrix);

    // 5. Invert Covariance (EWA) - 保持不变
    float det = (cov.x * cov.z - cov.y * cov.y);
    if (det == 0.0f) return;
    float det_inv = 1.f / det;
    float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

    // 6. Compute Extent and Tiles - 保持不变
    float mid = 0.5f * (cov.x + cov.z);
    float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
    float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
    float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
    
    uint2 rect_min, rect_max;
    getRect(point_image, my_radius, rect_min, rect_max, grid);
    if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
        return;

    // 7. Color handling - 保持不变
    if (colors_precomp == nullptr)
    {
        glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
        rgb[idx * C + 0] = result.x;
        rgb[idx * C + 1] = result.y;
        rgb[idx * C + 2] = result.z;
    }

    // Store Output
    float dist_r = sqrt(p_view.x * p_view.x + p_view.z * p_view.z + p_view.y * p_view.y);
	depths[idx] = dist_r;
    radii[idx] = my_radius;
    points_xy_image[idx] = point_image;
    conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
    tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ semantics,
	const float* __restrict__ depths, 
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_feature_map,
	float* __restrict__ out_depth,
	float* __restrict__ out_alpha) 
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	float prev_depp = 0.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	float SF[NUM_SEMANTIC_CHANNELS] = { 0 };
	float D = { 0 };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}

		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++){
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
			}

			float depp = depths[collected_id[j]];
			float w = alpha*T;
			D += depp*w;
			
			for (int ch = 0; ch < NUM_SEMANTIC_CHANNELS; ch++){
				SF[ch] += semantics[collected_id[j] * NUM_SEMANTIC_CHANNELS + ch] * alpha * T; 
			}

			T = test_T;
			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		// rgb
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		// depth
		out_depth[pix_id] = D;
		// feature
		for (int ch = 0; ch < NUM_SEMANTIC_CHANNELS; ch++)                 
			out_feature_map[ch * H * W + pix_id] = SF[ch];

		out_alpha[pix_id] = 1.0f - T;
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float* semantic_feature,
	const float* depths, 
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* out_feature_map,
	float* out_depth,
	float* out_alpha) 
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		semantic_feature,
		depths, 
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		out_feature_map,
		out_depth,
		out_alpha);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}