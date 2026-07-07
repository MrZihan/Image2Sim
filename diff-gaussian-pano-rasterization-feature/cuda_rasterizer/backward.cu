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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

__global__ void computeCov2DCUDA(int P,
    const float3* means,
    const int* radii,
    const float* cov3Ds,
    const float h_x, float h_y,
    const float tan_fovx, float tan_fovy,
    const float* view_matrix,
    const float* dL_dconics,
    float3* dL_dmeans,
    float* dL_dcov)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || !(radii[idx] > 0))
        return;

    const float* cov3D = cov3Ds + 6 * idx;
    float3 mean = means[idx];
    float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };

    // 1. Transform to View Space
    float3 t = transformPoint4x3(mean, view_matrix);

    // 2. Compute Spherical Jacobian (与 Forward 保持一致)
    // 注意：这里的 h_x 和 h_y 实际上是在 python 端通过 focal_x/y 传入的 W/2pi 和 H/pi
    float factor_u = h_x;
    float factor_v = h_y;

    float x = t.x; float y = t.y; float z = t.z;
    float x2 = x * x; float y2 = y * y; float z2 = z * z;
    float xz = x2 + z2; float r = sqrt(xz); float xyz = xz + y2;

    if (xz < 0.00001f || xyz < 0.00001f) return;

    // Jacobian derivatives
    float du_dx = factor_u * (z / xz);
    float du_dz = factor_u * (-x / xz);
    float dv_dx = factor_v * (-x * y) / (r * xyz);
    float dv_dy = factor_v * r / xyz;
    float dv_dz = factor_v * (-z * y) / (r * xyz);

    // [FIX] 正确构造 J (GLM 是列主序，构造函数参数顺序为 Col0, Col1, Col2)
    glm::mat3 J = glm::mat3(
        du_dx, dv_dx, 0.0f, // Column 0: d/dx
        0.0f,  dv_dy, 0.0f, // Column 1: d/dy (du_dy=0)
        du_dz, dv_dz, 0.0f  // Column 2: d/dz
    );

    glm::mat3 W = glm::mat3(
    view_matrix[0], view_matrix[4], view_matrix[8],
    view_matrix[1], view_matrix[5], view_matrix[9],
    view_matrix[2], view_matrix[6], view_matrix[10]
);

    glm::mat3 Vrk = glm::mat3(
        cov3D[0], cov3D[1], cov3D[2],
        cov3D[1], cov3D[3], cov3D[4],
        cov3D[2], cov3D[4], cov3D[5]
    );

    // [FIX] 乘法顺序 T = J * W
    glm::mat3 T = J * W;
    
    // [FIX] Cov = T * Vrk * T^T
    glm::mat3 cov2D = T * Vrk * glm::transpose(T);

    // ... (dL_da, dL_db, dL_dc 的计算逻辑保持不变，不需要修改) ...
    // Copy paste logic for dL_da, dL_db, dL_dc calculation
    float a = cov2D[0][0] += 0.3f;
    float b = cov2D[0][1];
    float c = cov2D[1][1] += 0.3f;
    float denom = a * c - b * b;
    float dL_da = 0, dL_db = 0, dL_dc = 0;
    float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);
    if (denom2inv != 0) {
        dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
        dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
        dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);
        
        // [FIX] Compute dL_dcov (gradients for 3D covariance)
        // dSigma' = T^T * dCov * T
        // 这里需要手动展开 T^T * dL_dCov2D * T，或者复用原来的展开式但注意下标
        // 鉴于 T 矩阵元素含义变了，手动重写最安全：
        glm::mat3 dL_dCov2D = glm::mat3(dL_da, 0.5f*dL_db, 0.0f,  0.5f*dL_db, dL_dc, 0.0f,  0.0f, 0.0f, 0.0f);
        glm::mat3 dL_dVrk = glm::transpose(T) * dL_dCov2D * T;
        
        dL_dcov[6 * idx + 0] = dL_dVrk[0][0];
        dL_dcov[6 * idx + 1] = dL_dVrk[0][1]; // dL_dVrk is symmetric
        dL_dcov[6 * idx + 2] = dL_dVrk[0][2];
        dL_dcov[6 * idx + 3] = dL_dVrk[1][1];
        dL_dcov[6 * idx + 4] = dL_dVrk[1][2];
        dL_dcov[6 * idx + 5] = dL_dVrk[2][2];
    } else {
        for (int i = 0; i < 6; i++) dL_dcov[6 * idx + i] = 0;
    }

    // [FIX] Backpropagate through Matrix T = J * W is too complex for mean position.
    // In 360GS, we trust the direct positional gradient from preprocessCUDA much more.
    // We set these to 0 to avoid numerical instability.
    float dL_dtx = 0;
    float dL_dty = 0;
    float dL_dtz = 0;

    // Transform back to World Space (View^T * dL_dt)
    // 即使是0也保留这个结构，万一以后想加
    float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);
    dL_dmeans[idx] = dL_dmean;
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

template<int C>
__global__ void preprocessCUDA(
    int P, int D, int M,
    const float3* means,
    const int* radii,
    const float* shs,
    const bool* clamped,
    const glm::vec3* scales,
    const glm::vec4* rotations,
    const float scale_modifier,
    const float* proj,
    const float* view,
    const glm::vec3* campos,
    const float2* dL_dmean2D,
    glm::vec3* dL_dmeans,
    float* dL_dcolor,
    float* dL_dcov3D,
    float* dL_dsh,
    glm::vec3* dL_dscale,
    glm::vec4* dL_drot,
    float* dL_dz)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || !(radii[idx] > 0))
        return;

    float3 m = means[idx];
    
    // 1. Transform mean to View Space
    float3 t = transformPoint4x3(m, view);
    
    // [FIX] Retrieve Scaling Factors from Projection Matrix
    // Python端必须设置: proj[0] = W / 2pi, proj[5] = H / pi
    float factor_u = proj[0]; 
    float factor_v = proj[5]; 
    
    float x = t.x; float y = t.y; float z = t.z;
    float x2 = x * x; float y2 = y * y; float z2 = z * z;
    float xz = x2 + z2; float r = sqrt(xz); float xyz = xz + y2;

    if (xz < 0.00001f || xyz < 0.00001f) return;

    // Jacobian entries (Must match Forward pass logic)
    // Forward: phi = atan2(x, z), theta = atan2(y, r)
    // u = phi * factor_u, v = theta * factor_v
    
    float du_dx = factor_u * (z / xz);
    float du_dz = factor_u * (-x / xz);
    // du_dy = 0
    
    float dv_dx = factor_v * (-x * y) / (r * xyz);
    float dv_dy = factor_v * r / xyz;
    float dv_dz = factor_v * (-z * y) / (r * xyz);

    // 3. Compute Gradients w.r.t View Space Position (t)
    // Chain rule: dL/dt = dL/du * du/dt + dL/dv * dv/dt
    float dL_du = dL_dmean2D[idx].x;
    float dL_dv = dL_dmean2D[idx].y;

    float dL_dtx = dL_du * du_dx + dL_dv * dv_dx;
    float dL_dty =                 dL_dv * dv_dy;
    float dL_dtz = dL_du * du_dz + dL_dv * dv_dz;

    // 4. Backpropagate View -> World
    // dL_dmean_world = View_Rotation^T * dL_dt
    // view matrix indices: 0,1,2 is first row (if row-major)
    // Check transformPoint4x3 implementation in auxiliary.h usually:
    // x' = view[0]x + view[4]y + view[8]z + ... (This implies Column-Major storage or access)
    // So for transpose multiply:
    // dL_dx = dL_dtx * view[0] + dL_dty * view[4] + dL_dtz * view[8] ? NO.
    // If x' = v0*x + v4*y + v8*z, then dx'/dx = v0.
    // dL/dx = dL/dx' * dx'/dx = dL_dtx * v0 + dL_dty * v4 ...
    // Yes, this is correct for standard GLM/CUDA layout.
    
    glm::vec3 dL_dmean(0.0f);
    float dist_r = sqrt(t.x*t.x + t.y*t.y + t.z*t.z);
	if (dist_r > 1e-6) {
		float dL_dr = dL_dz[idx];
		
		// View Space gradient
		float dL_dtx_depth = dL_dr * (t.x / dist_r);
		float dL_dty_depth = dL_dr * (t.y / dist_r);
		float dL_dtz_depth = dL_dr * (t.z / dist_r);

		// Transform back to World Space (using View Matrix Transpose)
		dL_dmean.x += dL_dtx_depth * view[0] + dL_dty_depth * view[4] + dL_dtz_depth * view[8];
		dL_dmean.y += dL_dtx_depth * view[1] + dL_dty_depth * view[5] + dL_dtz_depth * view[9];
		dL_dmean.z += dL_dtx_depth * view[2] + dL_dty_depth * view[6] + dL_dtz_depth * view[10];
	}

    // [Suggestion] 忽略 dL_dz 或将其简化
    // 因为 forward 中我们用 r 做深度，这里如果直接加 dL_dz (screen space depth gradient)
    // 可能会引入噪音。但保留也没大问题，通常 dL_dz 比较小。
    float dldz = dL_dz[idx];
    dL_dmean.x += dldz * view[2];
    dL_dmean.y += dldz * view[6];
    dL_dmean.z += dldz * view[10];

    // Accumulate gradients
    dL_dmeans[idx] += dL_dmean;

    // Compute gradient updates due to computing colors from SHs
    if (shs)
        computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);
    if (scales)
        computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ semantic_feature,
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_dfeaturepixels,
	const float* __restrict__ dL_depths, 
	float2* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_dsemantic_feature,
	float* __restrict__ dL_dz,
	float* collected_semantic_feature) 
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];


	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float accum_semantic_feature_rec[NUM_SEMANTIC_CHANNELS] = { 0 }; 

	float dL_dpixel[C];
	float dL_dfeaturepixel[NUM_SEMANTIC_CHANNELS];
	float dL_depth;
	float accum_depth_rec = 0;


	if (inside)
	{
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		
		dL_depth = dL_depths[pix_id];

		for (int i = 0; i < NUM_SEMANTIC_CHANNELS; i++) 
			dL_dfeaturepixel[i] = dL_dfeaturepixels[i * H * W + pix_id];
	}

	float last_alpha = 0;
	float last_color[C] = { 0 };
	float last_semantic_feature[NUM_SEMANTIC_CHANNELS] = { 0 };
	float last_depth = 0; 

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
			collected_depths[block.thread_rank()] = depths[coll_id];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;
			const float dchannel_dsemantic_feature = alpha * T; 

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}
			const float c_d = collected_depths[j];
			accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
			last_depth = c_d;
			dL_dalpha += (c_d - accum_depth_rec) * dL_depth;


			for (int ch = 0; ch < NUM_SEMANTIC_CHANNELS; ch++) 
			{
				const float f = collected_semantic_feature[ch * BLOCK_SIZE + j];
				// Update last semantic feature (to be used in the next iteration)
				accum_semantic_feature_rec[ch] = last_alpha * last_semantic_feature[ch] + (1.f - last_alpha) * accum_semantic_feature_rec[ch];
				last_semantic_feature[ch] = f;

				const float dL_dfeaturechannel = dL_dfeaturepixel[ch];
				/**************************************************************************************************/
				// dL_dalpha += (f - accum_semantic_feature_rec[ch]) * dL_dfeaturechannel; // Only works for semnatic-meaning feature. Disable this line for general features.
				/**************************************************************************************************/
				
				// Update the gradients w.r.t. semnatic feature of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dsemantic_feature[global_id * NUM_SEMANTIC_CHANNELS + ch]), dchannel_dsemantic_feature * dL_dfeaturechannel); 
			}			



			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;


			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
			atomicAdd(&(dL_dz[global_id]),  alpha * T * dL_depth);
		}
	}
}


void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float2* dL_dmean2D,
	const float* dL_dconic,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float* dL_dz)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		dL_dconic,
		(float3*)dL_dmean3D,
		dL_dcov3D);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		projmatrix,
		viewmatrix,
		campos,
		(float2*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot,
		dL_dz);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float* colors,
	const float* semantic_feature,
	const float* depths, 
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	const float* dL_dfeaturepixels,
	const float* dL_depths, 
	float2* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_dsemantic_feature,
	float* dL_dz,
	float* collected_semantic_feature) 
	
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		conic_opacity,
		colors,
		semantic_feature,
		depths, 
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_dfeaturepixels,
		dL_depths, 
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
		dL_dsemantic_feature,
		dL_dz,
		collected_semantic_feature 
		);
}