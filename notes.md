- f_dc_0 .. f_dc_2
    - "DC" or constant componetns of SH for each color
- f_rest_0 .. f_rest_44
    - 15 coefficients each for red, green and blue

- rot_0 .. rot_3, scale_0 .. scale_2 -> camera params?

- nx, ny, nz and x, y, z models the 3D gaussian
    - zwicker et. all 2001 says how to project 3D gaussian into 2D to compute a Jacobian
    - This is further combined witht eh camera projection matrix W (with position and rotation)
    - the projected covariance matrix (2x2 matrix E' in image space), determines the size and shape of the 2D splats when rendering
    - No explicit raycasting - unclear how that works

- 3D gaussians use four parameters: position (x, y, z), covariance (how it’s stretched/scaled: 3x3), color (RGB), and alpha to represent the Gaussian.
- SH harmonic functions are used to represent view-dependant color
    - spherical harmonics refer to a set of functions that take inputs of distance from the center (r), polar angle (θ), and azimuthal angle (φ) and produce an output value (P) at a point on a sphere’s surface.


- Rasterize (Algorithm 2) in paper (M, S, C, A, V)
    - M, S -> gaussian means and covariances
    - C, A -> Color and alpha
    - V -> Camera
    - "Frustum Culling"
        - involves keeping only the 3D Gaussians that can be observed from the given camera V.
        - involves creating a frustum from the camera center and removing Gaussians that are not visible (culled) because they are outside the frustum
        - “Our method starts by splitting the screen into 16×16 tiles, and then proceeds to cull 3D Gaussians against the view frustum and each tile. Specifically, we only keep Gaussians with a 99% confidence interval intersecting the view frustum. Additionally, we use a guard band to trivially reject Gaussians at extreme positions (i.e., those with means close to the near plane and far outside the frustum), since computing their 2D covariance would be unstable.”
    - ScreenspaceGaussian(M, S, V) to project 3D gaussians into 2D (formula on page 4)
        - `$\sigma' = J W \sigma W^T J^T$`
        - J is the Jacobian of the projective transformation matrix that transforms camera coordinates into image coordinates (https://xoft.tistory.com/49) - camera specific?
        - W is the viewing transformation matrix that converts the world coordinate system into the camera coordinate system.
        - related to camera calibration
        - references: https://www.youtube.com/watch?v=2XM2Rb2pfyQ , https://towardsdatascience.com/what-are-intrinsic-and-extrinsic-camera-parameters-in-computer-vision-7071b72fb8ec
    - Divides the image into 16x16 tiles using CreateTiles.
        - the DuplicateWithKeys step duplicates Gaussians that overlap (between tiles) as many times as necessary and assigns each of them a 64-bit key consisting of a 32-bit view space depth and a 32-bit tile ID.
        - Gaussians with keys are then sorted using single GPU radix sort, as shown in the SortByKeys step.
        - blending is performed based on the initial sorting (no further ray-casting)
        - These approximations become negligible as splats approach the size of individual pixels
        - After SortByKeys, each tile maintains a list of Gaussians sorted by depth (depth from where to where?)
        - Next, in the IdentifyTileRange step, we identify the start and end of Gaussians with the same tile ID, and create and manage Gaussian lists for each tile based on this information (tile-ranges called R).
    - For each tile t, it receives the tile range R obtained earlier, which is essentially the list of Gaussians.
    - Blend the gaussians
        - For each pixel i within a tile, it accumulates color and alpha values from front to back, maximizing parallelism for data loading/sharing and processing. When a pixel reaches a target saturation of alpha (i.e., alpha goes to 1), the corresponding thread stops. At regular intervals, threads within a tile are queried, and processing of the entire tile terminates when all pixels have saturated.

- Implementation in code of rendering once we have sorted gaussians
    - Use scale, rot, and position to define the 3D covariance matrix
    - Use camera intrinsincs to project it into a 2D matrix
    - Convert 2D matrix into coefficients of an ellipse equation (??)
    - Use SH coefficients to compute color from a unit vector (gaussian_pos.xyz - cam_pos)
    - Use ellipse coefficients to compute "power" based on position in image frame (quad?)
        - multiply by opacity
    - Example does not seem to use tiling, and simply renders the gaussians in sorted order


** Random thoughts **
- FPGA optimized for gaussians rather than triangles?
- Use conical shapes instead of gaussians?

testing


** Code **
- extracted out three gaussians into a file for easier testing
- Code below from shaders in tinygs - https://github.com/limacv/GaussianSplattingViewer
- gaussian cov defined in terms of pos, rot, scale and opacity
```c
mat3 computeCov3D(vec3 scale, vec4 q)  // should be correct
{
    mat3 S = mat3(0.f);
    S[0][0] = scale.x;
	S[1][1] = scale.y;
	S[2][2] = scale.z;
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

    mat3 R = mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

    mat3 M = S * R;
    mat3 Sigma = transpose(M) * M;
    return Sigma;
}
```
- Rotated to image frame using camera intrinsics ``

- SH impl from python-cuda example - seems to use direction vector instead of angles
```c
// Spherical functions from svox2
__device__ __constant__ const float C0 = 0.28209479177387814;
__device__ __constant__ const float C1 = 0.4886025119029199;
__device__ __constant__ const float C2[] = {
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
};

__device__ __constant__ const float C3[] = {
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
};

__device__ __inline__ void calc_sh(
    const int basis_dim,
    const float* __restrict__ dir,
    float* __restrict__ out) {
    out[0] = C0;
    const float x = dir[0], y = dir[1], z = dir[2];
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, yz = y * z, xz = x * z;
    switch (basis_dim) {
        case 9:
            out[4] = C2[0] * xy;
            out[5] = C2[1] * yz;
            out[6] = C2[2] * (2.0 * zz - xx - yy);
            out[7] = C2[3] * xz;
            out[8] = C2[4] * (xx - yy);
            [[fallthrough]];
        case 4:
            out[1] = -C1 * y;
            out[2] = C1 * z;
            out[3] = -C1 * x;
    }
}
```

- https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/sh_utils.py#L117
    - SH2RGB


## Vertex shader version
- Computes the color/opacity 4 times per gaussian because of how vertex shaders work
- N instances of a "quad" (rectangle) the size of the viewing area is rendered
    - the vertex shader computes the color and opacity for each of the four vertices for each instance
    - the fragment shader computes the color by sampling the gaussian at the interior points of the quad
    - the bitmaps produced like this are squashed down using the alpha channel to generate the image
    - gaussians are sorted by depth from camera
