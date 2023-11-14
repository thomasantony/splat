use nalgebra as na;
use euc::{Pipeline, TriangleList, Empty};
use na::{Vector2, Vector3, Vector4, SVector, VectorView3};
use crate::{gaussians::{Gaussian, self, GaussianList}, camera::Camera};


const VERTICES: &[Vector2<f32>] = &[
    Vector2::new(-1.,  1.),
    Vector2::new(-1.,  -1.),
    Vector2::new(1., -1.),
    Vector2::new(1., 1.),
];

const INDICES: &[usize] = &[0, 1, 2, 0, 2, 3];

fn gaussian_vertex_shader(vert: &Vector2<f32>, g_position_w: &Vector3<f32>, cov2d: &na::Matrix2<f32>, camera: &Camera) -> (Vector4<f32>, Vector3<f32>, Vector2<f32>)
{
    // Create conic, coordxy, gl_position for passing to fragment shader
    let mut gl_position = Vector4::zeros();

    let cov2d_inv =  cov2d.try_inverse().unwrap();
    let conic = Vector3::new(cov2d_inv[(0,0)], cov2d_inv[(0, 1)], cov2d_inv[(1, 1)]);

    // compute 3-sigma bounding box size
    // Divide out the camera width and height to get in NDC
    let bboxsize_cam = Vector2::new(3.0 * cov2d[(0,0)].sqrt(), 3.0 * cov2d[(1,1)].sqrt());

    // Divide out camera plane size to get bounding box size in NDC
    let wh = Vector2::new(camera.w, camera.h);
    let bboxsize_ndc = bboxsize_cam.component_div(&wh) * 2.0;

    // Coordxy value (used to evaluate gaussian, also in camera space coordinates)
    let coordxy = vert.component_mul(&bboxsize_cam);

    // compute g_pos_screen and gl_position
    let view_matrix = camera.get_view_matrix();
    let projection_matrix = camera.get_project_matrix();
    let position4 = Vector4::new(g_position_w[0], g_position_w[1], g_position_w[2], 1.0);
    let g_pos_view = view_matrix * position4;

    let mut g_pos_screen = projection_matrix * g_pos_view;
    g_pos_screen = g_pos_screen / g_pos_screen[3];

    gl_position[0] = vert[0] * bboxsize_ndc[0] + g_pos_screen[0];
    gl_position[1] = vert[1] * bboxsize_ndc[1] + g_pos_screen[1];
    gl_position[2] = g_pos_screen[2];
    gl_position[3] = g_pos_screen[3];

    (gl_position, conic, coordxy)
}


pub struct GaussianSplatPipeline01 {
    pub gaussians: Vec<Gaussian>,
    pub camera: Camera,
}

#[derive(Clone, Debug)]
pub struct VertexInstance {
    pub vert_idx: usize,
    pub gaussian_idx: usize,
}

impl GaussianSplatPipeline01{
    pub fn render_to_buffer(&self, color: &mut euc::Buffer<u32, 2>)
    {
        // Create vector of VertexInstances from combining vertices and gaussians
        let mut vertex_instances = Vec::new();
        let sorted_gaussians = gaussians::sort_gaussians(&self.gaussians, &self.camera.get_view_matrix());
        for g_idx in sorted_gaussians.iter() {
            // For each instance, push all the vertices
            for vert_idx in INDICES {
                vertex_instances.push(VertexInstance {
                    vert_idx: *vert_idx,
                    gaussian_idx: *g_idx,
                });
            }
        }
        self.render(
            vertex_instances,
            color,
            &mut Empty::default(),
        );

    }
}

impl<'r> Pipeline<'r> for GaussianSplatPipeline01 {
    type Vertex = VertexInstance;
    type VertexData = SVector<f32, 9>; // color, alpha, conic, coordxy
    type Primitives = TriangleList;
    type Fragment = Vector4<f32>;
    type Pixel = u32;

    fn vertex(&self, vert_inst: &Self::Vertex) -> ([f32; 4], Self::VertexData) {
        let gaussian = &self.gaussians[vert_inst.gaussian_idx];
        let world_position = &gaussian.position;
        let ray_direction = (world_position - self.camera.position).normalize();
        let (color, alpha) = gaussian.color(15, &ray_direction);

        let cov2d = gaussian.project_cov3d_to_screen(&self.camera);
        let (gl_position, conic, coordxy) = gaussian_vertex_shader(
            &VERTICES[vert_inst.vert_idx],
            world_position,
            &cov2d,
            &self.camera,
        );
        // Placeholder outputs for now
        let vertex_data = SVector::<f32, 9>::from_row_slice(
            &[
            color[0],
            color[1],
            color[2],
            alpha,
            conic[0],
            conic[1],
            conic[2],
            coordxy[0],
            coordxy[1],
            ]
        );

        ([gl_position[0], gl_position[1], gl_position[2], gl_position[3]], vertex_data)
    }

    fn fragment(&self, vert_output: Self::VertexData) -> Self::Fragment {
        let color_rgb  = Vector3::new(vert_output[0], vert_output[1], vert_output[2]);
        let alpha = vert_output[3];
        let conic = Vector3::new(vert_output[4], vert_output[5], vert_output[6]);
        let coordxy = Vector2::new(vert_output[7], vert_output[8]);

        // let power = -0.5 * (conic.x * coordxy.x * coordxy.x + conic.z * coordxy.y * coordxy.y) - conic.y * coordxy.x * coordxy.y;
        let power = -0.5 * (conic[0] * coordxy[0] * coordxy[0] + conic[2] * coordxy[1] * coordxy[1]) - conic[1] * coordxy[0] * coordxy[1];
        if power > 0.0
        {
            return Self::Fragment::zeros();
        }
        let alpha = 0.99f32.min(alpha * power.exp());
        if alpha < 1.0 / 255.0
        {
            return Self::Fragment::zeros();
        }
        Vector4::new(color_rgb[0], color_rgb[1], color_rgb[2], alpha)
    }

    fn blend(&self, old_color: Self::Pixel, new_color: Self::Fragment) -> Self::Pixel {
        // Do alpha blending and return blended color
        // old color also has alpha in highest byte
        let old_alpha = (old_color >> 24) & 0xff;
        let old_color = Vector4::new(
            ((old_color >> 16) & 0xff) as f32 / 255.0,
            ((old_color >> 8) & 0xff) as f32 / 255.0,
            (old_color & 0xff) as f32 / 255.0,
            old_alpha as f32 / 255.0,
        );
        let alpha = new_color[3];
        let blended_color = (1.0 - alpha) * old_color + alpha * new_color;
        let r = (blended_color[0] * 255.0) as u8;
        let g = (blended_color[1] * 255.0) as u8;
        let b = (blended_color[2] * 255.0) as u8;
        u32::from_le_bytes([
            b,
            g,
            r,
            (alpha * 255.0) as u8,
        ])
    }
}

// Gaussian splat pipeline for GaussianList type
pub struct GaussianSplatPipeline02 {
    pub gaussians: GaussianList,
    pub camera: Camera,
}

impl<'r> Pipeline<'r> for GaussianSplatPipeline02 {
    type Vertex = VertexInstance;
    type VertexData = SVector<f32, 9>; // color, alpha, conic, coordxy
    type Primitives = TriangleList;
    type Fragment = Vector4<f32>;
    type Pixel = u32;

    fn vertex(&self, vertex: &Self::Vertex) -> ([f32; 4], Self::VertexData) {
        let g_idx = vertex.gaussian_idx;
        let world_position = self.gaussians.positions.fixed_view::<3,1>(0,g_idx);

        let ray_direction = (world_position - self.camera.position).normalize();
        let (color, alpha) = self.gaussians.color(g_idx, 15, &ray_direction);

        let cov2d = self.gaussians.project_cov3d_to_screen(g_idx, &self.camera);
        let (gl_position, conic, coordxy) = gaussian_vertex_shader(
            &VERTICES[vertex.vert_idx],
            &world_position.into(),
            &cov2d,
            &self.camera,
        );
        let vertex_data = SVector::<f32, 9>::from_row_slice(
            &[
            color[0],
            color[1],
            color[2],
            alpha,
            conic[0],
            conic[1],
            conic[2],
            coordxy[0],
            coordxy[1],
            ]
        );
        ([gl_position[0], gl_position[1], gl_position[2], gl_position[3]], vertex_data)

    }

    fn fragment(&self, vert_output: Self::VertexData) -> Self::Fragment {
        let color_rgb  = Vector3::new(vert_output[0], vert_output[1], vert_output[2]);
        let alpha = vert_output[3];
        let conic = Vector3::new(vert_output[4], vert_output[5], vert_output[6]);
        let coordxy = Vector2::new(vert_output[7], vert_output[8]);

        // let power = -0.5 * (conic.x * coordxy.x * coordxy.x + conic.z * coordxy.y * coordxy.y) - conic.y * coordxy.x * coordxy.y;
        let power = -0.5 * (conic[0] * coordxy[0] * coordxy[0] + conic[2] * coordxy[1] * coordxy[1]) - conic[1] * coordxy[0] * coordxy[1];
        if power > 0.0
        {
            return Self::Fragment::zeros();
        }
        let alpha = 0.99f32.min(alpha * power.exp());
        if alpha < 1.0 / 255.0
        {
            return Self::Fragment::zeros();
        }
        Vector4::new(color_rgb[0], color_rgb[1], color_rgb[2], alpha)
    }

    fn blend(&self, old_color: Self::Pixel, new_color: Self::Fragment) -> Self::Pixel {
        // Do alpha blending and return blended color
        // old color also has alpha in highest byte
        let old_alpha = (old_color >> 24) & 0xff;
        let old_color = Vector4::new(
            ((old_color >> 16) & 0xff) as f32 / 255.0,
            ((old_color >> 8) & 0xff) as f32 / 255.0,
            (old_color & 0xff) as f32 / 255.0,
            old_alpha as f32 / 255.0,
        );
        let alpha = new_color[3];
        let blended_color = (1.0 - alpha) * old_color + alpha * new_color;
        let r = (blended_color[0] * 255.0) as u8;
        let g = (blended_color[1] * 255.0) as u8;
        let b = (blended_color[2] * 255.0) as u8;
        u32::from_le_bytes([
            b,
            g,
            r,
            (alpha * 255.0) as u8,
        ])
    }
}

impl GaussianSplatPipeline02{
    pub fn render_to_buffer(&self, color: &mut euc::Buffer<u32, 2>)
    {
        // Create vector of VertexInstances from combining vertices and gaussians
        let mut vertex_instances = Vec::new();
        let sorted_gaussians = self.gaussians.sort(&self.camera);
        for g_idx in sorted_gaussians.iter() {
            // For each instance, push all the vertices
            for vert_idx in INDICES {
                vertex_instances.push(VertexInstance {
                    vert_idx: *vert_idx,
                    gaussian_idx: *g_idx,
                });
            }
        }
        self.render(
            vertex_instances,
            color,
            &mut Empty::default(),
        );

    }
}
