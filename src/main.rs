use euc::{buffer::Buffer2d, rasterizer, Pipeline, TriangleList, Empty};
use nalgebra as na;
use splat::{gaussians::{Gaussian, self}, camera::Camera};
use na::{Vector2, Vector3, Vector4, Matrix3, Matrix4, Matrix3x2, RowVector3, Vector6, SVector};

struct Triangle {
    pub vertices: Vec<Vector2<f32>>,
    pub gaussians: Vec<Gaussian>,
    pub camera: Camera,
}

#[derive(Clone, Debug)]
pub struct VertexInstance {
    pub vert_idx: usize,
    pub gaussian_idx: usize,
}

impl<'r> Pipeline<'r> for Triangle {
    type Vertex = VertexInstance;
    type VertexData = SVector<f32, 9>; // color, alpha, conic, coordxy
    type Primitives = TriangleList;
    type Fragment = Vector4<f32>;
    type Pixel = u32;

    fn vertex(&self, vert_inst: &Self::Vertex) -> ([f32; 4], Self::VertexData) {
        let gaussian = &self.gaussians[vert_inst.gaussian_idx];
        let world_position: na::Matrix<f32, na::Const<3>, na::Const<1>, na::ArrayStorage<f32, 3, 1>> = gaussian.position;
        let ray_direction = (world_position - self.camera.position).normalize();
        let (color, alpha) = gaussian.color(15, &ray_direction);

        let vert = &self.vertices[vert_inst.vert_idx];

        // Create conic, coordxy, gl_position for passing to fragment shader
        let mut gl_position: na::Matrix<f32, na::Const<4>, na::Const<1>, na::ArrayStorage<f32, 4, 1>> = Vector4::zeros();

        let cov2d = gaussian.project_cov3d_to_screen(&self.camera);

        let cov2d_inv =  cov2d.try_inverse().unwrap();
        let conic = Vector3::new(cov2d_inv[(0,0)], cov2d_inv[(0, 1)], cov2d_inv[(1, 1)]);

        // compute 3-sigma bounding box size
        // Divide out the camera width and height to get in NDC
        let bboxsize_cam = Vector2::new(3.0 * cov2d[(0,0)].sqrt(), 3.0 * cov2d[(1,1)].sqrt());

        // Divide out camera plane size to get bounding box size in NDC
        let wh = Vector2::new(self.camera.w, self.camera.h);
        let bboxsize_ndc = bboxsize_cam.component_div(&wh) * 2.0;

        // Coordxy value (used to evaluate gaussian, also in camera space coordinates)
        let coordxy = vert.component_mul(&bboxsize_cam);

        // compute g_pos_screen and gl_position
        let view_matrix = self.camera.get_view_matrix();
        let projection_matrix = self.camera.get_project_matrix();
        let position4 = Vector4::new(world_position[0], world_position[1], world_position[2], 1.0);
        let g_pos_view = view_matrix * position4;

        let mut g_pos_screen = projection_matrix * g_pos_view;
        g_pos_screen = g_pos_screen / g_pos_screen[3];

        gl_position[0] = vert[0] * bboxsize_ndc[0] + g_pos_screen[0];
        gl_position[1] = vert[1] * bboxsize_ndc[1] + g_pos_screen[1];
        gl_position[2] = g_pos_screen[2];
        gl_position[3] = g_pos_screen[3];

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
        let mut alpha = 0.99f32.min(alpha * power.exp());
        if alpha < 1.0 / 255.0
        {
            alpha = 0.0;
        }
        Vector4::new(color_rgb[0]*alpha, color_rgb[1]*alpha, color_rgb[2]*alpha, alpha)
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
        // println!("Blended color: {:?}", blended_color);
        let r = (blended_color[0] * 255.0) as u32;
        let g = (blended_color[1] * 255.0) as u32;
        let b = (blended_color[2] * 255.0) as u32;
        (r << 16) | (g << 8) | b
    }
}

const W: usize = 1280/2;
const H: usize = 720/2;

fn main() {
    let mut color = Buffer2d::fill([W, H], 0);
    const VERTICES: &[Vector2<f32>] = &[
        Vector2::new(-1.,  1.),
        Vector2::new(-1.,  -1.),
        Vector2::new(1., -1.),
        Vector2::new(1., 1.),
    ];

    const INDICES: &[usize] = &[0, 1, 2, 0, 2, 3];

    println!("Loading gaussians from ply file");
    let mut gaussians = gaussians::load_from_ply("notes/point_cloud.ply");
    // let mut gaussians = gaussians::naive_gaussians();

    // Compute cov3d for each gaussian
    println!("Computing cov3d for each gaussian");
    for gaussian in gaussians.iter_mut() {
        gaussian.compute_cov3d();
    }

    let camera_pos = Vector3::new(-0.57651054, 2.99040512, -0.03924271);
    let camera = Camera::new(H as f32, W as f32, Some(camera_pos));
    let sorted_gaussians = gaussians::sort_gaussians(&gaussians, &camera.get_view_matrix());
    // Create vector of VertexInstances from combining vertices and gaussians
    let mut vertex_instances = Vec::new();
    for g_idx in sorted_gaussians.iter() {
        // For each instance, push all the vertices
        for vert_idx in INDICES {
            vertex_instances.push(VertexInstance {
                vert_idx: *vert_idx,
                gaussian_idx: *g_idx,
            });
        }
    }

    let pipeline = Triangle {
        vertices: VERTICES.to_vec(),
        gaussians: gaussians.to_vec(),
        camera,
    };

    println!("Rendering");
    pipeline.render(
        vertex_instances,
        &mut color,
        &mut Empty::default(),
    );

    let mut win = minifb::Window::new("Splat", W, H, minifb::WindowOptions::default()).unwrap();
    while win.is_open() {
        win.update_with_buffer(color.raw(), W, H).unwrap();
        // win.update_with_buffer(color.raw(), W, H).unwrap();
    }
}
