use euc::{buffer::Buffer2d, rasterizer, Pipeline, TriangleList, Empty};
use nalgebra as na;
use splat::{gaussians::{Gaussian, self}, camera::Camera};
use na::{Vector2, Vector3, Vector4, Matrix3, Matrix4, Matrix3x2, RowVector3, Vector6};

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

pub struct VertexOutput {
    pub gl_position: Vector4<f32>,
}

impl<'r> Pipeline<'r> for Triangle {
    type Vertex = VertexInstance;
    type VertexData = Vector6<f32>; // color, alpha, texcoord
    type Primitives = TriangleList;
    type Fragment = Vector4<f32>;
    type Pixel = u32;

    fn vertex(&self, vert_inst: &Self::Vertex) -> ([f32; 4], Self::VertexData) {
        let gaussian = &self.gaussians[vert_inst.gaussian_idx];
        let world_position: na::Matrix<f32, na::Const<3>, na::Const<1>, na::ArrayStorage<f32, 3, 1>> = gaussian.position;
        let ray_direction = (world_position - self.camera.position).normalize();

        let (color, alpha) = gaussian.color(15, &ray_direction);

        let covariance = gaussian.projected_covariance_of_ellipsoid(&self.camera);
        let semi_axes = gaussian.extract_scale_of_covariance(&covariance);


        let rotation = gaussian.extract_rotation_of_ellipse(&covariance);
        let translation = gaussian.extract_translation_of_ellipse(&covariance);

        const ELLIPSE_SIZE_BIAS: f32 = 0.2f32;
        let mut transformation = Matrix3x2::<f32>::zeros();
        let transform_row0 = Vector2::new(rotation[1], -rotation[0]) * (ELLIPSE_SIZE_BIAS + semi_axes[0]);
        let transform_row1 = Vector2::new(rotation[0], rotation[1]) * (ELLIPSE_SIZE_BIAS + semi_axes[1]);
        transformation.set_row(0, &transform_row0.transpose());
        transformation.set_row(1, &transform_row1.transpose());
        transformation.set_row(2, &translation.transpose());

        let T = Matrix3::from_rows(
            &[
                RowVector3::new(transformation[(0,0)], transformation[(0,1)], 0.0),
                RowVector3::new(transformation[(1,0)], transformation[(1,1)], 0.0),
                RowVector3::new(transformation[(2,0)], transformation[(2,1)], 1.0)
            ]
        );

        let field_of_view_y = std::f32::consts::PI * 0.5;
        let view_height = (field_of_view_y * 0.5).tan();
        let view_width = (512 as f32 / 512 as f32) / view_height;
        let view_size =  Vector2::new(view_width, view_height);

        let vert_pos = self.vertices[vert_inst.vert_idx];
        let v3 = T * Vector3::new(vert_pos[0], vert_pos[1], 1.0);
        let v2 = v3.fixed_rows::<2>(0).component_div(&view_size);

        let gl_texcoord = vert_pos;
        let gl_position = Vector4::new(v2[0], v2[1], 0.0, 1.0);

        ([gl_position[0], gl_position[1], gl_position[2], gl_position[3]], Vector6::new(color[0], color[1], color[2], alpha, gl_texcoord[0], gl_texcoord[1]))
    }

    fn fragment(&self, col_texcoord: Self::VertexData) -> Self::Fragment {
        // println!("Fragment: {:?}", col);
        let color = Vector4::new(col_texcoord[0], col_texcoord[1], col_texcoord[2], col_texcoord[3]);
        let gl_texcoord = Vector2::new(col_texcoord[4], col_texcoord[5]);
        let power = gl_texcoord.dot(&gl_texcoord);
        let alpha = color[3] * (-0.5 * power).exp();
        if alpha < 1.0/255.0 {
            // Return black with zero alpha
            Vector4::new(0.0, 0.0, 0.0, 0.0)
        }else{
            let color = Vector4::new(color[0]*alpha, color[1]*alpha, color[2]*alpha, alpha);
            color
        }
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

const W: usize = 640;
const H: usize = 480;

fn main() {
    let [w, h] = [640, 480];
    let mut color = Buffer2d::fill([w, h], 0);
    const VERTICES: &[Vector2<f32>] = &[
        Vector2::new(-1.,  1.),
        Vector2::new(-1.,  -1.),
        Vector2::new(1., -1.),
        Vector2::new(1., 1.),
    ];

    const INDICES: &[usize] = &[0, 1, 2, 0, 2, 3];

    let gaussians = gaussians::load_from_ply("simple2.ply");
    let camera = Camera::new(w as f32, h as f32);
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

    Triangle{
        vertices: VERTICES.to_vec(),
        gaussians: gaussians.to_vec(),
        camera,
    }.render(
        vertex_instances,
        &mut color,
        &mut Empty::default(),
    );

    let mut win = minifb::Window::new("Splat", W, H, minifb::WindowOptions::default()).unwrap();
    while win.is_open() {
        win.update_with_buffer(color.raw(), W, H).unwrap();
    }
}
