use euc::{buffer::Buffer2d, rasterizer, Pipeline, TriangleList, Empty};
use nalgebra as na;
use splat::{gaussians::{Gaussian, self}, camera::Camera};
use na::{Vector2, Vector3, Vector4, Matrix3, Matrix4};

struct Triangle {
    pub vertices: Vec<Vector2<f32>>,
    pub gaussians: Vec<Gaussian>,
}

#[derive(Clone, Debug)]
pub struct VertexInstance {
    pub vert_idx: usize,
    pub gaussian_idx: usize,
}
impl<'r> Pipeline<'r> for Triangle {
    type Vertex = VertexInstance;
    type VertexData = Vector4<f32>;
    type Primitives = TriangleList;
    type Fragment = Vector4<f32>;
    type Pixel = u32;

    fn vertex(&self, vert_inst: &Self::Vertex) -> ([f32; 4], Self::VertexData) {
        let pos = self.vertices[vert_inst.vert_idx];
        ([pos[0], pos[1], 0.0, 1.0], Vector4::new(1.0, 0.0, 0.0, 1.0))
    }

    fn fragment(&self, col: Self::VertexData) -> Self::Fragment {
        // println!("Fragment: {:?}", col);
        col
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
    for g_idx in sorted_gaussians {
        // For each instance, push all the vertices
        for vert_idx in INDICES {
            vertex_instances.push(VertexInstance {
                vert_idx: *vert_idx,
                gaussian_idx: g_idx,
            });
        }
    }

    Triangle{
        vertices: VERTICES.to_vec(),
        gaussians: gaussians.to_vec(),
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
