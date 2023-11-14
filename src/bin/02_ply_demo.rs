use euc::buffer::Buffer2d;
use nalgebra as na;
use splat::{gaussians, camera::Camera};
use na:: Vector3;
use splat::pipelines::GaussianSplatPipeline01;

const W: usize = 1280;
const H: usize = 720;

fn main() {
    let mut color: euc::Buffer<u32, 2> = Buffer2d::fill([W, H], 0);

    println!("Loading gaussians from ply file");
    let mut gaussians = gaussians::load_from_ply("notes/point_cloud.ply");

    // Compute cov3d for each gaussian
    println!("Computing cov3d for each gaussian");
    for gaussian in gaussians.iter_mut() {
        gaussian.compute_cov3d();
    }

    let camera_pos = Vector3::new(-0.57651054, 2.99040512, -0.03924271);
    let camera = Camera::new(H as f32, W as f32, Some(camera_pos));

    let pipeline = GaussianSplatPipeline01 {
        gaussians: gaussians.to_vec(),
        camera,
    };

    println!("Rendering");
    pipeline.render_to_buffer(
        &mut color,
    );

    let mut win = minifb::Window::new("Splat", W, H, minifb::WindowOptions::default()).unwrap();
    while win.is_open() {
        win.update_with_buffer(color.raw(), W, H).unwrap();
        // win.update_with_buffer(color.raw(), W, H).unwrap();
    }
}
