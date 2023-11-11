use ply::parser;
use ply_rs as ply;
use splat::{gaussians::Gaussian, camera::Camera};
use euc::{Buffer2d, IndexedVertices, Pipeline, Target, TriangleList};
use nalgebra as na;
use na::{Vector4, Vector3, Vector2};
use minifb::{Key, Window, WindowOptions};

struct GaussianSplatPipeline<'a> {
    pub camera: &'a Camera,
    pub gaussians: &'a Vec<Gaussian>,
}

impl<'r> Pipeline<'r> for GaussianSplatPipeline<'r> {
    type Vertex = (Vector4<f32>, usize);
    type VertexData = Vector4<f32>;
    type Primitives = TriangleList;
    type Pixel = u32;
    type Fragment = Vector4<f32>;

    #[inline(always)]
    fn vertex(&self, (pos, gau_idx): &Self::Vertex) -> ([f32; 4], Self::VertexData) {
        let gau = &self.gaussians[*gau_idx];
        let projection = self.camera.get_view_matrix();
        let pos = projection * pos;
        let cam_pos = self.camera.position;
        let (color, alpha) = gau.color(0, &cam_pos);
        let color_rgba = Vector4::new(color[0], color[1], color[2], alpha);
        ([pos.x, pos.y, pos.z, pos.w], color_rgba)
    }

    #[inline(always)]
    fn fragment(&self, color: Self::VertexData) -> Self::Fragment {
        color
    }

    fn blend(&self, _: Self::Pixel, color: Self::Fragment) -> Self::Pixel {
        // convert nalgebra vector to u8 array
        let color_rgb = color * 255.0;
        let color: [u8; 4] = [color_rgb[0] as u8, color_rgb[1] as u8, color_rgb[2] as u8, color_rgb[3] as u8];
        u32::from_le_bytes(color)
    }
}


struct GaussianData {
    pub pos_screen_xy: Vector2<f32>,
    pub conic: Vector3<f32>,
    pub bounds: [Vector2<f32>; 4],
    pub color: (na::Vector3<f32>, f32),
}

fn main() {
    // Parse payload into a GaussianCollection
    let mut gaussians = Vec::new();
    // let file = std::fs::File::open("simple2.ply").unwrap();
    let file = std::fs::File::open("output/point_cloud/iteration_30000/point_cloud.ply").unwrap();
    let mut file = std::io::BufReader::new(file);
    let gaus_parser = parser::Parser::<Gaussian>::new();
    let header = gaus_parser.read_header(&mut file).unwrap();

    // Depending on the header, read the data into our structs..
    for (_ignore_key, element) in &header.elements {
        // we could also just parse them in sequence, but the file format might change
        match element.name.as_ref() {
            "vertex" => {
                gaussians = gaus_parser.read_payload_for_element(&mut file, &element, &header).unwrap();
            },
            _ => panic!("Unexpected element!"),
        }
    }

    let [w, h] = [512, 512];

    let mut color_buffer = Buffer2d::fill([w, h], 0u32);
    let mut depth = Buffer2d::fill([w, h], 1.0);

    let camera = Camera::new(512.0, 512.0);
    let view = camera.get_view_matrix();
    let positions_w = splat::gaussians::get_positions(&gaussians);
    let sorted = splat::gaussians::sort_gaussians_by_depth(&positions_w, &view);

    // wh = 2 * hfovxy_focal.xy * hfovxy_focal.z;
    let htanfovxy_focal = camera.get_htanfovxy_focal();
    let wh = 2.0 * htanfovxy_focal.fixed_view::<2,1>(0,0) * camera.get_htanfovxy_focal().z;

    // Process gaussians into GaussianData
    let gaussian_data = gaussians.iter_mut().map(|gau|{
        // Compute the cov3d
        gau.compute_cov3d();
            // Project the cov3d into the screen space (2d)
        let (cov2d, pos_screen) = gau.project_cov3d_to_screen(&camera);

        // screen space half quad height and width (3-sigma)
        let bb_size = na::Vector2::<f32>::new(3.0 * cov2d.x.sqrt(), 3.0 * cov2d.z.sqrt());
        let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
        let conic;
        if det == 0.0 {
            return None;
        } else {
            let det_inv = 1.0 / det;
            conic = Vector3::new(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);
        }

        // Pos screen is from -1 to 1, convert to 0 to 1
        let pos_screen_xy = pos_screen.fixed_view::<2, 1>(0, 0);
        let pos_screen_xy = (pos_screen_xy + Vector2::new(1.0, 1.0)) / 2.0;

        // Multiply by screen dimensions to get pixel coordinates
        let pos_screen_xy = pos_screen_xy.component_mul(&na::Vector2::new(camera.w, camera.h));
        let bounds = [
            na::Vector2::new(-bb_scr.x, bb_scr.y),  // top left
            na::Vector2::new(bb_scr.x, bb_scr.y),   // top right
            na::Vector2::new(bb_scr.x, -bb_scr.y),  // bottom right
            na::Vector2::new(-bb_scr.x, -bb_scr.y), // bottom left
        ];

        let color = gau.color(0, &camera.position);
        Some(GaussianData{
            pos_screen_xy: pos_screen_xy.into(),
            conic,
            bounds,
            color,
        })
    }).collect::<Vec<_>>();

    // Iterated over sorted gaussians, splatting them into the buffers
    // for gau_idx in sorted {
    //     let gau = &gaussians[gau_idx];
    //     let gau_data = &gaussian_data[gau_idx];
    //     if let Some(gau_data) = gau_data{
    //         let bounds = gau_data.bounds;
    //         let conic = gau_data.conic;
    //         let pos_screen_xy = gau_data.pos_screen_xy;

    //         let color: na::Matrix<f32, na::Const<3>, na::Const<1>, na::ArrayStorage<f32, 3, 1>> = gau.color(0, &camera.position).0;

            // Multiply bounds by screen dimensions to get pixel coordinates
            // Forget bounds, we'll compute for the full screen

            for y in (0..(camera.h as usize))
            {
                for x in (0..(camera.w as usize))
                {
                    // Iterate over the gaussians, compute "power" for each pixel and output color and alpha
                    let outputs: Vec<(na::Vector3<f32>, f32)> = gaussian_data.iter().take(1000).flatten()

                    .filter(|gau|{
                        let pos_screen_xy = &gau.pos_screen_xy;
                        // return true only if pixel within bounds
                        pos_screen_xy.x >= 0.0 && pos_screen_xy.x < camera.w && pos_screen_xy.y >= 0.0 && pos_screen_xy.y < camera.h
                    })
                    .flat_map(|gau|{
                        let conic = &gau.conic;

                        let coordxy = na::Vector2::new(x as f32, y as f32) - gau.pos_screen_xy;
                        let power = -0.5 * (conic.x * coordxy.x * coordxy.x + conic.z * coordxy.y * coordxy.y) - conic.y * coordxy.x * coordxy.y;
                        if power > 0.0f32
                        {
                            return None;
                        }
                        let (color, alpha) = gau.color;
                        let opacity = 0.99f32.min(alpha * power.exp());
                        if opacity < 1.0 / 255.0
                        {
                            return None;
                        }
                        Some((color*255.0, opacity))
                    }).collect();
                    if outputs.len() == 0
                    {
                        continue;
                    }
                    // Alpha blend (using the "over" operator) the outputs and apply to the pixel
                    let mut color = na::Vector3::new(0.0, 0.0, 0.0);
                    let mut alpha = 0.0;

                    for (color_rgb, opacity) in outputs
                    {
                        let alpha_new = alpha + opacity * (1.0 - alpha);
                        let color_new = (color * alpha + color_rgb * opacity * (1.0 - alpha)) / alpha_new;
                        alpha = alpha_new;
                        color = color_new;
                    }
                    // Convert color to u8 array
                    let color_u32 = u32::from_le_bytes([color[0] as u8, color[1] as u8, color[2] as u8, 255]);
                    let linear_index = color_buffer.linear_index([x, y]);
                    color_buffer.raw_mut()[linear_index] = color_u32;
                    if x % 10 == 0
                    {
                        println!("x: {}, y: {}", x, y)
                    }
                }
            }
    //     }
    // }
    let window = Window::new(
        "Splat",
        w,
        h,
        WindowOptions {
            resize: true,
            ..WindowOptions::default()
        },
    );
    let mut window = window.unwrap();
    while window.is_open() && !window.is_key_down(Key::Escape) {
        window.update_with_buffer(&color_buffer.raw(), w, h).unwrap();
    }

}
