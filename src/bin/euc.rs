// use ply::parser;
// use ply_rs as ply;
// use splat::{gaussians::Gaussian, camera::Camera};
// use euc::{Buffer2d, IndexedVertices, Pipeline, Target, TriangleList};
// use nalgebra as na;
// use na::Vector4;
// use minifb::{Key, Window, WindowOptions};

// struct GaussianSplatPipeline<'a> {
//     pub camera: &'a Camera,
//     pub gaussians: &'a Vec<Gaussian>,
// }

// impl<'r> Pipeline<'r> for GaussianSplatPipeline<'r> {
//     type Vertex = (Vector4<f32>, usize);
//     type VertexData = Vector4<f32>;
//     type Primitives = TriangleList;
//     type Pixel = u32;
//     type Fragment = Vector4<f32>;

//     #[inline(always)]
//     fn vertex(&self, (pos, gau_idx): &Self::Vertex) -> ([f32; 4], Self::VertexData) {
//         let gau = &self.gaussians[*gau_idx];
//         let projection = self.camera.get_view_matrix();
//         let pos = projection * pos;
//         let cam_pos = self.camera.position;
//         let (color, alpha) = gau.color(0, &cam_pos);
//         let color_rgba = Vector4::new(color[0], color[1], color[2], alpha);
//         ([pos.x, pos.y, pos.z, pos.w], color_rgba)
//     }

//     #[inline(always)]
//     fn fragment(&self, color: Self::VertexData) -> Self::Fragment {
//         color
//     }

//     fn blend(&self, _: Self::Pixel, color: Self::Fragment) -> Self::Pixel {
//         // convert nalgebra vector to u8 array
//         let color_rgb = color * 255.0;
//         let color: [u8; 4] = [color_rgb[0] as u8, color_rgb[1] as u8, color_rgb[2] as u8, color_rgb[3] as u8];
//         u32::from_le_bytes(color)
//     }
// }


// fn main() {
//     // Parse payload into a GaussianCollection
//     let mut gaussians = Vec::new();
//     let file = std::fs::File::open("simple2.ply").unwrap();
//     let mut file = std::io::BufReader::new(file);
//     let gaus_parser = parser::Parser::<Gaussian>::new();
//     let header = gaus_parser.read_header(&mut file).unwrap();

//     // Depending on the header, read the data into our structs..
//     for (_ignore_key, element) in &header.elements {
//         // we could also just parse them in sequence, but the file format might change
//         match element.name.as_ref() {
//             "vertex" => {
//                 gaussians = gaus_parser.read_payload_for_element(&mut file, &element, &header).unwrap();
//             },
//             _ => panic!("Unexpected element!"),
//         }
//     }

//     let [w, h] = [512, 512];

//     let mut color = Buffer2d::fill([w, h], 0);
//     let mut depth = Buffer2d::fill([w, h], 1.0);


//     use nalgebra as na;
//     const VERTICES: &[na::Vector2<f32>] = &[
//             na::Vector2::new(-1.,  1.),
//             na::Vector2::new(1.,  1.),
//             na::Vector2::new(1., -1.),
//             na::Vector2::new(-1., -1.),
//     ];
//     const INDICES: &[usize] = &[0, 1, 2, 0, 2, 3];


//     let camera = Camera::new(512.0, 512.0);
//     let view = camera.get_view_matrix();
//     let positions_w = splat::gaussians::get_positions(&gaussians);
//     let sorted = splat::gaussians::sort_gaussians_by_depth(&positions_w, &view);

//     let new_vertices: Vec<_> = sorted
//     .iter().flat_map(|gau_idx|{
//         VERTICES.iter().map(|v| (v.clone(), *gau_idx))
//     })
//     .collect();

//     let new_indices = sorted.iter().enumerate().flat_map(|(i, _)|{
//         INDICES.iter().map(move |ind| ind + i * VERTICES.len())
//     }).collect::<Vec<_>>();

//     println!("new_indices: {:?}", new_indices);

//     let mut win = Window::new("Gaussian Splats", w, h, WindowOptions::default()).unwrap();
//     let mut i = 0;
//     while win.is_open() && !win.is_key_down(Key::Escape) {
//         color.clear(0);
//         depth.clear(1.0);

//         let inde = IndexedVertices::new(new_indices.clone(), new_vertices.clone());
//         GaussianSplatPipeline {camera: &camera, gaussians: &gaussians }.render(
//             inde,
//             &mut color,
//             &mut depth,
//         );

//         win.update_with_buffer(color.raw(), w, h).unwrap();

//         i += 1;
//     }
// }

fn main() {}
