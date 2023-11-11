use ply_rs as ply;
use ply_rs::ply::{ Ply, DefaultElement };
use ply_rs::writer::{ Writer };

// A struct for holding parameters of a 3D gaussian
#[derive(Debug)]
struct Gaussian {
    mean: [f64; 3],
    variance: [f64; 3],
}

fn extract_three_points(ply: &Ply<DefaultElement>)
{
      // Copy the first three splats into a vector and then write into a new PLY file
    let (_, points) = ply.payload.front().unwrap();
    let mut output_points = Vec::new();
    for (i, splat) in points.iter().enumerate()
    {
        if i > 2
        {
            break;
        }
        output_points.push(splat.clone());
    }
    // Append the splats to the new PLY file
    let mut output_ply = Ply::<DefaultElement>::new();
    output_ply.payload.insert("vertex".to_string(), output_points);
    // Set header to match the original
    output_ply.header = ply.header.clone();

    // set up a writer
    let mut outfile = std::fs::File::create("simple2.ply").unwrap();
    let w = Writer::new();
    let written = w.write_ply(&mut outfile, &mut output_ply).unwrap();
    println!("Wrote {} bytes", written);
}
fn main() {
    // Open ply file at sparse/0/points3D.ply
    // let mut file = std::fs::File::open("output/point_cloud/iteration_30000/point_cloud.ply").unwrap();
    // let mut file = std::fs::File::open("simple.ply").unwrap();
    let mut file = std::fs::File::open("garden.splat").unwrap();
    let p = ply::parser::Parser::<ply::ply::DefaultElement>::new();

    // use the parser: read the entire file
    let ply = p.read_ply(&mut file);

    // make sure it did work
    assert!(ply.is_ok());
    let ply = ply.unwrap();

    println!("PLY header: {:#?}", ply.header);
    println!("PLY payload: {:#?}", ply.payload);
    // extract_three_points(&ply);

    // x, y, z, nx, ny, nz
    // f_dc_0, f_dc_1, f_dc_2
    // f_rest_0 ... f_rest_44
    // scale_0, scale_1, scale_2
    // opacity,
    // rot_0, rot_1, rot_2, rot_3
}
