/// trims and existing ply file to extract out the first three splats
use std::env;


use ply_rs as ply;
use ply_rs::ply::{ Ply, DefaultElement };
use ply_rs::writer::Writer;

fn extract_three_points(ply: &Ply<DefaultElement>) -> Ply<DefaultElement>
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
    output_ply
}

fn main()
{
    // Load input and output file names from the command line
    // Show error message with usage if not enough arguments

    let args: Vec<String> = env::args().collect();
    if args.len() < 3
    {
        println!("Usage: trim <input_file> <output_file>");
        return;
    }

    let input_file_name = &args[1];
    let output_file_name = &args[2];

    let mut file = std::fs::File::open(input_file_name).unwrap();

    let p = ply::parser::Parser::<ply::ply::DefaultElement>::new();

    // use the parser: read the entire file
    let ply = p.read_ply(&mut file);

    // make sure it did work
    assert!(ply.is_ok());
    let ply = ply.unwrap();

    let mut output_ply = extract_three_points(&ply);

    // set up a writer
    let mut outfile = std::fs::File::create(output_file_name).unwrap();
    let w = Writer::new();
    let written = w.write_ply(&mut outfile, &mut output_ply).unwrap();
    println!("Wrote {} bytes", written);
}
