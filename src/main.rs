use ply::parser;
use ply::ply::Property;
use ply_rs as ply;
use ply_rs::ply::{ Ply, DefaultElement };
use ply_rs::writer::{ Writer };

use nalgebra as na;

#[derive(Clone, Debug)]
pub struct Gaussian {
    pub position: na::Point3<f32>,  // x, y, z
    pub scale: na::Vector3<f32>,                // scale_0, scale_1, scale_2
    pub opacity: f32,               // opacity
    pub rotation: na::Quaternion<f32>,  // rot_0, rot_1, rot_2, rot_3
    pub color: na::Vector3<f32>,    // f_dc_0, f_dc_1, f_dc_2
    pub sh: na::SVector<f32, 45>,   // f_rest_0 ... f_rest_44
}

impl ply::ply::PropertyAccess for Gaussian {
    fn new() -> Self {
        Self {
            position: na::Point3::new(0.0, 0.0, 0.0),
            scale: na::Vector3::new(0.0, 0.0, 0.0),
            opacity: 0.0,
            rotation: na::Quaternion::identity(),
            color: na::Vector3::new(0.0, 0.0, 0.0),
            sh: na::SVector::<f32, 45>::zeros(),
        }
    }
    // Set properties from the PLY file
    fn set_property(&mut self, property_name: String, property: Property)
    {
        //Assign values based on property name
        match (property_name.as_ref(), property) {
            ("x", ply::ply::Property::Float(v)) => self.position[0] = v,
            ("y", ply::ply::Property::Float(v)) => self.position[1] = v,
            ("z", ply::ply::Property::Float(v)) => self.position[2] = v,
            ("scale_0", ply::ply::Property::Float(v)) => self.scale[0] = v,
            ("scale_1", ply::ply::Property::Float(v)) => self.scale[1] = v,
            ("scale_2", ply::ply::Property::Float(v)) => self.scale[2] = v,
            ("opacity", ply::ply::Property::Float(v)) => self.opacity = v,
            ("rot_0", ply::ply::Property::Float(v)) => self.rotation[0] = v,
            ("rot_1", ply::ply::Property::Float(v)) => self.rotation[1] = v,
            ("rot_2", ply::ply::Property::Float(v)) => self.rotation[2] = v,
            ("rot_3", ply::ply::Property::Float(v)) => self.rotation[3] = v,
            ("f_dc_0", ply::ply::Property::Float(v)) => self.color[0] = v,
            ("f_dc_1", ply::ply::Property::Float(v)) => self.color[1] = v,
            ("f_dc_2", ply::ply::Property::Float(v)) => self.color[2] = v,
            (s, ply::ply::Property::Float(v)) if s.starts_with("f_rest_") => {
                let index = s[7..].parse::<usize>().unwrap();
                self.sh[index] = v;
            }
            _ => {}
        }
    }
}

// Array of structures (AoS) representation of 3D gaussians with spherical harmonic coefficients
#[derive(Clone, Debug)]
pub struct GaussianCollection
{
    pub positions: Vec<na::Point3<f32>>,
    pub scales: Vec<na::Vector3<f32>>,
    pub opacities: Vec<f32>,
    pub rotations: Vec<na::UnitQuaternion<f32>>,
    pub sh: Vec<na::SVector<f32, 45>>,
}
impl GaussianCollection{
    pub fn new() -> Self
    {
        Self{
            positions: Vec::new(),
            scales: Vec::new(),
            opacities: Vec::new(),
            rotations: Vec::new(),
            sh: Vec::new(),
        }
    }
    pub fn extend(&mut self, other: &Vec<Gaussian>)
    {
        for gaussian in other
        {
            self.push(gaussian);
        }
    }
    pub fn push(&mut self, gaussian: &Gaussian)
    {
        self.positions.push(gaussian.position);
        self.scales.push(gaussian.scale);
        self.opacities.push(gaussian.opacity);
        self.rotations.push(na::UnitQuaternion::from_quaternion(gaussian.rotation));
        self.sh.push(gaussian.sh);
    }
}

pub struct Covariance3DCollection(pub Vec<na::SMatrix<f32, 3, 3>>);


impl GaussianCollection{

    pub fn compute_covariance_3D(&self) -> Covariance3DCollection
    {
        let mut covariances = Vec::new();
        for i in 0..self.positions.len()
        {

            let rotation = self.rotations[i].to_rotation_matrix();
            // square the scales and make a diagonal matrix
            let scale = na::Matrix3::new(
                self.scales[i][0] * self.scales[i][0], 0.0, 0.0,
                0.0, self.scales[i][1] * self.scales[i][1], 0.0,
                0.0, 0.0, self.scales[i][2] * self.scales[i][2],
            );
            let covariance = rotation.transpose() * scale * rotation;
            covariances.push(rotation * covariance * rotation.transpose());
        }
        Covariance3DCollection(covariances)
    }
}

fn main() {

    // Parse payload into a GaussianCollection
    let mut gaussians = GaussianCollection::new();
    let file = std::fs::File::open("simple2.ply").unwrap();
    let mut file = std::io::BufReader::new(file);
    let gaus_parser = parser::Parser::<Gaussian>::new();
    let header = gaus_parser.read_header(&mut file).unwrap();

    // Depending on the header, read the data into our structs..
    for (_ignore_key, element) in &header.elements {
        // we could also just parse them in sequence, but the file format might change
        match element.name.as_ref() {
            "vertex" => {
                let gaus = gaus_parser.read_payload_for_element(&mut file, &element, &header).unwrap();
                println!("Gaussians: {:#?}", gaus);
                gaussians.extend(&gaus);

            },
            _ => panic!("Unexpected element!"),
        }
    }
}
