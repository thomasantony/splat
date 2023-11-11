use na::{UnitQuaternion, Vector3};
use nalgebra as na;
use ply_rs::ply::{Property, PropertyAccess};

const SH_C0: f32 = 0.28209479177387814f32;
const SH_C1: f32 = 0.4886025119029199f32;

const SH_C2_0: f32 = 1.0925484305920792f32;
const SH_C2_1: f32 = -1.0925484305920792f32;
const SH_C2_2: f32 = 0.31539156525252005f32;
const SH_C2_3: f32 = -1.0925484305920792f32;
const SH_C2_4: f32 = 0.5462742152960396f32;

const SH_C3_0: f32 = -0.5900435899266435f32;
const SH_C3_1: f32 = 2.890611442640554f32;
const SH_C3_2: f32 = -0.4570457994644658f32;
const SH_C3_3: f32 = 0.3731763325901154f32;
const SH_C3_4: f32 = -0.4570457994644658f32;
const SH_C3_5: f32 = 1.445305721320277f32;
const SH_C3_6: f32 = -0.5900435899266435f32;

const HALF: na::Vector3<f32> = na::Vector3::new(0.5, 0.5, 0.5);


#[derive(Clone, Debug)]
pub struct Gaussian {
    pub position: na::Vector3<f32>,  // x, y, z
    pub scale: na::Vector3<f32>,    // scale_0, scale_1, scale_2
    pub opacity: f32,               // opacity
    pub rotation: na::Quaternion<f32>,  // rot_0, rot_1, rot_2, rot_3
    pub color: na::Vector3<f32>,    // f_dc_0, f_dc_1, f_dc_2
    pub sh: na::SVector<f32, 45>,   // f_rest_0 ... f_rest_44
}

impl Gaussian {
    pub fn to_covariance_3d(&self) -> na::Matrix3<f32>
    {
        let rotation = UnitQuaternion::from_quaternion(self.rotation).to_rotation_matrix();        // square the scales and make a diagonal matrix
        // square the scales and make a diagonal matrix
        let scale = na::Matrix3::new(
            self.scale[0] * self.scale[0], 0.0, 0.0,
            0.0, self.scale[1] * self.scale[1], 0.0,
            0.0, 0.0, self.scale[2] * self.scale[2],
        );
        // compute the covariance matrix
        let covariance = rotation * scale * rotation.transpose();
        covariance
    }
    pub fn color(&self, sh_dim: usize, cam_pos: &na::Vector3<f32>) -> (na::Vector3<f32>, f32)
    {
        let dir = (self.position - cam_pos).normalize();

        // Compute color from the spherical harmonics for the given direction
        // The coefficients are laid out as:
        // C0_R, C0_B, C0_G, C1_R, C1_B, C1_G ...

        let c0 = na::Vector3::new(self.sh[0], self.sh[1], self.sh[2]);
        let mut color = SH_C0 * c0.component_mul(&self.color);

        if sh_dim > 3
        {
            // Add the first order spherical harmonics
            let c1 = na::Vector3::new(self.sh[3], self.sh[4], self.sh[5]);
            let c2 = na::Vector3::new(self.sh[6], self.sh[7], self.sh[8]);
            let c3 = na::Vector3::new(self.sh[9], self.sh[10], self.sh[11]);

            let x = dir[0];
            let y = dir[1];
            let z = dir[2];

            color = color - SH_C1 * y * c1 + SH_C1 * z * c2 - SH_C1 * x * c3;

            if sh_dim > 12
            {
                let c4 = na::Vector3::new(self.sh[12], self.sh[13], self.sh[14]);
                let c5 = na::Vector3::new(self.sh[15], self.sh[16], self.sh[17]);
                let c6 = na::Vector3::new(self.sh[18], self.sh[19], self.sh[20]);
                let c7 = na::Vector3::new(self.sh[21], self.sh[22], self.sh[23]);
                let c8 = na::Vector3::new(self.sh[24], self.sh[25], self.sh[26]);

                let (xx, yy, zz) = (x * x, y * y, z * z);
			    let (xy, yz, xz) = (x * y, y * z, x * z);
			    color = color +	SH_C2_0 * xy * c4 +
				SH_C2_1 * yz * c5 +
				SH_C2_2 * (2.0f32 * zz - xx - yy) * c6 +
				SH_C2_3 * xz * c7 +
				SH_C2_4 * (xx - yy) * c8;

                if sh_dim > 27 {
                    let c9 = na::Vector3::new(self.sh[27], self.sh[28], self.sh[29]);
                    let c10 = na::Vector3::new(self.sh[30], self.sh[31], self.sh[32]);
                    let c11 = na::Vector3::new(self.sh[33], self.sh[34], self.sh[35]);
                    let c12 = na::Vector3::new(self.sh[36], self.sh[37], self.sh[38]);
                    let c13 = na::Vector3::new(self.sh[39], self.sh[40], self.sh[41]);
                    let c14 = na::Vector3::new(self.sh[42], self.sh[43], self.sh[44]);
                    let c15 = na::Vector3::new(self.sh[45], self.sh[46], self.sh[47]);

                    color = color +
					SH_C3_0 * y * (3.0f32 * xx - yy) * c9 +
					SH_C3_1 * xy * z * c10 +
					SH_C3_2 * y * (4.0f32 * zz - xx - yy) * c11 +
					SH_C3_3 * z * (2.0f32 * zz - 3.0f32 * xx - 3.0f32 * yy) * c12 +
					SH_C3_4 * x * (4.0f32 * zz - xx - yy) * c13 +
					SH_C3_5 * z * (xx - yy) * c14 +
					SH_C3_6 * x * (xx - 3.0f32 * yy) * c15;
                }
            }
        }
        color += HALF;
        (color, self.opacity)
    }
}

impl PropertyAccess for Gaussian {
    fn new() -> Self {
        Self {
            position: na::Vector3::new(0.0, 0.0, 0.0),
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
            ("x", Property::Float(v)) => self.position[0] = v,
            ("y", Property::Float(v)) => self.position[1] = v,
            ("z", Property::Float(v)) => self.position[2] = v,
            ("scale_0", Property::Float(v)) => self.scale[0] = v,
            ("scale_1", Property::Float(v)) => self.scale[1] = v,
            ("scale_2", Property::Float(v)) => self.scale[2] = v,
            ("opacity", Property::Float(v)) => self.opacity = v,
            ("rot_0", Property::Float(v)) => self.rotation[0] = v,
            ("rot_1", Property::Float(v)) => self.rotation[1] = v,
            ("rot_2", Property::Float(v)) => self.rotation[2] = v,
            ("rot_3", Property::Float(v)) => self.rotation[3] = v,
            ("f_dc_0", Property::Float(v)) => self.color[0] = v,
            ("f_dc_1", Property::Float(v)) => self.color[1] = v,
            ("f_dc_2", Property::Float(v)) => self.color[2] = v,
            (s, Property::Float(v)) if s.starts_with("f_rest_") => {
                let index = s[7..].parse::<usize>().unwrap();
                self.sh[index] = v;
            }
            _ => {}
        }
    }
}

// Convert positions into a 4xN with the last row being 1.0
pub fn get_positions(gaussians: &Vec<Gaussian>) -> na::Matrix4xX<f32> {
    let mut positions = na::Matrix4xX::zeros(gaussians.len());
    for (i, g) in gaussians.iter().enumerate() {
        positions[(0, i)] = g.position[0];
        positions[(1, i)] = g.position[1];
        positions[(2, i)] = g.position[2];
        positions[(3, i)] = 1.0;
    }
    positions
}

pub fn sort_gaussians(gaussians: &Vec<Gaussian>, view: &na::Matrix4<f32>) -> Vec<usize> {
    // Project positions into the camera space
    let positions = view * get_positions(gaussians);

    // Sort by depth and get indices of columns
    let mut indices: Vec<usize> = (0..gaussians.len()).collect();
    indices.sort_by(|a, b| positions[(2, *a)].partial_cmp(&positions[(2, *b)]).unwrap());

    indices
}

pub fn sort_gaussians_by_depth(positions_w: &na::Matrix4xX<f32>, view: &na::Matrix4<f32>) -> Vec<usize> {
    // Project positions into the camera space
    let positions = view * positions_w;

    // Sort by depth and get indices of columns
    let mut indices: Vec<usize> = (0..positions_w.shape().1).collect();
    indices.sort_by(|a, b| positions[(2, *a)].partial_cmp(&positions[(2, *b)]).unwrap());

    indices
}

pub fn load_from_ply(filename: &str) -> Vec<Gaussian>
{
        let mut gaussians = Vec::new();
    let file = std::fs::File::open(filename).unwrap();
    let mut file = std::io::BufReader::new(file);
    let gaus_parser = ply_rs::parser::Parser::<Gaussian>::new();
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
    gaussians

}
