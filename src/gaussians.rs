use std::ops::MulAssign;

use nalgebra as na;
use na::{UnitQuaternion, Vector3, Matrix3, Vector4, Vector2, OMatrix, VectorView};
use ply_rs::ply::{Property, PropertyAccess};

use crate::camera::Camera;

type Matrix48xX<T> = OMatrix<T, na::U48, na::Dyn>;

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
    pub sh: na::SVector<f32, 48>,   // f_dc_0..3 and f_rest_0 ... f_rest_44
    pub cov3d: na::Matrix3<f32>,
}

#[inline]
pub fn eval_spherical_harmonics(sh: &VectorView<f32, na::U48>, sh_dim: usize, dir: &Vector3<f32>) -> Vector3<f32>
{
    let c0 = sh.fixed_rows::<3>(0);
    let mut color = SH_C0 * c0;

    if sh_dim > 3
    {
        // Add the first order spherical harmonics
        let c1 = sh.fixed_view::<3,1>(3,0);
        let c2 = sh.fixed_view::<3,1>(6,0);
        let c3 = sh.fixed_view::<3,1>(9,0);

        // Get a 3x3 view into the elements 3..12 of sh

        let x = dir[0];
        let y = dir[1];
        let z = dir[2];

        color = color - SH_C1 * y * c1 + SH_C1 * z * c2 - SH_C1 * x * c3;

        if sh_dim > 12
        {
            let c4 = sh.fixed_view::<3,1>(12,0);
            let c5 = sh.fixed_view::<3,1>(15,0);
            let c6 = sh.fixed_view::<3,1>(18,0);
            let c7 = sh.fixed_view::<3,1>(21,0);
            let c8 = sh.fixed_view::<3,1>(24,0);

            let (xx, yy, zz) = (x * x, y * y, z * z);
            let (xy, yz, xz) = (x * y, y * z, x * z);
            color = color +	SH_C2_0 * xy * c4 +
            SH_C2_1 * yz * c5 +
            SH_C2_2 * (2.0f32 * zz - xx - yy) * c6 +
            SH_C2_3 * xz * c7 +
            SH_C2_4 * (xx - yy) * c8;

            if sh_dim > 27 {
                let c9 =  sh.fixed_view::<3,1>(27,0);
                let c10 = sh.fixed_view::<3,1>(30, 0);
                let c11 = sh.fixed_view::<3,1>(33, 0);
                let c12 = sh.fixed_view::<3,1>(36, 0);
                let c13 = sh.fixed_view::<3,1>(39, 0);
                let c14 = sh.fixed_view::<3,1>(42, 0);
                let c15 = sh.fixed_view::<3,1>(45, 0);

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
    color
}
impl Gaussian {
    pub fn compute_cov3d(&mut self)
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
        self.cov3d = covariance;
    }
    pub fn project_cov3d_to_screen(&self, camera: &Camera) -> na::Matrix2<f32>
    {
        let viewmatrix = camera.get_view_matrix();

        // Project the Gaussian center to the camera space
        let pos_w = na::Vector4::new(self.position[0], self.position[1], self.position[2], 1.0);

        let pos_cam = viewmatrix * pos_w;

        // Compute the focal length and the tangent of the fov
        let htanfovxy = camera.get_htanfovxy_focal();
        let tan_fovx = htanfovxy[0];
        let tan_fovy = htanfovxy[1];
        let focal = htanfovxy[2];
        let focal_x = focal;
        let focal_y = focal;

        // Compute the tangent of the Gaussian center
        let mut t = pos_cam.clone();
        let limx = 1.3 * tan_fovx;
        let limy = 1.3 * tan_fovy;

        let txtz = pos_cam.x/pos_cam.z;
        let tytz = pos_cam.y/pos_cam.z;

        t.x = limx.min((-limx).max(txtz)) * pos_cam.z;
        t.y = limy.min((-limy).max(tytz)) * pos_cam.z;

        // Compute the Jacobian
        let J = na::Matrix3::new(
            focal_x / t.z, 0.0, -(focal_x * t.x) / (t.z * t.z),
            0.0, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
            0.0, 0.0, 0.0
        );

        let W = viewmatrix.fixed_view::<3, 3>(0, 0).transpose();
        let T = W * J;
        let mut cov = T.transpose() * self.cov3d.transpose() * T;

        // Apply low-pass filter: every Gaussian should be at least
        // one pixel wide/high. Discard 3rd row and column.
        let mut cov2d = cov.fixed_view_mut::<2,2>(0,0);
        cov2d[(0, 0)] += 0.01;
        cov2d[(1, 1)] += 0.01;

        // Convert into vector to return, along with position in screen space
        cov2d.into()
    }

    pub fn color(&self, sh_dim: usize, dir: &na::Vector3<f32>) -> (na::Vector3<f32>, f32)
    {
        // Compute color from the spherical harmonics for the given direction
        // The coefficients are laid out as:
        // C0_R, C0_B, C0_G, C1_R, C1_B, C1_G ...
        let color = eval_spherical_harmonics(&self.sh.as_view(), sh_dim, dir);
        (color, self.opacity)
    }

    pub fn projected_covariance_of_ellipsoid(&self, camera: &Camera) -> Matrix3<f32> {
        let view_matrix = camera.get_view_matrix(); // world -> camera converter
        let camera_matrix = view_matrix.try_inverse().unwrap();
        let camera_matrix = camera_matrix.fixed_view::<3,3>(0,0); // camera -> world converter

        let mut transform = Matrix3::from(UnitQuaternion::from_quaternion(self.rotation).to_rotation_matrix());
        transform.row_mut(0).mul_assign(self.scale[0]);
        transform.row_mut(1).mul_assign(self.scale[1]);
        transform.row_mut(2).mul_assign(self.scale[2]);

        let translation = Vector4::new(self.position[0], self.position[1], self.position[2], 1.0);
        // 3D Covariance
        let mut view_pos = view_matrix * translation;
        view_pos[0] = (view_pos[0] / view_pos[2]).clamp(-1.0, 1.0) * view_pos[2];
        view_pos[1] = (view_pos[1] / view_pos[2]).clamp(-1.0, 1.0) * view_pos[2];

        let J = Matrix3::<f32>::new(
            1.0 / view_pos[2], 0.0, -view_pos[0] / (view_pos[2] * view_pos[2]),
            0.0, 1.0 / view_pos[2], -view_pos[1] / (view_pos[2] * view_pos[2]),
            0.0, 0.0, 0.0,
        );
        let T = transform.transpose() * camera_matrix * J;
        let covariance_matrix = T.transpose() * T;

        return covariance_matrix;
    }

    pub fn extract_scale_of_covariance(&self, cov: &Matrix3<f32>) -> Vector2<f32>
    {
        let a = (cov[(0,0)] - cov[(1,1)]) * (cov[(0,0)] - cov[(1,1)]);
        let b = (a + 4.0 * cov[(0,1)] * cov[(0,1)]).sqrt();
        let semi_major_axis = ((cov[(0,0)] + cov[(1,1)] + b) * 0.5).sqrt();
        let semi_minor_axis = ((cov[(0,0)] + cov[(1,1)] - b) * 0.5).sqrt();
        return Vector2::new(semi_major_axis, semi_minor_axis);
    }

    pub fn extract_rotation_of_ellipse(&self, cov: &Matrix3<f32>) -> Vector2<f32> {
        /*
            phi = atan(2.0 * cov[(0,1)] / (cov[(0,0)] - cov[(1,1)])) / 2
            k = cos(phi)
            j = sin(phi)
            Insert angle phi into cos() and then apply the half-angle identity to get:
        */
        let a = (cov[(0,0)] - cov[(1,1)]) * (cov[(0,0)] - cov[(1,1)]);
        let b = a + 4.0 * cov[(0,1)] * cov[(0,1)];
        let c = 0.5 * (a / b).sqrt();
        let mut j = (0.5 - c).sqrt();
        let mut k = -(0.5 + c).sqrt() * (cov[(0,1)]).signum() * (cov[(0,0)] - cov[(1,1)]).signum();
        if cov[(0,1)] < 0.0 || cov[(0,0)] - cov[(1,1)] < 0.0 {
            k = -k;
            j = -j;
        }
        if cov[(0,0)] - cov[(1,1)] < 0.0 {
            let t = j;
            j = -k;
            k = t;
        }
        return Vector2::new(j, k);
    }
    pub fn extract_translation_of_ellipse(&self, cov: &Matrix3<f32>) -> Vector2<f32> {
        /*
            The center of the ellipse is at the extremum (minimum / maximum) of the implicit curve.
            So, take the partial derivative in x and y, which is: (2.0 * cov[(0,0)] * x + M.x.y * y + cov[(0,2)], M.x.y * x + 2.0 * cov[(1,1)] * y + cov[(1,2)])
            And the roots of that partial derivative are the position of the extremum, thus the translation of the ellipse.
        */
        let discriminant = cov[(0,0)] * cov[(1,1)] - cov[(0,1)] * cov[(0,1)];
        let inverse_discriminant = 1.0 / discriminant;
        return Vector2::new(
            cov[(0,1)] * cov[(1,2)] - cov[(1,1)] * cov[(0,2)],
            cov[(0,1)] * cov[(0,2)] - cov[(0,0)] * cov[(1,2)],
        ) * inverse_discriminant;
    }
}

impl PropertyAccess for Gaussian {
    fn new() -> Self {
        Self {
            position: na::Vector3::new(0.0, 0.0, 0.0),
            scale: na::Vector3::new(0.0, 0.0, 0.0),
            opacity: 0.0,
            rotation: na::Quaternion::identity(),
            sh: na::SVector::<f32, 48>::zeros(),
            cov3d: na::Matrix3::<f32>::zeros(),
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
            ("scale_0", Property::Float(v)) => self.scale[0] = v.exp(),
            ("scale_1", Property::Float(v)) => self.scale[1] = v.exp(),
            ("scale_2", Property::Float(v)) => self.scale[2] = v.exp(),
            ("opacity", Property::Float(v)) => self.opacity = 1.0 / (1.0 + (-v).exp()),
            ("rot_0", Property::Float(v)) => self.rotation[0] = v,
            ("rot_1", Property::Float(v)) => self.rotation[1] = v,
            ("rot_2", Property::Float(v)) => self.rotation[2] = v,
            ("rot_3", Property::Float(v)) => self.rotation[3] = v,
            ("f_dc_0", Property::Float(v)) => self.sh[0] = v,
            ("f_dc_1", Property::Float(v)) => self.sh[1] = v,
            ("f_dc_2", Property::Float(v)) => self.sh[2] = v,
            (s, Property::Float(v)) if s.starts_with("f_rest_") => {
                let index = s[7..].parse::<usize>().unwrap();
                self.sh[3+index] = v;
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
    indices.sort_by(|a, b| positions[(2, *a)].partial_cmp(&positions[(2, *b)]).unwrap_or(std::cmp::Ordering::Equal));

    indices
}

pub fn sort_gaussians_by_depth(positions_w: &na::Matrix4xX<f32>, view: &na::Matrix4<f32>) -> Vec<usize> {
    // Project positions into the camera space
    let positions = view * positions_w;

    // Sort by depth and get indices of columns
    let mut indices: Vec<usize> = (0..positions_w.shape().1).collect();
    indices.sort_by(|a, b| positions[(2, *a)].partial_cmp(&positions[(2, *b)]).unwrap_or(std::cmp::Ordering::Equal));

    indices
}

pub fn naive_gaussians() -> Vec<Gaussian>
{
    // Generates 4 gaussians used for testing
    let mut output = Vec::new();

    let mut g = Gaussian::new();
    g.position = Vector3::new(0.0, 0.0, 0.0);
    g.scale = Vector3::new(0.03, 0.03, 0.03);
    g.opacity = 1.0;
    g.rotation = na::Quaternion::new(1.0, 0.0, 0.0, 0.0);
    let mut color = Vector3::new(1.0, 0.0, 1.0);
    color = (color - HALF) / 0.28209;
    g.sh[0] = color[0];
    g.sh[1] = color[1];
    g.sh[2] = color[2];
    output.push(g);

    let mut g = Gaussian::new();
    g.position = Vector3::new(1.0, 0.0, 0.0);
    g.scale = Vector3::new(0.2, 0.03, 0.03,);
    g.opacity = 1.0;
    g.rotation = na::Quaternion::new(1.0, 0.0, 0.0, 0.0);
    let mut color = Vector3::new(1.0, 0.0, 0.0);
    color = (color - HALF) / 0.28209;
    g.sh[0] = color[0];
    g.sh[1] = color[1];
    g.sh[2] = color[2];
    output.push(g);

    let mut g = Gaussian::new();
    g.position = Vector3::new(0.0, 1.0, 0.0);
    g.scale = Vector3::new(0.03, 0.2, 0.03);
    g.opacity = 1.0;
    g.rotation = na::Quaternion::new(1.0, 0.0, 0.0, 0.0);
    let mut color = Vector3::new(0.0, 1.0, 0.0);
    color = (color - HALF) / 0.28209;
    g.sh[0] = color[0];
    g.sh[1] = color[1];
    g.sh[2] = color[2];
    output.push(g);

    let mut g = Gaussian::new();
    g.position = Vector3::new(0.0, 0.0, 1.0);
    g.scale = Vector3::new(0.03, 0.03, 0.2);
    g.opacity = 1.0;
    g.rotation = na::Quaternion::new(1.0, 0.0, 0.0, 0.0);
    let mut color = Vector3::new(0.0, 0.0, 1.0);
    color = (color - HALF) / 0.28209;
    g.sh[0] = color[0];
    g.sh[1] = color[1];
    g.sh[2] = color[2];
    output.push(g);

    output

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

    // Find average position of all gaussians and subtract it from all positions
    let mut avg_pos = Vector3::new(0.0, 0.0, 0.0);
    for g in gaussians.iter() {
        avg_pos += g.position;
    }
    avg_pos /= gaussians.len() as f32;
    for g in gaussians.iter_mut() {
        g.position -= avg_pos;
    }
    gaussians

}


pub struct GaussianList {
    pub positions: na::Matrix4xX<f32>,
    pub scales: na::Matrix3xX<f32>,
    pub opacities: na::DVector<f32>,
    pub rotations: na::Matrix4xX<f32>,
    pub sh: Matrix48xX<f32>,
    pub num_gaussians: usize,
    cov3d: na::Matrix3xX<f32>,
}

impl GaussianList{
    pub fn from_vec(gaussians: Vec<Gaussian>) -> Self
    {
        let num_gaussians = gaussians.len();
        // Make position 4D by adding a 1.0 to the end of each position vector
        let positions = na::Matrix4xX::from_iterator(num_gaussians, gaussians.iter().flat_map(|g| g.position.iter().cloned().chain([1.0])));
        let scales = na::Matrix3xX::from_iterator(num_gaussians, gaussians.iter().flat_map(|g| g.scale.iter().cloned()));
        let opacities = na::DVector::from_iterator(num_gaussians, gaussians.iter().map(|g| g.opacity));
        let rotations = na::Matrix4xX::from_iterator(num_gaussians, gaussians.iter().flat_map(|g| g.rotation.as_vector().iter().cloned()));
        let sh = Matrix48xX::from_iterator(num_gaussians, gaussians.iter().flat_map(|g| g.sh.iter().cloned()));
        let cov3d = na::Matrix3xX::zeros(num_gaussians*3);
        let mut out = Self {
            positions,
            scales,
            opacities,
            rotations,
            sh,
            num_gaussians,
            cov3d,
        };
        out.compute_cov3d();
        out
    }

    pub fn naive_gaussians() -> Self {
        let gaussians = naive_gaussians();
        Self::from_vec(gaussians)
    }
    pub fn compute_cov3d(&mut self)
    {
        // Iterate over all gaussians and compute the covariance matrix
        for i in 0..self.num_gaussians
        {
            let rotation = UnitQuaternion::from_quaternion(na::Quaternion::from_vector(self.rotations.column(i).into())).to_rotation_matrix();
            // square the scales and make a diagonal matrix
            let scale = na::Matrix3::new(
                self.scales[(0, i)] * self.scales[(0, i)], 0.0, 0.0,
                0.0, self.scales[(1, i)] * self.scales[(1, i)], 0.0,
                0.0, 0.0, self.scales[(2, i)] * self.scales[(2, i)],
            );
            // compute the covariance matrix
            let covariance = rotation * scale * rotation.transpose();
            self.cov3d.fixed_view_mut::<3,3>(0, i*3).copy_from(&covariance);
        }
    }

    pub fn sort(&self, camera: &Camera) -> Vec<usize>
    {
        // Converts positions to camera space and sorts by depth
        let positions = camera.get_view_matrix() * &self.positions;
        let mut indices: Vec<usize> = (0..self.num_gaussians).collect();
        indices.sort_by(|a, b| positions[(2, *a)].partial_cmp(&positions[(2, *b)]).unwrap());
        indices
    }

    pub fn project_cov3d_to_screen(&self, idx: usize, camera: &Camera) -> na::Matrix2<f32>
    {
        let position = self.positions.column(idx);
        let viewmatrix = camera.get_view_matrix();

        // Project the Gaussian center to the camera space
        let pos_w = na::Vector4::new(position[0], position[1], position[2], 1.0);

        let pos_cam = viewmatrix * pos_w;

        // Compute the focal length and the tangent of the fov
        let htanfovxy = camera.get_htanfovxy_focal();
        let tan_fovx = htanfovxy[0];
        let tan_fovy = htanfovxy[1];
        let focal = htanfovxy[2];
        let focal_x = focal;
        let focal_y = focal;

        // Compute the tangent of the Gaussian center
        let mut t = pos_cam.clone();
        let limx = 1.3 * tan_fovx;
        let limy = 1.3 * tan_fovy;

        let txtz = pos_cam.x/pos_cam.z;
        let tytz = pos_cam.y/pos_cam.z;

        t.x = limx.min((-limx).max(txtz)) * pos_cam.z;
        t.y = limy.min((-limy).max(tytz)) * pos_cam.z;

        // Compute the Jacobian
        let J = na::Matrix3::new(
            focal_x / t.z, 0.0, -(focal_x * t.x) / (t.z * t.z),
            0.0, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
            0.0, 0.0, 0.0
        );

        let W = viewmatrix.fixed_view::<3, 3>(0, 0).transpose();
        let T = W * J;
        let cov3d = self.cov3d.fixed_view::<3,3>(0, 3*idx);
        let mut cov = T.transpose() * cov3d.transpose() * T;

        // Apply low-pass filter: every Gaussian should be at least
        // one pixel wide/high. Discard 3rd row and column.
        let mut cov2d = cov.fixed_view_mut::<2,2>(0,0);
        cov2d[(0, 0)] += 0.3;
        cov2d[(1, 1)] += 0.3;

        // Convert into vector to return, along with position in screen space
        cov2d.into()
    }

    pub fn color(&self, idx: usize, sh_dim: usize, direction: &Vector3<f32>) -> (Vector3<f32>, f32) {
        // Compute color from the spherical harmonics for the given direction
        // The coefficients are laid out as:
        // C0_R, C0_B, C0_G, C1_R, C1_B, C1_G ...
        let color = eval_spherical_harmonics(&self.sh.column(idx).as_view(), sh_dim, direction);
        (color, self.opacities[idx])
    }
}
