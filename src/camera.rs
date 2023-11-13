use nalgebra as na;

pub struct Camera {
    znear: f32,
    zfar: f32,
    pub h: f32,
    pub w: f32,
    fovy: f32,
    pub position: na::Vector3<f32>,
    target: na::Vector3<f32>,
    up: na::Vector3<f32>,
    yaw: f32,
    pitch: f32,
    is_pose_dirty: bool,
    is_intrin_dirty: bool,
    roll_sensitivity: f32,
}

impl Camera {
    pub fn new(h: f32, w: f32, position: Option<na::Vector3<f32>>) -> Self {
        Self {
            znear: 0.01,
            zfar: 100.0,
            h,
            w,
            fovy: std::f32::consts::PI / 2.0,
            position: position.unwrap_or(na::Vector3::new(0.0, 0.0, 3.0)),
            target: na::Vector3::new(0.0, 0.0, 0.0),
            up: na::Vector3::new(0.0, -1.0, 0.0),
            yaw: -std::f32::consts::PI / 2.0,
            pitch: 0.0,
            is_pose_dirty: true,
            is_intrin_dirty: true,
            roll_sensitivity: 0.03,
        }
    }

    pub fn get_view_matrix(&self) -> na::Matrix4<f32> {
        let position = na::Point3::from(self.position);
        let target = na::Point3::from(self.target);
        let view = na::Matrix4::look_at_rh(
            &position,
            &target,
            &self.up,
        );
        view

    }

    pub fn update_resolution(&mut self, height: f32, width: f32) {
        self.h = height;
        self.w = width;
        self.is_intrin_dirty = true;
    }

    pub fn get_project_matrix(&self) -> na::Matrix4<f32> {
        na::Perspective3::new(self.w / self.h, self.fovy, self.znear, self.zfar).into_inner()
    }

    pub fn get_htanfovxy_focal(&self) -> na::Vector3<f32> {
        let htany = (self.fovy / 2.0).tan();
        let htanx = htany / self.h * self.w;
        let focal = self.h / (2.0 * htany);
        na::Vector3::new(htanx, htany, focal)
    }

    pub fn get_focal(&self) -> f32 {
        self.h / (2.0 * (self.fovy / 2.0).tan())
    }
}
