use nalgebra as na;

#[derive(Clone, Debug)]
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
    pub is_pose_dirty: bool,
    is_intrin_dirty: bool,
}

impl Camera {
    pub fn new(h: f32, w: f32, start_position: Option<na::Vector3<f32>>) -> Self {
        Self {
            znear: 0.01,
            zfar: 1000.0,
            h,
            w,
            fovy: std::f32::consts::PI / 2.0,
            position: start_position.unwrap_or(na::Vector3::new(0.0, 0.0, 3.0)),
            target: na::Vector3::new(0.0, 0.0, 0.0),
            up: na::Vector3::new(0.0, -1.0, 0.0),
            yaw: 0.0,
            pitch: 0.0,
            is_pose_dirty: true,
            is_intrin_dirty: true,
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
    pub fn update_pitch_angle(&mut self, delta: f32) {
        self.pitch = delta;
        self.is_pose_dirty = true;
    }
    pub fn update_yaw_angle(&mut self, delta: f32) {
        self.yaw = delta;
        self.is_pose_dirty = true;
    }

    pub fn update_camera_pose(&mut self) {
        let camera_x = na::Vector3::x_axis();    // Pitch axis
        let camera_y = na::Vector3::y_axis();    // Yaw axis

        // Camera's Z axis is facing in the -self.position direction in the world frame
        let transform_c2w = na::Rotation::<f32, 3>::rotation_between(&-self.position, &na::Vector3::z()).unwrap_or(na::Rotation3::identity());
        let pitch_axis = transform_c2w * camera_y;
        let yaw_axis = transform_c2w * camera_x;

        println!("Pitch axis in world space: {:?}", pitch_axis);
        println!("Yaw axis in world space: {:?}", yaw_axis);

        let pitch_rotation = na::Rotation3::from_axis_angle(&pitch_axis, self.pitch);
        let yaw_rotation = na::Rotation3::from_axis_angle(&yaw_axis, self.yaw);

        let rotation = yaw_rotation * pitch_rotation;

        self.position = rotation * self.position;
        println!("Camera position: {:?}", self.position);

        self.yaw = 0.0;
        self.pitch =0.0;

        self.is_pose_dirty = false;
    }
}
