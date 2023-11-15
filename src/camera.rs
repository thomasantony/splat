use nalgebra as na;
use nalgebra_glm as glm;
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
    view_matrix: na::Matrix4<f32>,
    projection_matrix: na::Matrix4<f32>,
}

impl Camera {
    pub fn new(h: f32, w: f32, start_position: Option<na::Vector3<f32>>) -> Self {
        Self {
            znear: 0.01,
            zfar: 100.0,
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
            view_matrix: na::Matrix4::identity(),
            projection_matrix: na::Matrix4::identity(),
        }
    }

    pub fn compute_matrices(&mut self)
    {
        // Ref: https://asliceofrendering.com/camera/2019/11/30/ArcballCamera/
        // Get position and target as homogeneous coordinates
        let position = glm::vec4(self.position.x, self.position.y, self.position.z, 1.0);
        let pivot = glm::vec4(self.target.x, self.target.y, self.target.z, 1.0);

        // Extra step to handle the problem when the camera direction is the same as the up vector
        let viewdir = glm::normalize(&(self.position - &self.target));
        let cos_angle = viewdir.dot(&self.up);
        if cos_angle * self.pitch.signum() > 0.99f32
        {
            self.pitch = 0.0;
        }

        // step 2: Rotate the camera around the pivot point on the first axis.
        let rotation_x = glm::rotation(self.yaw, &self.up);
        let position = (rotation_x * (position - pivot)) + pivot;

        // step 3: Rotate the camera around the pivot point on the second axis.
        let right = glm::cross(&self.up, &self.position);
        let rotation_y = glm::rotation(self.pitch, &right);
        let final_position = (rotation_y * (position - pivot)) + pivot;

        self.view_matrix = glm::look_at(&final_position.xyz(), &self.target, &self.up);

        self.projection_matrix = glm::perspective(self.w / self.h, self.fovy, self.znear, self.zfar);
    }

    pub fn get_view_matrix(&self) -> &na::Matrix4<f32> {
        return &self.view_matrix;
    }

    pub fn update_resolution(&mut self, height: f32, width: f32) {
        self.h = height;
        self.w = width;
        self.is_intrin_dirty = true;
    }

    pub fn get_project_matrix(&self) -> &na::Matrix4<f32> {
        return &self.projection_matrix;
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
        self.pitch += delta;
        self.is_pose_dirty = true;
    }
    pub fn update_yaw_angle(&mut self, delta: f32) {
        self.yaw += delta;
        self.is_pose_dirty = true;
    }

    pub fn update_camera_pose(&mut self) {
        // let camera_x = na::Vector3::x_axis();    // Pitch axis
        // let camera_y = na::Vector3::y_axis();    // Yaw axis

        // // Camera's Z axis is facing in the -self.position direction in the world frame
        // let transform_c2w = na::Rotation::<f32, 3>::rotation_between(&-self.position, &na::Vector3::z()).unwrap_or(na::Rotation3::identity());
        // let pitch_axis = transform_c2w * camera_x;
        // let yaw_axis = transform_c2w * camera_y;

        // let pitch_rotation = na::Rotation3::from_axis_angle(&pitch_axis, self.pitch);
        // let yaw_rotation = na::Rotation3::from_axis_angle(&yaw_axis, self.yaw);

        // let rotation = yaw_rotation * pitch_rotation;

        // self.position = rotation * self.position;

        // self.yaw = 0.0;
        // self.pitch =0.0;
        self.compute_matrices();
        // println!("positon: {}", self.position);
        // println!("view_matrix: {}", self.view_matrix.fixed_slice::<1,4>(0,0));

        self.is_pose_dirty = false;
    }
}
