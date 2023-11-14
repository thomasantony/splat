# splat

This project is a Rust implementation of a Gaussian Splatting Viewer, inspired by the OpenGL shader code found [here](https://github.com/limacv/GaussianSplattingViewer).

The project was first prototyped in a Jupyter notebook to validate the approach and algorithms. After the prototype was successful, the project was then implemented in Rust using the `euc` software rendering crate.

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

- Rust: You can download Rust from [the official website](https://www.rust-lang.org/tools/install).
- Some pre-trained 3D Gaussian splat models. Some examples:
    - https://media.reshot.ai/models/plush_sledge/output.zip
    - https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip  - 13GB

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
2. Build and Run the project
    ```sh
    cargo run --release splat
    ```

### License
Distributed under the MIT License. See LICENSE for more information.
