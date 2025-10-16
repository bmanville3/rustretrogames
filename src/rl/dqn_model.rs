use crate::deep::{linear::Linear, relu::ReLU, sequential::Sequential};

pub struct DQN {
    net: Sequential,
}

impl DQN {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut net = Sequential::new();
        net.add(Linear::new(input_dim, 128));
        net.add(ReLU::new());
        net.add(Linear::new(128, 64));
        net.add(ReLU::new());
        net.add(Linear::new(64, output_dim));
        net.add(ReLU::new());
        DQN { net }
    }

    pub fn forward(&mut self, input: Vec<f32>) -> Vec<f32> {
        self.net.forward(input)
    }
}
