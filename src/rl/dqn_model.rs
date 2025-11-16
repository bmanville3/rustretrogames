use crate::deep::{layer::Layer, linear::{Linear, WeightInit}, relu::ReLU, sequential::{LayerEnum, Sequential}};

#[derive(Debug, Clone)]
pub struct DQN {
    net: Sequential,
}

impl DQN {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut net = Sequential::new();
        net.add(LayerEnum::Linear(Linear::new(input_dim, 128, WeightInit::He)));
        net.add(LayerEnum::ReLU(ReLU::new()));
        net.add(LayerEnum::Linear(Linear::new(128, 64, WeightInit::He)));
        net.add(LayerEnum::ReLU(ReLU::new()));
        net.add(LayerEnum::Linear(Linear::new(64, output_dim, WeightInit::He)));
        net.add(LayerEnum::ReLU(ReLU::new()));
        DQN { net }
    }

    pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        self.net.forward(input)
    }

    pub fn backward(&mut self, grad_loss: &[f32], lr: f32) {
        self.net.backward(grad_loss, lr);
    }
}
