pub trait Layer {
    fn forward(&mut self, input: &[f32]) -> Vec<f32>;
    fn backward(&mut self, grad_output: &[f32], learning_rate: f32) -> Vec<f32>;
}
