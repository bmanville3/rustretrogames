use serde::{Deserialize, Serialize};
use crate::deep::layer::{StatefulLayer, StatelessLayer};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatelessReLU;

impl StatelessReLU {
    pub fn new() -> Self {
        Self
    }
}

impl StatelessLayer<f32> for StatelessReLU {
    fn _forward(&mut self, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
        (input.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect(), vec![])
    }

    fn _backward(&mut self, input: &[f32], grad_output: &[f32], _lr: f32, _intermediate_caches: &Vec<f32>) -> Vec<f32> {
        input.iter()
            .zip(grad_output.iter())
            .map(|(&x, &g)| if x > 0.0 { g } else { 0.0 })
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReLU {
    mask: Vec<bool>,
    stateless: StatelessReLU,
}

impl ReLU {
    pub fn new() -> Self {
        Self {
            mask: vec![],
            stateless: StatelessReLU::new(),
        }
    }
}

impl StatefulLayer for ReLU {
    fn _forward(&mut self, input: &[f32]) -> Vec<f32> {
        self.mask = input.iter().map(|&x| x > 0.0).collect();
        let (out, _) = self.stateless._forward(input);
        out
    }

    fn _backward(&mut self, grad_output: &[f32], _learning_rate: f32) -> Vec<f32> {
        grad_output.iter()
            .zip(self.mask.iter())
            .map(|(&g, &m)| if m { g } else { 0.0 })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stateless_relu_forward_and_backward() {
        let mut relu = StatelessReLU::new();
        let input = vec![-1.0, 0.0, 2.0, -3.5, 4.0];
        let (output, _cache) = relu._forward(&input);
        assert_eq!(output, vec![0.0, 0.0, 2.0, 0.0, 4.0]);

        let grad_output = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let grad_input = relu._backward(&input, &grad_output, 0.0, &vec![]);
        assert_eq!(grad_input, vec![0.0, 0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_stateful_relu_forward_and_backward() {
        let mut relu = ReLU::new();
        let input = vec![-1.0, 0.0, 2.0, -3.5, 4.0];
        let output = relu._forward(&input);
        assert_eq!(output, vec![0.0, 0.0, 2.0, 0.0, 4.0]);

        let grad_output = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let grad_input = relu._backward(&grad_output, 0.0);
        assert_eq!(grad_input, vec![0.0, 0.0, 1.0, 0.0, 1.0]);
    }
}
