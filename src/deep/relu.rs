use crate::deep::layer::Layer;

pub struct ReLU {
    mask: Vec<bool>,
}

impl ReLU {
    pub fn new() -> Self {
        Self { mask: vec![] }
    }
}

impl Layer for ReLU {
    fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        self.mask = input.iter().map(|&x| x > 0.0).collect();
        input.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect()
    }

    fn backward(&mut self, grad_output: &[f32], _lr: f32) -> Vec<f32> {
        grad_output
            .iter()
            .zip(self.mask.iter())
            .map(|(&g, &m)| if m { g } else { 0.0 })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_forward_and_backward() {
        let mut relu = ReLU::new();

        let input = vec![-1.0, 0.0, 2.0, -3.5, 4.0];
        let output = relu.forward(&input);
        assert_eq!(output, vec![0.0, 0.0, 2.0, 0.0, 4.0]);

        // Gradient should pass through only for positive elements
        let grad_output = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let grad_input = relu.backward(&grad_output, 0.0);
        assert_eq!(grad_input, vec![0.0, 0.0, 1.0, 0.0, 1.0]);
    }
}
