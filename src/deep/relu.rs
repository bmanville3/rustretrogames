use serde::{Deserialize, Serialize};
use crate::deep::layer::{Layer, StatefulLayerWrapper, StatelessLayer};

/// Stateless ReLU: core forward/backward logic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatelessReLU;

impl StatelessReLU {
    pub fn new() -> Self {
        Self
    }
}

impl StatelessLayer for StatelessReLU {
    type ParamGrad = Vec<f32>; // ReLU has no parameters, so ParamGrad is empty

    fn forward(&self, input: &[f32]) -> (Vec<f32>, Vec<Vec<f32>>) {
        let output: Vec<f32> = input.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect();
        (output, vec![]) // ReLU has no intermediate caches
    }

    fn backward_no_update(
        &self,
        input: &[f32],
        grad_output: &[f32],
        _caches: &Vec<Vec<f32>>,
    ) -> (Vec<f32>, Self::ParamGrad) {
        let grad_input: Vec<f32> = input.iter()
            .zip(grad_output.iter())
            .map(|(&x, &g)| if x > 0.0 { g } else { 0.0 })
            .collect();
        (grad_input, vec![]) // No parameters to update
    }

    fn apply_update(&mut self, _grad: &Self::ParamGrad, _learning_rate: f32) {
        // Nothing to update
    }
    
    fn accumulate_param_grads(_grad1: Self::ParamGrad, _grad2: Self::ParamGrad) -> Self::ParamGrad {
        vec![]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReLU {
    pub layer: StatefulLayerWrapper<StatelessReLU>,
}

impl ReLU {
    pub fn new() -> Self {
        Self {
            layer: StatefulLayerWrapper::new(StatelessReLU::new()),
        }
    }
}

impl Layer<StatelessReLU> for ReLU {
    fn get_stateful_wrapper_mut(&mut self) -> &mut StatefulLayerWrapper<StatelessReLU> {
       &mut self.layer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_forward() {
        let relu = StatelessReLU::new();
        let input = vec![-1.0, 0.0, 2.0, -3.5, 4.0];
        let (output, _cache) = relu.forward(&input);
        assert_eq!(output, vec![0.0, 0.0, 2.0, 0.0, 4.0]);

        let grad_output = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let (grad_input, _grad_params) = relu.backward_no_update(&input, &grad_output, &vec![]);
        assert_eq!(grad_input, vec![0.0, 0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_relu_backward() {
        let mut relu = ReLU::new();
        let input = vec![-1.0, 0.0, 2.0, -3.5, 4.0];
        let output = relu.forward(&input);
        assert_eq!(output, vec![0.0, 0.0, 2.0, 0.0, 4.0]);

        let grad_output = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let grad_input = relu.backward(&grad_output, 0.0);
        assert_eq!(grad_input, vec![0.0, 0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_relu_batch_forward_backward() {
        let mut relu = ReLU::new();

        // Create a batch of 4 inputs
        let batch_inputs: Vec<Vec<f32>> = vec![
            vec![-1.0, 2.0, 0.0],
            vec![3.0, -4.0, 1.0],
            vec![0.0, 0.0, 0.0],
            vec![-2.0, 5.0, -3.0],
        ];

        // Convert to slices for batch API
        let batch_slices: Vec<&[f32]> = batch_inputs.iter().map(|v| v.as_slice()).collect();

        // Forward batch
        let batch_outputs: Vec<Vec<f32>> = batch_slices
            .iter()
            .map(|x| relu.forward(x))
            .collect();

        // Expected outputs explained:
        // ReLU(x) = max(0, x)
        // Batch 0: [-1, 2, 0] -> [0, 2, 0]
        // Batch 1: [3, -4, 1] -> [3, 0, 1]
        // Batch 2: [0, 0, 0] -> [0, 0, 0]
        // Batch 3: [-2, 5, -3] -> [0, 5, 0]
        let expected_outputs: Vec<Vec<f32>> = vec![
            vec![0.0, 2.0, 0.0],
            vec![3.0, 0.0, 1.0],
            vec![0.0, 0.0, 0.0],
            vec![0.0, 5.0, 0.0],
        ];

        assert_eq!(batch_outputs, expected_outputs);

        // Grad outputs (same shape as batch outputs)
        let batch_grad_outputs: Vec<Vec<f32>> = vec![
            vec![1.0, 1.0, 1.0],
            vec![0.5, 0.5, 0.5],
            vec![2.0, 2.0, 2.0],
            vec![3.0, 3.0, 3.0],
        ];

        let batch_grad_slices: Vec<&[f32]> = batch_grad_outputs.iter().map(|v| v.as_slice()).collect();

        // Expected backward gradients explained:
        // Batch 0 input: [-1, 2, 0], grad_out: [1, 1, 1] -> grad_input: [0, 1, 0]
        // Batch 1 input: [3, -4, 1], grad_out: [0.5, 0.5, 0.5] -> grad_input: [0.5, 0, 0.5]
        // Batch 2 input: [0, 0, 0], grad_out: [2, 2, 2] -> grad_input: [0, 0, 0]
        // Batch 3 input: [-2, 5, -3], grad_out: [3, 3, 3] -> grad_input: [0, 3, 0]
        let batch_grad_inputs: Vec<Vec<f32>> = batch_slices
            .iter()
            .zip(batch_grad_slices.iter())
            .map(|(input, grad_out)| {
                relu.layer.inner.backward_no_update(input, grad_out, &vec![]).0
            })
            .collect();

        let expected_grad_inputs: Vec<Vec<f32>> = vec![
            vec![0.0, 1.0, 0.0],
            vec![0.5, 0.0, 0.5],
            vec![0.0, 0.0, 0.0],
            vec![0.0, 3.0, 0.0],
        ];

        assert_eq!(batch_grad_inputs, expected_grad_inputs);
    }
}
