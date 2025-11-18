use rand::Rng;
use rand_distr::{Normal, Distribution};
use serde::{Deserialize, Serialize};
use crate::deep::layer::{Layer, StatelessLayer, StatefulLayerWrapper};

/// Weight initialization schemes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WeightInit {
    He,
    Uniform(f32),
}

/// Stateless linear layer: implements core forward/backward logic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatelessLinear {
    pub weights: Vec<Vec<f32>>, // [output_size][input_size]
    pub biases: Vec<f32>,
}

impl StatelessLinear {
    pub fn new(input_size: usize, output_size: usize, init: WeightInit) -> Self {
        let mut rng = rand::thread_rng();
        let weights = match init {
            WeightInit::He => {
                let std = (2.0 / input_size as f32).sqrt();
                let normal = Normal::new(0.0, std).unwrap();
                (0..output_size)
                    .map(|_| (0..input_size).map(|_| normal.sample(&mut rng) as f32).collect())
                    .collect()
            }
            WeightInit::Uniform(limit) => {
                (0..output_size)
                    .map(|_| (0..input_size).map(|_| rng.gen_range(-limit..limit)).collect())
                    .collect()
            }
        };
        let biases = vec![0.0; output_size];
        Self { weights, biases }
    }
}

impl StatelessLayer for StatelessLinear {
    type ParamGrad = Vec<f32>; // Use the ParamGrad associated type

    fn accumulate_param_grads(mut grad1: Self::ParamGrad, grad2: Self::ParamGrad) -> Self::ParamGrad {
        for (a, b) in grad1.iter_mut().zip(grad2.into_iter()) {
            *a += b;
        }
        grad1
    }

    fn forward(&self, input: &[f32]) -> (Vec<f32>, Vec<Vec<f32>>) {
        if self.weights.len() > 0 && input.len() != self.weights[0].len() {
            panic!("Input does not match weights. Got input length {:#?} but weights had shape ({:#?}, {:#?})", input.len(), self.weights.len(), self.weights[0].len())
        }
        let output: Vec<f32> = self.weights
            .iter()
            .zip(self.biases.iter())
            .map(|(w_row, b)| w_row.iter().zip(input).map(|(w, x)| w * x).sum::<f32>() + b)
            .collect();
        (output, vec![]) // Linear has no intermediate caches
    }

    fn backward_no_update(
        &self,
        input: &[f32],
        grad_output: &[f32],
        _caches: &Vec<Vec<f32>>,
    ) -> (Vec<f32>, Self::ParamGrad) {
        // Gradient w.r.t input: W^T * grad_output
        let mut grad_input = vec![0.0; input.len()];
        for i in 0..self.weights.len() {
            for j in 0..input.len() {
                grad_input[j] += self.weights[i][j] * grad_output[i];
            }
        }

        // Gradient w.r.t parameters (flatten weights + biases)
        let mut grad_params = Vec::with_capacity(self.weights.len() * input.len() + self.biases.len());
        for i in 0..self.weights.len() {
            // weights gradient
            grad_params.extend(input.iter().map(|x| grad_output[i] * x));
            // bias gradient
            grad_params.push(grad_output[i]);
        }

        (grad_input, grad_params)
    }

    fn apply_update(&mut self, grad: &Self::ParamGrad, learning_rate: f32) {
        let input_size = self.weights[0].len();
        for (i, w_row) in self.weights.iter_mut().enumerate() {
            for j in 0..input_size {
                w_row[j] -= learning_rate * grad[i * (input_size + 1) + j];
            }
            self.biases[i] -= learning_rate * grad[i * (input_size + 1) + input_size];
        }
    }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Linear {
    pub layer: StatefulLayerWrapper<StatelessLinear>,
}

impl Linear {
    pub fn new(input_size: usize, output_size: usize, init: WeightInit) -> Self {
        Self {
            layer: StatefulLayerWrapper::new(StatelessLinear::new(input_size, output_size, init)),
        }
    }
}

impl Layer<StatelessLinear> for Linear {
    fn get_stateful_wrapper_mut(&mut self) -> &mut StatefulLayerWrapper<StatelessLinear> {
        &mut self.layer
    }
}

#[cfg(test)]
mod tests {
    use rand::seq::SliceRandom;

    use crate::deep::mse::mse_loss;

    use super::*;

    #[test]
    fn test_linear_forward() {
        let mut layer = Linear::new(2, 2, WeightInit::Uniform(1.0));
        layer.layer.inner.weights = vec![vec![0.5, -0.5], vec![1.0, 2.0]];
        layer.layer.inner.biases = vec![0.1, -0.2];

        let input = vec![2.0, 3.0];
        let output = layer.forward(&input);

        let expected = vec![-0.4, 7.8];
        for (o, e) in output.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-6);
        }
    }

    #[test]
    fn test_linear_backward() {
        let stateless_linear = StatelessLinear { weights: vec![vec![1.0, 2.0]], biases: vec![0.5] };
        let mut linear = Linear { layer: StatefulLayerWrapper::new(stateless_linear) };
        
        let input = vec![3.0, 4.0];
        let _ = linear.forward(&input);

        let grad_output = vec![2.0];
        let grad_input = linear.backward(&grad_output, 0.1);

        // Gradient w.r.t input = W^T * grad_output = [2, 4]
        assert_eq!(grad_input, vec![2.0, 4.0]);

        // Weight updates: w -= lr * grad_output * input
        //   dw = 0.1 * 2.0 * [3, 4] = [0.6, 0.8]
        //   => w = [1.0 - 0.6, 2.0 - 0.8] = [0.4, 1.2]
        assert!((linear.layer.inner.weights[0][0] - 0.4).abs() < 1e-6);
        assert!((linear.layer.inner.weights[0][1] - 1.2).abs() < 1e-6);

        // Bias update: b -= lr * grad_output = 0.5 - 0.2 = 0.3
        assert!((linear.layer.inner.biases[0] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_linear_batch_forward_backward() {
        let mut linear = Linear::new(2, 1, WeightInit::Uniform(1.0));
        linear.layer.inner.weights = vec![vec![1.0, 2.0]];
        linear.layer.inner.biases = vec![0.5];

        let batch_inputs: Vec<&[f32]> = vec![
            &[3.0, 4.0],
            &[1.0, -1.0],
            &[0.0, 2.0],
        ];

        // Forward batch
        let outputs = linear.forward_batch(&batch_inputs);

        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0], vec![11.5]); // 1*3 + 2*4 + 0.5
        assert_eq!(outputs[1], vec![-0.5]); // 1*1 + 2*-1 + 0.5
        assert_eq!(outputs[2], vec![4.5]); // 1*0 + 2*2 + 0.5

        // Grad outputs
        let batch_grad_outputs: Vec<&[f32]> = vec![
            &[1.0],
            &[2.0],
            &[3.0],
        ];

        // Backward batch (weights should be updated)
        let grad_inputs = linear.backward_batch(&batch_grad_outputs, 0.1);

        // Check gradient inputs
        assert_eq!(grad_inputs.len(), 3);
        assert_eq!(grad_inputs[0], vec![1.0, 2.0]); // W^T * grad
        assert_eq!(grad_inputs[1], vec![2.0, 4.0]);
        assert_eq!(grad_inputs[2], vec![3.0, 6.0]);

        // Check weight update
        // average grad for weights = [(1*3 + 2*1 + 3*0)/3, (1*4 + 2*-1 + 3*2)/3] * lr
        let expected_w0 = 1.0 - 0.1 * ((1.0*3.0 + 2.0*1.0 + 3.0*0.0)/3.0);
        let expected_w1 = 2.0 - 0.1 * ((1.0*4.0 + 2.0*-1.0 + 3.0*2.0)/3.0);
        assert!((linear.layer.inner.weights[0][0] - expected_w0).abs() < 1e-6);
        assert!((linear.layer.inner.weights[0][1] - expected_w1).abs() < 1e-6);

        // Bias update
        let expected_b = 0.5 - 0.1 * ((1.0 + 2.0 + 3.0)/3.0);
        assert!((linear.layer.inner.biases[0] - expected_b).abs() < 1e-6);
    }

    #[test]
    fn test_linear_learns_linear_mapping() {
        let mut linear = Linear::new(2, 2, WeightInit::He);

        // Define a linear mapping: y = 2*x0 - 3*x1 + 0.5
        let target_fn = |x: &[f32]| -> f32 { 2.0 * x[0] - 3.0 * x[1] + 0.5 };

        let mut rng = rand::thread_rng();
        let n_samples = 1000;

        // Generate random input samples
        let batch_inputs: Vec<Vec<f32>> = (0..n_samples)
            .map(|_| vec![
                rng.gen_range(-10.0..10.0),
                rng.gen_range(-10.0..10.0),
            ])
            .collect();

        let learning_rate = 0.01;
        let epochs = 1000;
        let batch_size = 10;

        for _epoch in 0..epochs {
            // Sample random batch indices
            let sampled_indices: Vec<usize> = (0..batch_inputs.len())
                .collect::<Vec<_>>()
                .choose_multiple(&mut rng, batch_size)
                .cloned()
                .collect();

            // Prepare batch slices
            let batch_slices: Vec<&[f32]> = sampled_indices
                .iter()
                .map(|&i| batch_inputs[i].as_slice())
                .collect();

            // Forward pass
            let outputs = linear.forward_batch(&batch_slices);

            // Compute gradients for the sampled batch
            let batch_grad_outputs: Vec<Vec<f32>> = sampled_indices
                .iter()
                .zip(outputs.iter())
                .map(|(&i, y_pred)| {
                    let y_true = target_fn(&batch_inputs[i]);
                    mse_loss(y_pred, &[y_true, y_true]).1
                })
                .collect();

            let batch_grad_slices: Vec<&[f32]> = batch_grad_outputs.iter().map(|v| v.as_slice()).collect();

            // Backward batch update
            let _ = linear.backward_batch(&batch_grad_slices, learning_rate);
        }

        // Evaluate on all samples
        let batch_slices: Vec<&[f32]> = batch_inputs.iter().map(|v| v.as_slice()).collect();
        let outputs = linear.forward_batch(&batch_slices);

        for (x, y_pred) in batch_inputs.iter().zip(outputs.iter()) {
            let y_true = target_fn(x);
            let y_true = [y_true, y_true];
            let rel_error = ((y_pred[0] - y_true[0]).abs()) / y_true[0].abs();
            assert!(
                rel_error < 0.05,
                "Prediction error too high: {} vs {}",
                y_pred[0],
                y_true[0]
            );
        }
    }
}
