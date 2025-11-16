use rand::Rng;
use rand_distr::{Normal, Distribution};
use serde::{Deserialize, Serialize};

use crate::deep::layer::{StatefulLayer, StatelessLayer};

/// Weight initialization schemes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WeightInit {
    He,
    Uniform(f32),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatelessLinear {
    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<f32>,
}

impl StatelessLinear {
    pub fn new(input_size: usize, output_size: usize, init: WeightInit) -> Self {
        let mut rng = rand::thread_rng();
        let weights: Vec<Vec<f32>> = match init {
            WeightInit::He => {
                // He initialization: Normal(0, sqrt(2 / fan_in))
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
        Self {
            weights,
            biases,
        }
    }
}

impl StatelessLayer<f32> for StatelessLinear {
    fn _forward(&mut self, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
        (self.weights
            .iter()
            .zip(self.biases.iter())
            .map(|(w_row, b)| w_row.iter().zip(input).map(|(w, x)| w * x).sum::<f32>() + b)
            .collect(), vec![])
    }

    fn _backward(&mut self, input: &[f32], grad_output: &[f32], lr: f32, _intermediate_caches: &Vec<f32>) -> Vec<f32> {
        // Gradient w.r.t input
        let mut grad_input = vec![0.0; input.len()];

        for i in 0..self.weights.len() {
            for j in 0..input.len() {
                grad_input[j] += self.weights[i][j] * grad_output[i];
            }
        }

        // Update weights and biases
        for i in 0..self.weights.len() {
            for j in 0..input.len() {
                self.weights[i][j] -= lr * grad_output[i] * input[j];
            }
            self.biases[i] -= lr * grad_output[i];
        }

        grad_input
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Linear {
    pub stateless_linear: StatelessLinear,
    pub input_cache: Vec<f32>,
}

impl Linear {
    pub fn new(input_size: usize, output_size: usize, init: WeightInit) -> Self {
        Self {
            stateless_linear: StatelessLinear::new(input_size, output_size, init),
            input_cache: vec![],
        }
    }
}

impl StatefulLayer for Linear {
    fn _forward(&mut self, input: &[f32]) -> Vec<f32> {
        self.input_cache = input.to_vec();
        let (out, _) = self.stateless_linear._forward(input);
        out
    }

    fn _backward(&mut self, grad_output: &[f32], learning_rate: f32) -> Vec<f32> {
        let input = &self.input_cache;
        self.stateless_linear._backward(input, grad_output, learning_rate, &vec![])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward() {
        let mut layer = Linear {
            stateless_linear: StatelessLinear { weights: vec![vec![0.5, -0.5], vec![1.0, 2.0]], biases: vec![0.1, -0.2] },
            input_cache: vec![],
        };

        let input = vec![2.0, 3.0];
        let output = layer._forward(&input);

        // Expected:
        // neuron 0: 0.5*2 + (-0.5)*3 + 0.1 = -0.4
        // neuron 1: 1*2 + 2*3 - 0.2 = 7.8
        let expected = vec![-0.4, 7.8];
        for (o, e) in output.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-6);
        }
    }

    #[test]
    fn test_linear_backward_updates_weights() {
        let mut layer = Linear {
            stateless_linear: StatelessLinear {weights: vec![vec![1.0, 2.0]],
            biases: vec![0.5]},
            input_cache: vec![3.0, 4.0],
        };

        let grad_output = vec![2.0];
        let lr = 0.1;

        // Gradient w.r.t input = weights^T * grad_output = [2, 4]
        let grad_input = layer._backward(&grad_output, lr);
        assert_eq!(grad_input, vec![2.0, 4.0]);

        // Weight updates: w -= lr * grad_output * input
        //   dw = 0.1 * 2.0 * [3, 4] = [0.6, 0.8]
        //   => w = [1.0 - 0.6, 2.0 - 0.8] = [0.4, 1.2]
        assert!((layer.stateless_linear.weights[0][0] - 0.4).abs() < 1e-6);
        assert!((layer.stateless_linear.weights[0][1] - 1.2).abs() < 1e-6);

        // Bias update: b -= lr * grad_output = 0.5 - 0.2 = 0.3
        assert!((layer.stateless_linear.biases[0] - 0.3).abs() < 1e-6);
    }
}
