use rand::Rng;

use crate::deep::layer::Layer;

pub struct Linear {
    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<f32>,
    pub input_cache: Vec<f32>,
}

impl Linear {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..output_size)
            .map(|_| (0..input_size).map(|_| rng.gen_range(-0.1..0.1)).collect())
            .collect();
        let biases = vec![0.0; output_size];
        Self {
            weights,
            biases,
            input_cache: vec![],
        }
    }
}

impl Layer for Linear {
    fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        self.input_cache = input.to_vec();
        self.weights
            .iter()
            .zip(self.biases.iter())
            .map(|(w_row, b)| w_row.iter().zip(input).map(|(w, x)| w * x).sum::<f32>() + b)
            .collect()
    }

    fn backward(&mut self, grad_output: &[f32], lr: f32) -> Vec<f32> {
        let input = &self.input_cache;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward() {
        let mut layer = Linear {
            weights: vec![vec![0.5, -0.5], vec![1.0, 2.0]],
            biases: vec![0.1, -0.2],
            input_cache: vec![],
        };

        let input = vec![2.0, 3.0];
        let output = layer.forward(&input);

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
            weights: vec![vec![1.0, 2.0]],
            biases: vec![0.5],
            input_cache: vec![3.0, 4.0],
        };

        let grad_output = vec![2.0];
        let lr = 0.1;

        // Gradient w.r.t input = weights^T * grad_output = [2, 4]
        let grad_input = layer.backward(&grad_output, lr);
        assert_eq!(grad_input, vec![2.0, 4.0]);

        // Weight updates: w -= lr * grad_output * input
        //   dw = 0.1 * 2.0 * [3, 4] = [0.6, 0.8]
        //   => w = [1.0 - 0.6, 2.0 - 0.8] = [0.4, 1.2]
        assert!((layer.weights[0][0] - 0.4).abs() < 1e-6);
        assert!((layer.weights[0][1] - 1.2).abs() < 1e-6);

        // Bias update: b -= lr * grad_output = 0.5 - 0.2 = 0.3
        assert!((layer.biases[0] - 0.3).abs() < 1e-6);
    }
}
