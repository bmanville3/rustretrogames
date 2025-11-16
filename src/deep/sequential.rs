use serde::{Deserialize, Serialize};
use crate::deep::layer::{StatefulLayer, StatelessLayer};
use crate::deep::linear::StatelessLinear;
use crate::deep::relu::StatelessReLU;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StatelessLayerEnum {
    Linear(StatelessLinear),
    ReLU(StatelessReLU),
    Sequential(StatelessSequential),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerCache {
    Output(Vec<f32>),
    Sequential(Vec<LayerCache>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatelessSequential {
    layers: Vec<StatelessLayerEnum>,
}

impl StatelessLayer<LayerCache> for StatelessSequential {
    fn _forward(&mut self, input: &[f32]) -> (Vec<f32>, Vec<LayerCache>) {
        if self.layers.is_empty() {
            panic!("There must be at least one layer before calling forward")
        }
        let mut output = input.to_vec();
        let mut caches = Vec::new();

        // input is never added to cached
        // adds the output (so input next round is already added)
        for layer in &mut self.layers {
            match layer {
                StatelessLayerEnum::Linear(l) => {
                    let (out, _sub_caches) = l.forward(&output);
                    output = out;
                    caches.push(LayerCache::Output(output.clone()));
                }
                StatelessLayerEnum::ReLU(r) => {
                    let (out, _sub_caches) = r.forward(&output);
                    output = out;
                    caches.push(LayerCache::Output(output.clone()));
                }
                StatelessLayerEnum::Sequential(s) => {
                    let (out, sub_caches) = s.forward(&output);
                    output = out;
                    caches.push(LayerCache::Sequential(sub_caches));
                    caches.push(LayerCache::Output(output.clone()))
                }
            }
        }
        // last element will be the last output
        let _ = caches.pop();
        (output, caches)
    }

    fn _backward(&mut self, input: &[f32], grad_output: &[f32], learning_rate: f32, intermediate_caches: &Vec<LayerCache>) -> Vec<f32> {
        let mut grad = grad_output.to_vec();

        for (layer, cache) in self.layers.iter_mut().rev().zip(intermediate_caches.iter().rev()) {
            grad = match (layer, cache) {
                (StatelessLayerEnum::Linear(l), LayerCache::Output(input)) => {
                    l.backward(input, &grad, learning_rate, input)
                }
                (StatelessLayerEnum::ReLU(r), LayerCache::Output(input)) => {
                    r.backward(input, &grad, learning_rate, input)
                }
                (StatelessLayerEnum::Sequential(s), LayerCache::Sequential(sub_caches)) => {
                    s.backward(input, &grad, learning_rate, sub_caches)
                }
                _ => panic!("Layer / cache mismatch"),
            };
        }

        grad
    }
}

impl StatelessSequential {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add(&mut self, layer: StatelessLayerEnum) {
        self.layers.push(layer);
    }
}

/// Stateful Sequential that stores caches internally
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sequential {
    stateless: StatelessSequential,
    caches: Vec<LayerCache>,
    input: Vec<f32>,
}

impl Sequential {
    pub fn new() -> Self {
        Self {
            stateless: StatelessSequential::new(),
            caches: vec![],
            input: vec![],
        }
    }

    pub fn add(&mut self, layer: StatelessLayerEnum) {
        self.stateless.add(layer);
    }
}

impl StatefulLayer for Sequential {
    fn _forward(&mut self, input: &[f32]) -> Vec<f32> {
        self.input = input.to_vec();
        let (output, caches) = self.stateless._forward(input);
        self.caches = caches;
        output
    }

    fn _backward(&mut self, grad_output: &[f32], learning_rate: f32) -> Vec<f32> {
        self.stateless._backward(&self.input, grad_output, learning_rate, &self.caches)
    }
}


#[cfg(test)]
mod tests {
    use crate::deep::{linear::WeightInit, mse::mse_loss};

    use super::*;
    use rand::Rng;

    #[test]
    fn test_stateful_sequential_forward_linear_relu() {
        let linear = StatelessLinear {
            weights: vec![vec![0.5, -0.5], vec![1.0, 2.0]],
            biases: vec![0.1, -0.2],
        };

        let relu = StatelessReLU::new();

        let mut seq = Sequential::new();
        seq.add(StatelessLayerEnum::Linear(linear));
        seq.add(StatelessLayerEnum::ReLU(relu));

        let input = vec![2.0, 3.0];

        let output = seq.forward(&input);
        assert_eq!(seq.caches.len(), 1);

        let linear_out = vec![-0.4, 7.8];
        assert!(matches!(seq.caches[0], LayerCache::Output(_)));
        match seq.caches[0].clone() {
            LayerCache::Output(items) => {
                assert_eq!(linear_out.len(), items.len());
                for i in 0..linear_out.len() {
                    assert!((linear_out[i] - items[i]).abs() < 1e-5);
                }
            },
            _ => panic!("Should have faile earleir"),
        }

        let expected_output = vec![0.0, 7.8];

        assert_eq!(output.len(), expected_output.len());
        for (o, e) in output.iter().zip(expected_output.iter()) {
            assert!((o - e).abs() < 1e-6, "Expected {}, got {}", e, o);
        }
    }

    #[test]
    fn test_stateful_sequential_forward_nested_complex() {
        let inner_relu = StatelessReLU::new();
        let inner_linear = StatelessLinear {
            weights: vec![vec![0.2, -0.3], vec![0.5, 0.5]],
            biases: vec![0.0, 0.1],
        };

        let mut inner_seq = Sequential::new();
        inner_seq.add(StatelessLayerEnum::ReLU(inner_relu));
        inner_seq.add(StatelessLayerEnum::Linear(inner_linear));

        let inner_layer_enum = StatelessLayerEnum::Sequential(inner_seq.stateless.clone());

        let outer_linear = StatelessLinear {
            weights: vec![vec![0.5, -0.5], vec![1.0, 2.0]],
            biases: vec![0.1, -0.2],
        };
        let outer_relu = StatelessReLU::new();

        let mut outer_seq = Sequential::new();
        outer_seq.add(StatelessLayerEnum::Linear(outer_linear));
        outer_seq.add(inner_layer_enum);
        outer_seq.add(StatelessLayerEnum::ReLU(outer_relu));

        let input = vec![2.0, 3.0];

        let output = outer_seq.forward(&input);

        println!("{:#?}", outer_seq.caches);
        assert_eq!(outer_seq.caches.len(), 3);
        assert!(matches!(outer_seq.caches[0], LayerCache::Output(_)));
        assert!(matches!(outer_seq.caches[1], LayerCache::Sequential(_)));
        assert!(matches!(outer_seq.caches[2], LayerCache::Output(_)));
        if let LayerCache::Output(items) = &outer_seq.caches[0] {
            let expected_linear = vec![-0.4, 7.8];
            assert_eq!(items.len(), expected_linear.len());
            for i in 0..expected_linear.len() {
                assert!((expected_linear[i] - items[i]).abs() < 1e-5);
            }
        }

        if let LayerCache::Sequential(inner_caches) = &outer_seq.caches[1] {
            assert_eq!(inner_caches.len(), 1);

            if let LayerCache::Output(items) = &inner_caches[0] {
                let expected_relu = vec![0.0, 7.8];
                assert_eq!(items.len(), expected_relu.len());
                for i in 0..expected_relu.len() {
                    assert!((expected_relu[i] - items[i]).abs() < 1e-5);
                }
            } else {
                panic!("Should have hit")
            }
        } else {
            panic!("Should have hit")
        }

        if let LayerCache::Output(items) = &outer_seq.caches[2] {
            let prev = vec![0.0, 7.8];
            let n0 = prev[0]*0.2 + prev[1]*(-0.3) + 0.0;
            let n1 = prev[0]*0.5 + prev[1]*0.5 +0.1;
            let expected_linear = vec![n0, n1];
            assert_eq!(items.len(), expected_linear.len());
            for i in 0..expected_linear.len() {
                assert!((expected_linear[i] - items[i]).abs() < 1e-5);
            }
        } else {
            panic!("Should have hit")
        }

        let inner_output = {
            let prev = vec![0.0, 7.8];
            vec![prev[0]*0.2 + prev[1]*(-0.3) +0.0, prev[0]*0.5 + prev[1]*0.5 +0.1]
        };
        let expected_output = inner_output.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect::<Vec<_>>();

        assert_eq!(output.len(), expected_output.len());
        for (o, e) in output.iter().zip(expected_output.iter()) {
            assert!((o - e).abs() < 1e-6, "Expected {}, got {}", e, o);
        }
    }


    #[test]
    fn test_sequential_learns_nested_relu() {
        let mut rng = rand::thread_rng();

        let n_samples = 1000;
        let mut xs = Vec::with_capacity(n_samples);
        let mut ys = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            let mut x: f32 = rng.gen_range(-10.0..10.0);
            if -1.0 < x  || x <= 0.0 {
                x = -1.0
            } else if 0.0 < x || x < 1.0 {
                x = 1.0
            }

            let y = {
                let inner = (-1.5 * x).max(0.0);
                (1.5 * inner).max(0.0)
            };

            xs.push(vec![x]);
            ys.push(vec![y]);
        }

        let mut model = Sequential::new();
        model.add(StatelessLayerEnum::Linear(StatelessLinear::new(1, 8, WeightInit::He)));
        model.add(StatelessLayerEnum::ReLU(StatelessReLU::new()));
        model.add(StatelessLayerEnum::Linear(StatelessLinear::new(8, 1, WeightInit::He)));

        let epochs = 10;
        let lr = 0.0001;
        let acc_thresh = 0.03;

        let mut total_loss = 0.0;
        let mut correct = 0;
        for (x, y_true) in xs.iter().zip(ys.iter()) {
            let pred = model.forward(&x);
            let (loss, _grad) = mse_loss(&pred, &[y_true[0]]);
            let true_val = y_true[0];

            if (pred[0] - true_val).abs() / true_val.abs() <= acc_thresh {
                correct += 1;
            }
            total_loss += loss;
        }

        println!("Before Training: Loss {:.4}. Accuracy (within {:}) {:.4}.", total_loss / n_samples as f32, acc_thresh, correct as f32 / n_samples as f32);
        let mut final_accuracy = 0.0;
        for epoch in 0..epochs {
            let mut correct = 0;
            let mut total_loss = 0.0;

            for (x, y_true) in xs.iter().zip(ys.iter()) {
                let pred = model.forward(&x);
                let (loss, grad) = mse_loss(&pred, &[y_true[0]]);
                let true_val = y_true[0];

                if (pred[0] - true_val).abs() / true_val.abs() <= acc_thresh {
                    correct += 1;
                }
                total_loss += loss;

                model.backward(&grad, lr);
            }
            final_accuracy = correct as f32 / n_samples as f32;
            println!("Epoch {}: Loss {:.4}. Accuracy (within {:}) {:.4}.", epoch + 1, total_loss / n_samples as f32, acc_thresh, final_accuracy);
        }
        println!("Final accuracy (within {} of true value): {:.4}", acc_thresh, final_accuracy);

        assert!(final_accuracy >= 0.95, "Accuracy (within {} of true value): {:.4}", acc_thresh, final_accuracy);
    }
}
