use serde::{Deserialize, Serialize};
use crate::deep::{layer::{Layer, StatefulLayerWrapper, StatelessLayer, clip}, linear::StatelessLinear, relu::StatelessReLU};

/// ParamGrad type for sequential layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParamGrad {
    ReLU(Vec<f32>),
    Linear(Vec<f32>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SequentialLayer {
    ReLU(StatelessReLU),
    Linear(StatelessLinear),
}

impl StatelessLayer for SequentialLayer {
    type ParamGrad = ParamGrad;

    fn forward(&self, input: &[f32]) -> (Vec<f32>, Vec<Vec<f32>>) {
        match self {
            SequentialLayer::ReLU(layer) => layer.forward(input),
            SequentialLayer::Linear(layer) => layer.forward(input),
        }
    }

    fn backward_no_update(
        &self,
        input: &[f32],
        grad_output: &[f32],
        caches: &Vec<Vec<f32>>,
    ) -> (Vec<f32>, Self::ParamGrad) {
        match self {
            SequentialLayer::ReLU(layer) => {
                let (grad_input, grad_params) = layer.backward_no_update(input, grad_output, caches);
                (grad_input, ParamGrad::ReLU(grad_params))
            }
            SequentialLayer::Linear(layer) => {
                let (grad_input, grad_params) = layer.backward_no_update(input, grad_output, caches);
                (grad_input, ParamGrad::Linear(grad_params))
            }
        }
    }

    fn apply_update(&mut self, grad: &Self::ParamGrad, learning_rate: f32) {
        match (self, grad) {
            (SequentialLayer::ReLU(layer), ParamGrad::ReLU(g)) => layer.apply_update(g, learning_rate),
            (SequentialLayer::Linear(layer), ParamGrad::Linear(g)) => layer.apply_update(g, learning_rate),
            pair => panic!("Mismatched ParamGrad for SequentialLayer: {:#?}", pair),
        }
    }
    
    fn accumulate_param_grads(grad1: Self::ParamGrad, grad2: Self::ParamGrad) -> Self::ParamGrad {
        match (grad1, grad2) {
            (ParamGrad::ReLU(g1), ParamGrad::ReLU(g2)) => {
                ParamGrad::ReLU(StatelessReLU::accumulate_param_grads(g1, g2))
            }
            (ParamGrad::Linear(g1), ParamGrad::Linear(g2)) => {
                ParamGrad::Linear(StatelessLinear::accumulate_param_grads(g1, g2))
            }
            _ => panic!("Mismatched ParamGrad variants in accumulation"),
        }
    }
}

/// Stateless sequential container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatelessSequential {
    pub layers: Vec<SequentialLayer>,
    pub clip_grads: bool,
    pub max_clip: f32,
    pub min_clip: f32,
}

impl StatelessSequential {
    pub fn new() -> Self {
        Self { layers: Vec::new(), clip_grads: true, max_clip: 1e3, min_clip: -1e3 }
    }

    pub fn add(&mut self, layer: SequentialLayer) {
        self.layers.push(layer);
    }

    pub fn add_other_seq(&mut self, seq: StatelessSequential) {
        for layer in seq.layers {
            self.add(layer);
        }
    }
}

impl StatelessLayer for StatelessSequential {
    type ParamGrad = Vec<ParamGrad>;

    fn forward(&self, input: &[f32]) -> (Vec<f32>, Vec<Vec<f32>>) {
        let mut output = input.to_vec();
        let mut caches = Vec::new();

        // input is never added to cached
        // adds the output (so input next round is already added)
        for layer in &self.layers {
            caches.push(output.clone());
            let (out, sub_caches) = layer.forward(&output);
            if sub_caches.len() > 0 {
                panic!("Model with intermiedate caching not yet supportee.")
            }
            output = out;
        }
        // last element will be the last output
        (output, caches)
    }

    fn backward_no_update(
        &self,
        _input: &[f32],
        grad_output: &[f32],
        caches: &Vec<Vec<f32>>,
    ) -> (Vec<f32>, Self::ParamGrad) {
        let mut grad = grad_output.to_vec();
        let mut layer_grads = Vec::new();

        for (layer, cache) in self.layers.iter().rev().zip(caches.iter().rev()) {
            if self.clip_grads {
                clip(&mut grad, self.max_clip, self.min_clip);
            }
            let (grad_input, grad_params) = layer.backward_no_update(cache, &grad, &vec![]);
            grad = grad_input;
            layer_grads.push(grad_params);
        }
        if self.clip_grads {
            clip(&mut grad, self.max_clip, self.min_clip);
        }
        layer_grads.reverse();
        (grad, layer_grads)
    }


    fn apply_update(&mut self, grads: &Self::ParamGrad, learning_rate: f32) {
        for (layer, grad) in self.layers.iter_mut().zip(grads.iter()) {
            layer.apply_update(grad, learning_rate);
        }
    }
    
    fn accumulate_param_grads(grad1: Self::ParamGrad, grad2: Self::ParamGrad) -> Self::ParamGrad {
        if grad1.len() != grad2.len() {
            panic!("Mismatched lengths of Sequential ParamGrad vectors");
        }

        grad1.into_iter()
            .zip(grad2.into_iter())
            .map(|(g1, g2)| SequentialLayer::accumulate_param_grads(g1, g2))
            .collect()
    }
}

/// Stateful sequential wrapper
#[derive(Debug, Clone)]
pub struct Sequential {
    pub layer: StatefulLayerWrapper<StatelessSequential>,
}

impl Sequential {
    pub fn new() -> Self {
        Self { layer: StatefulLayerWrapper::new(StatelessSequential::new()) }
    }

    pub fn add(&mut self, layer: SequentialLayer) {
        self.layer.inner.add(layer);
    }

    pub fn add_other_seq(&mut self, seq: Sequential) {
        for layer in seq.layer.inner.layers {
            self.add(layer);
        }
    }
}

impl Layer<StatelessSequential> for Sequential {
    fn get_stateful_wrapper_mut(&mut self) -> &mut StatefulLayerWrapper<StatelessSequential> {
        &mut self.layer
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::deep::{linear::WeightInit, mse::mse_loss};
    use rand::{Rng, thread_rng};

    #[test]
    fn test_seq_forward() {
        let linear = StatelessLinear {
            weights: vec![vec![0.5, -0.5], vec![1.0, 2.0]],
            biases: vec![0.1, -0.2],
        };
        let relu = StatelessReLU::new();

        let mut seq = Sequential::new();
        seq.add(SequentialLayer::Linear(linear));
        seq.add(SequentialLayer::ReLU(relu));

        let input = vec![2.0, 3.0];
        let output = seq.layer.forward(&input); // caches populated inside wrapper

        // The first layer is linear: output = W*x + b
        let expected_linear = vec![-0.4, 7.8];
        // In your cache scheme, the first layer output should be caches[0]
        assert_eq!(seq.layer.inner.layers.len(), 2); // ensure we added 2 layers

        if let SequentialLayer::Linear(_) = &seq.layer.inner.layers[0] {
            let cached_linear_output = &seq.layer.inner.layers[0].forward(&input).0;
            assert_eq!(cached_linear_output.len(), expected_linear.len());
            for (o, e) in cached_linear_output.iter().zip(expected_linear.iter()) {
                assert!((o - e).abs() < 1e-5, "Expected {}, got {}", e, o);
            }
        } else {
            panic!("First layer should be linear");
        }

        // Final output after ReLU
        let expected_output = vec![0.0, 7.8];
        assert_eq!(output.len(), expected_output.len());
        for (o, e) in output.iter().zip(expected_output.iter()) {
            assert!((o - e).abs() < 1e-6, "Expected {}, got {}", e, o);
        }
    }


    #[test]
    fn test_seq_forward_extreme() {
        let input_data = [-5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let output_data = input_data.clone();
        let mut seq = Sequential::new();
        for i in 0..100 {
            if i % 2 == 0 {
                seq.add(SequentialLayer::Linear(StatelessLinear::new(2, 2, WeightInit::He)));
            } else {
                seq.add(SequentialLayer::ReLU(StatelessReLU::new()));
            }
        }
        seq.add(SequentialLayer::Linear(StatelessLinear::new(2, 1, WeightInit::He)));
        let epochs = 10;
        let lr = 0.0001;
        let acc_thresh = 0.03;
        for epoch in 0..epochs {
            let mut correct = 0;
            let mut total_loss = 0.0;

            for (x, y_true) in input_data.iter().zip(output_data.iter()) {
                let y_true = &[y_true.clone()];
                let pred = seq.forward(&[x.clone(), x.clone()]);
                let (loss, grad) = mse_loss(&pred, y_true);
                total_loss += loss;

                // Count correct predictions
                if (pred[0] - y_true[0]).abs() / y_true[0].abs() <= acc_thresh {
                    correct += 1;
                }

                seq.backward(&grad, lr);
            }

            let accuracy = correct as f32 / input_data.len() as f32;
            println!(
                "Epoch {}: Loss {:.4}, Accuracy (within {:.2}) {:.4}",
                epoch + 1,
                total_loss / input_data.len() as f32,
                acc_thresh,
                accuracy
            );
        }
    }


    #[test]
    fn test_sequential_learns_nested_relu() {
        let mut rng = rand::thread_rng();

        // Generate dataset
        let n_samples = 1000;
        let mut xs = Vec::with_capacity(n_samples);
        let mut ys = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            let x: f32 = rng.gen_range(-10.0..10.0);
            let mut inter = -1.5 * x + 2.0; // linear transform
            inter = inter.max(0.0); // relu
            let y = 1.5 * inter + 2.0; // linear transform

            xs.push(vec![x]);
            ys.push(vec![y]);
        }

        let mut model = Sequential::new();
        model.add(SequentialLayer::Linear(StatelessLinear::new(1, 20, WeightInit::He)));
        model.add(SequentialLayer::ReLU(StatelessReLU::new()));
        model.add(SequentialLayer::Linear(StatelessLinear::new(20, 20, WeightInit::He)));
        model.add(SequentialLayer::ReLU(StatelessReLU::new()));
        model.add(SequentialLayer::Linear(StatelessLinear::new(20, 1, WeightInit::He)));

        let epochs = 100;
        let lr = 0.001;
        let acc_thresh = 0.05;

        // Training loop
        for epoch in 0..epochs {
            let mut correct = 0;
            let mut total_loss = 0.0;

            for (x, y_true) in xs.iter().zip(ys.iter()) {
                let pred = model.forward(x);
                let (loss, grad) = mse_loss(&pred, y_true);
                total_loss += loss;

                // Count correct predictions
                if (pred[0] - y_true[0]).abs() / y_true[0].abs() <= acc_thresh {
                    correct += 1;
                }

                model.backward(&grad, lr);
            }

            let accuracy = correct as f32 / n_samples as f32;
            println!(
                "Epoch {}: Loss {:.4}, Accuracy (within {:.2}) {:.4}",
                epoch + 1,
                total_loss / n_samples as f32,
                acc_thresh,
                accuracy
            );
            if accuracy > 0.98 {
                break
            }
        }

        // Final check
        let mut correct = 0;
        for (x, y_true) in xs.iter().zip(ys.iter()) {
            let pred = model.forward(x);
            if (pred[0] - y_true[0]).abs() / y_true[0].abs() <= acc_thresh {
                correct += 1;
            }
        }

        let final_accuracy = correct as f32 / n_samples as f32;
        assert!(
            final_accuracy >= 0.95,
            "Expected accuracy >= 0.95, got {:.4}",
            final_accuracy
        );
    }

    #[test]
    fn compare_sgd_vs_minibatch_training() {
        let mut rng = thread_rng();
        // -----------------------------
        // Generate a HARD dataset
        // -----------------------------
        let n_samples = 5_000;
        let mut xs = Vec::with_capacity(n_samples);
        let mut ys = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            let x: f32 = rng.gen_range(-10.0..10.0);

            let mut inter1 = -1.8 * x + 3.5;
            inter1 = inter1.max(0.0);

            let mut inter2 = 0.75 * inter1 - 5.0;
            inter2 = inter2.max(0.0);

            let y = 2.25 * inter2 + 4.0;

            xs.push(vec![x]);
            ys.push(vec![y]);
        }

        let acc_thresh = 0.10;
        let epochs = 10;
        let sgd_lr = 0.0001;
        let mb_lr = 0.001;
        let batch_size = 16;

        // -----------------------------
        // Helper: create new model
        // -----------------------------
        fn new_model() -> Sequential {
            let mut model = Sequential::new();
            model.add(SequentialLayer::Linear(StatelessLinear::new(1, 52, WeightInit::He)));
            model.add(SequentialLayer::ReLU(StatelessReLU::new()));
            model.add(SequentialLayer::Linear(StatelessLinear::new(52, 52, WeightInit::He)));
            model.add(SequentialLayer::ReLU(StatelessReLU::new()));
            model.add(SequentialLayer::Linear(StatelessLinear::new(52, 1, WeightInit::He)));
            model
        }

        // -----------------------------
        // SGD TRAINING
        // -----------------------------
        let mut model_sgd = new_model();
        let start_sgd = Instant::now();
        let mut converged_epoch_sgd = epochs;
        let mut final_acc_sgd = 0.0;

        for epoch in 0..epochs {
            let mut correct = 0usize;
            let mut total_loss = 0.0;

            for (x, y_true) in xs.iter().zip(ys.iter()) {
                let pred = model_sgd.forward(x);
                let (loss, grad) = mse_loss(&pred, y_true);
                total_loss += loss;
                model_sgd.backward(&grad, sgd_lr);

                if (pred[0] - y_true[0]).abs() / y_true[0].abs() < acc_thresh {
                    correct += 1;
                }
            }

            let accuracy = correct as f32 / n_samples as f32;
            println!("[SGD] Epoch {epoch}: acc={accuracy:.4}. Avg loss (per samlpe): {:.4}", total_loss / n_samples as f32);
            final_acc_sgd = accuracy;

            if accuracy >= 0.95 {
                converged_epoch_sgd = epoch + 1;
                break;
            }
        }

        let time_sgd = start_sgd.elapsed();

        // -----------------------------
        // MINI-BATCH TRAINING
        // -----------------------------
        let mut model_mb = new_model();
        let start_mb = Instant::now();
        let mut converged_epoch_mb = epochs;
        let mut final_acc_mb = 0.0;

        for epoch in 0..epochs {
            let mut correct = 0usize;
            let mut i = 0;
            let mut total_loss = 0.0;
            let mut num_batches = 0;

            while i < n_samples {
                num_batches += 1;
                let end = (i + batch_size).min(n_samples);

                // Build pointers for forward pass
                let batch_inputs: Vec<&[f32]> = xs[i..end]
                    .iter()
                    .map(|x| x.as_slice())
                    .collect();

                // Forward on batch
                let preds = model_mb.forward_batch(&batch_inputs);

                // Build batch gradients for backward
                let mut batch_grads: Vec<Vec<f32>> = Vec::with_capacity(preds.len());
                for (pred, y_true) in preds.iter().zip(&ys[i..end]) {
                    let (loss, grad) = mse_loss(pred, y_true);
                    total_loss += loss;
                    batch_grads.push(grad);
                }

                let batch_grads_refs: Vec<&[f32]> =
                    batch_grads.iter().map(|g| g.as_slice()).collect();

                // Single update for the whole batch
                model_mb.backward_batch(&batch_grads_refs, mb_lr);

                i = end;
            }

            // Accuracy check
            for (x, y_true) in xs.iter().zip(ys.iter()) {
                let pred = model_mb.forward(x);
                if (pred[0] - y_true[0]).abs() / y_true[0].abs() < acc_thresh {
                    correct += 1;
                }
            }

            let accuracy = correct as f32 / n_samples as f32;
            println!("[MB] Epoch {epoch}: acc={accuracy:.4}. Avg loss per batch: {:.4}", total_loss / num_batches as f32);
            final_acc_mb = accuracy;

            if accuracy >= 0.95 {
                converged_epoch_mb = epoch + 1;
                break;
            }
        }


        let time_mb = start_mb.elapsed();

        // -----------------------------
        // Print summary
        // -----------------------------
        println!();
        println!("===== RESULTS =====");
        println!("SGD time: {:?}, epochs to {:.2}%: {}", time_sgd, final_acc_sgd, converged_epoch_sgd);
        println!("Mini-batch time: {:?}, epochs to {:.2}%: {}", time_mb, final_acc_mb, converged_epoch_mb);
    }

    #[test]
    fn test_input() {
        let input_size = 12836;
        let output_size = 4;
        let mut rng = thread_rng();
        let mut seq = Sequential::new();
        seq.add(SequentialLayer::Linear(StatelessLinear::new(input_size, 30, WeightInit::He)));
        seq.add(SequentialLayer::ReLU(StatelessReLU::new()));
        seq.add(SequentialLayer::Linear(StatelessLinear::new(30, 30, WeightInit::He)));
        seq.add(SequentialLayer::ReLU(StatelessReLU::new()));
        seq.add(SequentialLayer::Linear(StatelessLinear::new(30, output_size, WeightInit::He)));

        let mut data = Vec::new();
        for _ in 0..10 {
            let mut x = Vec::new();
            for _ in 0..input_size {
                x.push(rng.gen_range(-1.0..1.0));
            }
            data.push(x);
        }
        let mut y = Vec::new();
        for _ in 0..10 {
            let mut x = Vec::new();
            for _ in 0..output_size {
                x.push(rng.gen_range(-1.0..1.0));
            }
            y.push(x);
        }
        let out = seq.forward_batch(&data.iter().map(|d| d.as_slice()).collect::<Vec<&[f32]>>());
        let grads: Vec<Vec<f32>> = out.iter()
            .zip(y.iter())
            .map(|(p, t)| mse_loss(p, t).1)
            .collect();
        let grad_slices: Vec<&[f32]> = grads.iter()
            .map(|v| v.as_slice())
            .collect();
        seq.backward_batch(&grad_slices, 0.1);
    }
}
