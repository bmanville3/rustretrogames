use crate::deep::{layer::Layer, linear::Linear, relu::ReLU};

#[derive(Clone, Debug)]
pub enum LayerEnum {
    Linear(Linear),
    ReLU(ReLU),
    Sequential(Sequential),
}

impl LayerEnum {
    fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        match self {
            LayerEnum::Linear(l) => l.forward(input),
            LayerEnum::ReLU(r) => r.forward(input),
            LayerEnum::Sequential(s) => s.forward(input),
        }
    }

    fn backward(&mut self, grad: &[f32], lr: f32) -> Vec<f32> {
        match self {
            LayerEnum::Linear(l) => l.backward(grad, lr),
            LayerEnum::ReLU(r) => r.backward(grad, lr),
            LayerEnum::Sequential(s) => s.backward(grad, lr),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Sequential {
    layers: Vec<LayerEnum>,
}

impl Sequential {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add(&mut self, layer: LayerEnum) {
        self.layers.push(layer);
    }
}

impl Layer for Sequential {
    fn _forward(&mut self, input: &[f32]) -> Vec<f32> {
        let mut output: Vec<f32> = input.to_vec();
        for layer in self.layers.iter_mut() {
            output = layer.forward(&output);
        }
        output
    }

    fn _backward(&mut self, grad_output: &[f32], lr: f32) -> Vec<f32> {
        let mut grad: Vec<f32> = grad_output.to_vec();
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad, lr);
        }
        grad
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    // Assuming these exist in your crate
    use crate::deep::{linear::{Linear, WeightInit}, mse::mse_loss, relu::ReLU};

    #[test]
    fn test_sequential_learns_nested_relu() {
        let mut rng = rand::thread_rng();

        // -----------------------
        // Generate dataset
        // -----------------------
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
        model.add(LayerEnum::Linear(Linear::new(1, 8, WeightInit::He)));
        model.add(LayerEnum::ReLU(ReLU::new()));
        model.add(LayerEnum::Linear(Linear::new(8, 1, WeightInit::He)));

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

