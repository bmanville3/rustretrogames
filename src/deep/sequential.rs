use crate::deep::layer::Layer;

pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add<L: Layer + 'static>(&mut self, layer: L) {
        self.layers.push(Box::new(layer));
    }

    pub fn forward(&mut self, mut input: Vec<f32>) -> Vec<f32> {
        for layer in self.layers.iter_mut() {
            input = layer.forward(&input);
        }
        input
    }

    pub fn backward(&mut self, grad_output: Vec<f32>, lr: f32) {
        let mut grad = grad_output;
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad, lr);
        }
    }
}
