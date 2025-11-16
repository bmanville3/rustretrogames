pub trait Layer {
    /// Core forward implementation. Layers implement this.
    fn _forward(&mut self, input: &[f32]) -> Vec<f32>;
    /// Core backward implementation. Layers implement this.
    fn _backward(&mut self, grad_output: &[f32], learning_rate: f32) -> Vec<f32>;

    /// Public forward method.
    fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        // just a redirection right now but may be useful later
        self._forward(input)
    }

    /// Public backward method. Performs gradient clippings
    fn backward(&mut self, grad_output: &[f32], learning_rate: f32) -> Vec<f32> {
        let mut grad_input = self._backward(grad_output, learning_rate);

        for g in grad_input.iter_mut() {
            if g.is_nan() {
                *g = 0.0;
            } else if *g > 1e3 {
                *g = 1e3;
            } else if *g < -1e3 {
                *g = -1e3;
            }
        }

        grad_input
    }
}
