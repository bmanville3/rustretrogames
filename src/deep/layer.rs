pub trait StatelessLayer<T> {
    /// Core forward implementation. Layers implement this.
    /// 
    /// # Returns
    /// - The output in the first position.
    /// - All intermediate cached inputs EXCEPT the first one.
    ///     This will be empty for all layers that do not have
    ///     intermediate layers. For example, it will be empty
    ///     with Linear but have a value for Sequential.
    fn _forward(&mut self, input: &[f32]) -> (Vec<f32>, Vec<T>);

    /// Public forward method.
    fn forward(&mut self, input: &[f32]) -> (Vec<f32>, Vec<T>) {
        // just a redirection right now but may be useful later
        self._forward(input)
    }

    /// Core backward implementation. Layers implement this.
    fn _backward(&mut self, input: &[f32], grad_output: &[f32], learning_rate: f32, intermediate_caches: &Vec<T>) -> Vec<f32>;

    /// Public backward method. Performs gradient clippings
    fn backward(&mut self, input: &[f32], grad_output: &[f32], learning_rate: f32, intermediate_caches: &Vec<T>) -> Vec<f32> {
        let mut grad_input = self._backward(input, grad_output, learning_rate, intermediate_caches);

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

pub trait StatefulLayer {
    /// Core forward implementation. Layers implement this.
    fn _forward(&mut self, input: &[f32]) -> Vec<f32>;

    /// Public forward method.
    fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        // just a redirection right now but may be useful later
        self._forward(input)
    }

    /// Core backward implementation. Layers implement this.
    fn _backward(&mut self, grad_output: &[f32], learning_rate: f32) -> Vec<f32>;

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
