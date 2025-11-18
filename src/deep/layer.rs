use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub trait StatelessLayer: Sync + Serialize {
    type ParamGrad: Send;
    /// Core forward implementation.
    ///
    /// Returns:
    /// - Output vector.
    /// - Intermediate cached inputs (empty for simple layers like Linear/ReLU).
    fn forward(&self, input: &[f32]) -> (Vec<f32>, Vec<Vec<f32>>);

    /// Forward pass over a batch (multithreaded via Rayon)
    fn forward_batch(&self, batch: &[&[f32]]) -> Vec<(Vec<f32>, Vec<Vec<f32>>)> {
        batch.par_iter().map(|x| self.forward(x)).collect()
    }

    /// Performs a single-layer backpropagation computation without updating the layer’s parameters.
    ///
    /// # Purpose
    /// This function computes the gradients required for backpropagation:
    /// 1. The gradient of the loss with respect to the input (`grad_input`), which is propagated to the previous layer.
    /// 2. The gradients of the loss with respect to the layer’s parameters (`grad_params`), returned as the associated type `ParamGrad`.
    ///
    /// This separation of gradient computation and parameter update allows:
    /// - Mini-batch gradient accumulation
    /// - Parallel computation of per-sample gradients
    /// - Custom learning rate schedules or optimizers
    ///
    /// # Parameters
    /// - `input: &[f32]`
    ///     - The input vector that was fed into this layer during the forward pass.
    ///     - Used to compute parameter gradients. For example:
    ///       - Linear layer: gradient w.r.t weights = `grad_output * input`
    ///       - Biases: gradient w.r.t biases = `grad_output`
    ///
    /// - `grad_output: &[f32]`
    ///     - The gradient of the loss with respect to the layer’s output (dL/dy).
    ///     - Supplied by the next layer during backpropagation.
    ///     - This implements the chain rule, allowing the layer to propagate gradients backward.
    ///
    /// - `caches: &Vec<Vec<f32>>`
    ///     - Any intermediate values saved during the forward pass that are necessary for computing gradients.
    ///     - Examples:
    ///         - Sequential: outputs of sublayers for gradient computation.
    ///
    /// # Returns
    /// `(grad_input, grad_params)`
    ///
    /// - `grad_input: Vec<f32>`
    ///     - The gradient of the loss with respect to this layer’s input (dL/dx).
    ///     - This is what will be passed to the previous layer in the network.
    ///     - Calculated using the chain rule. Examples:
    ///         - Linear: `grad_input = W^T * grad_output`
    ///         - ReLU: `grad_input[i] = grad_output[i] * (input[i] > 0 ? 1 : 0)`
    ///
    /// - ``grad_params: ParamGrad`  
    ///   Gradients of the loss with respect to the layer’s trainable parameters. Can be nested for sequential layers.  
    ///   Examples:
    ///   - Linear: a vector representing `dL/dW` and `dL/db`  
    ///   - ReLU: empty, as it has no parameters  
    ///   - Sequential: one `ParamGrad` per sublayer, potentially deeply nested
    ///
    /// # Notes
    /// - This function does not modify the layer’s parameters; it only computes the gradients.
    /// - It is safe to call this function in parallel for multiple inputs, which enables batch gradient computation.
    /// - Use `apply_update` or `backward` (which wraps `backward_no_update` + `apply_update`) to perform parameter updates.
    ///
    /// # Example
    /// ```rust
    /// // Linear layer: y = Wx + b
    /// let (grad_input, grad_params) = linear_layer.backward_no_update(&x, &grad_output, &caches);
    /// // grad_input: dL/dx
    /// // grad
    fn backward_no_update(
        &self,
        input: &[f32],
        grad_output: &[f32],
        caches: &Vec<Vec<f32>>,
    ) -> (Vec<f32>, Self::ParamGrad);

    /// Apply a computed parameter update
    fn apply_update(&mut self, grad: &Self::ParamGrad, learning_rate: f32);

    /// Standard backward pass (compute gradient + update weights)
    fn backward(&mut self, input: &[f32], grad_output: &[f32], learning_rate: f32, caches: &Vec<Vec<f32>>) -> Vec<f32> {
        let (grad_input, pgrad) = self.backward_no_update(input, grad_output, caches);
        self.apply_update(&pgrad, learning_rate);
        grad_input
    }

    /// Backward pass over a batch (compute all sample gradients in parallel, then accumulate)
    /// Learning rate is normalized by the input size.
    fn backward_batch(
        &mut self,
        inputs: &[&[f32]],
        grad_outputs: &[&[f32]],
        caches: &[Vec<Vec<f32>>],
        learning_rate: f32,
    ) -> Vec<Vec<f32>> {
        // Compute per-sample gradients in parallel
        let per_sample: Vec<(Vec<f32>, Self::ParamGrad)> = inputs
            .par_iter()
            .zip(grad_outputs.par_iter())
            .zip(caches.par_iter())
            .map(|((input, g_out), c)| self.backward_no_update(input, g_out, c))
            .collect();

        let mut accumulated_grad: Option<Self::ParamGrad> = None;
        let mut grad_inputs = Vec::with_capacity(per_sample.len());

        for (grad_input, pgrad) in per_sample {
            grad_inputs.push(grad_input);

            accumulated_grad = Some(match accumulated_grad {
                None => pgrad,
                Some(total) => Self::accumulate_param_grads(total, pgrad),
            });
        }

        if let Some(total) = accumulated_grad {
            self.apply_update(&total, learning_rate / inputs.len() as f32);
        }

        grad_inputs
    }

    fn accumulate_param_grads(grad1: Self::ParamGrad, grad2: Self::ParamGrad) -> Self::ParamGrad;
}

/// A wrapper for a [`StatelessLayer`] that automatically stores inputs and caches.
/// Supports single-sample and batch forward/backward.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatefulLayerWrapper<L: StatelessLayer> {
    // traits are given public access so wrappers can access them if need be

    pub inner: L,
    pub cached_input: Vec<f32>,
    pub cached_caches: Vec<Vec<f32>>,

    pub cached_batch_inputs: Vec<Vec<f32>>,
    pub cached_batch_caches: Vec<Vec<Vec<f32>>>,
}

impl<L: StatelessLayer> StatefulLayerWrapper<L> {
    pub fn new(layer: L) -> Self {
        Self {
            inner: layer,
            cached_input: Vec::new(),
            cached_caches: Vec::new(),
            cached_batch_inputs: Vec::new(),
            cached_batch_caches: Vec::new(),
        }
    }

    pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        let (output, caches) = self.inner.forward(input);
        self.cached_input.clear();
        self.cached_input.extend_from_slice(input);
        self.cached_caches = caches;
        output
    }

    pub fn forward_batch(&mut self, batch_inputs: &[&[f32]]) -> Vec<Vec<f32>> {
        let outputs = self.inner.forward_batch(batch_inputs);
        self.cached_batch_inputs.clear();
        self.cached_batch_caches.clear();

        let mut return_outputs = Vec::with_capacity(outputs.len());
        for (input, (output, caches)) in batch_inputs.iter().zip(outputs) {
            return_outputs.push(output);
            self.cached_batch_inputs.push((*input).to_vec());
            self.cached_batch_caches.push(caches.clone());
        }

        return_outputs
    }

    pub fn backward(&mut self, grad_output: &[f32], learning_rate: f32) -> Vec<f32> {
        let (grad_input, param_grad) =
            self.inner.backward_no_update(&self.cached_input, grad_output, &self.cached_caches);
        self.inner.apply_update(&param_grad, learning_rate);
        grad_input
    }

    pub fn backward_batch(&mut self, batch_grad_outputs: &[&[f32]], learning_rate: f32) -> Vec<Vec<f32>> {
        self.inner.backward_batch(
            &self.cached_batch_inputs.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
            batch_grad_outputs,
            &self.cached_batch_caches,
            learning_rate,
        )
    }
}

pub trait Layer<L: StatelessLayer> {
    fn get_stateful_wrapper_mut(&mut self) -> &mut StatefulLayerWrapper<L>;

    fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        self.get_stateful_wrapper_mut().forward(input)
    }

    fn forward_batch(&mut self, batch_inputs: &[&[f32]]) -> Vec<Vec<f32>> {
        self.get_stateful_wrapper_mut().forward_batch(batch_inputs)
    }

    fn backward(&mut self, grad_output: &[f32], learning_rate: f32) -> Vec<f32> {
        self.get_stateful_wrapper_mut().backward(grad_output, learning_rate)
    }

    fn backward_batch(&mut self, batch_grad_outputs: &[&[f32]], learning_rate: f32) -> Vec<Vec<f32>> {
        self.get_stateful_wrapper_mut().backward_batch(batch_grad_outputs, learning_rate)
    }
}

pub fn clip(grad_output: &mut [f32], max: f32, min: f32) {
    for g in grad_output {
        if g.is_nan() {
            *g = 0.0;
        } else if *g > max {
            *g = max;
        } else if *g < min {
            *g = min;
        }
    }
}

pub fn clip_batch(batch_grad_output: &mut [&mut [f32]], max: f32, min: f32) {
    for grad_output in batch_grad_output {
        clip(grad_output, max, min);
    }
}
