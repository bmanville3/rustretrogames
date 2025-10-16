pub fn mse_loss(pred: &[f32], target: &[f32]) -> (f32, Vec<f32>) {
    let mut loss = 0.0;
    let mut grad = Vec::with_capacity(pred.len());
    let len = pred.len() as f32;
    for (p, t) in pred.iter().zip(target) {
        let diff = p - t;
        loss += diff * diff;
        grad.push(2.0 * diff / len);
    }
    (loss / len, grad)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss_and_gradient() {
        let pred = vec![0.5, 1.0, 1.5];
        let target = vec![1.0, 1.0, 0.0];
        let (loss, grad) = mse_loss(&pred, &target);

        // loss = mean((p - t)^2) = ((-0.5)^2 + 0^2 + 1.5^2)/3 = (0.25 + 0 + 2.25)/3 = 0.8333
        assert!((loss - 0.8333).abs() < 1e-3);

        // gradient = 2*(p - t)/N
        // grad = [-0.3333, 0.0, 1.0]
        let expected = vec![-0.3333, 0.0, 1.0];
        for (g, e) in grad.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-3);
        }
    }
}
