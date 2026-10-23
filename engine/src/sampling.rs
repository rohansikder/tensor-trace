//! Deterministic sampling. Displayed probabilities always use the original logits;
//! temperature/top-p affect only selection, never rewrite model evidence.

pub(crate) fn softmax(values: &mut [f32]) -> Result<(), String> {
    if values.is_empty() || values.iter().any(|x| !x.is_finite()) {
        return Err("Cannot normalize empty or non-finite scores".into());
    }
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err("Invalid probability normalization".into());
    }
    for value in values {
        *value /= sum;
    }
    Ok(())
}

pub(crate) struct Sampler {
    state: u64,
}
impl Sampler {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9e3779b97f4a7c15 } else { seed },
        }
    }
    fn uniform(&mut self) -> f32 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        let bits = (self.state.wrapping_mul(0x2545f4914f6cdd1d) >> 32) as u32;
        (bits >> 8) as f32 / 16_777_216.0
    }
    pub(crate) fn sample(
        &mut self,
        logits: &[f32],
        temperature: f32,
        top_p: f32,
    ) -> Result<usize, String> {
        if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
            return Err("Temperature must be finite and between 0 and 2".into());
        }
        if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
            return Err("Top-p must be in (0, 1]".into());
        }
        if logits.is_empty() || logits.iter().any(|x| !x.is_finite()) {
            return Err("Model produced invalid logits".into());
        }
        if temperature == 0.0 {
            let mut best = 0;
            for i in 1..logits.len() {
                if logits[i] > logits[best] {
                    best = i;
                }
            }
            return Ok(best);
        }
        // Subtract the maximum and divide in f64 so every positive f32
        // temperature, including subnormals, is valid without overflowing.
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let mut probabilities: Vec<f32> = logits
            .iter()
            .map(|&v| (((v as f64 - max) / temperature as f64).exp()) as f32)
            .collect();
        let sum = probabilities.iter().sum::<f32>();
        for value in &mut probabilities {
            *value /= sum;
        }
        let mut order: Vec<usize> = (0..probabilities.len()).collect();
        order.sort_unstable_by(|&a, &b| {
            probabilities[b]
                .total_cmp(&probabilities[a])
                .then(a.cmp(&b))
        });
        let mut cumulative = 0.0f32;
        let mut keep = order.len();
        for (i, &id) in order.iter().enumerate() {
            cumulative += probabilities[id];
            if cumulative >= top_p {
                keep = i + 1;
                break;
            }
        }
        let coin = self.uniform() * cumulative;
        let mut accumulated = 0.0;
        for &id in order.iter().take(keep) {
            accumulated += probabilities[id];
            if coin < accumulated {
                return Ok(id);
            }
        }
        Ok(order[keep - 1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stable_softmax_handles_large_logits() {
        let mut p = [10000.0, 10001.0, 9999.0];
        softmax(&mut p).unwrap();
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p[1] > p[0] && p[0] > p[2]);
    }
    #[test]
    fn greedy_is_stable_on_ties_and_ignores_seed() {
        for seed in [0, 1, 42, 99] {
            assert_eq!(
                Sampler::new(seed)
                    .sample(&[1.0, 3.0, 3.0], 0.0, 0.5)
                    .unwrap(),
                1
            );
        }
    }
    #[test]
    fn seeded_nucleus_sampling_is_repeatable_and_excludes_tail() {
        let mut a = Sampler::new(42);
        let mut b = Sampler::new(42);
        for _ in 0..100 {
            let x = a.sample(&[2.0, 1.9, -10.0], 1.0, 0.9).unwrap();
            assert_eq!(x, b.sample(&[2.0, 1.9, -10.0], 1.0, 0.9).unwrap());
            assert!(x < 2);
        }
    }
    #[test]
    fn sampler_rejects_nonfinite_parameters() {
        assert!(Sampler::new(1).sample(&[1.0], f32::NAN, 0.9).is_err());
        assert!(Sampler::new(1).sample(&[1.0], 1.0, 0.0).is_err());
        assert!(Sampler::new(1).sample(&[f32::INFINITY], 0.0, 1.0).is_err());
    }
}

#[cfg(test)]
mod extreme_tests {
    use super::*;
    #[test]
    fn subnormal_temperature_has_a_stable_limiting_distribution() {
        for _ in 0..10 {
            assert_eq!(
                Sampler::new(42)
                    .sample(&[-1.0, 2.0, 1.0], f32::from_bits(1), 0.9)
                    .unwrap(),
                1
            );
        }
    }
}
