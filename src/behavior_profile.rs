/// Behavioral profile of a ternary agent.
///
/// Captures the distribution of actions, performance metrics, and
/// cooperation tendencies that define an agent's strategy.
#[derive(Debug, Clone)]
pub struct BehaviorProfile {
    /// Fraction of actions that are "explore" (0..1)
    pub explore_rate: f64,
    /// Fraction of actions that are "cooperate" (0..1)
    pub cooperate_rate: f64,
    /// Fraction of actions that are "defect" (0..1)
    pub defect_rate: f64,
    /// Win rate across all encounters (0..1)
    pub win_rate: f64,
    /// Shannon entropy of the action distribution (0..~1.585 for ternary)
    pub entropy: f64,
}

impl BehaviorProfile {
    /// Create a new behavior profile with validated fields.
    ///
    /// Values are clamped to [0, 1] where applicable.
    pub fn new(
        explore_rate: f64,
        cooperate_rate: f64,
        defect_rate: f64,
        win_rate: f64,
        entropy: f64,
    ) -> Self {
        let profile = Self {
            explore_rate: explore_rate.clamp(0.0, 1.0),
            cooperate_rate: cooperate_rate.clamp(0.0, 1.0),
            defect_rate: defect_rate.clamp(0.0, 1.0),
            win_rate: win_rate.clamp(0.0, 1.0),
            entropy: entropy.max(0.0),
        };
        profile
    }

    /// Convert to a 5-element feature vector for ML operations.
    pub fn to_features(&self) -> Vec<f64> {
        vec![
            self.explore_rate,
            self.cooperate_rate,
            self.defect_rate,
            self.win_rate,
            self.entropy,
        ]
    }

    /// Create a profile from a feature vector (5 elements).
    ///
    /// Panics if the vector doesn't have exactly 5 elements.
    pub fn from_features(features: &[f64]) -> Self {
        assert_eq!(features.len(), 5, "Expected 5 features, got {}", features.len());
        Self::new(
            features[0],
            features[1],
            features[2],
            features[3],
            features[4],
        )
    }

    /// Compute Euclidean distance to another profile in feature space.
    pub fn distance_to(&self, other: &BehaviorProfile) -> f64 {
        let a = self.to_features();
        let b = other.to_features();
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Check if action rates approximately sum to 1.0 within tolerance.
    pub fn is_valid_distribution(&self, tolerance: f64) -> bool {
        let sum = self.explore_rate + self.cooperate_rate + self.defect_rate;
        (sum - 1.0).abs() < tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_clamps_values() {
        let p = BehaviorProfile::new(-0.5, 1.5, 0.3, 0.5, -1.0);
        assert!((p.explore_rate - 0.0).abs() < 1e-9);
        assert!((p.cooperate_rate - 1.0).abs() < 1e-9);
        assert!((p.defect_rate - 0.3).abs() < 1e-9);
        assert!((p.entropy - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_roundtrip_features() {
        let p = BehaviorProfile::new(0.3, 0.4, 0.3, 0.6, 1.1);
        let features = p.to_features();
        let p2 = BehaviorProfile::from_features(&features);
        assert!((p.explore_rate - p2.explore_rate).abs() < 1e-9);
        assert!((p.cooperate_rate - p2.cooperate_rate).abs() < 1e-9);
        assert!((p.win_rate - p2.win_rate).abs() < 1e-9);
        assert!((p.entropy - p2.entropy).abs() < 1e-9);
    }

    #[test]
    fn test_distance_to_self_is_zero() {
        let p = BehaviorProfile::new(0.3, 0.4, 0.3, 0.5, 1.0);
        assert!(p.distance_to(&p) < 1e-9);
    }

    #[test]
    fn test_valid_distribution() {
        let p = BehaviorProfile::new(0.3, 0.4, 0.3, 0.5, 1.0);
        assert!(p.is_valid_distribution(0.01));
        let p2 = BehaviorProfile::new(0.5, 0.5, 0.5, 0.5, 1.0);
        assert!(!p2.is_valid_distribution(0.01));
    }
}
