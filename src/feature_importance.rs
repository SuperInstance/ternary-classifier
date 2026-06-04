use crate::{BehaviorProfile, Species, SpeciesClassifier};

/// Ranks which behavioral features matter most for classification.
///
/// Uses a permutation-based approach: for each feature, randomly
/// shuffle that feature across all profiles and measure accuracy drop.
/// Larger drop = more important feature.
#[derive(Debug, Clone)]
pub struct FeatureImportance {
    /// Feature names in order
    pub feature_names: Vec<&'static str>,
    /// Importance scores (0..1, relative)
    pub scores: Vec<f64>,
}

impl FeatureImportance {
    /// The canonical feature names in order.
    pub fn feature_names() -> Vec<&'static str> {
        vec!["explore_rate", "cooperate_rate", "defect_rate", "win_rate", "entropy"]
    }

    /// Compute feature importance using permutation importance.
    ///
    /// - `profiles`: the test profiles
    /// - `labels`: their true species labels
    /// - `classifier`: the classifier to evaluate
    /// - `shuffle_seed`: base seed for reproducibility (each feature uses seed + feature_index)
    /// - `num_shuffles`: how many times to shuffle each feature (more = stable)
    pub fn compute(
        profiles: &[BehaviorProfile],
        labels: &[Species],
        classifier: &SpeciesClassifier,
        shuffle_seed: u64,
        num_shuffles: usize,
    ) -> Self {
        let feature_names = Self::feature_names();
        let dim = feature_names.len();
        assert_eq!(profiles.len(), labels.len());

        // Baseline accuracy
        let baseline_acc = compute_accuracy(profiles, labels, classifier);

        let mut importance = vec![0.0f64; dim];

        for feat_idx in 0..dim {
            let mut total_acc_drop = 0.0;

            for shuffle_i in 0..num_shuffles {
                // Seeded pseudo-random shuffle for this feature
                let seed = shuffle_seed.wrapping_add(feat_idx as u64).wrapping_add(shuffle_i as u64);
                let mut shuffled_profiles: Vec<BehaviorProfile> = profiles.to_vec();
                shuffle_feature(&mut shuffled_profiles, feat_idx, seed);

                let shuffled_acc = compute_accuracy(&shuffled_profiles, labels, classifier);
                total_acc_drop += baseline_acc - shuffled_acc;
            }

            importance[feat_idx] = total_acc_drop / num_shuffles as f64;
        }

        // Normalize to 0..1
        let max_importance = importance.iter().cloned().fold(0.0f64, f64::max);
        if max_importance > 0.0 {
            for score in &mut importance {
                *score /= max_importance;
            }
        }

        Self {
            feature_names,
            scores: importance,
        }
    }

    /// Get the most important feature.
    pub fn most_important(&self) -> (&str, f64) {
        let idx = self
            .scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        (self.feature_names[idx], self.scores[idx])
    }

    /// Get features ranked from most to least important.
    pub fn ranked(&self) -> Vec<(&str, f64)> {
        let mut ranked: Vec<_> = self
            .feature_names
            .iter()
            .zip(self.scores.iter())
            .map(|(n, &s)| (*n, s))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    /// Format as a readable report.
    pub fn format(&self) -> String {
        let ranked = self.ranked();
        let mut s = String::from("Feature Importance (ranked):\n");
        for (i, (name, score)) in ranked.iter().enumerate() {
            let bar_len = (score * 40.0) as usize;
            let bar: String = "█".repeat(bar_len);
            s.push_str(&format!("{:>2}. {:<20} {:.3} {}\n", i + 1, name, score, bar));
        }
        s
    }
}

fn compute_accuracy(
    profiles: &[BehaviorProfile],
    labels: &[Species],
    classifier: &SpeciesClassifier,
) -> f64 {
    let correct = profiles
        .iter()
        .zip(labels.iter())
        .filter(|(p, &l)| classifier.classify(p) == l)
        .count();
    correct as f64 / profiles.len() as f64
}

/// Simple seeded pseudo-random permutation of a single feature column.
fn shuffle_feature(profiles: &mut [BehaviorProfile], feat_idx: usize, seed: u64) {
    // Fisher-Yates shuffle with simple LCG PRNG
    let n = profiles.len();
    if n <= 1 {
        return;
    }

    // Extract the feature values
    let mut values: Vec<f64> = profiles.iter().map(|p| p.to_features()[feat_idx]).collect();

    // Shuffle using LCG
    let mut state = seed;
    for i in (1..n).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        values.swap(i, j);
    }

    // Write back
    for (i, profile) in profiles.iter_mut().enumerate() {
        let mut features = profile.to_features();
        features[feat_idx] = values[i];
        *profile = BehaviorProfile::from_features(&features);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<BehaviorProfile>, Vec<Species>) {
        vec![
            (BehaviorProfile::new(0.1, 0.7, 0.2, 0.5, 0.8), Species::Diplomat),
            (BehaviorProfile::new(0.6, 0.2, 0.2, 0.4, 1.3), Species::Explorer),
            (BehaviorProfile::new(0.2, 0.35, 0.45, 0.7, 0.8), Species::Marksman),
            (BehaviorProfile::new(0.1, 0.2, 0.7, 0.7, 0.7), Species::Climber),
            (BehaviorProfile::new(0.3, 0.3, 0.4, 0.4, 1.5), Species::Prospector),
        ]
            .into_iter()
            .unzip()
    }

    #[test]
    fn test_feature_importance_computes() {
        let (profiles, labels) = make_test_data();
        let classifier = SpeciesClassifier::new();
        let fi = FeatureImportance::compute(&profiles, &labels, &classifier, 42, 10);
        assert_eq!(fi.scores.len(), 5);
        // All scores should be >= 0
        assert!(fi.scores.iter().all(|&s| s >= 0.0));
    }

    #[test]
    fn test_most_important_returns_valid() {
        let (profiles, labels) = make_test_data();
        let classifier = SpeciesClassifier::new();
        let fi = FeatureImportance::compute(&profiles, &labels, &classifier, 42, 10);
        let (name, score) = fi.most_important();
        assert!(!name.is_empty());
        assert!(score >= 0.0);
    }

    #[test]
    fn test_ranked_ordering() {
        let (profiles, labels) = make_test_data();
        let classifier = SpeciesClassifier::new();
        let fi = FeatureImportance::compute(&profiles, &labels, &classifier, 42, 20);
        let ranked = fi.ranked();
        assert_eq!(ranked.len(), 5);
        // Should be in descending order
        for i in 1..ranked.len() {
            assert!(ranked[i].1 <= ranked[i - 1].1);
        }
    }

    #[test]
    fn test_format_output() {
        let (profiles, labels) = make_test_data();
        let classifier = SpeciesClassifier::new();
        let fi = FeatureImportance::compute(&profiles, &labels, &classifier, 42, 5);
        let output = fi.format();
        assert!(output.contains("explore_rate"));
        assert!(output.contains("Feature Importance"));
    }
}
