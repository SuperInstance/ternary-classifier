use crate::{SpeciesClassifier, BehaviorProfile, Species};

/// A linear decision boundary between two species in feature space.
///
/// Represented as a hyperplane: w·x + b = 0
#[derive(Debug, Clone)]
pub struct DecisionBoundary {
    /// The two species this boundary separates
    pub species_a: Species,
    pub species_b: Species,
    /// Normal vector (weights) of the hyperplane
    pub weights: Vec<f64>,
    /// Bias term
    pub bias: f64,
}

impl DecisionBoundary {
    /// Compute the signed distance from a profile to this boundary.
    ///
    /// Positive → species_a side, negative → species_b side.
    pub fn signed_distance(&self, profile: &BehaviorProfile) -> f64 {
        let features = profile.to_features();
        let dot: f64 = self
            .weights
            .iter()
            .zip(features.iter())
            .map(|(w, x)| w * x)
            .sum();
        dot + self.bias
    }

    /// Classify which side of the boundary a profile falls on.
    pub fn classify_side(&self, profile: &BehaviorProfile) -> Species {
        if self.signed_distance(profile) >= 0.0 {
            self.species_a
        } else {
            self.species_b
        }
    }
}

/// Compute decision boundaries between all species pairs.
///
/// Uses the species classifier's heuristic thresholds to derive
/// approximate linear boundaries in feature space.
pub fn compute_boundaries(
    classifier: &SpeciesClassifier,
    reference_profiles: &[(BehaviorProfile, Species)],
) -> Vec<DecisionBoundary> {
    let features_dim = 5;
    let species = Species::all();
    let mut boundaries = Vec::new();

    // For each pair of species, find representative profiles and
    // compute a separating hyperplane at the midpoint
    for i in 0..species.len() {
        for j in (i + 1)..species.len() {
            let sp_a = species[i];
            let sp_b = species[j];

            let profiles_a: Vec<&BehaviorProfile> = reference_profiles
                .iter()
                .filter(|(_, s)| *s == sp_a)
                .map(|(p, _)| p)
                .collect();
            let profiles_b: Vec<&BehaviorProfile> = reference_profiles
                .iter()
                .filter(|(_, s)| *s == sp_b)
                .map(|(p, _)| p)
                .collect();

            if profiles_a.is_empty() || profiles_b.is_empty() {
                continue;
            }

            // Compute centroids of each species
            let centroid_a = mean_features(&profiles_a, features_dim);
            let centroid_b = mean_features(&profiles_b, features_dim);

            // The boundary normal points from b→a
            let mut weights: Vec<f64> = centroid_a
                .iter()
                .zip(centroid_b.iter())
                .map(|(a, b)| a - b)
                .collect();

            // Normalize
            let norm: f64 = weights.iter().map(|w| w * w).sum::<f64>().sqrt();
            if norm > 1e-12 {
                for w in &mut weights {
                    *w /= norm;
                }
            }

            // Midpoint
            let midpoint: Vec<f64> = centroid_a
                .iter()
                .zip(centroid_b.iter())
                .map(|(a, b)| (a + b) / 2.0)
                .collect();

            // bias = -w · midpoint
            let bias: f64 = -weights
                .iter()
                .zip(midpoint.iter())
                .map(|(w, m)| w * m)
                .sum::<f64>();

            boundaries.push(DecisionBoundary {
                species_a: sp_a,
                species_b: sp_b,
                weights,
                bias,
            });
        }
    }

    boundaries
}

fn mean_features(profiles: &[&BehaviorProfile], dim: usize) -> Vec<f64> {
    let mut sum = vec![0.0; dim];
    for p in profiles {
        let f = p.to_features();
        for (i, &v) in f.iter().enumerate() {
            sum[i] += v;
        }
    }
    let n = profiles.len() as f64;
    sum.iter().map(|s| s / n).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_boundaries() {
        let classifier = SpeciesClassifier::new();
        let profiles = vec![
            (BehaviorProfile::new(0.1, 0.7, 0.2, 0.5, 0.8), Species::Diplomat),
            (BehaviorProfile::new(0.6, 0.2, 0.2, 0.4, 1.3), Species::Explorer),
            (BehaviorProfile::new(0.2, 0.3, 0.5, 0.7, 0.7), Species::Climber),
        ];
        let boundaries = compute_boundaries(&classifier, &profiles);
        assert!(!boundaries.is_empty());
        // Should have boundaries for Diplomat-Explorer, Diplomat-Climber, Explorer-Climber
        assert_eq!(boundaries.len(), 3);
    }

    #[test]
    fn test_signed_distance_classifies_correctly() {
        let classifier = SpeciesClassifier::new();
        let diplomat = BehaviorProfile::new(0.05, 0.85, 0.1, 0.5, 0.6);
        let explorer = BehaviorProfile::new(0.75, 0.1, 0.15, 0.4, 1.4);
        let profiles = vec![
            (diplomat.clone(), Species::Diplomat),
            (explorer.clone(), Species::Explorer),
        ];
        let boundaries = compute_boundaries(&classifier, &profiles);

        // Find the Diplomat-Explorer boundary
        let boundary = boundaries.iter().find(|b| {
            (b.species_a == Species::Diplomat && b.species_b == Species::Explorer)
                || (b.species_a == Species::Explorer && b.species_b == Species::Diplomat)
        }).unwrap();

        // Diplomat should be on the positive side relative to its species
        let dist_diplomat = boundary.signed_distance(&diplomat);
        let dist_explorer = boundary.signed_distance(&explorer);
        // They should be on opposite sides
        assert!(dist_diplomat * dist_explorer < 0.0);
    }
}
