use crate::{BehaviorProfile, Species};
use std::collections::HashMap;

/// A cluster centroid — the mean position of a discovered cluster.
#[derive(Debug, Clone)]
pub struct Centroid {
    pub features: Vec<f64>,
    pub label: Option<Species>,
    pub member_count: usize,
}

impl Centroid {
    pub fn new(features: Vec<f64>) -> Self {
        Self {
            features,
            label: None,
            member_count: 0,
        }
    }
}

/// Simple k-means clustering over behavior profiles.
///
/// Discovers species clusters automatically from unlabeled data.
/// No external dependencies — uses a straightforward Lloyd's algorithm.
#[derive(Debug, Clone)]
pub struct KMeansCluster {
    /// Number of clusters (k)
    pub k: usize,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance (sum of centroid shifts)
    pub tolerance: f64,
    /// Fitted centroids (None until fit is called)
    pub centroids: Option<Vec<Centroid>>,
}

impl KMeansCluster {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            max_iterations: 100,
            tolerance: 1e-6,
            centroids: None,
        }
    }

    /// Set max iterations.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set convergence tolerance.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Fit k-means to a set of behavior profiles.
    ///
    /// Uses random initialization from the data points.
    /// Returns the final centroids.
    pub fn fit(&mut self, profiles: &[BehaviorProfile]) -> Result<Vec<Centroid>, String> {
        if profiles.len() < self.k {
            return Err(format!(
                "Need at least {} profiles for k={}, got {}",
                self.k,
                self.k,
                profiles.len()
            ));
        }

        let data: Vec<Vec<f64>> = profiles.iter().map(|p| p.to_features()).collect();
        let dim = data[0].len();

        // Initialize centroids by picking k evenly-spaced profiles
        let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(self.k);
        let step = data.len() as f64 / self.k as f64;
        for i in 0..self.k {
            let idx = (step * i as f64).floor() as usize;
            centroids.push(data[idx].clone());
        }

        let mut assignments = vec![0usize; data.len()];

        for _iter in 0..self.max_iterations {
            // Assign each point to nearest centroid
            for (i, point) in data.iter().enumerate() {
                let mut best_dist = f64::MAX;
                let mut best_k = 0;
                for (k, centroid) in centroids.iter().enumerate() {
                    let dist = euclidean(point, centroid);
                    if dist < best_dist {
                        best_dist = dist;
                        best_k = k;
                    }
                }
                assignments[i] = best_k;
            }

            // Recompute centroids
            let mut new_centroids: Vec<Vec<f64>> = centroids
                .iter()
                .map(|_| vec![0.0; dim])
                .collect();
            let mut counts = vec![0usize; self.k];

            for (i, &cluster) in assignments.iter().enumerate() {
                counts[cluster] += 1;
                for (d, &val) in data[i].iter().enumerate() {
                    new_centroids[cluster][d] += val;
                }
            }

            // Average and handle empty clusters
            for k in 0..self.k {
                if counts[k] > 0 {
                    for d in 0..dim {
                        new_centroids[k][d] /= counts[k] as f64;
                    }
                } else {
                    // Reinitialize empty cluster from a random data point
                    let idx = assignments.iter().position(|&a| a == assignments[0]).unwrap_or(0);
                    new_centroids[k] = data[idx].clone();
                    counts[k] = 1;
                }
            }

            // Check convergence
            let total_shift: f64 = centroids
                .iter()
                .zip(new_centroids.iter())
                .map(|(old, new)| euclidean(old, new))
                .sum();

            centroids = new_centroids;

            if total_shift < self.tolerance {
                break;
            }
        }

        // Build Centroid structs with counts
        let result: Vec<Centroid> = centroids
            .into_iter()
            .enumerate()
            .map(|(k, features)| {
                let count = assignments.iter().filter(|&&a| a == k).count();
                Centroid {
                    features,
                    label: None,
                    member_count: count,
                }
            })
            .collect();

        self.centroids = Some(result.clone());
        Ok(result)
    }

    /// Predict the nearest cluster for a profile.
    ///
    /// Must call `fit` first.
    pub fn predict(&self, profile: &BehaviorProfile) -> Result<usize, String> {
        let centroids = self
            .centroids
            .as_ref()
            .ok_or("Must call fit() before predict()")?;

        let features = profile.to_features();
        let mut best_dist = f64::MAX;
        let mut best_k = 0;
        for (k, centroid) in centroids.iter().enumerate() {
            let dist = euclidean(&features, &centroid.features);
            if dist < best_dist {
                best_dist = dist;
                best_k = k;
            }
        }
        Ok(best_k)
    }

    /// Assign species labels to clusters based on majority vote from known labels.
    pub fn label_clusters(
        &mut self,
        profiles: &[BehaviorProfile],
        labels: &[Species],
    ) -> Result<HashMap<usize, Species>, String> {
        let centroids = self.centroids.as_ref().ok_or("Must call fit() first")?;
        if profiles.len() != labels.len() {
            return Err("profiles and labels must have same length".into());
        }

        let mut cluster_species: HashMap<usize, HashMap<Species, usize>> = HashMap::new();

        for (profile, &species) in profiles.iter().zip(labels.iter()) {
            let cluster = self.predict(profile)?;
            cluster_species
                .entry(cluster)
                .or_insert_with(HashMap::new)
                .entry(species)
                .and_modify(|c| *c += 1)
                .or_insert(1);
        }

        // Assign majority species to each cluster
        let mut result = HashMap::new();
        for (&cluster, species_counts) in &cluster_species {
            let best = species_counts
                .iter()
                .max_by_key(|(_, &count)| count)
                .map(|(&s, _)| s);
            if let Some(sp) = best {
                result.insert(cluster, sp);
            }
        }

        // Update centroid labels
        if let Some(ref mut centroids) = self.centroids {
            for (i, centroid) in centroids.iter_mut().enumerate() {
                centroid.label = result.get(&i).copied();
            }
        }

        Ok(result)
    }
}

fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_requires_enough_profiles() {
        let mut km = KMeansCluster::new(5);
        let profiles = vec![BehaviorProfile::new(0.3, 0.4, 0.3, 0.5, 1.0)];
        assert!(km.fit(&profiles).is_err());
    }

    #[test]
    fn test_fit_converges() {
        let mut km = KMeansCluster::new(2).with_max_iterations(50);
        let profiles: Vec<BehaviorProfile> = (0..20)
            .map(|i| {
                if i < 10 {
                    BehaviorProfile::new(0.1, 0.8, 0.1, 0.5, 0.5)
                } else {
                    BehaviorProfile::new(0.8, 0.1, 0.1, 0.5, 0.5)
                }
            })
            .collect();
        let result = km.fit(&profiles).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_predict_requires_fit() {
        let km = KMeansCluster::new(2);
        let p = BehaviorProfile::new(0.3, 0.4, 0.3, 0.5, 1.0);
        assert!(km.predict(&p).is_err());
    }

    #[test]
    fn test_predict_after_fit() {
        let mut km = KMeansCluster::new(2);
        let profiles: Vec<BehaviorProfile> = (0..20)
            .map(|i| {
                if i < 10 {
                    BehaviorProfile::new(0.1, 0.8, 0.1, 0.5, 0.5)
                } else {
                    BehaviorProfile::new(0.8, 0.1, 0.1, 0.5, 0.5)
                }
            })
            .collect();
        km.fit(&profiles).unwrap();

        let diplomat = BehaviorProfile::new(0.1, 0.8, 0.1, 0.5, 0.5);
        let explorer = BehaviorProfile::new(0.8, 0.1, 0.1, 0.5, 0.5);

        // They should be in different clusters
        assert_ne!(
            km.predict(&diplomat).unwrap(),
            km.predict(&explorer).unwrap()
        );
    }
}
