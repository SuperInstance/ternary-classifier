# ternary-classifier

Classifies ternary agent behavior into **strategy species** using multiple methods. Pure Rust, no unsafe, no external dependencies.

## Species Classification Guide

Ternary agents choose between three actions: **Explore**, **Cooperate**, and **Defect**. Their behavioral patterns over time reveal their underlying strategy — their *species*.

### The Five Species

| Species | Signature | Key Traits |
|---------|-----------|------------|
| **Explorer** 🧭 | High explore rate (≥ 0.4) | Discovers new strategies, moderate entropy, curious |
| **Diplomat** 🤝 | High cooperate rate (≥ 0.5) | Builds alliances, low defect rate, collaborative |
| **Marksman** 🎯 | Low entropy + high win rate (≥ 0.55) | Precision targeting, consistent strategy, efficient |
| **Climber** 🧗 | High defect rate (≥ 0.5) + high win rate | Exploits others, competitive, results-oriented |
| **Prospector** ⚖️ | Balanced / everything else | Opportunistic sampling, high entropy, adaptable |

### Classification Priority

The rule-based classifier checks in this order:

1. **Diplomat** — if `cooperate_rate ≥ 0.5`
2. **Climber** — if `defect_rate ≥ 0.5` AND `win_rate ≥ 0.55`
3. **Explorer** — if `explore_rate ≥ 0.4`
4. **Marksman** — if `entropy < 1.2` AND `win_rate ≥ 0.55`
5. **Prospector** — default (everything else)

### Behavior Features

Each agent is characterized by a `BehaviorProfile` with five features:

- **`explore_rate`** — Fraction of actions that explore (0–1)
- **`cooperate_rate`** — Fraction of actions that cooperate (0–1)
- **`defect_rate`** — Fraction of actions that defect (0–1)
- **`win_rate`** — Success rate across encounters (0–1)
- **`entropy`** — Shannon entropy of action distribution (0–~1.585)

## Usage

```rust
use ternary_classifier::{BehaviorProfile, SpeciesClassifier, Species};

// Create a profile
let profile = BehaviorProfile::new(0.1, 0.7, 0.2, 0.5, 0.8);

// Classify it
let classifier = SpeciesClassifier::new();
let species = classifier.classify(&profile);
assert_eq!(species, Species::Diplomat);
```

### K-Means Clustering

Discover species automatically from unlabeled data:

```rust
use ternary_classifier::{BehaviorProfile, KMeansCluster};

let profiles = vec![
    BehaviorProfile::new(0.1, 0.8, 0.1, 0.5, 0.5),
    BehaviorProfile::new(0.8, 0.1, 0.1, 0.5, 0.5),
    // ... more profiles
];

let mut km = KMeansCluster::new(3);
let centroids = km.fit(&profiles).unwrap();
let cluster = km.predict(&profiles[0]).unwrap();
```

### Decision Boundaries

Compute separating hyperplanes between species:

```rust
use ternary_classifier::{compute_boundaries, SpeciesClassifier, Species};

let boundaries = compute_boundaries(&classifier, &labeled_profiles);
for b in &boundaries {
    println!("{} vs {}: distance = {:.3}",
        b.species_a.name(), b.species_b.name(),
        b.signed_distance(&test_profile));
}
```

### Confusion Matrix

Evaluate classifier accuracy:

```rust
use ternary_classifier::ConfusionMatrix;

let mut cm = ConfusionMatrix::new();
for (profile, actual) in test_data {
    let predicted = classifier.classify(&profile);
    cm.record(actual, predicted);
}
println!("Accuracy: {:.1}%", cm.accuracy() * 100.0);
println!("{}", cm.format());
```

### Feature Importance

Rank which features matter most:

```rust
use ternary_classifier::FeatureImportance;

let fi = FeatureImportance::compute(&profiles, &labels, &classifier, 42, 20);
println!("{}", fi.format());
```

## Architecture

```
src/
├── lib.rs                  # Re-exports
├── behavior_profile.rs     # BehaviorProfile struct
├── species_classifier.rs   # Species enum + SpeciesClassifier
├── kmeans.rs               # KMeansCluster + Centroid
├── decision_boundary.rs    # DecisionBoundary + compute_boundaries()
├── confusion_matrix.rs     # ConfusionMatrix evaluation
└── feature_importance.rs   # FeatureImportance permutation ranking
```

## License

MIT

## See Also
- **ternary-ensemble** — related
- **ternary-fitness** — related
- **ternary-explain** — related
- **ternary-attention** — related
- **ternary-scoring** — related

