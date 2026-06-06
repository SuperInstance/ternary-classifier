# ternary-classifier

Species classification for ternary agents. From raw behavior profiles to labeled populations.

When you run a multi-agent system, every agent develops behavioral patterns: how much it explores, cooperates, defects, and wins. This crate turns those patterns into *species*—named strategy archetypes you can reason about, cluster, and evaluate. It includes a rule-based classifier, k-means clustering, decision boundary computation, confusion matrices, and feature importance analysis. All in ~400 lines, zero external dependencies.

## Why this exists

You can observe agents all day, but observation without classification is just noise. This crate answers the question: *"What kind of agent am I looking at?"* The answer isn't a floating-point score—it's a species label with semantic meaning.

The five species map naturally to strategic archetypes you find in iterated games, market participants, and evolved populations:

| Species | Strategy | Real-world analog |
|---------|----------|-------------------|
| Explorer | High explore rate, discovers new strategies | R&D team, early adopter |
| Diplomat | High cooperation, low defection | Alliance builder, open-source contributor |
| Marksman | Low entropy, high win rate | Specialist, precision trader |
| Climber | High defection, high win rate | Zero-sum competitor, arbitrageur |
| Prospector | Balanced, high entropy | Opportunist, generalist |

## The key insight

Classification isn't a single operation—it's a pipeline. You profile behavior, classify into species, validate with a confusion matrix, discover clusters in unlabeled data, compute decision boundaries to understand *where* the boundaries are, and rank feature importance to know *why* the boundaries are there. This crate gives you every stage of that pipeline.

```
BehaviorProfile → SpeciesClassifier → Species
       ↓                ↓                  ↓
  to_features()    KMeansCluster     ConfusionMatrix
       ↓                ↓                  ↓
  feature space    centroids         accuracy/precision/recall
       ↓                                   ↓
  DecisionBoundary                FeatureImportance
```

## Quick start

```rust
use ternary_classifier::*;

// Describe an agent's behavior
let profile = BehaviorProfile::new(
    0.1,   // explore_rate: rarely explores
    0.7,   // cooperate_rate: highly cooperative
    0.2,   // defect_rate: rarely defects
    0.5,   // win_rate: average
    0.8,   // entropy: low — predictable strategy
);

// Classify it
let classifier = SpeciesClassifier::new();
let species = classifier.classify(&profile);
assert_eq!(species, Species::Diplomat);
```

## Architecture

### BehaviorProfile

The fundamental data type. A 5-dimensional feature vector capturing an agent's strategic fingerprint:

```rust
let profile = BehaviorProfile::new(explore, cooperate, defect, win, entropy);
let features: Vec<f64> = profile.to_features();       // [explore, cooperate, defect, win, entropy]
let dist = profile.distance_to(&other_profile);        // Euclidean distance in feature space
let valid = profile.is_valid_distribution(0.01);       // do rates sum to ~1.0?
```

### SpeciesClassifier

Rule-based classifier using configurable thresholds:

```rust
let classifier = SpeciesClassifier::new();  // sensible defaults
// Or customize:
let classifier = SpeciesClassifier {
    high_entropy: 1.0,
    explore_threshold: 0.5,
    cooperate_threshold: 0.6,
    defect_threshold: 0.4,
    win_threshold: 0.6,
};

// Single classification
let species = classifier.classify(&profile);

// Batch classification
let results: Vec<(usize, Species)> = classifier.classify_batch(&profiles);
```

### KMeansCluster

Unsupervised clustering to discover species groups from unlabeled data:

```rust
let mut km = KMeansCluster::new(3)      // 3 clusters
    .with_max_iterations(50)
    .with_tolerance(1e-4);

km.fit(&profiles)?;                      // discover clusters
let cluster_id = km.predict(&profile)?;  // assign new profile

// Label clusters by majority vote from known labels
let mapping = km.label_clusters(&profiles, &labels)?;
```

### ConfusionMatrix

Evaluate classification accuracy:

```rust
let mut cm = ConfusionMatrix::new();
for (profile, true_label) in test_data {
    let predicted = classifier.classify(profile);
    cm.record(true_label, predicted);
}

println!("{}", cm.format());            // pretty-printed matrix
cm.accuracy();                          // overall accuracy
cm.precision(Species::Explorer);        // per-class precision
cm.recall(Species::Diplomat);           // per-class recall
cm.f1(Species::Climber);                // per-class F1
```

### DecisionBoundary

Linear decision boundaries between species pairs:

```rust
let boundaries = compute_boundaries(&classifier, &labeled_profiles);
// Each boundary separates two species with a hyperplane w·x + b = 0
for boundary in &boundaries {
    let side = boundary.classify_side(&profile);  // which species?
    let distance = boundary.signed_distance(&profile);  // how far from boundary?
}
```

### FeatureImportance

Permutation-based feature importance—rank which behavioral dimensions matter most:

```rust
let importance = FeatureImportance::compute(
    &profiles, &labels, &classifier,
    42,    // shuffle seed (reproducible)
    20,    // number of shuffles per feature
);

println!("{}", importance.format());
// Output:
// Feature Importance (ranked):
//  1. cooperate_rate       1.000 ████████████████████████████████████████
//  2. defect_rate          0.847 ████████████████████████████████████
//  3. win_rate             0.312 ████████████
//  4. entropy              0.105 ████
//  5. explore_rate         0.000 

let (top_feature, score) = importance.most_important();
```

## Real-world example: Population analysis

```rust
use ternary_classifier::*;

// You've tracked 100 agents through an iterated game.
// Each has a behavior profile from their action history.

let profiles: Vec<BehaviorProfile> = /* ... */;
let classifier = SpeciesClassifier::new();

// Classify the entire population
let species_counts: std::collections::HashMap<Species, usize> = profiles
    .iter()
    .map(|p| classifier.classify(p))
    .fold(std::collections::HashMap::new(), |mut acc, s| {
        *acc.entry(s).or_insert(0) += 1;
        acc
    });

// Discover if there are sub-populations the classifier misses
let mut km = KMeansCluster::new(7);  // try more clusters than known species
km.fit(&profiles)?;
let centroids = km.centroids.as_ref().unwrap();
println!("Discovered {} clusters", centroids.len());
for (i, c) in centroids.iter().enumerate() {
    println!("  Cluster {}: {} members", i, c.member_count);
}

// Evaluate classifier quality with a confusion matrix
let mut cm = ConfusionMatrix::new();
for (profile, true_label) in labeled_test_set {
    cm.record(true_label, classifier.classify(&profile));
}
println!("Accuracy: {:.1}%", cm.accuracy() * 100.0);
```

## API surface

| Type | Purpose |
|------|---------|
| `BehaviorProfile` | 5D feature vector for an agent's strategy |
| `Species` | 5 strategy archetypes (Explorer, Diplomat, Marksman, Climber, Prospector) |
| `SpeciesClassifier` | Rule-based classifier with configurable thresholds |
| `KMeansCluster` | Unsupervised k-means over behavior profiles |
| `Centroid` | Cluster center with optional species label |
| `DecisionBoundary` | Linear separator between two species |
| `ConfusionMatrix` | 5×5 evaluation matrix with precision/recall/F1 |
| `FeatureImportance` | Permutation-based feature ranking |

## Ecosystem connections

- **ternary-gauge** — generate `BehaviorProfile`s by gauging agent action streams over time
- **ternary-paxos** — classify how agents vote in consensus rounds
- **ternary-membrane** — model how agent populations flow and equilibrate

## Design decisions

**Why rule-based classification instead of a neural network?** Because the feature space is 5-dimensional and the boundaries are interpretable. A neural network would add complexity without adding insight. When your data has structure this clean, a well-tuned rules engine beats a black box.

**Why k-means for unsupervised discovery?** Because Lloyd's algorithm converges fast in low-dimensional spaces. With only 5 features, k-means finds real clusters in under 50 iterations. No GPU required.

**Why permutation importance?** Because it's model-agnostic and honest. It doesn't rely on internal model state—it measures actual accuracy degradation when you destroy a feature's signal. Works with any classifier, not just the built-in one.

## Stats

| Metric | Value |
|--------|-------|
| Public types | 7 |
| Public functions | ~30 |
| License | MIT |
| External dependencies | 0 |
| Unsafe | 0 |

## Installation

```toml
[dependencies]
ternary-classifier = "0.1.0"
```

## License

MIT
