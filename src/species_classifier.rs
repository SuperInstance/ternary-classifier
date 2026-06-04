use crate::BehaviorProfile;

/// Strategy species that a ternary agent can be classified as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Species {
    /// High explore rate, moderate entropy — discovers new strategies
    Explorer,
    /// High cooperate rate, low defect — builds alliances
    Diplomat,
    /// Low entropy, high win rate — precision targeting
    Marksman,
    /// High defect rate, high win rate — exploits others
    Climber,
    /// Balanced profile, high entropy — opportunistic sampling
    Prospector,
}

impl Species {
    /// All species variants in order.
    pub fn all() -> &'static [Species] {
        &[
            Species::Explorer,
            Species::Diplomat,
            Species::Marksman,
            Species::Climber,
            Species::Prospector,
        ]
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Species::Explorer => "Explorer",
            Species::Diplomat => "Diplomat",
            Species::Marksman => "Marksman",
            Species::Climber => "Climber",
            Species::Prospector => "Prospector",
        }
    }

    /// Index into a fixed-size array (0..4).
    pub fn index(&self) -> usize {
        match self {
            Species::Explorer => 0,
            Species::Diplomat => 1,
            Species::Marksman => 2,
            Species::Climber => 3,
            Species::Prospector => 4,
        }
    }

    /// Species from index (0..4). Returns None for out-of-range.
    pub fn from_index(i: usize) -> Option<Species> {
        match i {
            0 => Some(Species::Explorer),
            1 => Some(Species::Diplomat),
            2 => Some(Species::Marksman),
            3 => Some(Species::Climber),
            4 => Some(Species::Prospector),
            _ => None,
        }
    }
}

/// Rule-based species classifier.
///
/// Uses heuristic thresholds on the behavior profile features
/// to assign one of the five species.
#[derive(Debug, Clone)]
pub struct SpeciesClassifier {
    /// Entropy threshold for "high entropy" (default: 1.2)
    pub high_entropy: f64,
    /// Explore rate threshold for Explorer (default: 0.4)
    pub explore_threshold: f64,
    /// Cooperate rate threshold for Diplomat (default: 0.5)
    pub cooperate_threshold: f64,
    /// Defect rate threshold for Climber (default: 0.5)
    pub defect_threshold: f64,
    /// Win rate threshold for competitive species (default: 0.55)
    pub win_threshold: f64,
}

impl Default for SpeciesClassifier {
    fn default() -> Self {
        Self {
            high_entropy: 1.2,
            explore_threshold: 0.4,
            cooperate_threshold: 0.5,
            defect_threshold: 0.5,
            win_threshold: 0.55,
        }
    }
}

impl SpeciesClassifier {
    pub fn new() -> Self {
        Self::default()
    }

    /// Classify a behavior profile into a species.
    pub fn classify(&self, profile: &BehaviorProfile) -> Species {
        // Diplomat: high cooperation
        if profile.cooperate_rate >= self.cooperate_threshold {
            return Species::Diplomat;
        }

        // Climber: high defect rate + good win rate
        if profile.defect_rate >= self.defect_threshold
            && profile.win_rate >= self.win_threshold
        {
            return Species::Climber;
        }

        // Explorer: high explore rate
        if profile.explore_rate >= self.explore_threshold {
            return Species::Explorer;
        }

        // Marksman: low entropy + high win rate
        if profile.entropy < self.high_entropy
            && profile.win_rate >= self.win_threshold
        {
            return Species::Marksman;
        }

        // Prospector: everything else (balanced/opportunistic)
        Species::Prospector
    }

    /// Classify multiple profiles.
    pub fn classify_batch<'a>(
        &self,
        profiles: &'a [BehaviorProfile],
    ) -> Vec<(usize, Species)> {
        profiles
            .iter()
            .enumerate()
            .map(|(i, p)| (i, self.classify(p)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_classifier() -> SpeciesClassifier {
        SpeciesClassifier::new()
    }

    #[test]
    fn test_classify_diplomat() {
        let p = BehaviorProfile::new(0.1, 0.6, 0.3, 0.5, 0.9);
        assert_eq!(make_classifier().classify(&p), Species::Diplomat);
    }

    #[test]
    fn test_classify_climber() {
        let p = BehaviorProfile::new(0.1, 0.2, 0.7, 0.7, 0.8);
        assert_eq!(make_classifier().classify(&p), Species::Climber);
    }

    #[test]
    fn test_classify_explorer() {
        let p = BehaviorProfile::new(0.5, 0.25, 0.25, 0.4, 1.3);
        assert_eq!(make_classifier().classify(&p), Species::Explorer);
    }

    #[test]
    fn test_classify_marksman() {
        let p = BehaviorProfile::new(0.2, 0.35, 0.45, 0.7, 0.8);
        assert_eq!(make_classifier().classify(&p), Species::Marksman);
    }

    #[test]
    fn test_classify_prospector() {
        // Moderate everything, high entropy
        let p = BehaviorProfile::new(0.3, 0.3, 0.4, 0.4, 1.5);
        assert_eq!(make_classifier().classify(&p), Species::Prospector);
    }

    #[test]
    fn test_classify_batch() {
        let profiles = vec![
            BehaviorProfile::new(0.1, 0.6, 0.3, 0.5, 0.9), // Diplomat
            BehaviorProfile::new(0.5, 0.25, 0.25, 0.4, 1.3), // Explorer
        ];
        let results = make_classifier().classify_batch(&profiles);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], (0, Species::Diplomat));
        assert_eq!(results[1], (1, Species::Explorer));
    }

    #[test]
    fn test_species_roundtrip_index() {
        for sp in Species::all() {
            assert_eq!(Species::from_index(sp.index()), Some(*sp));
        }
    }
}
