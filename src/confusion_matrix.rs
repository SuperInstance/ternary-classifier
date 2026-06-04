use crate::Species;

/// Confusion matrix for evaluating species classifier accuracy.
///
/// Tracks predicted vs actual species classifications.
#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    /// 5x5 matrix: rows = actual, cols = predicted
    /// Index using Species::index()
    matrix: [[usize; 5]; 5],
    /// Total samples
    total: usize,
}

impl ConfusionMatrix {
    pub fn new() -> Self {
        Self {
            matrix: [[0; 5]; 5],
            total: 0,
        }
    }

    /// Record a prediction.
    pub fn record(&mut self, actual: Species, predicted: Species) {
        self.matrix[actual.index()][predicted.index()] += 1;
        self.total += 1;
    }

    /// Get the count at (actual, predicted).
    pub fn get(&self, actual: Species, predicted: Species) -> usize {
        self.matrix[actual.index()][predicted.index()]
    }

    /// Total number of samples.
    pub fn total(&self) -> usize {
        self.total
    }

    /// Compute overall accuracy (correct / total).
    pub fn accuracy(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let correct: usize = (0..5).map(|i| self.matrix[i][i]).sum();
        correct as f64 / self.total as f64
    }

    /// Precision for a species: TP / (TP + FP).
    pub fn precision(&self, species: Species) -> f64 {
        let i = species.index();
        let tp = self.matrix[i][i] as f64;
        let mut predicted_positive = 0;
        for row in 0..5 {
            predicted_positive += self.matrix[row][i];
        }
        if predicted_positive == 0 {
            return 0.0;
        }
        tp / predicted_positive as f64
    }

    /// Recall for a species: TP / (TP + FN).
    pub fn recall(&self, species: Species) -> f64 {
        let i = species.index();
        let tp = self.matrix[i][i] as f64;
        let actual_positive: usize = (0..5).map(|col| self.matrix[i][col]).sum();
        if actual_positive == 0 {
            return 0.0;
        }
        tp / actual_positive as f64
    }

    /// F1 score for a species (harmonic mean of precision and recall).
    pub fn f1(&self, species: Species) -> f64 {
        let p = self.precision(species);
        let r = self.recall(species);
        if p + r == 0.0 {
            return 0.0;
        }
        2.0 * p * r / (p + r)
    }

    /// Per-class accuracy: correct for species / total.
    pub fn per_class_accuracy(&self, species: Species) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.matrix[species.index()][species.index()] as f64 / self.total as f64
    }

    /// Format the matrix as a readable string.
    pub fn format(&self) -> String {
        let names = Species::all().iter().map(|s| s.name()).collect::<Vec<_>>();
        let mut s = String::from("          Predicted →\nActual ↓  ");
        for name in &names {
            s.push_str(&format!("{:>10}", name));
        }
        s.push('\n');

        for (i, name) in names.iter().enumerate() {
            s.push_str(&format!("{:<10}", name));
            for j in 0..5 {
                s.push_str(&format!("{:>10}", self.matrix[i][j]));
            }
            s.push('\n');
        }

        s.push_str(&format!("\nAccuracy: {:.2}%\n", self.accuracy() * 100.0));
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_matrix() {
        let cm = ConfusionMatrix::new();
        assert_eq!(cm.total(), 0);
        assert!((cm.accuracy() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_perfect_classification() {
        let mut cm = ConfusionMatrix::new();
        for sp in Species::all() {
            cm.record(*sp, *sp);
        }
        assert_eq!(cm.total(), 5);
        assert!((cm.accuracy() - 1.0).abs() < 1e-9);
        for sp in Species::all() {
            assert!((cm.precision(*sp) - 1.0).abs() < 1e-9);
            assert!((cm.recall(*sp) - 1.0).abs() < 1e-9);
            assert!((cm.f1(*sp) - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_misclassification() {
        let mut cm = ConfusionMatrix::new();
        cm.record(Species::Explorer, Species::Explorer);
        cm.record(Species::Explorer, Species::Diplomat);
        // Explorer: precision = 1/(1+0) = 1.0, recall = 1/2 = 0.5
        assert!((cm.precision(Species::Explorer) - 1.0).abs() < 1e-9);
        assert!((cm.recall(Species::Explorer) - 0.5).abs() < 1e-9);
        assert_eq!(cm.get(Species::Explorer, Species::Diplomat), 1);
    }

    #[test]
    fn test_format_output() {
        let mut cm = ConfusionMatrix::new();
        cm.record(Species::Explorer, Species::Explorer);
        let output = cm.format();
        assert!(output.contains("Accuracy"));
        assert!(output.contains("Explorer"));
    }
}
