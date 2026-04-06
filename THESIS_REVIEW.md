# Thesis Review: Corrections and Improvements

**Thesis:** "Explainable Brain Tumor Classification via Generation Using Conditional StyleGANs and CNNs"
**Author:** Junwei He Mai
**Reviewer:** Automated deep review (PDF + code + notebooks + results)
**Date:** 2026-04-06

---

## How to use this document

Each item includes:
- **Location:** Page number and section
- **Issue:** What's wrong
- **Original:** Exact text (when applicable)
- **Suggested:** Proposed correction
- **Priority:** CRITICAL / HIGH / MEDIUM / LOW

---

## 1. CRITICAL: Undefined Key Concepts

### 1.1 Data leakage never formally defined
- **Location:** Section 1.1 (p.1-2). Mentioned 16+ times across the thesis but never defined.
- **Issue:** Your MAIN contribution — patient-level leakage control — is never explained to the reader. The term is used assuming the reader already knows what it means.
- **Suggested addition (Section 1.1, after motivation paragraph):**
  > "Patient-level data leakage occurs when MRI slices from the same patient appear in both training and test sets. Since a single patient may contribute multiple highly correlated slices (axial, sagittal, coronal), models can learn to recognize patient-specific anatomical features rather than tumor-specific characteristics, leading to artificially inflated accuracy. This work enforces strict patient-disjoint partitions throughout all experiments to prevent this form of leakage."
- **Priority:** CRITICAL

### 1.2 FID not explained at first use
- **Location:** Abstract (p.2), line: "achieving an FID score of 46"
- **Issue:** Reader has no context for what FID is or whether 46 is good/bad.
- **Suggested change in Abstract:**
  > "achieving a Fréchet Inception Distance (FID) of 46.08 (lower is better), outperforming comparable conditional generative models in the literature"
- **Also add to Section 2.4.1:** A brief definition with interpretation guidelines.
- **Priority:** CRITICAL

### 1.3 CNN not expanded in Abstract
- **Location:** Abstract (p.2), line: "96.0% using a CNN with augmented synthetic data"
- **Suggested:** "96.0% using a Convolutional Neural Network (CNN) with augmented synthetic data"
- **Priority:** CRITICAL

### 1.4 "Reduced dataset" undefined in Methods
- **Location:** First appears in Results (Section 5.4.2, p.73) without clear definition in Methods (Chapter 4).
- **Issue:** Reader doesn't know what was filtered, by whom, or with what criteria.
- **Suggested addition (Section 4.2):**
  > "A reduced variant of the dataset was constructed by excluding MRI slices of poor diagnostic quality (e.g., slices with minimal tumor visibility or significant artifacts). This reduced dataset excludes [X patients / X images], corresponding to approximately 25% fewer images than the complete dataset. Both variants preserve patient-level leakage control."
- **Priority:** HIGH

### 1.5 Image resolution justification misplaced
- **Location:** Currently only in Discussion (p.104-105): "256×256 strikes a balance"
- **Issue:** Resolution choice should be justified in Methods, not first mentioned in Discussion.
- **Suggested:** Move resolution justification to Section 4.3 (StyleGAN-C Architecture) and add brief mention in Discussion as follow-up.
- **Priority:** MEDIUM

---

## 2. CRITICAL: Factual Inconsistencies

### 2.1 Metric format: percentages vs decimals
- **Location:** Abstract (p.2-3) vs Results (Ch.5) vs Conclusions (Ch.7)
- **Issue:**
  - Abstract: "92.3%", "83.2%", "83.8%", "96.0%", "50%", "32.7%"
  - Results/Conclusions: "0.923", "0.832", "0.838", "0.960"
- **Suggested:** Standardize ALL to decimal format (0.923 instead of 92.3%) for consistency with ML reporting conventions. Alternatively, use percentages consistently, but pick ONE format.
- **Priority:** CRITICAL

### 2.2 FID precision varies
- **Location:**
  - Pages 20, 78: "46.083" (3 decimal places)
  - Pages 110, 114: "46.08" (2 decimal places)
- **Suggested:** Use "46.08" everywhere (2 decimals is standard for FID reporting).
- **Priority:** MEDIUM

### 2.3 "761 patients" vs "761 images"
- **Location:** Page 88, Section 5.4.2
- **Original:** "excludes 761 patients, corresponding to approximately 25% fewer images"
- **Issue:** Unclear if 761 refers to patients or images. These are very different quantities. Verify against Table 4.1 and correct.
- **Priority:** HIGH

### 2.4 Complete dataset improvement NOT statistically significant — but presented as success
- **Location:** Tables 5.22-5.23 (p.91) vs Discussion (p.87-88)
- **Issue:**
  - Table 5.23: Global NLL p-value = 0.5075 (NOT significant) for complete dataset
  - Table 5.22: McNemar p = 0.080 (NOT significant at α=0.05)
  - But Discussion (p.87-88) emphasizes the 0.9% accuracy gain without clearly stating it is NOT statistically significant
- **Suggested addition to Discussion (Section 6.1.4):**
  > "It should be noted that the accuracy improvement on the complete dataset (0.947 → 0.956) does not reach statistical significance (McNemar p = 0.080; NLL t-test p = 0.5075), unlike the reduced dataset where global NLL improvement is significant (p = 0.0065). This suggests that synthetic augmentation provides its greatest benefit when real data is limited."
- **Priority:** CRITICAL

### 2.5 ECE vs NLL contradiction — not discussed
- **Location:** Notebook 12 results vs Discussion (Ch.6)
- **Issue:**
  - ECE (ensemble, reduced dataset): baseline = 0.0047, augmented = 0.0147 (3× WORSE)
  - NLL (ensemble, reduced dataset): augmented shows significant improvement for some classes
  - The thesis emphasizes NLL improvement but never discusses that ECE degrades
- **Suggested addition to Discussion (Section 6.1.4, after NLL discussion):**
  > "While the NLL analysis reveals improved probabilistic calibration at the class level, particularly for pituitary, the Expected Calibration Error (ECE) shows a different pattern: the augmented ensemble exhibits higher ECE (0.0147) compared to the real-only baseline (0.0047). This divergence arises because NLL penalizes low probability assigned to the correct class, whereas ECE measures the average gap between predicted confidence and observed accuracy across bins. The augmented model may assign higher probability to correct predictions (lower NLL) while simultaneously being less well-calibrated in aggregate (higher ECE). This distinction warrants further investigation, potentially through post-hoc calibration techniques such as temperature scaling."
- **Priority:** CRITICAL

### 2.6 Meningioma NLL worsens with augmentation on complete dataset
- **Location:** Table 5.23 (p.91)
- **Issue:** Meningioma NLL increases from 0.253 to 0.307 with synthetic augmentation on complete dataset (worse calibration), but this is not discussed as a negative result.
- **Suggested addition to Discussion:**
  > "For the meningioma class on the complete dataset, NLL increases from 0.246 to 0.307, suggesting that synthetic augmentation may introduce noise for this already under-represented class when ample real data is available."
- **Priority:** HIGH

---

## 3. HIGH: Unsupported or Overstated Claims

### 3.1 Grad-CAM claim based on n=3
- **Location:** Page 109: "Grad-CAM can substantially enhance the expert's confidence"
- **Issue:** Based on only 3 synthetic images with "full" heatmap rating (Table 5.30). With such tiny n, statistical inference is unreliable.
- **Suggested rewording:**
  > "Preliminary results suggest that when Grad-CAM correctly highlights the tumor region, it may enhance the expert's confidence and diagnostic accuracy. However, these findings are based on a very small sample (n=3 for synthetic images with correct heatmaps) and should be interpreted with caution."
- **Priority:** HIGH

### 3.2 Causal attribution claim
- **Location:** Page 109, Section 6.1.7: "the observed improvements can be causally attributed to enhanced generalization rather than data contamination"
- **Issue:** Patient-level splitting is necessary but not sufficient for causal claims. Other confounders (architecture, hyperparameters, random initialization) are not controlled.
- **Suggested rewording:**
  > "the observed improvements are consistent with enhanced generalization rather than data contamination, as ensured by the strict patient-level partitioning"
- **Priority:** HIGH

### 3.3 Counterfactuals framed as "complementary" despite inferior performance
- **Location:** Page 100, Section 6.1.2
- **Issue:** Counterfactual CAS F1 = 0.852 is LOWER than direct generation CAS (0.895–0.923), but text frames them as equally valuable.
- **Suggested addition:**
  > "While counterfactual classification underperforms direct synthetic generation for tumor classification (F1 0.852 vs 0.895–0.923), its primary value lies in providing interpretable visual explanations rather than raw predictive performance."
- **Priority:** MEDIUM

### 3.4 "Measurable gains even when underlying real dataset has been cleaned"
- **Location:** Page 88
- **Issue:** Improvement is 0.953 → 0.960 (0.7%), while strong geometric augmentation alone reaches 0.959. The difference between GAN augmentation (0.960) and strong augmentation (0.959) is 0.001 — likely within noise.
- **Suggested rewording:**
  > "The gains from StyleGAN-C augmentation on the reduced dataset (0.953 → 0.960) are modest and comparable in magnitude to those achieved by strong geometric augmentation alone (0.959). This suggests that while the generative approach provides a small additional benefit, much of the improvement can be attributed to increased data diversity rather than specifically to GAN-generated content."
- **Priority:** HIGH

### 3.5 HITL framed as pilot too late
- **Location:** Only acknowledged as "pilot feasibility study" on page 108.
- **Issue:** Should be stated upfront when Experiment 3 is introduced (Section 4.8).
- **Suggested addition to Section 4.8:**
  > "This experiment is designed as a pilot feasibility study with a single neuroradiologist, intended to provide initial evidence on the clinical plausibility of synthetic images and the utility of XAI overlays. The sample size (n=33 images) limits statistical power but establishes a protocol for future multi-reader studies."
- **Priority:** MEDIUM

---

## 4. HIGH: Statistical Issues

### 4.1 No Bonferroni correction for multiple comparisons
- **Location:** Tables 5.22-5.24 (p.91-92), Notebook 11
- **Issue:** 4 simultaneous tests (global + 3 per-class) at α=0.05 without correction. With Bonferroni, α should be 0.0125.
- **Impact check:**
  - Pituitary NLL: p = 4.73×10⁻⁵ → SURVIVES Bonferroni ✓
  - Global NLL (reduced): p = 0.0065 → SURVIVES Bonferroni ✓
  - Meningioma NLL (reduced): p = 0.2804 → Already non-significant
  - Glioma NLL (reduced): p = 0.8748 → Already non-significant
- **Suggested addition:**
  > "After Bonferroni correction for four simultaneous comparisons (adjusted α = 0.0125), the global NLL improvement (p = 0.0065) and the pituitary-specific improvement (p = 4.73 × 10⁻⁵) remain statistically significant."
- **Priority:** HIGH

### 4.2 HITL tables without confidence intervals
- **Location:** Tables 5.26-5.30 (p.80-82)
- **Issue:** Means over n=9 per group without standard deviations or 95% CI. With such small n, variability is critical information.
- **Suggested:** Add SD column to all HITL tables. Example: "Mean confidence: 3.00 ± 1.22"
- **Priority:** HIGH

### 4.3 Effect size not reported
- **Location:** Section 5.4.4 (Statistical Tests)
- **Issue:** Only p-values reported. Cohen's d would strengthen interpretation — shows magnitude of effect, not just significance.
- **Suggested addition:** Report Cohen's d for significant NLL comparisons.
- **Priority:** MEDIUM

---

## 5. MEDIUM: Structure and Transitions

### 5.1 Missing transition between Exp 2A and 2B
- **Location:** Between Sections 5.2 and 5.3 (p.68)
- **Issue:** Exp 2A operates at patient-level (1 slice per patient), Exp 2B at slice-level. This methodological switch is not motivated.
- **Suggested transition paragraph:**
  > "While Experiment 2A demonstrated that latent inversion can achieve non-trivial classification at the patient level, the computational cost of per-patient inversion limits scalability. Experiment 2B therefore adopts a slice-level approach, evaluating whether pre-computed feature representations — from a CNN, a supervised encoder, and the e4e encoder — capture discriminative tumor information amenable to linear classification."
- **Priority:** MEDIUM

### 5.2 Missing transition Ch 2 → Ch 3
- **Location:** End of Chapter 2 / beginning of Chapter 3
- **Suggested:** Add bridging paragraph summarizing theoretical foundations and motivating the literature review.
- **Priority:** LOW

### 5.3 Missing transition Ch 3 → Ch 4
- **Location:** End of Chapter 3 / beginning of Chapter 4
- **Suggested:** Add paragraph connecting identified gaps in literature to the proposed method.
- **Priority:** LOW

### 5.4 Ablation hypotheses not stated
- **Location:** Section 4.7 (p.53-57)
- **Issue:** Three ablation studies are listed but the specific hypothesis each tests is not declared.
- **Suggested addition for each ablation:**
  > "Ablation 1 (CAS): Tests whether the synthetic images alone contain sufficient class-discriminative information to train a classifier, independently of real data."
  > "Ablation 2 (Oversampling): Tests whether the observed accuracy improvement is due to class balancing alone or to the additional variability introduced by the generator."
  > "Ablation 3 (Augmentation strength): Tests whether the benefit of GAN augmentation persists when compared to strong geometric augmentation on real data."
- **Priority:** MEDIUM

### 5.5 Dataset statistics missing from Section 4.2
- **Location:** Section 4.2 (p.40-42)
- **Issue:** Patient count, total slices, per-class distribution, and split percentages are not immediately presented.
- **Suggested:** Add summary table early in Section 4.2:

  | Class | Patients | Images | Train | Test |
  |---|---|---|---|---|
  | Glioma | X | X | X | 446 |
  | Meningioma | X | X | X | 194 |
  | Pituitary | X | X | X | 271 |
  | **Total** | **X** | **3064** | **2153** | **911** |

- **Priority:** MEDIUM

---

## 6. MEDIUM: Writing Style

### 6.1 "This work" repeated 7 times
- **Location:** Throughout Introduction and Contribution sections
- **Suggested alternatives:** "The present study", "Our framework", "The proposed approach", "This thesis", "The current investigation"
- **Priority:** LOW

### 6.2 Passive voice obscures methodology
- **Location:** p.87: "The reduced dataset is filtered to exclude MRI slices with poor quality"
- **Issue:** Who filtered? By what criteria? Automated or manual?
- **Suggested:** "We manually filtered the dataset to exclude MRI slices with poor diagnostic quality, defined as slices where the tumor region occupied less than X% of the image area or where imaging artifacts obscured the lesion."
- **Priority:** MEDIUM

### 6.3 "data is" vs "data are"
- **Location:** Multiple instances throughout
- **Issue:** In formal academic English, "data" is traditionally plural. Consistency needed.
- **Suggested:** Choose one convention and apply throughout. "Data are" is more traditional; "data is" is increasingly accepted.
- **Priority:** LOW

### 6.4 Abstract too dense with numbers
- **Location:** Abstract (p.2)
- **Issue:** 7 different metrics in one paragraph (FID 46, 50%, CAS 92.3%, 83.2%, 83.8%, 96.0%, 32.7%). Hard to parse.
- **Suggested reorganization by contribution:**
  > "First, a conditional StyleGAN2-ADA generator is trained, achieving an FID of 46.08 and a CAS of 0.923.
  > Second, three classification approaches are compared: latent inversion via e4e encoder (0.832), optimization-based inversion (0.838), and CNN with synthetic augmentation (0.960).
  > Third, Grad-CAM heatmaps increase neuroradiologist diagnostic confidence by 32.7% on synthetic images."
- **Priority:** MEDIUM

### 6.5 Literature accuracy numbers lack context
- **Location:** Chapter 3 (Literature Review)
- **Issue:** Papers cited with 98%, 96.97%, 95.6% accuracy from different datasets and without patient control, making comparison misleading.
- **Suggested:** Add a note when citing high-accuracy papers:
  > "[46] reports 98.7% accuracy; however, patient-level leakage control is not reported (NR in Table 6.7)."
- **Priority:** MEDIUM

---

## 7. MEDIUM: Table Improvements

### 7.1 Table 5.23 (NLL comparison) — confusing layout
- **Location:** Page 91-92
- **Issue:** Nested headers "N(=#Reduced images/#Complete images)" are hard to parse. The comparison structure "A: real-only, B: synthetic" in caption but not in header.
- **Suggested:** Split into two separate tables (one for reduced, one for complete dataset) or add clear column headers: "Reduced Dataset (A vs B)" and "Complete Dataset (A vs B)".
- **Priority:** MEDIUM

### 7.2 Table 5.30 (Grad-CAM ratings) — ambiguous hierarchy
- **Location:** Page 82
- **Issue:** Subcategories yes → full/partial and then "no" are ambiguous. Is "yes (overall)" the sum of "full" + "partial"?
- **Suggested:** Add a note: "The XAI rating 'yes' is subdivided into 'full' (heatmap fully covers tumor) and 'partial' (heatmap partially overlaps tumor)."
- **Priority:** LOW

### 7.3 Tables 5.13-5.14 — "Support" column ambiguous
- **Location:** Pages 86-87
- **Issue:** "Support" column shows per-class sample counts but the "Total" row shows 911 (full dataset size), which is not the sum of per-class values in the same column context.
- **Suggested:** Rename to "Test samples" and add a note clarifying it represents the number of test samples per class.
- **Priority:** LOW

### 7.4 Table 5.6 — empty cells unexplained
- **Location:** Page 82-83
- **Issue:** "Accuracy" column is blank for per-class rows without explaining the convention.
- **Suggested:** Add dash (—) for non-applicable cells and a table note: "— indicates metric not applicable at per-class level."
- **Priority:** LOW

---

## 8. Consistency Cross-Check: Abstract ↔ Results ↔ Conclusions

| Metric | Abstract (p.2) | Results | Conclusions (p.114-115) | Status |
|---|---|---|---|---|
| FID | 46 | 46.083 (p.78) / 46.08 (p.110) | 46.08 | ⚠ Precision varies |
| CAS | 92.3% | 0.923 (Table 5.19) | 0.923 | ⚠ Format mismatch |
| e4e accuracy | 83.2% | 0.832 (Table 5.6) | 0.832 | ⚠ Format mismatch |
| Optimization accuracy | 83.8% | 0.838 (Table 5.4) | 0.838 | ⚠ Format mismatch |
| CNN augmented | 96.0% | 0.960 (Table 5.18/6.6) | 0.96 | ⚠ Format mismatch |
| Expert accuracy (fake) | 50% | 0.50 (Table 5.26) | 50% | ⚠ Format mismatch |
| Confidence increase | 32.7% | 2.47→3.28 (Table 5.28) | 1.44→3.0 | ⚠ Different comparison! |
| H1 | — | — | "partially supported" | ✓ |
| H2A | — | — | "fully supported" | ✓ |
| H2B | — | — | "fully supported" | ✓ |
| H3 | — | — | "partially supported" | ✓ |

### CRITICAL: Confidence increase number inconsistency
- **Abstract (p.2):** "32.7% increase in diagnostic confidence (from 2.47 to 3.28)"
- **Conclusions (p.115):** "confidence increased from 1.44 to 3.0"
- **Issue:** These are DIFFERENT comparisons!
  - 2.47 → 3.28 = XAI not-paired vs XAI paired (Table 5.28, all images)
  - 1.44 → 3.0 = synthetic without XAI vs synthetic with XAI (Table 5.28, synthetic only)
- **Suggested fix:** Use one consistent comparison throughout. The 1.44→3.0 comparison (synthetic-only) is more impressive and specific. If using the 2.47→3.28 comparison in the Abstract, clarify it includes both real and synthetic images.
- **Priority:** CRITICAL

---

## Summary by Priority

| Priority | Count | Estimated effort |
|---|---|---|
| CRITICAL | 7 | 3-4 hours |
| HIGH | 8 | 4-5 hours |
| MEDIUM | 14 | 4-6 hours |
| LOW | 7 | 1-2 hours |
| **Total** | **36** | **12-17 hours** |

### Top 5 fixes with highest impact-to-effort ratio:
1. Standardize metric format (% vs decimal) — 30 min, fixes 12+ instances
2. Add data leakage definition — 10 min, strengthens main contribution
3. Add ECE vs NLL discussion paragraph — 15 min, addresses biggest vulnerability
4. Fix confidence increase inconsistency (Abstract vs Conclusions) — 10 min
5. Add "not statistically significant" note for complete dataset — 10 min
