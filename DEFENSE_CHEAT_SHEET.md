# Thesis Defense Cheat Sheet

## 15 Most Likely Questions with Bullet-Point Answers

---

### Q1: What is the core problem your thesis addresses?

- Brain tumors (glioma, meningioma, pituitary) require **different treatments**
- Deep learning can automate MRI classification, but most studies have **data leakage**
- Limited labeled medical data available
- Clinicians need **explainable** models, not black boxes
- **One sentence:** "I built an end-to-end framework for brain tumor classification that addresses data leakage, data scarcity, and explainability."

---

### Q2: What is patient-level data leakage and why does it matter?

- Each patient has **multiple correlated MRI slices** (axial, sagittal, coronal)
- Image-level split = same patient in train AND test
- Model memorizes **patient anatomy**, not **tumor features**
- I proved it: without control **0.985**, with control **0.954** = **3 points inflation**
- Most published papers **don't report** whether they control this
- Only **3 multiclass studies** on Figshare enforce patient control
- **One sentence:** "Without patient-level control, accuracy inflates by about 3 percentage points, and most papers don't enforce it."

---

### Q3: Describe your framework and experiments.

- **Three stages:**
  1. **Generation** (Exp 1): Conditional StyleGAN2-ADA trained on Figshare dataset
  2. **Classification** (Exp 2A/2B/2C): Three approaches compared
  3. **Evaluation** (Exp 3): Grad-CAM XAI + neuroradiologist HITL
- Exp 2A = classification by latent inversion (counterfactuals + LPIPS)
- Exp 2B = SVM on features from 3 encoders (supervised, CNN, e4e)
- Exp 2C = **main experiment** = ResNet-50 + synthetic augmentation + K-Fold
- Exp 3 = blinded pilot study with neuroradiologist
- **One sentence:** "I trained a conditional GAN to generate synthetic MRI, used it to augment a CNN classifier, and validated with Grad-CAM and a neuroradiologist."

---

### Q4: What is CAS and what does 0.923 mean?

- **IMPORTANT: Train on SYNTHETIC, test on REAL** (not the other way around!)
- Measures: do synthetic images carry enough class information to learn from?
- Goes beyond FID (visual quality) to measure **functional utility**
- 0.923 = a model trained ONLY on fake images classifies **92.3% of real images correctly**
- First application of CAS to brain tumor MRI
- **One sentence:** "CAS trains only on synthetic images and tests on real ones -- 0.923 means our generated images preserve enough tumor information to classify real cases."

---

### Q5: What is FID and how does your score compare?

- FID = Frechet Inception Distance, **lower is better**
- Measures distribution distance between real and generated images
- My FID = **46.08**
- Comparison:
  - TumorGAN: 77.43
  - Diffusion Model: 70.58
  - SSimDCL: 92.36
  - Fetty StyleGAN: 13.0 (but trained >1 month, overfitting risk)
- Best among **comparable conditional** models
- **One sentence:** "Our FID of 46.08 is the lowest among comparable conditional generators in brain tumor MRI."

---

### Q6: How did you get 0.960 accuracy? Explain the setup.

- Model: **ResNet-50** (not just "CNN")
- Dataset: **reduced** dataset (higher quality images)
- Training: **5-fold cross-validation**, same seeds across all runs
- Augmentation: real images + synthetic images at **truncation psi = 1.3**
- Final model: **ensemble** of 5 fold models (predictions aggregated)
- Baseline (real-only ensemble): **0.953**
- Improvement: 0.953 --> 0.960 (+0.7%)
- **One sentence:** "It's a ResNet-50 ensemble from 5-fold cross-validation on the reduced dataset, augmented with StyleGAN-C images at psi 1.3."

---

### Q7: The improvement is less than 1%. Is it worth it?

- **Argument 1 -- Calibration:** NLL improves significantly (p = 0.0065 global, p = 4.73e-5 pituitary). Model gives **more reliable probabilities**, which matters clinically.
- **Argument 2 -- Ablations:** Oversampling only reaches **0.934**, GAN reaches **0.960**. That's a **2.6 point gap** = GAN adds real variability, not just class balancing.
- **Argument 3 -- Framework:** The GAN also enables counterfactual videos (2A), latent analysis (2B), and HITL evaluation (3). Accuracy is a secondary benefit.
- **One sentence:** "The main benefit is calibration, not accuracy -- and the GAN serves the entire framework, not just augmentation."

---

### Q8: Explain your ablation studies.

- **Ablation 1 (CAS):** Train on synthetic only --> 0.923 accuracy. Synthetic images carry class information.
- **Ablation 2 (Oversampling):** Duplicate real minority images --> 0.934. Class balancing alone is **not enough**. GAN adds variability (0.960).
- **Ablation 3 (Weak vs Strong augmentation):** Weak geometric = 0.932, Strong geometric = 0.959, GAN = 0.960. GAN provides small additional benefit over strong augmentation.
- **Key takeaway:** The improvement is NOT just from balancing classes or standard augmentation. The generator introduces **new visual variability**.
- **One sentence:** "Oversampling reaches 0.934, strong augmentation 0.959, and GAN 0.960 -- proving the generator adds variability beyond class balancing and geometric transforms."

---

### Q9: What statistical tests did you run?

- **McNemar test** (slice-level): Ensemble vs single-run = significant. Real-only vs GAN ensemble = **NOT significant** (p > 0.05 in accuracy).
- **NLL t-test** (calibration): Global reduced = **p = 0.0065** (significant). Pituitary = **p = 4.73e-5** (highly significant). Meningioma and glioma = not significant.
- **Interpretation:** Accuracy difference is modest, but **calibration significantly improves**. The model produces more reliable probability estimates.
- After **Bonferroni correction** (alpha = 0.0125): global and pituitary **still significant**.
- **One sentence:** "McNemar shows no significant accuracy difference, but NLL t-tests show the GAN-augmented model is significantly better calibrated, especially for pituitary."

---

### Q10: Explain the HITL experiment and main findings.

- **Design:** Pilot study, 1 neuroradiologist, **33 images** (15 real + 18 synthetic), blinded, random order
- Expert rates: tumor class, confidence (1-5), utility (1-5)
- Half images with **Grad-CAM overlay**, half without
- **Key numbers:**
  - Real images: 100% accuracy, confidence 3.73
  - Synthetic without XAI: 44% accuracy, confidence **1.44**
  - Synthetic with XAI: 56% accuracy, confidence **3.00**
- Grad-CAM **doubles confidence** on synthetic images (1.44 --> 3.00)
- Expert is conservative: when she answers, accuracy is **90%** on synthetics
- Pituitary best (0.83 accuracy), meningioma worst (0.17)
- **It's a pilot** -- first HITL in brain tumor classification
- **One sentence:** "Grad-CAM doubles expert confidence on synthetic images from 1.44 to 3.00, and when the expert commits to a diagnosis, accuracy reaches 90%."

---

### Q11: Only 50% of synthetic images are useful. Doesn't the generator fail?

- Expert is **conservative** -- doesn't guess. Answered cases: **90% accuracy**
- It's **class-dependent**: pituitary works well (0.83), meningioma doesn't (0.17)
- Even **real single slices** can be ambiguous -- radiologists normally use multiple planes
- Machine utility vs human utility are **different things**
- CAS = 0.923 proves the images work **for the classifier**
- This finding **motivates future work**: synthetic image curation/filtering pipeline
- **One sentence:** "The 50% reflects the expert's conservative approach and single-slice limitation -- among answered cases accuracy is 90%, and CAS proves the images are useful for machine learning."

---

### Q12: Why StyleGAN2-ADA and not Diffusion Models?

- StyleGAN2-ADA has an **interpretable latent space** (W/W+)
- This enables: latent inversion, counterfactual generation, feature extraction (Exp 2A, 2B)
- **ADA** is specifically designed for **small datasets** (adaptive discriminator augmentation)
- Diffusion models don't offer easy latent manipulation for counterfactuals
- Diffusion models are suggested as **future work**
- **One sentence:** "StyleGAN's interpretable latent space enables inversion, counterfactuals, and feature extraction -- capabilities that diffusion models don't easily provide."

---

### Q13: Why ResNet-50 and not a newer architecture?

- ResNet-50 is an **established backbone**, comparable with existing literature
- Transfer learning from ImageNet
- The goal was to evaluate **augmentation impact**, not maximize architecture
- ViT papers report 98.7% but **without patient control** = inflated
- Fair comparison requires same architecture focus on **augmentation effect**
- **One sentence:** "I chose ResNet-50 to isolate the effect of synthetic augmentation -- using a newer architecture would confuse whether the improvement comes from the model or the data."

---

### Q14: What are the main limitations?

- **Single dataset** (Figshare) -- no external validation
- **HITL: 1 expert, 33 images** -- pilot study, not conclusive
- **Modest accuracy improvement** (+0.7%) -- main benefit is calibration
- **50% synthetic images non-diagnostic** -- need curation pipeline
- **256x256 resolution** -- higher resolution could capture finer detail
- **One sentence:** "The main limitation is single-dataset validation with one expert -- future work needs multi-center data and larger HITL studies."

---

### Q15: What are your main contributions?

1. **Strict patient-level leakage control** across all experiments -- demonstrated 3% inflation without it
2. **First CAS application** to brain tumor MRI (0.923)
3. **CNN classifier achieving 0.960** accuracy -- surpasses all SOTA with patient control
4. **First HITL pilot** in brain tumor classification
5. **Grad-CAM validation** showing confidence doubles on synthetic images
6. **End-to-end framework**: generation --> classification --> XAI --> clinical validation
- **One sentence:** "My main contributions are strict leakage control throughout, the first CAS evaluation in brain tumors, and the first HITL pilot combining synthetic image assessment with XAI evaluation."

---

## Quick Reference: Key Numbers

| Metric | Value |
|---|---|
| FID | 46.08 |
| CAS | 0.923 |
| Best accuracy (ensemble, reduced) | 0.960 |
| Baseline accuracy (real-only) | 0.953 |
| Oversampling baseline | 0.934 |
| Strong augmentation | 0.959 |
| NLL p-value (global, reduced) | 0.0065 |
| NLL p-value (pituitary) | 4.73e-5 |
| Leakage inflation | ~3 percentage points |
| HITL confidence without XAI (synthetic) | 1.44 |
| HITL confidence with XAI (synthetic) | 3.00 |
| Supervised encoder SVM | 0.947 |
| CNN counterfactual accuracy | 0.838 |
| E4E SVM accuracy | 0.832 |
| Expert accuracy on synthetic (answered) | 0.90 |
| Images in dataset | 3,064 |
| Test set size | 911 |

---

## Emergency Phrases

If you don't understand a question:
> "Could you rephrase the question, please?"

If you don't know the exact number:
> "The exact value is in Table X.X of the thesis, but the key finding is..."

If you need time to think:
> "That's a good question. Let me think about that for a moment."

If challenged on a limitation:
> "I agree, and that's why I included it in the future work section."

If asked something outside your scope:
> "That's an interesting direction for future research, but it was outside the scope of this thesis."
