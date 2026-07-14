# Credit Card Fraud Detection: Orange3 Final Project

> A machine learning workflow built in Orange3 to detect fraudulent credit card transactions, comparing three different models to see which one actually catches fraud best.

---

## Table of Contents
- [About This Project](#about-this-project)
- [The Problem](#the-problem)
- [The Dataset](#the-dataset)
- [My Workflow](#my-workflow)
- [Metrics I Used](#metrics-i-used-and-why)
- [Results](#results)
- [Which Model I Picked](#which-model-i-picked-and-why)
- [What I Learned](#what-i-learned)
- [Files in This Repo](#files-in-this-repo)

---

## About This Project
For my final project, I built a machine learning workflow in Orange3 to detect fraudulent credit card transactions. The goal was to take an imbalanced dataset and see how well different machine learning models could actually catch fraud.

## The Problem
Credit card fraud is a real issue for banks and their customers, and one of the biggest challenges is that fraud is *rare* compared to normal transactions. In my dataset, only about **0.17%** of transactions were actually fraud. That makes this a much harder problem than it sounds, because a model can look "accurate" while still missing almost all the fraud, so I had to be careful about which metrics actually meant something.

## The Dataset

| | |
|---|---|
| **Source** | Kaggle Credit Card Fraud Detection dataset (anonymized European cardholder transactions, September 2013) |
| **Original size** | 284,807 transactions, too large for Orange to run smoothly |
| **Working sample** | `savedata.csv`, 14,241 transactions, stratified so it keeps the same ~0.17% fraud rate |
| **Features** | `Time`, `V1`–`V28` (anonymized PCA components), `Amount`, and `Class` (target) |
| **Class labels** | `0` = Legitimate, `1` = Fraud |
| **Problem type** | Binary Classification |

---

## My Workflow

### Step 1: Split the Data
The original file was too big for Orange to handle smoothly, so I used a Data Sampler to shrink it down to a manageable size and saved it as a new file (`savedata.csv`) to use for the rest of the project.

<img width="1069" height="265" alt="Split Data" src="https://github.com/user-attachments/assets/4a2c5403-83c2-43ba-a340-5528a38b837d" />

### Step 2: Explore the Data
Before building the workflow, I needed to understand my data first:

| Widget | What it told me |
|---|---|
| Data Info / Column Statistics | Each column's type, range, and confirmed there were no missing values |
| Data Table | The raw rows |
| Distributions | How many transactions were fraud vs. legit, and just how imbalanced it really was |
| Box Plot | How much money was involved in fraud vs. legit transactions |
| Scatter Plot | Whether fraud and legit transactions visually clustered apart from each other |
| Rank | Scored every feature by how well it predicts fraud, using two methods (Gini and ANOVA), so I could see which columns actually mattered |

<img width="1111" height="669" alt="Data Analysis" src="https://github.com/user-attachments/assets/2e81a7f4-f650-4f88-a95b-6915c95aa0ad" />

### Step 3: Build and Test the Models

1. **Set Class as target** so Orange knew what I was actually trying to predict
2. **Rank → top 10 features**, narrowed down to the 10 most useful columns instead of using all 30
3. **Preprocess → normalize to [0, 1]**, put every feature on the same scale so no single column (like Amount) could dominate just because its raw numbers are bigger
4. **Three models**, Logistic Regression, Random Forest, and kNN, all trained on the same data so I could fairly compare different approaches
5. **Test and Score (5-fold cross-validation)**, tested each model on data it hadn't already seen, so the results are honest and not just memorized answers
6. **Confusion Matrix**, broke each model's results down into True Positives, False Positives, True Negatives, and False Negatives
7. **ROC Analysis**, visualized the tradeoff each model makes between catching fraud and raising false alarms

<img width="1134" height="866" alt="Final Workflow" src="https://github.com/user-attachments/assets/7d569765-46a5-4e1c-b9a9-b4f6b5e8c2b7" />

## Metrics I Used, and Why

Accuracy alone doesn't work on a dataset this imbalanced. A model that never predicts fraud would still be "right" 99.8% of the time. So instead, I focused on:

- **Recall**, out of all the real fraud cases, what % did the model actually catch
- **F1**, balances Precision and Recall together, so a model can't fake a good score by being lopsided
- **MCC**, the most reliable single score for imbalanced data, since it considers all four outcomes (correctly caught fraud, correctly cleared legit, missed fraud, false alarms) at once

---

## Results

| Model | AUC | F1 | Precision | Recall | MCC | Fraud Caught (out of 24) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.969 | 0.438 | 0.875 | 0.292 | 0.505 | 7 |
| Random Forest | 0.854 | 0.667 | 0.778 | 0.583 | 0.673 | 14 |
| **kNN** | 0.875 | **0.714** | 0.833 | **0.625** | **0.721** | **15** |

> **Note:** Random Forest's exact numbers can shift slightly between runs since it involves some randomness. The overall conclusion stays the same.

---

## Which Model I Picked, and Why

I went with **kNN** as my best model. It had the highest F1-score, Recall, and MCC of the three, meaning it was best at actually catching real fraud without falling apart on the rare class.

Logistic Regression technically had the best AUC and Precision, but that's a bit misleading here: at the actual decision threshold being used, it only caught 7 out of 24 real fraud cases, compared to kNN's 15. In a real banking situation, missing fraud is way more costly than occasionally flagging a normal transaction for review, so I think kNN is the more practically useful model here.

---

## What I Learned
Going through this project taught me that picking the "best" model isn't as simple as looking at one score. Different metrics can tell different results, especially when your data is imbalanced. Accuracy alone would've been useless here, and even AUC ended up being misleading once I took a closer look at what was actually happening at the real decision threshold.

---

## Files in This Repo

| File | Description |
|---|---|
| `finalworkflow.ows` | My complete Orange3 workflow, open it directly in Orange3 to see everything |
| `savedata.csv` | The 14,241-row sampled dataset used throughout this project. Needed for the File widgets in `finalworkflow.ows` to work correctly |
| `README.md` | This summary of the project and results |

**To run this workflow yourself:** download `finalworkflow.ows` and `savedata.csv` into the same folder, then open the `.ows` file in [Orange3](https://orangedatamining.com/). If the File widgets show an error, double-click them and browse to wherever you saved `savedata.csv`.

**Note on the original dataset:** the full 284,807-row original dataset was too large to upload here (over GitHub's 100MB limit). It's publicly available on Kaggle: [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Tools
Built entirely using [Orange3](https://orangedatamining.com/), a visual data mining and machine learning tool.
