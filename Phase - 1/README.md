# ⚖️ AI-Driven Court Case Prioritization System

## 📌 Overview
This project is an AI-based judicial decision-support system designed to help courts prioritize cases based on fairness, urgency, and delay.

Instead of following traditional FIFO (First-In-First-Out), the system identifies:
- Over-detained undertrial prisoners
- Long-pending cases
- Vulnerable individuals needing urgent attention

---

## 🎯 Problem Statement
Judicial systems face:
- Large case backlogs
- Delayed hearings
- Undertrial prisoners serving beyond their sentence

Existing systems only track cases and do not:
- Prioritize intelligently
- Detect unfair detention
- Assist decision-making

---

## 🚀 Solution
We built a feature-engineered clustering system that:
- Converts raw case data into structured indicators
- Detects critical cases automatically
- Groups cases into priority levels
- Supports explainable decision-making

---

## 🧠 System Architecture

Raw Data → Cleaning → Feature Engineering → Feature Vector → Clustering → Priority Levels

---

## 📊 Dataset

### Type
Hybrid dataset (real-structure inspired + synthetic features)

### Based on
- National Judicial Data Grid (NJDG)
- eCourts system
- NCRB crime data
- Indian Kanoon legal data

### Why Hybrid?
No public dataset contains:
- detention duration
- sentence limits
- priority labels

So we created a legally grounded dataset.

---

## 🧹 Data Preprocessing

- Text normalization (IPC extraction using regex)
- Date standardization
- Missing value handling
- Category normalization
- Outlier removal
- Deduplication

---

## ⚙️ Feature Engineering

### 1. Detention & Fairness
- detention_days
- expected_sentence_days
- detention_ratio = detention / sentence
- overstay_flag

---

### 2. Legal Features
- ipc_section
- severity_score
- bailable / non-bailable
- legal_status

---

### 3. Temporal Features
- days_pending
- last_hearing_gap
- stale_case_flag

---

### 4. Vulnerability Features
- age
- gender
- health_flag
- disability_flag
- vulnerability_score

---

### 5. NLP Features
- urgency_nlp
- bail_eligibility_nlp
- case_complexity

---

### 6. Derived Metrics
- detention_ratio_exp_norm
- offense_severity_index

---

## 🔥 Key Feature

### Overstay Detection
- overstay = detention_days > expected_sentence_days
  
This identifies cases where a person has served more time than legally allowed.

---

## 🧩 Clustering Model

We use:
- KMeans / DBSCAN

### Input Features
- detention_ratio, days_pending, severity_score,
- vulnerability_score, legal_status, bailable


---

## 🏷️ Priority Levels

| Priority | Description |
|---------|------------|
| 🔴 Critical | Overstay / illegal detention |
| 🟠 High | Near limit + long delay |
| 🟡 Medium | Normal cases |
| 🟢 Low | Low urgency |

---

## 📈 Decision Logic
- IF overstay → CRITICAL
- ELSE IF high detention ratio + delay → HIGH
- ELSE IF moderate → MEDIUM
- ELSE → LOW


---

## 💡 Key Innovations

- Feature-based legal reasoning
- Overstay detection
- Fairness-aware prioritization
- Explainable AI (no black-box)
- Clustering-based classification

---

## 🏛️ Comparison

| Feature | Existing Systems | Our System |
|--------|----------------|-----------|
| Case tracking | Yes | Yes |
| AI prioritization | No | Yes |
| Overstay detection | No | Yes |
| Clustering | No | Yes |
| Fairness logic | No | Yes |

---

## ⚠️ Limitations

- Uses synthetic-enhanced dataset
- No real-time court integration
- Needs legal validation

---

## 🔮 Future Scope

- Real-time court integration
- Blockchain audit system
- AI-based document analysis
- Automated scheduling engine

---

## 🏁 Conclusion

This system transforms judicial workflows from:
passive tracking → intelligent prioritization

---

## 📢 One-Line Summary

AI system that identifies unfairly delayed and over-detained cases and prioritizes them for faster justice.
