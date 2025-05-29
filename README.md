# VOC Auto-Tagging System

## 📋 Overview

The VOC Auto-Tagging System is an intelligent tool designed to analyze large-scale user feedback. It automatically identifies key touchpoints in the user journey, issue types, and sentiment polarity, helping teams gain actionable insights efficiently.

## ✨ Key Features

* **Smart Tagging**: Automatically detects user journey stages, issue types, and sentiment.
* **Batch Processing**: Supports bulk data import via Excel or CSV.
* **Visualization Reports**: Generates visual dashboards and summary reports.
* **User-Friendly Interface**: No technical background required to operate.

---

## 🎯 Tagging Framework

### **User Journey Touchpoints** (multi-label)

* **Pre-Purchase**: Brand perception, website/App experience, in-store inquiry/test drive
* **Purchase Process**: Ordering flow, sales service attitude, pricing transparency
* **Delivery**: Handover speed/process, onboarding training, store environment
* **Driving Experience**: Acceleration, braking, handling, comfort, noise
* **Smart Features**: Navigation, voice assistant, HUD/control interaction, OTA
* **Charging & Energy**: Home charging, public stations, range performance
* **After-Sales Service**: Maintenance, customer support, trade-in

### **Issue Types** (multi-label)

* Stability | Performance | Usability | Compatibility
* Aesthetics | Interaction Logic | Safety | Service Experience | Expectation Gap

### **Sentiment Polarity** (single-label)

* Positive | Neutral | Negative

---

## 🚀 Quick Start

### **System Requirements**

* Windows 10 or later
* Recommended: 8GB+ RAM
* Stable internet connection

### **Launch Instructions**

**Method 1 (Recommended):**

* Double-click `voc_launcher.ps1` in the project root.

**Method 2:**

* Right-click `voc_launcher.ps1` → Run with PowerShell.

On first run, the system will auto-install required environments.

---

## 💡 Usage Guide

### **Choose Mode on Startup**

```
🎯 VOC Analyzer v3.0 - Main Menu

1. 🎪 Demo Mode (Quick preview)
2. 🔤 Interactive Mode (Recommended)
3. 🎓 Train New Model
4. 📊 Batch Process Files
5. 📈 Model Performance Evaluation
6. 🔄 System Initialization
7. ❌ Exit
```

### **Single Text Analysis**

Go to `Interactive Mode` → `Analyze a Single Text`.

Example input:

> "The voice assistant of Li Auto ONE is accurate, but the navigation sometimes takes a longer route."

**Output:**

* Touchpoints: Smart Navigation, Voice Assistant
* Issues: Usability
* Sentiment: Neutral
* Confidence Scores Included

---

## 📊 Batch File Processing

### Step 1: Prepare Input File

CSV or Excel format:

```
text,notes  
"The voice assistant is accurate, but the navigation is off-route.", Feedback 1  
"Charging is slower than advertised.", Feedback 2  
"Sales service was great, smooth delivery process.", Feedback 3  
```

### Step 2: Run Batch Mode

Choose `Batch Process Files` and follow prompts:

* File Path
* Text Column Name (default: `text`)
* Generate Report? (recommended: Yes)

### Step 3: View Results

* Tagged CSV output
* HTML report with visual analytics

---

## 🎓 Model Training

To train with custom data:

**Format Example:**

```
text,touchpoints,issue_types,sentiment  
"Navigation often takes longer routes, voice not recognized","Smart Navigation,Voice Assistant","Stability,Usability",Negative  
"Delivery staff gave detailed instructions, seat massage was impressive","Delivery Training,Suspension Comfort",,Positive  
```

**Notes:**

* Use commas to separate multiple labels
* Min. 500 labeled samples recommended
* Ensure clean, accurate labeling

---

## 📁 File Structure

```
voc-analyzer/
├── voc_launcher.ps1           # Launcher script
├── README.md                  # Documentation
├── src/                       # Core source files
├── data/
│   ├── voc_sample_data.csv    # Sample data
│   ├── models/                # Trained models
│   ├── logs/                  # System logs
│   └── reports/               # Generated reports
└── config/                    # Configuration files
```

---

## 🛠️ Troubleshooting

### Startup Issues

**Problem**: Cannot launch

* Confirm Windows 10+
* Right-click → "Run with PowerShell"

**Problem**: Execution policy error

* Run PowerShell as Admin:
  `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`
  Confirm with `Y`, then relaunch.

### Usage Issues

**File read error**

* Ensure file is CSV/XLSX and not open elsewhere
* File must be UTF-8 encoded

**Inaccurate tagging**

* Ensure clear and complete input text
* Consider retraining the model for domain-specific language

**Slow performance**

* First run may download models
* For large datasets, process in batches
* Check your network connection

---

## 📋 Output Details

### CSV Output

* Original text
* Predicted touchpoints
* Predicted issue types
* Sentiment polarity
* Confidence scores

### HTML Report

* Summary stats
* Charts by category
* Correlation heatmaps
* Quality diagnostics

---

## ✅ Best Practices

### Data Preparation

* Ensure clean, well-written text
* One feedback per line
* Remove irrelevant symbols or formatting

### Batch Processing

* Recommend ≤5000 entries per batch
* Split larger files for stability
* Clear temp files regularly

### Interpreting Results

* Focus on high-confidence results
* Review predictions below 0.5 manually
* Retrain periodically with new data

---

*Last updated: May 26, 2025*

---
