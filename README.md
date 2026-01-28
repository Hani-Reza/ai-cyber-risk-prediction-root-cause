# 🛡️ Cybersecurity Risk Prediction with Root Cause Analysis

<div align="center">

![Python](https://img.shields.io/badge/python-3.12.7-blue)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)
![ML](https://img.shields.io/badge/ML-Random%20Forest-orange)
![UAE](https://img.shields.io/badge/context-UAE%20Enterprise-green)
![License](https://img.shields.io/badge/license-MIT-yellow)

**Production-grade AI/ML system for SOC operations with explainable risk predictions**

</div>

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Installation & Setup](#-installation--setup)
- [Usage Examples](#-usage-examples)
- [Project Structure](#-project-structure)
- [ML Methodology](#-ml-methodology)
- [Sample Outputs](#-sample-outputs)
- [UAE Enterprise Compliance](#-uae-enterprise-compliance)
- [Interview Demonstration](#-interview-demonstration)

## 🎯 Project Overview

**Enterprise Problem**: SOC analysts face alert fatigue from thousands of security events daily, with limited tools to prioritize risks or explain why events are flagged.

**Our Solution**: A production-ready AI/ML system that:
1. **Predicts cybersecurity risk** using machine learning
2. **Categorizes risks** as LOW/MEDIUM/HIGH with SOC-aligned thresholds
3. **Explains root causes** with human-readable explanations
4. **Supports both CLI** (engineering validation) and **Streamlit UI** (stakeholder demo)

**Target Role**: AI/ML Engineer (model + logic heavy)
**Domain**: Cybersecurity Risk Prediction & Root Cause Analysis
**Region Context**: UAE Enterprise/Government Grade

## ✨ Key Features

### 🏗️ Production Engineering
- **Modular architecture** with clear separation of concerns
- **Deterministic execution** (seeded randomness for reproducibility)
- **Full ML pipeline** from data generation to deployment
- **Enterprise-grade error handling** and logging

### 🤖 ML & AI Capabilities
- **Random Forest Classifier** optimized for cybersecurity patterns
- **Feature engineering** with domain-specific risk indicators
- **Explainable AI** without external dependencies (SHAP-style)
- **Confidence scoring** for risk predictions

### 🎯 Cybersecurity Domain Integration
- **SOC-aligned risk thresholds** (LOW/MEDIUM/HIGH/CRITICAL)
- **UAE business context** multipliers (privilege, criticality, timing)
- **Attack pattern detection** (Brute Force, Credential Stuffing, etc.)
- **Root cause analysis** with mitigation recommendations

### 💻 Dual Interface
- **CLI application** for batch processing and engineering validation
- **Streamlit dashboard** for stakeholder demonstrations and SOC operations
- **JSON/CSV export** for integration with existing SOC tools

## 🏗️ System Architecture
┌─────────────────────────────────────────────────────────────────────────┐
│                    CYBERSECURITY RISK ANALYZER v1.0                     │
│                        Production Architecture                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   DATA      │    │   FEATURE   │    │   ML        │    │   RISK      │
│   GENERATION│    │   ENGINE    │    │   ENGINE    │    │   ANALYTICS │
│   LAYER     │    │   LAYER     │    │   LAYER     │    │   LAYER     │
├─────────────┤    ├─────────────┤    ├─────────────┤    ├─────────────┤
│• Deterministic│  │• Data       │    │• Model      │    │• Threshold  │
│  SOC event   │  │  validation  │    │  training   │    │  logic      │
│  simulation  │  │• Feature     │    │• Cross-val  │    │• Risk       │
│• Seeded      │  │  engineering │    │  scoring    │    │  scoring    │
│  randomness  │  │• Categorical │    │• Persistence│    │• SOC-aligned│
│• Realistic   │  │  encoding    │    │  (pickle)   │    │  categories │
│  distributions│  │• Scaling     │    │             │    │             │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                   │                   │
       ▼                  ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA FLOW PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ 1. GENERATE → 2. PROCESS → 3. TRAIN → 4. SCORE → 5. EXPLAIN → 6. OUTPUT │
│                                                                         │
│    ┌─────┐      ┌─────┐     ┌─────┐    ┌─────┐     ┌─────┐    ┌─────┐  │
│    │10K  │      │Clean│     │Model│    │Risk │     │Root │    │LOW/ │  │
│    │SOC  │─────▶│Feat.│────▶│Fit  │───▶│Level│────▶│Cause│───▶│MED/ │  │
│    │Events│     │Eng. │     │Eval.│    │Logic│     │Anal.│    │HIGH │  │
│    └─────┘      └─────┘     └─────┘    └─────┘     └─────┘    └─────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────┐    ┌─────────────────────────┐
│         CLI INTERFACE (Phase 7)         │    │ STREAMLIT UI (Phase 8)  │
├─────────────────────────────────────────┤    ├─────────────────────────┤
│• Batch processing                       │    │• File upload            │
│• Real-time simulation                   │    │• Visualization          │
│• SOC analyst workflow                   │    │• Stakeholder demo       │
│• Engineering validation                 │    │• No ML logic            │
│• Production logs                        │    │• Thin wrapper           │
└─────────────────────────────────────────┘    └─────────────────────────┘


## 🚀 Installation & Setup

### Prerequisites
- **Python 3.12.7** (64-bit)
- **Windows** (optimized for VS Code terminal)
- **8GB RAM** minimum

### Quick Installation (VS Code Terminal)

# 1. Clone repository
git clone https://github.com/yourusername/ai-cyber-risk-prediction-root-cause.git
cd ai-cyber-risk-prediction-root-cause

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate.bat

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate data and train model (one-time setup)
python src\data_generator.py
python src\preprocessing.py
python src\model_training.py

### 📖 Usage Examples
1. CLI Application (SOC Analyst Workflow)
bash
# Check system status
python cli\app.py status

# Analyze single event
python cli\app.py analyze --failed-logins 15 --login-velocity 25.7 --ip-reputation 20

# Batch process CSV file
python cli\app.py process --input data\raw\security_events.csv --output reports\analysis.json

# Simulate real-time monitoring
python cli\app.py monitor --count 10 --interval 2

# Generate sample data
python cli\app.py sample --count 100 --output data\sample_events.csv

2. Streamlit Dashboard (Stakeholder Demo)
bash
streamlit run visualization\streamlit_app.py
# Opens at http://localhost:8501


## Dashboard Pages:

📊 Dashboard Overview: Executive summary and KPIs

📁 Upload & Analyze: Batch CSV processing

🔍 Single Event Analysis: Real-time investigation

📈 Risk Trends: Historical analysis

⚙️ System Configuration: Admin and compliance info


## 📁 Project Structure
ai-cyber-risk-prediction-root-cause/
├── data/                           # Data storage
│   ├── raw/                       # Original generated data
│   └── processed/                 # Processed features
├── models/                        # Trained artifacts
│   ├── risk_classifier.pkl       # ML model
│   ├── preprocessor.pkl          # Feature transformer
│   └── model_metadata.json       # Training metadata
├── src/                          # Core business logic
│   ├── data_generator.py         # Phase 2: SOC event simulation
│   ├── preprocessing.py          # Phase 3: Feature engineering
│   ├── model_training.py         # Phase 4: ML training
│   ├── risk_scoring.py           # Phase 5: Risk categorization
│   ├── root_cause_analysis.py    # Phase 6: Explainability
│   └── utils.py                  # Shared utilities
├── cli/                          # Command-line interface
│   └── app.py                    # Phase 7: CLI orchestration
├── visualization/                # UI layer
│   └── streamlit_app.py          # Phase 8: Streamlit dashboard
├── reports/                      # Generated outputs
├── requirements.txt              # Python dependencies
├── install_vscode.bat            # Windows installer
└── README.md                     # This file

## 🧠 ML Methodology
Model Selection: Random Forest
Why Random Forest over alternatives?

✅ Non-linear patterns: Cyber attacks have complex feature interactions

✅ Feature importance: Critical for SOC analyst explainability

✅ Robustness: Handles imbalanced data (70% benign, 30% attacks)

✅ UAE compliance: Provides audit trail via feature contributions

Why NOT Deep Learning?

❌ Limited interpretability violates regulatory requirements

❌ 10K samples insufficient for deep learning

❌ Longer training/inference not justified for this problem


### Performance Metrics (Expected)
Metric	                Target	            Rationale
Recall	                >0.85	    Must detect majority of attacks
False Positive Rate	    <0.15	    Avoid SOC alert fatigue
ROC-AUC	                >0.85	    Good discrimination capability
CV Stability	        σ<0.05	    Consistent across data splits


### Feature Engineering
Engineered Feature	            Purpose	                        Cybersecurity Logic
compound_risk_score	            Weighted combination	        Mimics SOC analyst mental model
privilege_risk_multiplier	Amplifies risk for privileged accounts	UAE enterprise priority
non_business_hours_risk	        Flags unusual timing	        Attacks often outside 9-5 UAE time
device_ip_risk	        Combined device+IP risk	            Low trust + low reputation = high suspicion


## 📊 Sample Outputs
CLI Output Example

============================================================
EVENT ANALYSIS RESULTS
============================================================
Event ID: CLI_143025
Risk Level: HIGH
Probability: 0.723
Confidence: 0.856

Recommended Action: Immediate investigation - likely security incident

🎯 Most Likely Attack Pattern: Brute Force Attack

🔍 Top Contributing Factors:
  1. failed_login_attempts
     Value: 15
     Explanation: High number of failed attempts - possible brute force attack
     Contribution: 0.892
     Severity: HIGH

  2. login_velocity
     Value: 25.7
     Explanation: Unusually high login velocity - potential credential stuffing
     Contribution: 0.756
     Severity: HIGH


## 🇦🇪 UAE Enterprise Compliance
Regulatory Alignment
- Deterministic execution for audit compliance (seeded randomness)
- Feature importance tracking for regulatory explainability
- Business context integration for UAE enterprise priorities
- Full audit trail with timestamps and confidence scores

### SOC Operational Guidelines
Parameter	                    Value	            UAE Context
High-risk escalation time	    15 minutes	    UAE SOC SLA standard
Daily alert capacity	        100 events	    Prevents analyst fatigue
False positive tolerance	    20%	            Enterprise operational threshold
Business hours	                9 AM - 5 PM	    UAE timezone aligned


### Data Sovereignty
- All processing within project environment
- No external API calls or data exfiltration
- Generated data mimics UAE enterprise patterns
- Compliance with UAE IA and NESA frameworks


## 🔧 Development & Extension
Adding New Features
- New risk indicators: Add to data_generator.py and preprocessing.py
- Additional models: Extend model_training.py with new classifiers
- Custom thresholds: Modify risk_scoring.py business logic
- New visualizations: Add plots to streamlit_app.py

### Integration with Existing SOC Tools
- SIEM integration: CLI outputs JSON for Splunk/ArcSight ingestion
- Ticketing systems: Risk assessments can trigger Jira/ServiceNow tickets
- Threat intelligence: Enrich with external feeds via utils.py extensions

### Performance Optimization
- Batch processing: Already implemented for 10K+ events
- Model serialization: Joblib for efficient loading
- Caching: Streamlit session state for dashboard performance


## 👨‍💻 Author
Hani Reza
AI Engineer & Full-Stack Developer

https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white
https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white
https://img.shields.io/badge/Email-D14836?logo=gmail&logoColor=white

Looking For: AI/ML Engineering roles in UAE/GCC region with focus on government digital transformation.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
UAE Government for digital transformation inspiration

Streamlit team for the excellent web framework

Open-source community for continuous learning resources

<div align="center">
Built with ❤️ for the AI Engineering Community

Professional • Production-Ready • Portfolio Project

https://img.shields.io/github/stars/Hani-Reza/UAE-SMART-GOV-AI?style=social
https://img.shields.io/github/forks/Hani-Reza/UAE-SMART-GOV-AI?style=social

</div> ```
