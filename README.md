# VANTAGE-STR: Advanced Forensic DNA Intelligence Platform

![License](https://img.shields.io/badge/license-MIT-emerald)
![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Status](https://img.shields.io/badge/status-Production%20Ready-success)

**VANTAGE-STR** is a cutting-edge forensic intelligence system designed for high-stakes DNA profiling, kinship analytics, and biometric reconstruction. It bridges the gap between biological analysis and tactical intelligence by combining rigorous **Likelihood Ratio (LR)** statistical models with **Generative AI** and **Blockchain** immutability.

---

## 🛡️ Core Capabilities

### 1. Forensic Identity Panel
*   **Real-Time Phenotype Prediction**: Deterministic mapping of SNP markers (HERC2, MC1R, SLC24A5) to physical traits.
    *   **Ocular Pigmentation**: Eye color prediction (Blue, Hazel, Brown).
    *   **Dermal Classification**: Skin tone estimation (Type I-VI).
    *   **Hair Morphology**: Texture and color analysis.
*   **Coherence & Reliability**: Automated "Sync Status" scoring to validate genetic data against predicted phenotypes.
*   **Visual Forensics**: High-fidelity, dark-mode UI for rapid field assessment.

### 2. Geo-Forensic Intelligence
*   **Ancestry Heatmaps**: Interactive global map visualizing probable ancestral origins using k-NN and Random Forest classifiers.
*   **Region Targeting**: Auto-flyTo navigation focusing on high-probability geographic clusters.
*   **Confidence Rings**: Dynamic visualization of 95% confidence zones for geolocation.

### 3. Kinship & Pedigree Analytics
*   **Familial Relationship Detection**: Instant calculation of Kinship Indices (KI) for Parent-Child, Sibling, and Half-Sibling relationships.
*   **Pedigree Tree Visualization**: Graph-based rendering of multi-generational family links.
*   **Population Correction**: Balding-Nichols NRC II model integration for accurate statistical weighting.

### 4. Blockchain Audit Ledger
*   **Immutable Integrity**: Every analysis is hashed and anchored to the **Polygon/Sepolia** blockchain.
*   **On-Chain Verification**: Direct links to Etherscan for independent auditability of forensic reports.
*   **Zero-Knowledge Proofs (ZKP)**: Private verification of DNA matches without revealing raw genetic data (Circom/SnarkJS).

---

## 🏗️ Technical Architecture

### Frontend (Tactical UI)
*   **Framework**: Next.js 14 (App Router)
*   **Styling**: Tailwind CSS v4, Framer Motion (Animations)
*   **Maps**: Leaflet.js / React-Leaflet
*   **State**: React Hooks / Context API

### Backend (Intelligence Core)
*   **API**: FastAPI (Python 3.10+)
*   **ML Engines**: PyTorch, Scikit-learn (Phenotype/Ancestry Models)
*   **Database**: Supabase / PostgreSQL (Structured Data) & Milvus (Vector Embeddings)
*   **Agents**: DSPy for strategic reasoning and report generation

### Blockchain (Verification Layer)
*   **Smart Contracts**: Solidity (Hardhat Development Environment)
*   **Integration**: Wagmi / Viem hooks
*   **Network**: Polygon Amoy / Sepolia Testnet

---

## 🚀 Getting Started

### Prerequisites
*   **Docker & Docker Compose**
*   **Node.js 20+**
*   **Python 3.10+**

### 1. Infrastructure Setup
Spin up the local services (Database, Redis, Milvus):
```bash
docker-compose up -d
```

### 2. Backend Installation
Navigate to the backend directory and install dependencies:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 3. Frontend Installation
Navigate to the frontend directory and start the development server:
```bash
cd frontend
npm install
npm run dev
```

The dashboard will be available at `http://localhost:3000`.

---

## 📄 Documentation

### Security & Privacy
VANTAGE-STR is built with **Privacy by Design**.
*   **Data Isolation**: Raw STR markers are processed in secure, isolated environments.
*   **ZKP**: Identity verification uses Zero-Knowledge Proofs to prevent genetic data leakage.
*   **Audit Trails**: All access and modification events are logged to the immutable ledger.

### Contributing
We welcome contributions from the forensic and open-source community. Please read `CONTRIBUTING.md` for our code of conduct and pull request process.

---

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for more information.

---

## 👨‍💻 Author
**Yusuf Çalışır**
*   **LinkedIn**: [Yusuf Çalışır](https://www.linkedin.com/in/yusufcalisir/)
*   **Portfolio**: [yusufcalisir.me](https://yusufcalisir.me)

> *Deployed for the future of forensic technology.*