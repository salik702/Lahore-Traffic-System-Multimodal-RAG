<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=220&section=header&text=LAHORE%20TRAFFIC%20AI&fontSize=60&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Multimodal%20RAG%20Intelligence%20System%20%E2%80%A2%20Vision%20%2B%20Text%20%2B%20Knowledge%20Graphs&descAlignY=55&descSize=18&descColor=93c5fd" width="100%"/>

<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=24&duration=2600&pause=700&color=818CF8&center=true&vCenter=true&repeat=true&width=800&height=60&lines=Intelligent%20Traffic%20Analysis;Vision-Language%20RAG%20Pipeline;Accident%20Report%20Intelligence;Knowledge-Graph%20Grounded%20Insights" alt="Typing SVG" />
</a>

<br/>

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16-000000?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org/)
[![Groq](https://img.shields.io/badge/Groq-LPU-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=ultralytics&logoColor=black)](https://ultralytics.com/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)

**A production-grade AI system bridging the gap between raw traffic sensor data, accident reports, and actionable urban intelligence.**

[Explore the Architecture](#-system-architecture) • [Installation Guide](#-installation-and-setup) • [Core Features](#-core-features)

</div>

---

## 📌 Overview

**Lahore Traffic Intelligence AI** is a sophisticated multimodal Retrieval-Augmented Generation (RAG) system designed to analyze, synthesize, and query complex urban traffic datasets. Unlike traditional text-based RAG, this system implements a **Vision-Language pipeline** that integrates Computer Vision (YOLOv8), semantic image embeddings (CLIP), and structured Knowledge Graphs to provide grounded insights into traffic patterns and incident reporting.

The system transforms unstructured data—including accident PDFs, traffic sign imagery, and official penalty schedules—into a searchable, high-fidelity knowledge base.

## ✨ Core Features

### 👁️ Multimodal Perception Pipeline
- **Object Detection**: Integration of **YOLOv8** for high-precision detection of vehicles, motorcycles, and riders.
- **Specialized Vision Modules**: Dedicated detectors for **Helmet Compliance** and **License Plate Localization**.
- **OCR Intelligence**: Integration of **RapidOCR (PP-OCRv4)** for extracting text from license plates and traffic signage.
- **Semantic Image Search**: CLIP-based embeddings allowing users to retrieve visually similar accident evidence via natural language.

### 🧠 Intelligence Engine
- **Hybrid RAG-Anything**: A dual-retrieval strategy combining **Naive Vector Search** (semantic chunks) with **Graph-based Retrieval** (relational entities).
- **Knowledge Graph Integration**: Automatic extraction of entities and relationships, exportable via **GraphML** for structural analysis in tools like Gephi.
- **Groq-Accelerated Inference**: Leveraging LPU technology for ultra-low latency LLM responses, ensuring real-time query performance.
- **Strict Grounding**: A rigorous evidence-boundary system that prevents model hallucination by strictly adhering to sourced accident reports and official regulations.

### 🛠️ Evidence & Legal Pipeline
- **Violation Analysis**: Deterministic rules engine that maps visual detections (e.g., "no helmet") to official Punjab traffic fines.
- **Automated Report Generation**: Synthesis of visual evidence, detected violations, and legal citations into a professional report format.
- **Provenance Tracking**: Every insight is linked back to a specific source ID, page number, or image URL.

---

## 🧱 System Architecture

```mermaid
flowchart TD
    subgraph Data_Layer [Data Layer]
        A1[Accident PDFs] --> B
        A2[Traffic Sign Images] --> B
        A3[Official Fine JSONs] --> B
        A4[Accident Reports] --> B
    end

    subgraph Pipeline [Multimodal Ingestion Pipeline]
        B[Ingestion Engine] --> C1[YOLOv8 / RapidOCR]
        B --> C2[CLIP Embeddings]
        B --> C3[LightRAG Parser]
        C1 & C2 & C3 --> D[Multimodal Vector Store]
        D --> E[Knowledge Graph]
    end

    subgraph Retrieval [Intelligence Interface]
        F[User Query] --> G[Hybrid Retriever]
        E --> G
        D --> G
        G --> H[Contextual Prompt]
        H --> I[Groq LLM]
        I --> J[Grounded Intelligence Insight]
    end

    style Data_Layer fill:#f9f9f9,stroke:#333,stroke-width:2px
    style Pipeline fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style Retrieval fill:#fff3e0,stroke:#e65100,stroke-width:2px
```

---

## 🚀 Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend** | `Next.js 16`, `React 19`, `Tailwind CSS 4` | High-performance, motion-enhanced monitoring dashboard |
| **Backend** | `FastAPI`, `Pydantic` | Asynchronous API orchestration and state management |
| **RAG Framework** | `RAG-Anything`, `LightRAG` | Multimodal retrieval and Graph-based knowledge management |
| **LLM Engine** | `Groq` (LPU) | Ultra-fast inference for grounded text generation |
| **Vision** | `YOLOv8`, `ONNX Runtime` | Real-time object detection and violation analysis |
| **OCR** | `RapidOCR` | Text extraction from plates and signage |
| **Embeddings** | `Sentence-Transformers` | Local semantic vectorization (`all-MiniLM-L6-v2`) |
| **Storage** | `SQLite`, `NumPy (.npz)` | Conversation history and image embedding storage |

---

## 🔧 Installation and Setup

### 📋 Prerequisites
- **Python**: $\ge 3.12, < 3.14$
- **Node.js**: $\ge 18.0.0$
- **Hardware**: GPU recommended for YOLOv8 inference (CPU fallback supported via ONNX)

### 🛠️ Backend Setup
```bash
# Navigate to backend
cd backend

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r pyproject.toml
```

**Environment Configuration**
Create a `.env` file in the `backend/` directory:
```env
GROQ_API_KEY=your_groq_api_key_here
RAG_ENABLED=True
RAG_QUERY_MODE=hybrid
BACKEND_CORS_ORIGINS=http://localhost:3000
```

### 💻 Frontend Setup
```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

---

## 📖 Usage Examples

### 1. Incident Intelligence Query
**User**: *"What are the common causes of accidents in the Gulberg area of Lahore?"*  
**System**: Retrieves accident reports from the vector store $\to$ Extracts patterns from the Knowledge Graph $\to$ Synthesizes a response grounded in official reports.

### 2. Visual Evidence Analysis
**User**: (Uploads image of a motorcycle rider)  
**System**: 
1. **Detection**: Identifies `motorcycle` and `person`.
2. **Analysis**: Checks for `helmet` $\to$ Result: `Negative`.
3. **Legal Mapping**: Maps "No Helmet" to `Punjab Traffic Fine Schedule` $\to$ Identifies PKR amount.
4. **Output**: Generates a report with visual crops as evidence.

---

## 📁 Project Directory Structure

```text
Lahore-Traffic-AI/
├── backend/               # FastAPI Core
│   ├── app/
│   │   ├── api/           # REST Endpoints
│   │   ├── core/          # Global Config & Settings
│   │   ├── ingestion/     # Multimodal Data Pipeline
│   │   ├── rag/           # RAG-Anything / LightRAG Engine
│   │   └── services/      # Vision, Evidence, & Database Logic
│   ├── models/            # YOLOv8 & OCR Weights
│   └── rag_storage/       # Vector & Graph Indices
├── frontend/              # Next.js 16 Dashboard
│   ├── app/               # App Router & Pages
│   └── components/        # UI Components (Tailwind/Motion)
├── scripts/               # Data Acquisition & Eval Tools
├── data/                  # Raw & Processed Datasets
└── graphs/                # Exported .graphml Visualizations
```

---

## ⚠️ Technical Limitations & Considerations

<details>
<summary><b>Click to view Model Constraints</b></summary>

- **Temporal Data**: Single images do not establish movement; findings are static and require manual review for behavioral violations.
- **Class Granularity**: The current COCO-based weights do not reliably distinguish between vans and rickshaws.
- **OCR Accuracy**: Plate text is unverified and must be cross-checked against the provided image crop.
- **Context**: Punjab fine amounts are based on the most recently available structured registry; local jurisdictional variations may exist.
</details>

---

## 🤝 Contributing

Contributions to improve the **Vision-RAG** accuracy or expand the **Knowledge Graph** depth are welcome.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'feat: Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is open-source and provided "as-is" for research and public safety enhancement.

<div align="center">

### Built with obsession by **Salik Ahmad** 🤖

<p>
  <a href="https://salikahmad.vercel.app/" target="_blank">
    <img src="https://img.shields.io/badge/Website-salikahmad.vercel.app-818CF8?style=for-the-badge&logo=vercel&logoColor=white&labelColor=0d0d0d" alt="Website" />
  </a>
  <a href="https://www.linkedin.com/in/salik-ahmad-programmer/" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-Salik%20Ahmad-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white&labelColor=0d0d0d" alt="LinkedIn" />
  </a>
  <a href="https://github.com/SalikAhmad702" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-SalikAhmad702-181717?style=for-the-badge&logo=github&logoColor=white&labelColor=0d0d0d" alt="GitHub" />
  </a>
</p>

<br/>

<a href="https://salikahmad.vercel.app/">
  <img src="https://readme-typing-svg.herokuapp.com?font=JetBrains+Mono&size=16&duration=4000&pause=1000&color=818CF8&center=true&vCenter=true&width=700&lines=AI%2FML+Engineer;Traffic+Intelligence+Specialist;Multimodal+RAG+Architect;Copyright+©+2026+Salik+Ahmad." alt="Footer Typing" />
</a>

<br/>

<img src="https://capsule-render.vercel.app/api?type=shark&color=0:0f0c29,50:302b63,100:24243e&height=140&section=footer" width="100%"/>

</div>
