<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=220&section=header&text=LAHORE%20TRAFFIC%20AI&fontSize=60&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Multimodal%20RAG%20Intelligence%20System%20%E2%80%A2%20Vision%20%2B%20Text%20%2B%20Knowledge%20Graphs&descAlignY=55&descSize=18&descColor=93c5fd" width="100%"/>

<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=20&duration=2600&pause=700&color=818CF8&center=true&vCenter=true&repeat=true&width=750&height=50&lines=Intelligent%20Traffic%20Analysis;Vision-Language%20RAG%20Pipeline;Accident%20Report%20Intelligence;Knowledge-Graph%20Grounded%20Insights" alt="Typing SVG" />
</a>

<br/>

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=next.js&logoColor=white" />
  <img src="https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white" />
  <img src="https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=ultralytics&logoColor=black" />
  <img src="https://img.shields.io/badge/Tailwind-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white" />
</p>

</div>

<br/>

## 📌 Overview

**Lahore Traffic Intelligence AI** is a sophisticated multimodal RAG (Retrieval-Augmented Generation) system designed to analyze and synthesize traffic data, accident reports, and visual evidence. Unlike standard text-based chatbots, this system integrates **Computer Vision (YOLOv8)**, **CLIP embeddings**, and **Knowledge Graphs** to provide grounded, intelligent insights into urban traffic patterns and incident reporting.

> Built to bridge the gap between raw sensor/image data and actionable intelligence, utilizing a high-performance pipeline of Vision-Language Models (VLMs) and Graph-based retrieval.

<br/>

## ✨ Features

<table>
<tr>
<td width="50%" valign="top">

### 👁️ Multimodal Perception
- **Visual Analysis**: YOLOv8 integration for traffic sign and object detection.
- **Semantic Image Search**: CLIP-based embeddings for retrieving visually similar accident evidence.
- **OCR Integration**: RapidOCR for extracting text from traffic signs and official reports.
- **Vision-RAG**: Querying image data alongside text for comprehensive evidence analysis.

</td>
<td width="50%" valign="top">

### 🧠 Intelligence Engine
- **RAG-Anything**: Flexible retrieval across text, images, and structured data.
- **Knowledge Graphs**: Exporting RAG relations to GraphML for structural traffic analysis.
- **Groq Acceleration**: Ultra-fast LLM inference for real-time traffic querying.
- **Grounded Generation**: Strict adherence to sourced accident reports and data.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 🛠️ Data Pipeline
- **Automated Acquisition**: Custom scripts for acquiring accident imagery and reports.
- **Knowledge Ingestion**: Processing raw PDFs and images into a searchable vector store.
- **Evaluation Suite**: Dedicated scripts to verify vision accuracy and chat groundedness.
- **Synthetic Augmentation**: Tools to improve and refresh vision assets.

</td>
<td width="50%" valign="top">

### 🎨 Modern Interface
- **Next.js 16 Architecture**: Cutting-edge frontend with React 19 and TypeScript.
- **Motion-Enhanced UI**: Glassmorphism and fluid animations via `motion`.
- **Dark Theme**: Professional, high-contrast interface for intelligence monitoring.
- **Responsive Design**: Optimized for both desktop dashboards and field devices.

</td>
</tr>
</table>

<br/>

## 🧱 System Architecture

```mermaid
flowchart LR
    A["Raw Data (PDF/Images)"] --> B["Ingestion Pipeline"]
    B --> C["YOLOv8 / CLIP / OCR"]
    C --> D["Multimodal Vector Store"]
    D --> E["Knowledge Graph"]
    F["User Query"] --> G["Hybrid Retriever"]
    E --> G
    D --> G
    G --> H["Contextual Prompt"]
    H --> I["Groq LLM"]
    I --> J["Traffic Intelligence Insight"]
```

<br/>

## 🛠️ Tech Stack

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Next.js-000000?style=flat-square&logo=next.js&logoColor=white" />
  <img src="https://img.shields.io/badge/Groq-F55036?style=flat-square&logo=groq&logoColor=white" />
  <img src="https://img.shields.io/badge/Ultralytics-00FFFF?style=flat-square" />
  <img src="https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=typescript&logoColor=white" />
</p>

**Core techniques applied:** Multimodal Embeddings · Vision-Language Models · RAG-Anything · GraphML Export · Object Detection · OCR

<br/>

## 📁 Project Structure

```
Lahore-Traffic-AI/
├── backend/               # FastAPI server, RAG logic, and Model workers
│   ├── app/               # Core application logic
│   ├── models/            # ML model weights and configs
│   └── rag_storage/       # Vector and Knowledge store
├── frontend/              # Next.js 16 dashboard
│   ├── app/               # Pages and Routing
│   └── components/        # UI components
├── scripts/               # Data acquisition and evaluation tools
├── data/                  # Raw and processed traffic datasets
└── graphs/                # Exported Knowledge Graphs (.graphml, .svg)
```

<br/>

## 🔧 Installation

### Backend Setup
```bash
cd backend
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r pyproject.toml # Or use uv sync
```

Create a `.env` file in `backend/`:
```
GROQ_API_KEY=your_groq_key_here
# Add other required environment variables
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

<br/>

## 📖 Usage

| Action | Component | Description |
|---|---|---|
| **Ingest Data** | `scripts/` | Run `acquire_data.py` to gather traffic and accident reports |
| **Index Vision** | `scripts/` | Use `index_images.py` to create CLIP embeddings |
| **Query System** | `Frontend` | Use the chat interface to ask about specific traffic zones |
| **Analyze Graph** | `graphs/` | Open `.graphml` files in Gephi to visualize accident relations |
| **Verify AI** | `scripts/` | Run `verify_assistant.py` to test groundedness |

<br/>

## 📊 Core Components

| Component | Responsibility |
|---|---|
| `embedding_worker.py` | Background process for generating multimodal embeddings |
| `rag_storage` | Manages the persistence of text and image vectors |
| `Vision Pipeline` | Combines YOLOv8 and OCR to turn images into searchable text |
| `FastAPI App` | Orchestrates retrieval and generation for the frontend |

<br/>

## 🗺️ Roadmap

- [x] Multimodal RAG (Text + Image) pipeline
- [x] YOLOv8 Traffic Sign detection
- [x] Knowledge Graph export functionality
- [x] Next.js 16 high-performance dashboard
- [ ] Real-time traffic camera stream integration
- [ ] Predictive accident hotspot mapping
- [ ] Automated report generation for authorities
- [ ] Multi-agent collaboration for urban planning

<br/>

## 🤝 Contributing

Contributions are welcome to improve the Vision-RAG accuracy and the knowledge graph depth.

```bash
git checkout -b feature/your-improvement
git commit -m "feat: describe your change"
git push origin feature/your-improvement
```

<br/>

## 📄 License

This project is **free and open source**.

- ✅ Free to use, modify, and distribute
- ✅ Free for research and public safety use
- ⚠️ Provided "as-is" without warranty

---

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%"/>

## 📧 Let's Connect

<div align="center">

<h3>Built with obsession by <b>Salik Ahmad</b> 🤖</h3>

<p>
  <a href="https://salikahmad.vercel.app/" target="_blank">
    <img src="https://img.shields.io/badge/Website-salikahmad.vercel.app-818CF8?style=for-the-badge&logo=vercel&logoColor=white&labelColor=0d0d0d" />
  </a>
  <a href="https://www.linkedin.com/in/salik-ahmad-programmer/" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-Salik%20Ahmad-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white&labelColor=0d0d0d" />
  </a>
  <a href="https://github.com/SalikAhmad702" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-SalikAhmad702-181717?style=for-the-badge&logo=github&logoColor=white&labelColor=0d0d0d" />
  </a>
</p>

<br/>

<a href="https://salikahmad.vercel.app/">
  <img src="https://readme-typing-svg.herokuapp.com?font=JetBrains+Mono&size=14&duration=4000&pause=1000&color=818CF8&center=true&vCenter=true&width=700&lines=AI%2FML+Engineer;Traffic+Intelligence+Specialist;Copyright+©+2026+Salik+Ahmad." alt="Footer Typing" />
</a>

<br/><br/>

<img src="https://capsule-render.vercel.app/api?type=shark&color=0:0f0c29,50:302b63,100:24243e&height=140&section=footer" width="100%"/>

</div>
