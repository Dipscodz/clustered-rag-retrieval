# Clustered RAG Retrieval

## 📌 Overview

This project presents an enhanced Retrieval-Augmented Generation (RAG) framework that improves document retrieval using embedding-based semantic clustering. By organizing documents into clusters before retrieval, the system ensures that only the most relevant context is provided to the language model, leading to more accurate and meaningful responses.

---

## 🚀 Key Features

* Sentence embeddings using SentenceTransformers
* KMeans-based semantic clustering
* Cluster-aware document retrieval
* Improved relevance in RAG pipeline
* Modular and scalable architecture

---

## 🧠 Motivation

Traditional RAG systems rely solely on vector similarity, which may retrieve loosely relevant documents. This project introduces a clustering-based retrieval strategy to:

* Reduce irrelevant context
* Improve semantic relevance
* Enhance overall response accuracy

---

## ⚙️ System Architecture

```
Documents → Embeddings → Clustering → Cluster Selection → Retrieval → Response
```

---

## 📂 Project Structure

```
clustered-rag-retrieval/
│
├── data/
├── embeddings/
├── models/
├── src/
│   ├── preprocess.py
│   ├── embed.py
│   ├── cluster.py
│   ├── retrieve.py
│   ├── generate.py
│   └── evaluate.py
│
├── results/
├── requirements.txt
├── main.py
└── README.md
```

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python main.py
```

---

## 📊 Methodology

1. Convert documents into embeddings
2. Apply KMeans clustering to group similar documents
3. Identify the most relevant cluster for a query
4. Retrieve documents from the selected cluster
5. Generate responses based on retrieved context

---

## 📈 Expected Results

* Improved retrieval precision
* Better semantic relevance
* Enhanced answer quality compared to baseline RAG

---

## 🔬 Future Work

* Integration with large language models (LLMs)
* Hybrid re-ranking techniques
* Dynamic clustering methods
* Evaluation on large-scale datasets

---

## 📄 Research Context

This project is being developed as part of a research effort focusing on improving Retrieval-Augmented Generation systems using advanced retrieval strategies.

---

## 👨‍💻 Author

Derin Denny Mathew

---
