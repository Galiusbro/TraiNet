# TrainNet.ai

> Decentralized AI training platform — train models on your data, reward the world for helping.

---

## 🚀 What is TrainNet.ai?

TrainNet.ai is a decentralized platform for training machine learning models using volunteer compute resources.

It lets users:
- Upload their datasets
- Select a pre-configured model architecture
- Automatically distribute training to worker nodes
- Merge results into a final model
- Download trained weights or integrate via API

And it rewards contributors (workers) who train shards of the model.

---

## 🎯 Why it matters

- AI training is expensive, centralized, and cloud-locked.
- Millions of GPUs and CPUs around the world sit idle.
- TrainNet.ai connects compute with need — like mining, but actually useful.

---

## 🛠️ Features (MVP)

- ✅ Task queue with Redis
- ✅ Docker-based worker agents (PyTorch)
- ✅ Dataset sharding & aggregation
- ✅ Finalizer with weight merging
- ✅ FastAPI backend + REST API
- ✅ React frontend (upload, status, download)
- ✅ Supabase for storage and metadata

---

## 🧠 Architecture

![TrainNet Architecture](./trainnet_architecture_dark.png)

---

## 💡 Example Workflow

1. User uploads dataset via `/train` endpoint or React UI
2. Backend shards the data and creates Redis queues
3. Distributed workers fetch shards and train models
4. Finalizer checks for completion and merges models
5. User downloads their trained model

---

## 🔒 Security & Future Plans

- Add model verification / validation
- Reputation and reward system for workers
- Custom hyperparameter selection
- Token-based marketplace
- Federated / ZK training support (R&D)

---

## 📂 Repository Structure

```
trainnet-ai/
├── backend/       # FastAPI backend: API, task logic, file mgmt
├── worker/        # Python shard trainer (Dockerized)
├── finalizer/     # Weight aggregator
├── frontend/      # React client UI
├── deployments/   # Docker Compose configs
└── README.md      # You're here
```

---

## 👤 About the Author

Ayaal Santaev — backend engineer, open-source enthusiast, and believer in a fairer, more decentralized AI infrastructure.

---

## 📬 Contact & Updates

This is a solo-built MVP.  
Currently seeking grant support, early adopters, and contributors.

📫 Contact: ayaalsantaev@gmail.com  
📎 Pitch deck: [TrainNet_PitchDeck.pdf](./TrainNet_PitchDeck.pdf)  
📁 GitHub: https://github.com/Galiusbro/TraiNet

---

## 🧪 Try It Locally

Coming soon: Docker quickstart + demo mode.
