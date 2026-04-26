# 💪myGymBro Attribution

This file documents external data sources, knowledge-base sources, third-party tools, and AI assistance used in myGymBro.

## Data Sources

### Workout Dataset

The seeded workout dataset and model training data are based on the Kaggle weightlifting dataset:

- Kaggle dataset: <https://www.kaggle.com/datasets/joep89/weightlifting>

The raw workout CSV is stored in:

- `data/raw/weightlifting_721_workouts.csv`

The project cleaning pipeline converts this data into:

- `data/user/user_workouts.csv`
- `data/processed/workouts.csv`
- `data/processed/train_workouts.csv`
- `data/processed/test_workouts.csv`

In the deployed demo, `data/user/user_workouts.csv` is used as a session seed only. User edits are kept in Streamlit session state and are not written back to the shared dataset.

### RAG Knowledge Base Links

The retrieval-augmented generation knowledge base uses article and YouTube sources listed in:

- Article links: `knowledge_base/links/article-links.txt`
- YouTube links: `knowledge_base/links/yt-links.txt`

Extracted article text is stored in:

- `data/raw/articles/`

Extracted video transcript text is stored in:

- `data/raw/videos/`

The generated RAG artifacts are stored in:

- `data/processed/chunks.json`
- `data/processed/embeddings.npy`
- `data/processed/index_config.json`

## Third-Party Packages and Tools

The project uses Python packages listed in `requirements.txt`, including:

- Streamlit
- pandas
- numpy
- XGBoost
- PyTorch and Hugging Face Transformers
- OpenAI Python SDK
- matplotlib

The article and video extraction scripts use:

- `newspaper3k` for article extraction in `src/rag/extract_article_links.py`
- `youtube-transcript-api` and YouTube oEmbed/watch-page metadata lookup in `src/rag/extract_yt_links.py`

## AI Assistance

AI development tools were used as programming assistance during the project. I used ChatGPT/Codex to brainstorm approaches, debug errors, review code paths, and draft portions of implementation. I reviewed and modified the generated code before including it in the final project.

### AI Links Used

- Packages for article extraction and YouTube video extraction: <https://chatgpt.com/s/t_69e832a2047881919917414c33ea787d>
- Improving RAG context alignment with user prompts: <https://chatgpt.com/s/t_69eae6442be88191804fe8e54169a7c6>
- Models for predicting next workout weight: <https://chatgpt.com/share/69eae5ce-77b0-83ea-95d5-23b9f4b240c8>
- Open-source workout CSVs for model training: <https://chatgpt.com/share/69eb02b5-2b7c-83ea-b625-5b2df60f825e>
- Getting started with XGBoost from a CSV: <https://chatgpt.com/share/69ebec44-59d0-83ea-938d-ffa1d72a428d>
- Getting started with the Duke API key / LiteLLM setup: <https://chatgpt.com/share/69ec60e4-4d38-83ea-80cc-5c1858085991>

### What AI Helped Generate or Draft

AI assistance contributed to implementation ideas for:

- The Streamlit multipage app structure and session-state helpers
- The coach chat flow, mainly how to stream LLM output and feed context to the LLM
- Methods for RAG context reranking beyond similarity scores between context and query
- Feature engineering on the dataset via exercise group categories
- The layout of the baseline, XGBoost, and XGBoost-plus scripts
- Webscraping article and Youtube links