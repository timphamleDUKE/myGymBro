# 💪myGymBro Setup

This setup guide explains how to open the deployed app, run it locally, and optionally reproduce the data, RAG, model, and evaluation artifacts included in the repository.

The repository already contains the generated outputs needed for the app to function, including processed workout files, RAG chunks/embeddings, trained model artifacts, prediction CSVs, plots, and evaluation reports. You only need to rerun the pipeline scripts if you want to regenerate or verify those artifacts from the raw inputs.

## Use the Deployed App

Open the Streamlit app:

```text
https://mygymbroduke.streamlit.app/
```

The deployed demo is session-only. Each visitor starts with the same seeded workout history from `data/user/user_workouts.csv`, but any profile changes, new workout logs, chat history, and prompt usage are private to that browser session and are not permanently saved.

## Run Locally

1.  **Clone the repository**
    ```bash
    git clone https://github.com/timphamleDUKE/myGymBro.git
    cd myGymBro
    ```
2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Add an API key for the chat model.**

    Create `.streamlit/secrets.toml` locally:

    ```toml
    LITELLM_TOKEN = "your-token-here"
    ```

    The app also supports `OPENAI_API_KEY` if you are using a compatible OpenAI endpoint. Do not commit `.streamlit/secrets.toml`; it is intentionally ignored by git.

4.  Launch the app
    ```bash
    streamlit run app/app.py
    ```

## Reproduce Pipeline Artifacts

The repository already contains the final generated artifacts. The commands below are only needed if you want to rerun the full pipeline from source files.

Run commands from the repository root.

### 1. Webscrape for Raw Knowledge Base Text

Article links are listed in:

```text
knowledge_base/links/article-links.txt
```

Youtube video links are listed in:

```text
knowledge_base/links/yt-links.txt
```

**Note:** You can add additional raw knowledge by adding new article links and youtube video links in their respective .txt files

To scrape article text into `data/raw/articles/`:

```bash
python -m src.rag.extract_article_links
```

To fetch YouTube transcript text into `data/raw/videos/`:

```bash
python -m src.rag.extract_yt_links
```

These scripts depend on external websites and may be blocked, rate-limited, or return different text over time. The submitted repository includes the extracted raw text files so the rest of the project does not require rerunning the scrapers.

### 2. Build RAG Embeddings

After raw article/video text exists in `data/raw/articles/` and `data/raw/videos/`, build the RAG index:

```bash
python -m src.rag.build_embeddings
```

This writes:

```text
data/processed/chunks.json
data/processed/embeddings.npy
data/processed/index_config.json
```

These files are included in the submitted project.

### 3. Clean Workout Data

The raw demo workout dataset is:

```text
data/raw/weightlifting_721_workouts.csv
```

To regenerate the seeded workout data and train/test splits:

```bash
python -m src.ml.clean_data
```

This writes:

```text
data/user/user_workouts.csv
data/processed/workouts.csv
data/processed/train_workouts.csv
data/processed/test_workouts.csv
```

In the deployed demo, `data/user/user_workouts.csv` is used only as the starting seed for each new session. User edits are not written back to this file.

### 4. Generate Model Predictions

Run the baseline model:

```bash
python -m src.ml.baseline
```

This writes:

```text
src/test/baseline/baseline_predictions.csv
```

Train/evaluate the standard XGBoost model:

```bash
python -m src.ml.train_xgboost
```

This writes:

```text
src/ml/models/xgboost_next_weight.json
src/ml/models/xgboost_next_weight_features.json
src/test/xgboost/xgboost_predictions.csv
```

Train/evaluate the grouped XGBoost-plus models:

```bash
python -m src.ml.train_xgboost_plus
```

This writes:

```text
src/ml/models/xgboost_plus_next_weight_global.json
src/ml/models/xgboost_plus_next_weight_global_features.json

src/ml/models/xgboost_plus_next_weight_upper_compound.json
src/ml/models/xgboost_plus_next_weight_upper_compound_features.json

src/ml/models/xgboost_plus_next_weight_lower_compound.json
src/ml/models/xgboost_plus_next_weight_lower_compound_features.json

src/ml/models/xgboost_plus_next_weight_isolation.json
src/ml/models/xgboost_plus_next_weight_isolation_features.json

src/ml/models/xgboost_plus_next_weight_bodyweight.json
src/ml/models/xgboost_plus_next_weight_bodyweight_features.json

src/test/xgboost/xgboost_plus_predictions.csv
```

### 5. Test and Compare Models

After prediction CSVs exist, we can generate metrics, plots, and the final combined report:

```bash
python -m src.test.test_models
```

This writes per-model metrics and plots under:

```text
src/test/baseline/
src/test/xgboost/
src/test/xgboost-plus/
```

It also writes the combined reports:

```text
src/test/model_final_report.csv
src/test/model_final_report.md
```

## Recommended Reproduction Order

If starting from raw inputs and regenerating everything, use:

```bash
python -m src.rag.extract_article_links
python -m src.rag.extract_yt_links
python -m src.rag.build_embeddings
python -m src.ml.clean_data
python -m src.ml.baseline
python -m src.ml.train_xgboost
python -m src.ml.train_xgboost_plus
python -m src.test.test_models
```

For normal app usage, you can skip all of those scripts and run:

```bash
streamlit run app/app.py
```

**Note:** The XGBoost-plus model is **NOT** used during inference time in the chatbox, refer to Evaluation Section in [README.md](./README.md) for more information