# 🐾 WildGuard AI: Animal Identifier

A Streamlit prototype that uses OpenAI CLIP (`openai/clip-vit-base-patch32`) to compare uploaded images with descriptions of five supported animal classes (cat, elephant, lion, panda, tiger) and a broad non-animal category.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first launch downloads the pretrained model, so it can take a few minutes and requires an internet connection.

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and `README.md` to a GitHub repository.
2. Open https://share.streamlit.io/
3. Choose **Create app**, select your repository and branch, and set the main file path to `app.py`.
4. Deploy.

## Important limitations

- Similarity scores are **not probabilities**.
- The non-animal decision is a simple prototype heuristic, not a validated open-set detector.
- Test with many images from all five classes and many unrelated objects before relying on predictions.
- This is a learning project, not a wildlife safety or conservation monitoring system.
