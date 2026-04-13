"""Streamlit app for interactive image similarity search using LanceDB.

Demonstrates how LanceDB's inline image storage makes it trivial to build
a self-contained image search app — no separate image server needed.
"""

import io
from pathlib import Path

import lancedb
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMAGES_DIR = DATA_DIR / "coco_images"
LANCEDB_DIR = DATA_DIR / "lancedb"

st.set_page_config(page_title="SigLIP 2 Image Search", layout="wide")


@st.cache_resource
def load_table():
    db = lancedb.connect(str(LANCEDB_DIR))
    return db.open_table("coco_clip_embeddings")


@st.cache_data
def load_embeddings_df():
    return pd.read_parquet(DATA_DIR / "embeddings" / "image_embeddings.parquet")


def image_from_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes))


def main():
    st.title("SigLIP 2 Image Similarity Search")
    st.caption(
        "Powered by LanceDB — images are stored inline with vectors, no external storage needed."
    )

    with st.expander("How this works"):
        st.markdown("""
        **This app reads images directly from LanceDB search results.**

        When you search, LanceDB returns the image bytes alongside the vectors and metadata.
        There's no separate API call to S3 or a CDN — the images live with the data.

        ```python
        results = table.search(query_vec).limit(8).to_pandas()
        for row in results:
            image = Image.open(io.BytesIO(row["image_bytes"]))  # that's it!
        ```

        Compare this to OpenSearch, where you'd need to:
        1. Query OpenSearch for matching document IDs
        2. Extract the image path/URL from each result
        3. Fetch the actual image from a separate storage system

        With LanceDB, it's one step.
        """)

    table = load_table()
    df = load_embeddings_df()

    # Sidebar: pick a query image
    st.sidebar.header("Select a Query Image")

    # Show a grid of sample images to choose from
    num_samples = 20
    sample_indices = np.linspace(0, len(df) - 1, num_samples, dtype=int)

    selected_idx = st.sidebar.selectbox(
        "Choose an image",
        options=list(range(num_samples)),
        format_func=lambda i: f"{df.iloc[sample_indices[i]]['file_name']} — {str(df.iloc[sample_indices[i]]['caption'])[:40]}...",
    )

    actual_idx = sample_indices[selected_idx]
    query_row = df.iloc[actual_idx]

    # Show the query image
    query_img_path = IMAGES_DIR / query_row["file_name"]
    if query_img_path.exists():
        st.sidebar.image(str(query_img_path), caption=query_row["caption"], use_container_width=True)

    # Search controls
    k = st.sidebar.slider("Number of results", min_value=3, max_value=100, value=8)

    # Run search
    query_vec = query_row["vector"]
    results = table.search(query_vec).limit(k).to_pandas()

    # Display results
    st.header(f"Top {len(results)} Similar Results")

    cols = st.columns(4)
    for i, (_, row) in enumerate(results.iterrows()):
        col = cols[i % 4]
        with col:
            # LanceDB advantage: images come directly from the search results
            if row["image_bytes"] and len(row["image_bytes"]) > 0:
                img = image_from_bytes(row["image_bytes"])
                st.image(img, use_container_width=True)
            else:
                st.write("*(no image data)*")

            st.caption(
                f"dist: {row['_distance']:.4f}\n\n"
                f"{row['caption']}"
            )



if __name__ == "__main__":
    main()
