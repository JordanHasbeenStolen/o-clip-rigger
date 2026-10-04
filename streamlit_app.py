import requests
import streamlit as st

API_URL = "http://localhost:8000"

MODEL_LABELS = {
    "openclip": "OpenCLIP (classic)",
    "siglip2": "SigLIP2 (modern, more accurate)",
}

st.set_page_config(page_title="o-clip-rigger", layout="wide")
st.title("o-clip-rigger")
st.write(
    "This app searches your personal photo collection using natural language, "
    "or looks at a single photo and suggests tags for it. Pick what you want to do below."
)

model = st.radio(
    "Model",
    list(MODEL_LABELS.keys()),
    format_func=lambda key: MODEL_LABELS[key],
    horizontal=True,
)

task = st.radio(
    "What do you want to do?",
    ["Search photos by text", "Get tags for a photo"],
    horizontal=True,
)

if task == "Search photos by text":
    with st.sidebar:
        st.header("Search settings")
        top_k = st.slider("Number of results", 1, 20, 1)
        preview_width = st.slider("Preview width (px)", 200, 800, 400, step=50)

    with st.form("search_form"):
        query = st.text_input("Query", placeholder="cat on the sofa")
        submitted = st.form_submit_button("Search", type="primary")

    if submitted and query:
        try:
            with st.spinner("Searching..."):
                resp = requests.post(
                    f"{API_URL}/search",
                    json={"query": query, "top_k": top_k, "model": model},
                    timeout=60,
                )
            resp.raise_for_status()
            data = resp.json()

            st.subheader(f"Results for: {query} ({MODEL_LABELS[model]})")

            def render_item(item, width):
                url = f"{API_URL}{item['url']}"
                st.image(url, width=width)
                st.caption(f"{item['path']}  |  score: {item['score']:.4f}")

                img_bytes = requests.get(url, timeout=30).content
                filename = item["path"].split("/")[-1]
                st.download_button(
                    label="Download original",
                    data=img_bytes,
                    file_name=filename,
                    mime="image/jpeg",
                    key=f"dl_{item['url']}",
                )

            if top_k == 1:
                render_item(data["results"][0], preview_width)
            else:
                cols = st.columns(3)
                for i, item in enumerate(data["results"]):
                    with cols[i % 3]:
                        render_item(item, preview_width)

        except requests.exceptions.ConnectionError:
            st.error("Cannot reach API. Is uvicorn running on port 8000?")
        except requests.exceptions.HTTPError as e:
            st.error(f"API error: {e}")

else:
    uploaded = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp"])
    submitted = st.button("Get tags", type="primary", disabled=uploaded is None)

    if submitted and uploaded is not None:
        try:
            with st.spinner("Tagging..."):
                resp = requests.post(
                    f"{API_URL}/tag",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    data={"model": model},
                    timeout=60,
                )
            resp.raise_for_status()
            data = resp.json()

            cols = st.columns([1, 1])
            with cols[0]:
                st.image(uploaded, width=400)
            with cols[1]:
                st.subheader(f"Tags ({MODEL_LABELS[data['model']]})")
                for t in data["tags"]:
                    st.write(f"**{t['tag']}** — {t['score']:.3f}")

        except requests.exceptions.ConnectionError:
            st.error("Cannot reach API. Is uvicorn running on port 8000?")
        except requests.exceptions.HTTPError as e:
            st.error(f"API error: {e}")
