import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="o-clip-rigger", layout="wide")
st.title("o-clip-rigger")
st.caption("Search photo collection with natural language")

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
                json={"query": query, "top_k": top_k},
                timeout=60,
            )
        resp.raise_for_status()
        data = resp.json()

        st.subheader(f"Results for: {query}")

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