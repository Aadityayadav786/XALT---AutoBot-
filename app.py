from utils.session_utils import init_user_session, get_user_and_session

# Initialize session
init_user_session()
user_id, session_id = get_user_and_session()

import uuid
import streamlit as st
import time
import os

from tools.scrap_and_filter import crawl_website
from vector_database import build_or_update_vector_db
from rag_pipeline import get_rag_response
from agents.deployment_agent import DeploymentAgent

# --- Streamlit Page Config ---
st.set_page_config(page_title="Auto AI Chatbot Builder", page_icon="🤖")
st.title("🤖 Auto Chatbot Builder from Website")

# Initialize session state
if "qa_chain_ready" not in st.session_state:
    st.session_state.qa_chain_ready = False

# --- Step 1: Input URL ---
url = st.text_input("🔗 Enter Website URL")

# --- Step 2: Run Scraper & Embedding Pipeline ---
if st.button("🚀 Start Process") and url:
    with st.status("⏳ Running pipeline...", expanded=True) as status:
        try:
            # Scrape
            st.write(f"🔍 Scraping website: `{url}`")
            txt_path = crawl_website(url)
            st.success("✅ Web scraping completed!")

            # (File is already saved by the scraper.)
            st.write("📄 Saving text data...")
            time.sleep(0.5)
            st.success("✅ Data saved successfully!")

            # Embed
            st.write("📦 Generating vector embeddings...")
            build_or_update_vector_db(txt_path)
            st.success("✅ Vector database created!")

            # Mark RAG as ready
            st.session_state.qa_chain_ready = True

            status.update(label="🎉 Pipeline completed!", state="complete", expanded=True)

        except Exception as e:
            status.update(label=f"❌ Pipeline failed: {e}", state="error")


# --- Step 3: Chat Interface (RAG) ---
if st.session_state.qa_chain_ready:
    st.markdown("---")
    st.subheader("💬 Chat with Us ")

    user_query = st.text_input("Ask a question about the site:")
    if user_query:
        # lazy-load + catch missing index
        try:
            with st.spinner("🧠 Generating answer..."):
                session_id = st.session_state.get("session_id", str(uuid.uuid4()))
                user_id = st.session_state.get("user_id", str(uuid.uuid4()))
                st.session_state["session_id"] = session_id
                st.session_state["user_id"] = user_id

                answer , docs = get_rag_response(user_query, session_id , user_id)
        except FileNotFoundError as err:
            st.error(str(err))
        else:
            st.markdown(f"**🤖 Answer:**  {answer}")
            with st.expander("🔍 Sources"):
                for i, doc in enumerate(docs, start=1):
                    st.markdown(f"**Doc {i}:**")
                    st.code(doc.page_content[:500], language="text")

st.markdown("---")
st.subheader("Satisfied with the chatbot?")
if st.button("🚀 Deploy it"):
    with st.status("📤 Pushing code to GitHub...", expanded=True) as status:
        try:
            deployment_agent = DeploymentAgent()
            github_repo_url = deployment_agent.deploy_now()  # Only pushes to GitHub

            status.update(label="✅ Code pushed to GitHub!", state="complete")
            st.success(f"🎉 Your chatbot repo is live at: [🔗 {github_repo_url}]({github_repo_url})")

        except Exception as e:
            status.update(label="❌ Deployment failed", state="error")
            st.error(f"Deployment Error: {e}")

# --- If Embedding Not Done Yet ---
elif not os.path.exists(os.path.join("vectorstore", "index.faiss")):
    st.warning("⚠️ Vectorstore not found — please run the pipeline above first.")
