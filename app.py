import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PDF Chatbot with RAG",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF Chatbot with RAG")
st.write("Upload a PDF and ask questions about its content.")


# =========================================================
# LOAD AI MODELS
# =========================================================

@st.cache_resource
def load_models():

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-small"
    )

    return embedding_model, generator


# =========================================================
# LOAD MODELS
# =========================================================

with st.spinner("Loading AI models... Please wait."):

    embedding_model, generator = load_models()

st.success("✅ AI models loaded successfully!")


# =========================================================
# PDF UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📤 Upload your PDF",
    type=["pdf"]
)


# =========================================================
# MAIN PROGRAM
# =========================================================

if uploaded_file is not None:

    # Read PDF
    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"


    # Check PDF text
    if not text.strip():

        st.error(
            "❌ No readable text was found in this PDF."
        )

        st.stop()


    st.success(
        f"✅ PDF uploaded successfully! "
        f"Pages: {len(reader.pages)}"
    )


    # =====================================================
    # SPLIT TEXT INTO CHUNKS
    # =====================================================

    chunk_size = 500

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunk = text[i:i + chunk_size]

        if chunk.strip():
            chunks.append(chunk)


    st.info(
        f"📚 PDF divided into {len(chunks)} text chunks."
    )


    # =====================================================
    # CREATE EMBEDDINGS
    # =====================================================

    with st.spinner("Creating document embeddings..."):

        embeddings = embedding_model.encode(
            chunks,
            convert_to_numpy=True
        )

        embeddings = embeddings.astype("float32")


    # =====================================================
    # CREATE FAISS INDEX
    # =====================================================

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)


    st.success(
        "✅ PDF processed successfully!"
    )


    # =====================================================
    # QUESTION INPUT
    # =====================================================

    st.subheader("💬 Ask a Question")

    question = st.text_input(
        "Enter your question about the PDF:"
    )


    # =====================================================
    # RAG SEARCH AND ANSWER
    # =====================================================

    if question:

        # -------------------------------------------------
        # Convert question into embedding
        # -------------------------------------------------

        with st.spinner("🔍 Searching the PDF..."):

            question_embedding = embedding_model.encode(
                [question],
                convert_to_numpy=True
            )

            question_embedding = question_embedding.astype(
                "float32"
            )


            # -------------------------------------------------
            # Search relevant chunks
            # -------------------------------------------------

            number_of_results = min(
                3,
                len(chunks)
            )

            distances, indices = index.search(
                question_embedding,
                number_of_results
            )


            relevant_chunks = []

            for i in indices[0]:

                if i >= 0 and i < len(chunks):

                    relevant_chunks.append(
                        chunks[i]
                    )


            context = "\n\n".join(
                relevant_chunks
            )


        # -------------------------------------------------
        # Generate Answer
        # -------------------------------------------------

        with st.spinner("🤖 Generating answer..."):

            prompt = f"""
Answer the question using only the information
provided in the context.

If the answer is not available in the context,
say "The answer is not available in the PDF."

Context:
{context}

Question:
{question}

Answer:
"""


            result = generator(
                prompt,
                max_new_tokens=150,
                do_sample=False
            )


            answer = result[0]["generated_text"]


        # -------------------------------------------------
        # Display Answer
        # -------------------------------------------------

        st.subheader("🤖 Answer")

        st.write(answer)


        # -------------------------------------------------
        # Display Retrieved Content
        # -------------------------------------------------

        with st.expander(
            "🔍 View Retrieved PDF Content"
        ):

            st.write(context)


else:

    st.info(
        "👆 Please upload a PDF to start chatting."
    )