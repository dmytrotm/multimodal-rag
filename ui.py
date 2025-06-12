import streamlit as st
import os
from pathlib import Path
from PIL import Image


# Import your RAG processor
try:
    from data_processor import DataProcessor  # Adjust import based on your file name
except ImportError:
    st.error("Please ensure your RAG processor code is in a file named 'data_processor.py' or update the import statement.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="THE BATCH SEARCH",
    page_icon="🤖",
    layout="wide"
)

# Simple CSS for clean styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    
    .answer-box {
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_rag_processor():
    """Load the RAG processor."""
    try:
        processor = DataProcessor(
            output_dir=None,  
            vectorstore_path="./chroma_db_final",
            docstore_path="./docstore_final",
            use_vision_model=True,  
            verbose=True,
            google_api_key=os.getenv("GOOGLE_API_KEY_1")  
        )
        return processor
    except Exception as e:
        st.error(f"Failed to load RAG processor: {e}")
        return None

def main():
    # Title
    st.markdown('<h1 class="main-title">🤖 THE BATCH SEARCH </h1>', unsafe_allow_html=True)
    
    # Load processor
    processor = load_rag_processor()
    if not processor:
        st.stop()
    
    # Input section
    st.subheader("💬 Ask a Question")
    query = st.text_input(
        "Enter your query:",
        placeholder="Ask anything about The Batch newsletters...",
        key="user_query"
    )
    
    # Number of results
    num_results = st.slider("Number of results to retrieve:", 1, 10, 5)
    
    # Search button
    if st.button("🔍 Search", type="primary") or query:
        if query.strip():
            with st.spinner("Searching..."):
                try:
                    # Get results from RAG processor
                    result = processor.query(query, k=num_results)
                    
                    # Display answer
                    st.subheader("📝 Answer")
                    st.markdown(f'<div class="answer-box">{result.get("answer", "No answer generated.")}</div>', 
                              unsafe_allow_html=True)
                    
                    # Display images if any
                    images = result.get('images', [])
                    if images:
                        st.subheader("🖼️ Related Images")
                        
                        # Create columns for images (max 2 per row)
                        for i in range(0, len(images), 2):
                            cols = st.columns(2)
                            
                            for j, col in enumerate(cols):
                                if i + j < len(images):
                                    img_data = images[i + j]
                                    image_path = img_data.get('path', '')
                                    description = img_data.get('description', 'No description available')
                                    ref_id = img_data.get('ref_id', '')
                                    
                                    with col:                                        
                                        # Try to display the image
                                        if image_path and not image_path.startswith('http'):
                                            if Path(image_path).exists():
                                                try:
                                                    image = Image.open(image_path)
                                                    st.image(image, caption=f"Reference {ref_id}", use_container_width=True)
                                                except Exception as e:
                                                    st.warning(f"Could not load image: {e}")
                                            else:
                                                st.warning(f"Image not found: {Path(image_path).name}")
                                        elif image_path.startswith('http'):
                                            st.image(image_path, caption=f"Reference {ref_id}", use_container_width=True)
                                        
                                        # Show description
                                        st.write(f"**Description:** {description}")
                                        
                                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Optional: Show sources in an expandable section
                    sources = result.get('sources', [])
                    if sources:
                        with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
                            for source in sources:
                                st.write(f"**[{source['ref_id']}]** {source.get('content_type', 'Unknown').replace('_', ' ').title()}")
                                st.write(f"Date: {source.get('date', 'Unknown')}")
                                if source.get('news_title'):
                                    st.write(f"Title: {source['news_title']}")
                                if source.get('source_url'):
                                    st.write(f"[Source URL]({source['source_url']})")
                                st.write("---")
                    
                except Exception as e:
                    st.error(f"Error during search: {e}")
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()