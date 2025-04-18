import streamlit as st
import asyncio
from textstat import textstat
from BlogEngine import BlogEngine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Blog Generator",
    page_icon="✍️",
    layout="wide"
)

# Initialize session state
if 'generated_blog' not in st.session_state:
    st.session_state.generated_blog = None

async def generate_blog_async(topic: str, tone: str):
    """Async wrapper for blog generation."""
    engine = BlogEngine()
    return await engine.generate_blog(topic, tone)

def main():
    st.title("✍️ Blog Generator")
    st.markdown("Generate SEO-optimized blog posts with AI")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        tone = st.selectbox(
            "Writing Tone",
            ["professional", "casual", "educational", "informative", "technical"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This tool uses AI to generate SEO-optimized blog posts.
        - Enter a topic
        - Choose a writing tone
        - Get a complete blog post with metadata
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        topic = st.text_input("Enter your blog topic", placeholder="e.g., Python Web Development")
        
        if st.button("Generate Blog"):
            if not topic:
                st.error("Please enter a topic")
            else:
                with st.spinner("Generating your blog post..."):
                    # Run async function
                    result = asyncio.run(generate_blog_async(topic, tone))
                    st.session_state.generated_blog = result
    
    if st.session_state.generated_blog:
        blog = st.session_state.generated_blog
        
        with col1:
            st.markdown("### Generated Blog")
            st.markdown(blog['content'])
            
            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download Markdown",
                    blog['content'],
                    file_name=f"{blog['slug']}.md",
                    mime="text/markdown"
                )
            with col2:
                st.download_button(
                    "Download Metadata",
                    str(blog['metadata']),
                    file_name=f"{blog['slug']}_meta.json",
                    mime="application/json"
                )
        
        with col2:
            st.markdown("### Blog Details")
            
            # Basic info
            st.markdown("#### Basic Information")
            st.markdown(f"**Title:** {blog['title']}")
            st.markdown(f"**Reading Time:** {blog['reading_time']} minutes")
            
            # Keywords
            st.markdown("#### Keywords")
            st.markdown(", ".join(blog['keywords']))
            
            # Readability scores
            st.markdown("#### Readability Scores")
            flesch_score = textstat.flesch_reading_ease(blog['content'])
            grade_level = textstat.flesch_kincaid_grade(blog['content'])
            
            st.markdown(f"**Flesch Reading Ease:** {flesch_score:.1f}")
            st.markdown(f"**Flesch-Kincaid Grade Level:** {grade_level:.1f}")
            
            # SEO metadata
            st.markdown("#### SEO Metadata")
            st.markdown(f"**Slug:** {blog['slug']}")
            st.markdown(f"**Meta Description:** {blog['meta_description']}")

if __name__ == "__main__":
    main() 