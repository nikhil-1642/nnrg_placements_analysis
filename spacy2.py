import spacy
from spacy import displacy
import streamlit as st

# Load spaCy model (only once, caching it)
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")

nlp = load_model()

# App UI
st.title("Named Entity Recognition (NER) with spaCy")

# Text input
text = st.text_area("Enter text for NER", height=200, key="text_area")

# Only process if there's input
if text.strip():
    doc = nlp(text)

    # Render NER with displaCy
    html = displacy.render(doc, style="ent", jupyter=False)
    st.markdown("**Detected Entities:**", unsafe_allow_html=True)
    st.markdown(html, unsafe_allow_html=True)

    # Display entities in a table
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    if entities:
        st.markdown("**Entity Table:**")
        st.table(entities)
    else:
        st.info("No named entities found.")
else:
    st.info("Enter or paste text above to analyze named entities.")
