

import streamlit as st
import google.generativeai as genai
import os

# Initialize Gemini API Client
API_KEY = st.secrets.get("GEMINI_API_KEY")  

if not API_KEY:
    st.error("⚠️ API key not found! Please add it to Streamlit secrets.")
    st.stop()  
else:
    genai.configure(api_key=API_KEY)  

class ResearcherAgent:
    """AI Agent responsible for researching and collecting data."""

    def __init__(self, query):
        self.query = query

    def collect_data(self):
        """Uses Gemini AI to gather research data."""
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")  
            response = model.generate_content(
                f"Find the latest research on {self.query}. Provide a detailed and structured summary. Also memtion the source of the information."
            )
            return response.text if response else "No research data found."
        except Exception as e:
            return f"Error in ResearcherAgent: {str(e)}"

class ValidatorAgent:
    """AI Agent responsible for validating, trimming, and refining the research data."""

    def __init__(self, research_data):
        self.research_data = research_data

    def validate_and_refine(self):
        """Uses Gemini AI to validate and enhance research data."""
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")  
            response = model.generate_content(
                f"""
                Validate and refine the following data:
                
                {self.research_data}
                
                - Ensure accuracy and up-to-date information (2023 onwards).
                - Trim unnecessary information while keeping key insights.
                - Identify any potential bias or conflicting claims.
                - If the query is about national policy or health, refer to government sources (.gov).
                - If the query is about technology, refer to reputable sites like Wikipedia.org or arXiv.org.
                - For general topics, use the latest, most trusted sources.
                - Structure it in a readable and concise format.
                - Make sure the source of information is reliable and trusted, if not then do a research from a reliable source and list them in the results.

                Provide the final refined version.
                """
            )
            return response.text if response else "No validated data available."
        except Exception as e:
            return f"Error in ValidatorAgent: {str(e)}"

def multi_agent_ai_system(user_query):
    """Main function to coordinate Researcher and Validator Agents."""
    
    with st.spinner("🔍 Researching..."):
        researcher = ResearcherAgent(user_query)
        research_data = researcher.collect_data()
    
    with st.spinner("✅ Validating research..."):
        validator = ValidatorAgent(research_data)
        validated_data = validator.validate_and_refine()

    return validated_data

# Streamlit UI
st.set_page_config(page_title="AI Research Assistant", layout="centered")

st.title("🧠 Your Personalized AI Researcher")
st.write("Enter your query to get structured and validated AI research.")

user_input = st.text_area("🔍 Enter your topic of research:", placeholder="E.g. Latest Generative AI advancements")

if st.button("Generate Research"):
    if user_input.strip():
        result = multi_agent_ai_system(user_input)
        st.subheader("🎯 Validated Research Data:")
        st.write(result)

        # Provide option to download as .txt file
        st.download_button(
            label="📥 Download Research as .txt",
            data=result,
            file_name="validated_research.txt",
            mime="text/plain"
        )
    else:
        st.warning("⚠️ Please enter a topic to research.")

st.markdown("---")
st.info("Built using Google Gemini AI")
