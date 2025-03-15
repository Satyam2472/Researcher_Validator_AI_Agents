

import streamlit as st
from google import genai
import os

# Initialize Gemini API Client
API_KEY = st.secrets["GEMINI_API_KEY"]
if not API_KEY:
    st.error("There is an error in the API, check your Streamlit secrets.")
else:
    client = genai.Client(api_key=API_KEY)


class ResearcherAgent:
    """AI Agent responsible for researching and collecting data."""

    def __init__(self, query):
        self.query = query

    def collect_data(self):
        """Uses Gemini AI to gather research data."""
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            # contents=f"Find the latest AI advancements in {self.query}. Provide a detailed and structured summary."
            contents=f"{self.query}. Provide a detailed and structured summary."
        )
        return response.text if response else "No research data found."


class ValidatorAgent:
    """AI Agent responsible for validating, trimming, and refining the research data."""

    def __init__(self, research_data):
        self.research_data = research_data

    def validate_and_refine(self):
        """Uses Gemini AI to validate and enhance research data."""
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"""
            Validate and refine the following data:
            
            {self.research_data}
            
            - Ensure accuracy and up-to-date information (2023 onwards).
            - Trim unnecessary information while keeping key insights.
            - Identify any potential bias or conflicting claims.
            - If the query is regarding of some national policy or health refer to government sources like websites that have .gov as domain name.
            - If the query is regarding the some technology refer to pages which as wikipedia.org or arxiv.org.
            - If its about some general issue you still refer to most recently published and trusted sources.
            - Structure it in a readable and concise format.
            
            Provide the final refined version.
            """
        )
        return response.text if response else "No validated data available."


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

st.title("🧠 Your Personalized AI Researcher.")
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

