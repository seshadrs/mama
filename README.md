# Medical Archive Memory Aid (MAMA)

This is a basic WIP app that can load PDF medical documents in a directory into a vector DB and allow search and QnA over those docs. It is important to have a unique Pinecone namespace for each patient.

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   export OPENAI_API_KEY="..."
   export PINECONE_API_KEY="..."
   ```

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. Open your web browser and go to the URL displayed in the terminal (usually `http://localhost:8501`).
