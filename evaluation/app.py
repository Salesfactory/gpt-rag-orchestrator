"""
Streamlit app for evaluating RAG system responses.
"""

import os
import re
import time
import uuid
from io import BytesIO
from pathlib import Path
import streamlit as st
import pandas as pd
import requests
from dotenv import load_dotenv
from evaluator import RAGEvaluator

# Load environment variables from parent .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


# Page config
st.set_page_config(
    page_title="RAG Evaluation Tool",
    page_icon="RAG",
    layout="wide"
)

# Title and description
st.title("RAG System Evaluation Tool")
st.markdown("""
Upload an Excel file with test questions and evaluate your RAG system's responses.
The file should contain columns: `question`, `fact`, and optionally `source`.
""")


def parse_streaming_response(response_text: str) -> str:
    """
    Extract the answer from the streaming response.

    The streaming format is:
    __PROGRESS__{...}__PROGRESS__
    __THINKING__{...}__THINKING__
    actual response text here
    __PROGRESS__{"step":"complete",...}__PROGRESS__
    __METADATA__{...}__METADATA__

    We need to extract everything that's NOT wrapped in markers.

    Args:
        response_text: Raw streaming response text

    Returns:
        Extracted answer text

    Raises:
        ValueError: If response format is invalid
    """
    # Remove all __PROGRESS__ blocks
    text = re.sub(r'__PROGRESS__.*?__PROGRESS__', '', response_text, flags=re.DOTALL)

    # Remove all __THINKING__ blocks
    text = re.sub(r'__THINKING__.*?__THINKING__', '', text, flags=re.DOTALL)

    # Remove all __METADATA__ blocks
    text = re.sub(r'__METADATA__.*?__METADATA__', '', text, flags=re.DOTALL)

    # Strip whitespace
    answer = text.strip()

    if not answer:
        raise ValueError("Empty response extracted from streaming output")

    return answer


def query_rag_endpoint(
    question: str,
    client_principal_id: str,
    client_principal_name: str,
    client_principal_organization: str,
    conversation_id: str,
    endpoint_url: str = "http://localhost:7071/api/orc",
    is_agentic_search_mode: bool = False
) -> str:
    """
    Send a question to the RAG orchestrator endpoint.

    Args:
        question: The question to ask
        client_principal_id: User ID
        client_principal_name: User name
        client_principal_organization: Organization ID
        conversation_id: Unique conversation ID for tracking
        endpoint_url: RAG endpoint URL
        is_agentic_search_mode: Force agentic search mode

    Returns:
        The AI-generated answer
    """
    payload = {
        "question": question,
        "client_principal_id": client_principal_id,
        "client_principal_name": client_principal_name,
        "client_principal_organization": client_principal_organization,
        "conversation_id": conversation_id,
        "is_agentic_search_mode": is_agentic_search_mode
    }

    response = requests.post(
        endpoint_url,
        json=payload,
        headers={"Content-Type": "application/json"},
        stream=True,
        timeout=120
    )

    response.raise_for_status()

    full_response = ""
    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
        if chunk:
            full_response += chunk

    answer = parse_streaming_response(full_response)
    return answer


def load_excel(file) -> pd.DataFrame:
    """Load Excel file and validate columns."""
    df = pd.read_excel(file, engine='openpyxl')

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Clean up text columns (strip trailing whitespace/newlines)
    for col in ['question', 'fact', 'source']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    required_cols = ['question', 'fact']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    if 'source' not in df.columns:
        df['source'] = ''

    if 'ai_answer' not in df.columns:
        df['ai_answer'] = ''

    if 'score' not in df.columns:
        df['score'] = 0

    if 'reasoning' not in df.columns:
        df['reasoning'] = ''

    if 'factual_accuracy' not in df.columns:
        df['factual_accuracy'] = ''

    return df


st.sidebar.header("Configuration")

endpoint_url = st.sidebar.text_input(
    "RAG Endpoint URL",
    value="http://localhost:7071/api/orc",
    help="URL of the RAG orchestrator endpoint"
)

st.sidebar.subheader("Client Credentials")
client_principal_id = st.sidebar.text_input(
    "Client Principal ID",
    value=os.getenv("EVAL_USER_ID"),
    help="User ID for authentication"
)

client_principal_name = st.sidebar.text_input(
    "Client Principal Name",
    value=os.getenv("EVAL_USER_NAME", "sheep"),
    help="User name for authentication"
)

client_principal_organization = st.sidebar.text_input(
    "Client Organization ID",
    value=os.getenv("EVAL_ORGANIZATION_ID"),
    help="Organization ID for authentication"
)

st.sidebar.subheader("Evaluation Options")
is_agentic_search_mode = st.sidebar.checkbox(
    "Force Agentic Search",
    value=False,
    help="Force the orchestrator to use agentic search for all questions"
)

# File upload
st.header("1. Upload Test Data")
uploaded_file = st.file_uploader(
    "Upload Excel file (.xlsx)",
    type=['xlsx'],
    help="File should contain: question, fact, and optionally source"
)

if uploaded_file:
    try:
        df = load_excel(uploaded_file)
        st.success(f"Loaded {len(df)} questions from Excel file")

        st.subheader("Preview")
        st.dataframe(df[['question', 'fact', 'source']].head(10), use_container_width=True)

        # Run evaluation
        st.header("2. Run Evaluation")

        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Start Evaluation", type="primary", use_container_width=True):
                st.session_state.run_evaluation = True

        with col2:
            st.info("This will query the RAG endpoint for each question and evaluate responses.")

        # Evaluation process
        if st.session_state.get('run_evaluation', False):
            st.subheader("Progress")

            progress_bar = st.progress(0)
            status_text = st.empty()

            # Initialize evaluator
            try:
                evaluator = RAGEvaluator()

                results = []
                total = len(df)

                for idx, row in df.iterrows():
                    status_text.text(f"Processing question {idx + 1}/{total}: {row['question'][:50]}...")

                    try:
                        # Query RAG endpoint
                        ai_answer = query_rag_endpoint(
                            question=row['question'],
                            client_principal_id=client_principal_id,
                            client_principal_name=client_principal_name,
                            client_principal_organization=client_principal_organization,
                            conversation_id=uuid.uuid4().hex,
                            endpoint_url=endpoint_url,
                            is_agentic_search_mode=is_agentic_search_mode
                        )

                        # Evaluate the answer
                        eval_result = evaluator.evaluate(
                            question=row['question'],
                            fact=row['fact'],
                            ai_answer=ai_answer
                        )

                        # Store results
                        results.append({
                            'question': row['question'],
                            'fact': row['fact'],
                            'source': row.get('source', ''),
                            'ai_answer': ai_answer,
                            'score': eval_result.score,
                            'reasoning': eval_result.reasoning,
                            'factual_accuracy': eval_result.factual_accuracy
                        })

                    except Exception as e:
                        st.error(f"Error processing question {idx + 1}: {str(e)}")
                        results.append({
                            'question': row['question'],
                            'fact': row['fact'],
                            'source': row.get('source', ''),
                            'ai_answer': f"ERROR: {str(e)}",
                            'score': 0,
                            'reasoning': f"Failed to process: {str(e)}",
                            'factual_accuracy': 'inaccurate'
                        })

                    # Update progress
                    progress_bar.progress((idx + 1) / total)
                    time.sleep(0.1)  # Small delay to avoid rate limits

                # Create results dataframe
                results_df = pd.DataFrame(results)
                st.session_state.results_df = results_df
                st.session_state.run_evaluation = False

                status_text.text("Evaluation complete!")

            except Exception as e:
                st.error(f"Failed to initialize evaluator: {str(e)}")
                st.info("Make sure O1_KEY and O1_ENDPOINT environment variables are set.")

        # Display results
        if 'results_df' in st.session_state:
            st.header("3. Results")

            results_df = st.session_state.results_df

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                avg_score = results_df['score'].mean()
                st.metric("Average Score", f"{avg_score:.2f}/5")

            with col2:
                perfect_count = len(results_df[results_df['score'] == 5])
                st.metric("Perfect Answers", f"{perfect_count}/{len(results_df)}")

            with col3:
                accurate_count = len(results_df[results_df['factual_accuracy'] == 'accurate'])
                st.metric("Accurate", f"{accurate_count}/{len(results_df)}")

            with col4:
                failing_count = len(results_df[results_df['score'] <= 2])
                st.metric("Failing (<=2)", f"{failing_count}/{len(results_df)}")

            # Results table
            st.subheader("Detailed Results")
            display_df = results_df[['question', 'fact', 'ai_answer', 'reasoning', 'score']]
            st.dataframe(
                display_df,
                use_container_width=True,
                column_config={
                    "ai_answer": st.column_config.TextColumn(
                        "Response",
                    ),
                    "score": st.column_config.NumberColumn(
                        "Score",
                        format="%d",
                        min_value=0,
                        max_value=5
                    )
                }
            )

            # Export results
            st.header("4. Export Results")

            col1, col2 = st.columns([1, 3])

            with col1:
                # Convert dataframe to Excel
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    results_df.to_excel(writer, index=False, sheet_name='Evaluation Results')

                excel_data = output.getvalue()

                st.download_button(
                    label="Download Results (Excel)",
                    data=excel_data,
                    file_name=f"rag_evaluation_results_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

            with col2:
                st.info("Download the results as an Excel file with all evaluations.")

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.info("Make sure your Excel file has 'question' and 'fact' columns.")

else:
    st.info("Upload an Excel file to get started")

# Footer
st.markdown("---")
