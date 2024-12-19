import streamlit as st
import pandas as pd
import os
from langchain_core.prompts import PromptTemplate
import json
from langchain_openai import ChatOpenAI
import evaluate
from typing import List, Dict
from prompts_v1 import *
import tempfile
from langchain_groq import ChatGroq

import os


os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["GROQ_API_KEY"]= os.getenv('GROQ_API_KEY')




# Configure page settings
st.set_page_config(
    page_title="RAG Evaluator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .metric-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None

class RAGEvaluator:
    def __init__(self):
        #self.llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.2) 
        self.llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
        self.eval_prompts = {
            "diversity_metrics": diversity_metrics,
            "creativity_metric": creativity_metric,
            "groundedness_metric": groundedness_metric,
            "coherence_metric": coherence_metric,
            "pointwise_metric":pointwise_metric,
            # "pairwise_metric":pairwise_metric
        }
        
    def evaluate_custom_metrics(self, df: pd.DataFrame, selected_metrics: List[str]) -> pd.DataFrame:
        for metric in selected_metrics:
            prompt = self.eval_prompts.get(metric)
            if not prompt:
                continue
                
            review_template = PromptTemplate.from_template(prompt)
            eval_score = []
            explanation = []
            
            progress_bar = st.progress(0)
            for idx in range(len(df)):
                progress = (idx + 1) / len(df)
                progress_bar.progress(progress)
                
                question = df["question"][idx]
                answer = df["answer"][idx]
                context = df["context"][idx]
                
                final_prompt = review_template.format(
                    question=question, 
                    answer=answer, 
                    context=context
                )
                
                response = self.llm.invoke(final_prompt).content
                data_dict = json.loads(response)
                
                eval_score.append(data_dict["eval_score"])
                explanation.append(data_dict["explanation"])
            
            df[f"{metric}_score"] = eval_score
            df[f"{metric}_explanation"] = explanation
            progress_bar.empty()
            
        return df
    
    def evaluate_traditional_metrics(self, df: pd.DataFrame, selected_metrics: List[str]) -> pd.DataFrame:
        if "BLEU" in selected_metrics:
            bleu = evaluate.load('bleu')
            scores = []
            for _, row in df.iterrows():
                score = bleu.compute(
                    predictions=[row['answer']], 
                    references=[row['context']], 
                    max_order=2
                )
                scores.append(score['bleu'])
            df['bleu_score'] = scores
            
        if "ROUGE" in selected_metrics:
            rouge = evaluate.load("rouge")
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            
            for _, row in df.iterrows():
                scores = rouge.compute(
                    predictions=[row['answer']],
                    references=[row['context']],
                    rouge_types=['rouge1', 'rouge2', 'rougeL']
                )
                rouge1_scores.append(scores['rouge1'])
                rouge2_scores.append(scores['rouge2'])
                rougeL_scores.append(scores['rougeL'])
                
            df['rouge1_score'] = rouge1_scores
            df['rouge2_score'] = rouge2_scores
            df['rougeL_score'] = rougeL_scores
            
        if "Perplexity" in selected_metrics:
            try:
                perplexity = evaluate.load("perplexity", module_type="metric")
                scores = []
                for _, row in df.iterrows():
                    try:
                        score = perplexity.compute(
                            model_id="gpt2", 
                            add_start_token=False,
                            predictions=[row['answer']]
                        )
                        scores.append(score['mean_perplexity'])
                    except KeyError:
                        # If mean_perplexity is not available, try perplexity
                        scores.append(score.get('perplexity', 0))
                    except Exception as e:
                        st.warning(f"Skipping perplexity calculation for one row due to: {str(e)}")
                        scores.append(0)
                df['perplexity_score'] = scores
            except Exception as e:
                st.error(f"Error calculating perplexity: {str(e)}")
                df['perplexity_score'] = [0] * len(df)
            
        return df

def main():
    st.title("🎯 RAG Evaluator")
    st.write("Upload your data and select evaluation metrics to analyze your RAG system's performance.")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload your evaluation data (CSV/Excel)", 
        type=['csv', 'xlsx']
    )
    
    # Metric selection
    st.sidebar.subheader("Select Evaluation Metrics")
    
    custom_metrics = st.sidebar.expander("Custom Metrics", expanded=True)
    selected_custom_metrics = custom_metrics.multiselect(
        "Choose custom metrics:",
        ["diversity_metrics", "creativity_metric", "groundedness_metric", "coherence_metric","pointwise_metric"],
        default=["coherence_metric"]
    )
    
    traditional_metrics = st.sidebar.expander("Traditional Metrics", expanded=True)
    selected_traditional_metrics = traditional_metrics.multiselect(
        "Choose traditional metrics:",
        ["BLEU", "ROUGE", "Perplexity"],
        default=["BLEU"]
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Display data preview
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Initialize evaluator
            evaluator = RAGEvaluator()
            
            # Evaluation button
            if st.button("🚀 Start Evaluation", type="primary"):
                with st.spinner("Evaluating..."):
                    # Perform evaluations
                    if selected_custom_metrics:
                        df = evaluator.evaluate_custom_metrics(df, selected_custom_metrics)
                    
                    if selected_traditional_metrics:
                        df = evaluator.evaluate_traditional_metrics(df, selected_traditional_metrics)
                    
                    st.session_state.evaluation_results = df
                    
                    # Save results
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                        df.to_excel(tmp.name, index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=open(tmp.name, 'rb'),
                            file_name="rag_evaluation_results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            
            # Display results if available
            if st.session_state.evaluation_results is not None:
                st.subheader("📈 Evaluation Results")
                
                # Create tabs for different result views
                tab1, tab2 = st.tabs(["📊 Metrics Overview", "📝 Detailed Results"])
                
                with tab1:
                    # Display metric summaries
                    cols = st.columns(len(selected_custom_metrics) + len(selected_traditional_metrics))
                    
                    metric_idx = 0
                    for metric in selected_custom_metrics:
                        with cols[metric_idx]:
                            avg_score = st.session_state.evaluation_results[f"{metric}_score"].mean()
                            st.metric(
                                label=metric.replace('_', ' ').title(),
                                value=f"{avg_score:.2f}"
                            )
                        metric_idx += 1
                    
                    if "BLEU" in selected_traditional_metrics:
                        with cols[metric_idx]:
                            avg_bleu = st.session_state.evaluation_results['bleu_score'].mean()
                            st.metric(label="BLEU Score", value=f"{avg_bleu:.2f}")
                        metric_idx += 1
                    
                    if "ROUGE" in selected_traditional_metrics:
                        with cols[metric_idx]:
                            avg_rouge = st.session_state.evaluation_results['rouge1_score'].mean()
                            st.metric(label="ROUGE-1 Score", value=f"{avg_rouge:.2f}")
                        metric_idx += 1

                    if "Perplexity" in selected_traditional_metrics:
                        with cols[metric_idx]:
                            avg_rouge = st.session_state.evaluation_results['perplexity_score'].mean()
                            st.metric(label="perplexity Score", value=f"{avg_rouge:.2f}")
                        metric_idx += 1
                
                with tab2:
                    st.dataframe(
                        st.session_state.evaluation_results,
                        use_container_width=True,
                        height=400
                    )
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            
    else:
        # Display welcome message and instructions
        st.info("👈 Please upload your evaluation data file (CSV/Excel) from the sidebar to begin.")
        
        # Display sample format
        st.subheader("📋 Expected Data Format")
        sample_data = pd.DataFrame({
            'question': ['What is RAG?', 'How does RAG work?'],
            'answer': ['RAG is...', 'RAG works by...'],
            'context': ['RAG (Retrieval-Augmented Generation)...', 'The RAG process involves...']
        })
        st.dataframe(sample_data, use_container_width=True)

if __name__ == "__main__":
    main()