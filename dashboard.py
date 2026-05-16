import os
import time
import random
import sys                     # <-- ADD THIS LINE
import concurrent.futures
from typing import Dict, Any
import html
import streamlit as st
import pandas as pd
import plotly.express as px
import base64
from tigergraph_graphrag.entity_extractor import extract_entities
# sys.path.insert(0, '/Users/mac/Documents/Projects/GraphRAGHackathon/tigergraph_graphrag')
from tigergraph_graphrag.load_data import graphrag_query
try:
    from llm_only import run_llm_only
except Exception as e:
    st.warning(f"LLM-only module unavailable: {e}")
    run_llm_only = None

basic_rag = None
index_papers = None
basic_rag_error = None

def load_basic_rag_module():
    global basic_rag, index_papers, basic_rag_error
    if basic_rag is not None or index_papers is not None or basic_rag_error is not None:
        return
    try:
        import basic_rag_2m as basic_rag_mod
        basic_rag = basic_rag_mod.basic_rag
        index_papers = basic_rag_mod.index_papers
    except Exception as e:
        basic_rag_error = str(e)
        basic_rag = None
        index_papers = None

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def calculate_cost(tokens: int, price_per_token: float = 0.00002) -> float:
    return tokens * price_per_token

def run_llm_only_task(query: str) -> Dict[str, Any]:
    if run_llm_only is None:
        return {"error": "LLM-only not available"}
    try:
        result = run_llm_only(query)
        return {
            "answer": result.get("answer", ""),
            "tokens": result.get("tokens_used", 0),
            "latency": result.get("latency_ms", 0),
            "cost": result.get("cost_usd", 0),
            "error": None
        }
    except Exception as e:
        return {"error": str(e)}

def run_basic_rag_task(query: str) -> Dict[str, Any]:
    if basic_rag is None:
        load_basic_rag_module()
    if basic_rag is None:
        return {"error": basic_rag_error or "Basic RAG not available"}
    try:
        result = basic_rag(query)
        return {
            "answer": result.get("answer", ""),
            "tokens": result.get("tokens", 0),
            "latency": result.get("latency_ms", 0),
            "cost": calculate_cost(result.get("tokens", 0)),
            "error": None
        }
    except Exception as e:
        return {"error": str(e)}

def run_graphrag_task(query: str) -> Dict[str, Any]:
    if graphrag_query is None:
        return {"error": "GraphRAG not available"}
    try:
        result = graphrag_query(query)
        tokens = result.get("tokens_used", 200)
        return {
            "answer": result.get("answer", ""),
            "tokens": tokens,
            "latency": result.get("latency_ms", 0),
            "cost": calculate_cost(tokens),
            "entities": result.get("entities", []),
            "reasoning_path": result.get("reasoning_path", ""),
            "error": None
        }
    except Exception as e:
        return {"error": str(e)}

def run_all(query: str) -> Dict[str, Any]:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        f1 = executor.submit(run_llm_only_task, query)
        f2 = executor.submit(run_basic_rag_task, query)
        f3 = executor.submit(run_graphrag_task, query)
        return {
            "LLM-only": f1.result(),
            "Basic RAG": f2.result(),
            "GraphRAG": f3.result()
        }

st.set_page_config(page_title="Quantum GraphRAG", layout="wide")

if "expanded" not in st.session_state:
    st.session_state.expanded = {}
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_results" not in st.session_state:
    st.session_state.last_results = None

st.markdown("""
<style>

    .stApp {
        background: radial-gradient(circle at 10% 20%, #0a0f2a, #030617);
    }
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top, #0B0F2A, #030617) !important;
    }

    .hero {
        width: 50rem;
    }
    .hero h1 {
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #A78BFA, #6EE7F9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero h3 {
        font-size: 2rem;
        background: linear-gradient(45deg, #A78BFA, #6EE7F9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1.5rem 0rem;
    }
    .hero p {
        color: rgba(255,255,255,0.8);
        font-size: 1rem;
        line-height: 1.4;
        padding: 1.5rem 0rem;
        justify-content: space-evenly;
    }

    .glass-card {
        background: rgba(15, 23, 42, 0.72);
        backdrop-filter: blur(16px);
        border-radius: 24px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        padding: 2rem;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
        height: 100%;
    }

    .glass-card:hover {
        border-color: rgba(165, 180, 252, 0.35);
        box-shadow: 0 20px 45px -10px rgba(165, 180, 252, 0.2);
        transform: translateY(-3px);
    }

    .stTextInput > div > div > input {
        background: rgba(15, 25, 45, 0.75);
        border: 1px solid #6EE7F9 !important;
        border-radius: 50px;
        padding: 0.5rem 1rem;
        color: white;
        font-size: 1rem;
        outline: none !important;
        box-shadow: none !important;
    }

    .stTextInput > div > div > input:focus {
        border: 1px solid #A78BFA !important;
        outline: none !important;
        box-shadow: 0 0 12px rgba(167, 139, 250, 0.45) !important;
    }

    .stTextInput > div > div > input:focus-visible {
        outline: none !important;
        box-shadow: 0 0 12px rgba(167, 139, 250, 0.45) !important;
    }

    .stButton > button {
        background: rgba(20, 30, 55, 0.45);
        backdrop-filter: blur(12px);
        border-radius: 10px;
        border: 1px solid rgba(110, 231, 249, 0.2);
        padding: 0.7rem 1rem;
        margin: 0.8rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        transition: all 0.2s ease;
        height: auto;
        font-size: 1rem;
        color: white;
        width: 100%;
    }

    .stButton > button:hover {
        transform: scale(1.02);
        border-color: rgba(167, 139, 250, 0.4);
        box-shadow: 0 12px 40px rgba(167, 139, 250, 0.15);
    }

    .answer-text {
        color: #e0e7ff;
        line-height: 1.5;
        margin: 1rem 0 0.5rem 0;
    }

    .entity-chip {
        background: rgba(110, 231, 249, 0.12);
        border-radius: 30px;
        padding: 0.2rem 0.9rem;
        display: inline-block;
        margin: 0.2rem 0.4rem;
        font-size: 0.8rem;
        color: #c4e5ff;
        border: 0.5px solid rgba(110,231,249,0.3);
    }

    .metrics-row {
        display: flex;
        justify-content: space-between;
        margin-top: 1rem;
        padding-top: 0.8rem;
        border-top: 1px solid rgba(110,231,249,0.2);
        font-size: 0.85rem;
        color: #a0b4ff;
    }

    .video-container {
        border-radius: 28px;
        overflow: hidden;
        border: 1px solid rgba(110,231,249,0.3);
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        background: #00000022;
    }
</style>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1.2, 1], gap="large")

with col_left:
    st.markdown("""
    <div class="hero">
        <h1>Query the Quantum</h1>
        <h3><i>Intelligent Exploration of Quantum Computing Research</i></h3>
        <p>Leverage advanced LLM pipelines to analyze, compare, and extract structured insights from complex 
        quantum computing papers, enabling deeper understanding of algorithms, circuits, and error correction techniques.</p>
    </div>
    """, unsafe_allow_html=True)
    query = st.text_input("", placeholder="e.g., quantum error correction with surface codes", label_visibility="collapsed")
    run = st.button("Start Analysis", use_container_width=False)

with col_right:
    try:
        with open("quantum_circuit.mp4", "rb") as f:
            video_bytes = base64.b64encode(f.read()).decode()
        st.markdown(f"""
        <div class="video-container">
            <video autoplay loop muted playsinline style="width:100%; border-radius:28px;">
                <source src="data:video/mp4;base64,{video_bytes}" type="video/mp4">
            </video>
        </div>
        """, unsafe_allow_html=True)
    except:
        st.markdown(f"""
        <div class="video-container" style="height:200px; display:flex; align-items:center; justify-content:center; background:linear-gradient(145deg, #1a2350, #0b1030);">
            <span style="color:#6EE7F9; font-size:1.2rem;">🌀 Quantum Circuit Visualization (demo)</span>
        </div>
        """, unsafe_allow_html=True)


if run and query:
    with st.spinner("Analyzing quantum knowledge graphs..."):
        st.session_state.last_query = query
        st.session_state.last_results = run_all(query)
        st.session_state.expanded = {}

if st.session_state.last_results and st.session_state.last_query:
    results = st.session_state.last_results
    pipelines = list(results.keys())
    
    st.markdown(f"### Results for: {st.session_state.last_query}")
    
    cols = st.columns(3, gap="medium")
    
    for idx, pipeline in enumerate(pipelines):
        data = results[pipeline]
        if data.get("error"):
            with cols[idx]:
                st.error(f"{pipeline} failed: {data['error']}")
            continue
        
        
        full_answer = html.escape(
            str(data.get("answer", "No answer generated."))
        )
        is_expanded = st.session_state.expanded.get(pipeline, False)
        truncated = full_answer[:160] + "..." if len(full_answer) > 160 else full_answer
        
        with cols[idx]:
            card_html = f'''
            <div class="glass-card">
                <h3 style="
                    margin-top:0;
                    margin-bottom:1.2rem;
                    text-decoration: underline;
                    text-underline-offset: 6px;
                    font-weight: 600;
                ">
                    {pipeline}
                </h3>
            '''
            if is_expanded:
                card_html += f'<div class="answer-text">{full_answer}</div>'
            else:
                card_html += f'<div class="answer-text">{truncated}</div>'
            
            # Metrics
            card_html += f'''
            <div class="metrics-row">
                <span>Tokens: {data['tokens']}</span>
                <span>Latency: {data['latency']} ms</span>
            </div>
            </div>
            '''
            st.markdown(card_html, unsafe_allow_html=True)
            
            # Read more / Show less button
            if not is_expanded:
                clicked = st.button("Read more", key=f"more_{pipeline}")
            else:
                clicked = st.button("Show less", key=f"less_{pipeline}")
            
            if clicked:
                st.session_state.expanded[pipeline] = not is_expanded
                st.rerun()

    st.markdown("## 📊 Pipeline Analytics")
    
    df_metrics = pd.DataFrame([
        {"Pipeline": k, "Tokens": v.get("tokens", 0), "Latency (ms)": v.get("latency", 0)}
        for k, v in results.items() if not v.get("error")
    ])
    
    def plotly_dark(fig):
        fig.update_layout(
            plot_bgcolor="#0a0f2a",
            paper_bgcolor="#0a0f2a",
            font=dict(color="#e0e7ff", size=12),
            title_font=dict(size=15, color="#A78BFA"),
            legend=dict(font=dict(color="#cbd5ff")),
            margin=dict(t=40, l=20, r=20, b=20)
        )
        fig.update_traces(marker=dict(line=dict(width=0)))
        return fig
    
    graph_cols = st.columns(3, gap="small")
    
    with graph_cols[0]:
        with st.container():
            st.markdown('<div class="graph-wrapper">', unsafe_allow_html=True)
            fig1 = px.bar(df_metrics, x="Pipeline", y="Tokens", color="Pipeline",
                          title="🎯 Token efficiency", color_discrete_sequence=["#6EE7F9", "#A78BFA", "#F472B6"])
            fig1.update_layout(showlegend=False)
            st.plotly_chart(plotly_dark(fig1), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with graph_cols[1]:
        with st.container():
            st.markdown('<div class="graph-wrapper">', unsafe_allow_html=True)
            fig2 = px.pie(df_metrics, names="Pipeline", values="Tokens", title="🍩 Token distribution",
                          color_discrete_sequence=["#6EE7F9", "#A78BFA", "#F472B6"])
            fig2.update_traces(hole=0.45, textposition="inside", textinfo="percent+label")
            fig2.update_layout(showlegend=False)
            st.plotly_chart(plotly_dark(fig2), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with graph_cols[2]:
        with st.container():
            st.markdown('<div class="graph-wrapper">', unsafe_allow_html=True)
            fig3 = px.bar(df_metrics, x="Pipeline", y="Latency (ms)", color="Pipeline",
                          title="⚡ Latency comparison", color_discrete_sequence=["#6EE7F9", "#A78BFA", "#F472B6"])
            fig3.update_layout(showlegend=False)
            st.plotly_chart(plotly_dark(fig3), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

