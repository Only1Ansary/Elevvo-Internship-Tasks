import streamlit as st
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

MODEL_DIR = "Best Model"

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_DIR).to(device).eval()
    return tokenizer, model, device

def run_qa(tokenizer, model, device, question, context, max_answer_len=30):
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation="only_second",
        return_offsets_mapping=True,
    )
    offsets = inputs.pop("offset_mapping")[0].tolist()
    seq_ids = inputs.sequence_ids(0)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    start_probs = torch.softmax(outputs.start_logits[0], dim=-1)
    end_probs = torch.softmax(outputs.end_logits[0], dim=-1)

    # Restrict answer spans strictly to context tokens (sequence_id == 1)
    context_mask = torch.tensor([1.0 if s == 1 else 0.0 for s in seq_ids], device=device)
    start_probs = start_probs * context_mask
    end_probs = end_probs * context_mask

    # Score all valid spans within max_answer_len
    score_matrix = torch.outer(start_probs, end_probs)
    n = len(start_probs)
    idx = torch.arange(n, device=device)
    valid_span = (idx.unsqueeze(1) <= idx.unsqueeze(0)) & ((idx.unsqueeze(0) - idx.unsqueeze(1)) < max_answer_len)
    score_matrix = score_matrix * valid_span

    best_idx = torch.argmax(score_matrix).item()
    start_tok, end_tok = divmod(best_idx, n)
    confidence = score_matrix[start_tok, end_tok].item()

    start_char = offsets[start_tok][0]
    end_char = offsets[end_tok][1]
    answer = context[start_char:end_char]

    return {"answer": answer, "score": confidence, "start": start_char, "end": end_char}

st.set_page_config(page_title="Question Answering", page_icon="❓")
st.title("Question Answering with Transformers")
st.caption("Paste a passage, ask a question, and the model extracts the answer span.")

tokenizer, model, device = load_model()
context = st.text_area("Passage", height=250, placeholder="Paste your text here...")
question = st.text_input("Question")

if st.button("Get answer", type="primary"):
    if not context.strip() or not question.strip():
        st.warning("Please provide both a passage and a question.")
    else:
        out = run_qa(tokenizer, model, device, question=question, context=context, max_answer_len=30)
        st.success(out["answer"])
        st.progress(min(max(float(out["score"]), 0.0), 1.0), text=f"Confidence: {out['score']:.1%}")
        s, e = out["start"], out["end"]
        st.markdown("**Answer in context:**")
        st.markdown(
            f"{context[:s]}<mark>{context[s:e]}</mark>{context[e:]}",
            unsafe_allow_html=True,
        )
