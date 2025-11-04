# import streamlit as st
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import json
# import random

# # --- Model Definition ---
# # This MUST match the model you trained
# class MLPNextWord(nn.Module):
#     def __init__(self, vocab_size, emb_dim, block_size, hidden_dim):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, emb_dim)
#         self.lin1 = nn.Linear(emb_dim * block_size, hidden_dim)
#         self.relu = nn.ReLU()
#         self.lin2 = nn.Linear(hidden_dim, hidden_dim) # 2nd layer
#         self.lin_out = nn.Linear(hidden_dim, vocab_size)

#     def forward(self, x):
#         x = self.embedding(x)
#         x = x.view(x.shape[0], -1)
#         x = self.relu(self.lin1(x))
#         x = self.relu(self.lin2(x)) # Pass through 2nd layer
#         x = self.lin_out(x)
#         return x

# # --- Helper Functions ---

# @st.cache_resource
# def load_vocab():
#     with open('c_code_vocab.json', 'r') as f:
#         vocab_data = json.load(f)
#     return vocab_data['stoi'], vocab_data['itos'], vocab_data['vocab_size']

# @st.cache_resource
# def load_model(model_path, vocab_size, emb_dim, block_size, hidden_dim):
#     model = MLPNextWord(vocab_size, emb_dim, block_size, hidden_dim)
#     model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
#     model.eval()
#     return model

# def generate_next_tokens(model, seed_text, n_tokens_to_gen, block_size, stoi, itos, temperature, random_seed):
#     torch.manual_seed(random_seed)
#     random.seed(random_seed)

#     model.eval()
#     unk_idx = stoi['<UNK>']

#     # Simple tokenizer (as C code tokens are space-separated)
#     initial_tokens = seed_text.lower().split()
#     tokens = []
#     for t in initial_tokens:
#         if t in stoi:
#             tokens.append(t)
#         else:
#             tokens.append('<UNK>') # Handle OOV words

#     generated_tokens = []

#     for _ in range(n_tokens_to_gen):
#         # Pad context if needed
#         if len(tokens) < block_size:
#             context_tokens = ['<UNK>'] * (block_size - len(tokens)) + tokens
#         else:
#             context_tokens = tokens[-block_size:]

#         context_ix = [stoi.get(t, unk_idx) for t in context_tokens]
#         context = torch.tensor([context_ix], dtype=torch.long)

#         with torch.no_grad():
#             logits = model(context)
#             if temperature == 0.0:
#                 next_token_ix = torch.argmax(logits, dim=1).item()
#             else:
#                 probs = F.softmax(logits / temperature, dim=1)
#                 next_token_ix = torch.multinomial(probs, num_samples=1).item()

#         next_token = itos.get(str(next_token_ix), '<UNK>') # itos keys are strings from json

#         tokens.append(next_token)
#         generated_tokens.append(next_token)

#     # Format output
#     output = seed_text
#     for token in generated_tokens:
#         if token == '\n':
#             output += "\n"
#         else:
#             output += " " + token
#     return output

# # --- Streamlit App UI ---
# st.title("Next-Token Predictor (C++ Code)")

# # --- Load Vocab and Model ---
# try:
#     stoi, itos, vocab_size = load_vocab()

#     # --- THESE MUST MATCH YOUR TRAINED MODEL ---
#     MODEL_EMB_DIM = 32
#     MODEL_HIDDEN_SIZE = 1024
#     MODEL_BLOCK_SIZE = 8
#     # ------------------------------------------

#     model = load_model(
#         'model_cpp.pth',
#         vocab_size,
#         MODEL_EMB_DIM,
#         MODEL_BLOCK_SIZE,
#         MODEL_HIDDEN_SIZE
#     )
#     st.sidebar.success("Model loaded successfully!")
#     st.sidebar.text(f"Params: {MODEL_EMB_DIM} (Emb), {MODEL_HIDDEN_SIZE} (Hidden)")

# except FileNotFoundError:
#     st.error("Error: Model file `model_cpp.pth` or vocab `c_code_vocab.json` not found.")
#     st.stop()
# except Exception as e:
#     st.error(f"An error occurred loading the model: {e}")
#     st.stop()


# # --- App Controls (Sidebar) ---
# st.sidebar.header("Generator Settings")
# k_words = st.sidebar.number_input("Tokens to Generate", min_value=5, max_value=100, value=25)
# temperature = st.sidebar.slider("Temperature (Randomness)", min_value=0.0, max_value=2.0, value=0.8, step=0.1)
# random_seed = st.sidebar.number_input("Random Seed", min_value=0, value=42)

# # --- Main Page UI ---
# seed_text_input = st.text_area("Enter seed text:", value="if ( ptr == null ) { \n return", height=100)

# if st.button("Generate Code"):
#     if not seed_text_input:
#         st.warning("Please enter some seed text.")
#     else:
#         with st.spinner("Generating..."):
#             generated_code = generate_next_tokens(
#                 model=model,
#                 seed_text=seed_text_input,
#                 n_tokens_to_gen=k_words,
#                 block_size=MODEL_BLOCK_SIZE,
#                 stoi=stoi,
#                 itos=itos,
#                 temperature=temperature,
#                 random_seed=random_seed
#             )
#         st.subheader("Generated Code")
#         st.code(generated_code, language='c')

# %%writefile app.py

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import random

# --- Model Definition ---
# This MUST match the model you trained
class MLPNextWord(nn.Module):
    def __init__(self, vocab_size, emb_dim, block_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(emb_dim * block_size, hidden_dim)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, hidden_dim) # 2nd layer
        self.lin_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.shape[0], -1)
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x)) # Pass through 2nd layer
        x = self.lin_out(x)
        return x

# --- Helper Functions ---

@st.cache_resource
def load_vocab():
    with open('c_code_vocab.json', 'r') as f:
        vocab_data = json.load(f)
    return vocab_data['stoi'], vocab_data['itos'], vocab_data['vocab_size']

# --- MODIFIED: Load function is now cached per model path ---
@st.cache_resource
def load_model(model_path, vocab_size, emb_dim, block_size, hidden_dim):
    try:
        model = MLPNextWord(vocab_size, emb_dim, block_size, hidden_dim)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{model_path}' not found. Make sure it's in the directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred loading the model: {e}")
        return None

def generate_next_tokens(model, seed_text, n_tokens_to_gen, block_size, stoi, itos, temperature, random_seed):
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    model.eval()
    unk_idx = stoi['<UNK>']
    
    initial_tokens = seed_text.lower().split()
    tokens = [t if t in stoi else '<UNK>' for t in initial_tokens]
    
    generated_tokens = []
    
    for _ in range(n_tokens_to_gen):
        if len(tokens) < block_size:
            context_tokens = ['<UNK>'] * (block_size - len(tokens)) + tokens
        else:
            context_tokens = tokens[-block_size:]
            
        context_ix = [stoi.get(t, unk_idx) for t in context_tokens]
        context = torch.tensor([context_ix], dtype=torch.long)
        
        with torch.no_grad():
            logits = model(context)
            if temperature == 0.0:
                next_token_ix = torch.argmax(logits, dim=1).item()
            else:
                probs = F.softmax(logits / temperature, dim=1)
                next_token_ix = torch.multinomial(probs, num_samples=1).item()
        
        next_token = itos.get(str(next_token_ix), '<UNK>')
        tokens.append(next_token)
        generated_tokens.append(next_token)
        
    output = seed_text
    for token in generated_tokens:
        output += " " + token if token != '\n' else "\n"
    return output

# --- Streamlit App UI ---
st.title("Next-Token Predictor (C++ Code)")

# --- Load Vocab ---
try:
    stoi, itos, vocab_size = load_vocab()
    
    # These MUST match your trained model
    MODEL_EMB_DIM = 32
    MODEL_HIDDEN_SIZE = 1024
    MODEL_BLOCK_SIZE = 8
except Exception as e:
    st.error(f"Error loading vocabulary 'c_code_vocab.json': {e}")
    st.stop()

# --- NEW: Sidebar Controls for Model Selection ---
st.sidebar.header("1. Choose Model Configuration")

# Map display names to the real .pth files
model_options = {
    "Low Epochs (10) - Underfit": "model_cpp_low.pth",
    "Medium Epochs (50) - Good Fit": "model_cpp_medium.pth",
    "High Epochs (100) - Overfit": "model_cpp_high.pth"
}

model_choice = st.sidebar.selectbox(
    "Select model (based on training duration):",
    options=list(model_options.keys())
)

# Get the corresponding filename
model_to_load = model_options[model_choice]

# --- Load the SELECTED Model ---
model = load_model(
    model_to_load, 
    vocab_size, 
    MODEL_EMB_DIM, 
    MODEL_BLOCK_SIZE, 
    MODEL_HIDDEN_SIZE
)

if model:
    st.sidebar.success(f"Loaded: `{model_to_load}`")
else:
    st.stop() # Don't run the app if the model failed to load

# --- Generator Settings (unchanged) ---
st.sidebar.header("2. Generator Settings")
k_words = st.sidebar.number_input("Tokens to Generate", min_value=5, max_value=100, value=25)
temperature = st.sidebar.slider("Temperature (Randomness)", min_value=0.0, max_value=2.0, value=0.8, step=0.1)
random_seed = st.sidebar.number_input("Random Seed", min_value=0, value=42)

# --- Main Page UI (unchanged) ---
seed_text_input = st.text_area("Enter seed text:", value="if ( ptr == null ) { \n return", height=100)

if st.button("Generate Code"):
    if not seed_text_input:
        st.warning("Please enter some seed text.")
    else:
        with st.spinner(f"Generating with '{model_choice}'..."):
            generated_code = generate_next_tokens(
                model=model, 
                seed_text=seed_text_input, 
                n_tokens_to_gen=k_words, 
                block_size=MODEL_BLOCK_SIZE, 
                stoi=stoi, 
                itos=itos, 
                temperature=temperature,
                random_seed=random_seed
            )
        st.subheader("Generated Code")
        st.code(generated_code, language='c')
