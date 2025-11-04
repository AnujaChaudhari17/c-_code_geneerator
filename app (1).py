# import streamlit as st
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import json
# import re

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
#     with open('sherlock_vocab.json', 'r') as f:
#         vocab_data = json.load(f)
#     return vocab_data['stoi'], vocab_data['itos'], vocab_data['vocab_size']

# @st.cache_resource
# def load_model(model_path, vocab_size, emb_dim, block_size, hidden_dim):
#     try:
#         model = MLPNextWord(vocab_size, emb_dim, block_size, hidden_dim)
#         model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
#         model.eval()
#         return model
#     except FileNotFoundError:
#         st.error(f"Error: Model file '{model_path}' not found.")
#         return None

# # --- UPDATED Generation Function (Fixes KeyError) ---
# def generate_next_tokens(model, seed_text, n_tokens_to_gen, block_size, stoi, itos, temperature, random_seed):
#     torch.manual_seed(random_seed)
#     model.eval()
    
#     # --- FIX for KeyError: '<UNK>' ---
#     # We will use '.' as our padding token, as it's guaranteed to be in the vocab.
#     pad_idx = stoi.get('.', 0) # Get index for '.' or 0 as a fallback
#     pad_token = itos.get(str(pad_idx), '.') # Get the token for that index
#     # --- END FIX ---
    
#     # Simple tokenizer
#     initial_tokens = re.sub(r'[^a-zA-Z0-9 \.]', '', seed_text.lower()).split()
    
#     # --- FIX: Filter out any unknown words from user input ---
#     tokens = [t for t in initial_tokens if t in stoi]
    
#     # Handle edge case where user input has NO known words
#     if not tokens:
#         st.warning(f"Input text contains no known words. Starting with a default token ('{pad_token}').")
#         tokens = [pad_token]
#     # --- END FIX ---
    
#     generated_tokens = []
    
#     for _ in range(n_tokens_to_gen):
#         # Prepare the context block
#         if len(tokens) < block_size:
#             # Pad with our safe padding token
#             context_tokens = [pad_token] * (block_size - len(tokens)) + tokens
#         else:
#             context_tokens = tokens[-block_size:]
            
#         # Convert context to indices
#         # All tokens in context_tokens are guaranteed to be in stoi
#         context_ix = [stoi[t] for t in context_tokens]
#         context = torch.tensor([context_ix], dtype=torch.long)
        
#         with torch.no_grad():
#             logits = model(context)
            
#             if temperature == 0.0:
#                 next_token_ix = torch.argmax(logits, dim=1).item()
#             else:
#                 probs = F.softmax(logits / temperature, dim=1)
#                 next_token_ix = torch.multinomial(probs, num_samples=1).item()
        
#         next_token = itos.get(str(next_token_ix), pad_token) # Fallback to pad_token
#         tokens.append(next_token)
#         generated_tokens.append(next_token)
        
#     # Re-join the text
#     output = seed_text
#     for token in generated_tokens:
#         output += " " + token
#     output = re.sub(r' +([.,!?])', r'\1', output) # Fix punctuation spacing
#     return output

# # --- Streamlit App UI ---
# st.title("Sherlock Holmes Next-Word Generator")

# # --- Load Vocab ---
# try:
#     stoi, itos, vocab_size = load_vocab()
#     MODEL_EMB_DIM = 32
#     MODEL_HIDDEN_SIZE = 1024
#     MODEL_BLOCK_SIZE = 8
# except Exception as e:
#     st.error(f"Error loading vocabulary 'sherlock_vocab.json': {e}")
#     st.stop()

# # --- Sidebar: Model Selection ---
# st.sidebar.header("1. Choose Model Configuration")
# model_options = {
#     "Low Epochs (Underfit)": "model_sherlock_low.pth",
#     "Medium Epochs (Good Fit)": "model_sherlock_medium.pth",
#     "High Epochs (Overfit)": "model_sherlock_high.pth"
# }
# model_choice = st.sidebar.selectbox(
#     "Select model (based on training duration):",
#     options=list(model_options.keys())
# )
# model_to_load = model_options[model_choice]

# # --- Load the SELECTED Model ---
# model = load_model(
#     model_to_load, 
#     vocab_size, 
#     MODEL_EMB_DIM, 
#     MODEL_BLOCK_SIZE, 
#     MODEL_HIDDEN_SIZE
# )
# if not model:
#     st.stop()

# # --- Sidebar: Generator Settings ---
# st.sidebar.header("2. Generator Settings")
# k_words = st.sidebar.number_input("Words to Generate", min_value=5, max_value=100, value=50)

# temperature = st.sidebar.slider(
#     "Temperature (Randomness)", 
#     min_value=0.0, max_value=2.0, value=0.8, step=0.1
# )
# st.sidebar.caption("0.0 = deterministic, 2.0 = very random")

# random_seed = st.sidebar.number_input(
#     "Random Seed", 
#     min_value=0, value=42
# )
# st.sidebar.caption("Change this to get a different random generation.")

# # --- Main Page UI ---
# seed_text_input = st.text_area("Enter seed text:", value="holmes sat in his chair", height=100)

# if st.button("Generate Text"):
#     if not seed_text_input:
#         st.warning("Please enter some seed text.")
#     else:
#         with st.spinner(f"Generating with '{model_choice}'..."):
#             generated_text = generate_next_tokens(
#                 model=model, 
#                 seed_text=seed_text_input, 
#                 n_tokens_to_gen=k_words, 
#                 block_size=MODEL_BLOCK_SIZE, 
#                 stoi=stoi, 
#                 itos=itos,
#                 temperature=temperature,
#                 random_seed=random_seed
#             )
#         st.subheader("Generated Text")
#         st.write(generated_text)


import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import re
import os       # <-- Add this
import requests # <-- Add this

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

# --- NEW DOWNLOAD LOGIC ---
# This function downloads the model file from the URL if it doesn't already exist
def download_model(model_path, url):
    if not os.path.exists(model_path):
        with st.spinner(f"Downloading {model_path}... This is a one-time setup."):
            try:
                r = requests.get(url, stream=True)
                r.raise_for_status() # Check for http errors
                with open(model_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                st.success(f"Downloaded {model_path}!")
            except Exception as e:
                st.error(f"Error downloading {model_path}: {e}")
                st.stop()

# --- Helper Functions ---
@st.cache_resource
def load_vocab():
    with open('sherlock_vocab.json', 'r') as f:
        vocab_data = json.load(f)
    return vocab_data['stoi'], vocab_data['itos'], vocab_data['vocab_size']

@st.cache_resource
def load_model(model_path, vocab_size, emb_dim, block_size, hidden_dim):
    try:
        model = MLPNextWord(vocab_size, emb_dim, block_size, hidden_dim)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"An error occurred loading the model: {e}")
        return None

# --- Generation Function (with KeyError fix) ---
def generate_next_tokens(model, seed_text, n_tokens_to_gen, block_size, stoi, itos, temperature, random_seed):
    torch.manual_seed(random_seed)
    model.eval()
    
    pad_idx = stoi.get('.', 0) 
    pad_token = itos.get(str(pad_idx), '.') 
    
    initial_tokens = re.sub(r'[^a-zA-Z0-9 \.]', '', seed_text.lower()).split()
    tokens = [t for t in initial_tokens if t in stoi]
    
    if not tokens:
        st.warning(f"Input text contains no known words. Starting with a default token ('{pad_token}').")
        tokens = [pad_token]
    
    generated_tokens = []
    
    for _ in range(n_tokens_to_gen):
        if len(tokens) < block_size:
            context_tokens = [pad_token] * (block_size - len(tokens)) + tokens
        else:
            context_tokens = tokens[-block_size:]
            
        context_ix = [stoi[t] for t in context_tokens]
        context = torch.tensor([context_ix], dtype=torch.long)
        
        with torch.no_grad():
            logits = model(context)
            
            if temperature == 0.0:
                next_token_ix = torch.argmax(logits, dim=1).item()
            else:
                probs = F.softmax(logits / temperature, dim=1)
                next_token_ix = torch.multinomial(probs, num_samples=1).item()
        
        next_token = itos.get(str(next_token_ix), pad_token)
        tokens.append(next_token)
        generated_tokens.append(next_token)
        
    output = seed_text
    for token in generated_tokens:
        output += " " + token
    output = re.sub(r' +([.,!?])', r'\1', output)
    return output

# --- Streamlit App UI ---
st.title("Sherlock Holmes Next-Word Generator")

# --- Load Vocab ---
try:
    stoi, itos, vocab_size = load_vocab()
    MODEL_EMB_DIM = 32
    MODEL_HIDDEN_SIZE = 1024
    MODEL_BLOCK_SIZE = 8
except Exception as e:
    st.error(f"Error loading vocabulary 'sherlock_vocab.json': {e}")
    st.stop()

# --- Sidebar: Model Selection ---
st.sidebar.header("1. Choose Model Configuration")

# --- NEW: URLs from your GitHub Release page ---
# !! REPLACE THESE with your actual GitHub Release URLs !!
MODEL_URLS = {
    "Low Epochs (10)": "https://github.com/AnujaChaudhari17/c-_code_geneerator/releases/download/v1.0/model_sherlock_low.pth
    "Medium Epochs (100)": "https://github.com/AnujaChaudhari17/c-_code_geneerator/releases/download/v1.0/model_sherlock_medium.pth",
    "High Epochs (200 - Final)": "https://github.com/AnujaChaudhari17/c-_code_geneerator/releases/download/v1.0/model_sherlock_high.pth"
}

# --- NEW: Maps dropdown name to local filename ---
model_options = {
    "Low Epochs (10)": "model_sherlock_low.pth",
    "Medium Epochs (100)": "model_sherlock_medium.pth",
    "High Epochs (200 - Final)": "model_sherlock_high.pth"
}

model_choice = st.sidebar.selectbox(
    "Select model (based on training duration):",
    options=list(MODEL_URLS.keys()) # Dropdown options
)

# --- NEW: Download and Load the SELECTED Model ---
# 1. Get the local filename and remote URL
model_to_load_name = model_options[model_choice]
model_url = MODEL_URLS[model_choice]

# 2. Download the model (if it's not already here)
download_model(model_to_load_name, model_url)

# 3. Load the downloaded model into memory
model = load_model(
    model_to_load_name, 
    vocab_size, 
    MODEL_EMB_DIM, 
    MODEL_BLOCK_SIZE, 
    MODEL_HIDDEN_SIZE
)
if not model:
    st.stop()

# --- Sidebar: Generator Settings ---
st.sidebar.header("2. Generator Settings")
k_words = st.sidebar.number_input("Words to Generate", min_value=5, max_value=100, value=50)

temperature = st.sidebar.slider(
    "Temperature (Randomness)", 
    min_value=0.0, max_value=2.0, value=0.8, step=0.1
)
st.sidebar.caption("0.0 = deterministic, 2.0 = very random")

random_seed = st.sidebar.number_input(
    "Random Seed", 
    min_value=0, value=42
)
st.sidebar.caption("Change this to get a different random generation.")

# --- Main Page UI ---
seed_text_input = st.text_area("Enter seed text:", value="holmes sat in his chair", height=100)

if st.button("Generate Text"):
    if not seed_text_input:
        st.warning("Please enter some seed text.")
    else:
        with st.spinner(f"Generating with '{model_choice}'..."):
            generated_text = generate_next_tokens(
                model=model, 
                seed_text=seed_text_input, 
                n_tokens_to_gen=k_words, 
                block_size=MODEL_BLOCK_SIZE, 
                stoi=stoi, 
                itos=itos,
                temperature=temperature,
                random_seed=random_seed
            )
        st.subheader("Generated Text")
        st.write(generated_text)
