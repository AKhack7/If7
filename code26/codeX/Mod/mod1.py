
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    ISHA - INTELLIGENT SYSTEM FOR HUMAN ASSISTANCE             ║
║                          ULTIMATE PERSONAL AI ASSISTANT                        ║
║                    WITH BEAUTIFUL ANIMATED GUI & ADVANCED LLM                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import pyttsx3
import speech_recognition as sr
import datetime
import os
import webbrowser
import pyautogui
import time
import psutil
import subprocess
import re
import threading
import socket
import logging
import queue
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import ctypes
import requests
import cv2
import numpy as np
import json
import pickle
from collections import defaultdict, Counter
import math
import random
import glob
import hashlib
import base64
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any, Union
import warnings
warnings.filterwarnings('ignore')

# === ADVANCED DEEP LEARNING IMPORTS ===
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, random_split
    import torchaudio
    import torchvision.transforms as transforms
    from transformers import (
        BertModel, 
        BertTokenizer,
        GPT2LMHeadModel,
        GPT2Tokenizer,
        pipeline,
        AutoModelForCausalLM,
        AutoTokenizer,
        Conversation
    )
    from sentence_transformers import SentenceTransformer
    DEEP_LEARNING_AVAILABLE = True
except Exception as e:
    DEEP_LEARNING_AVAILABLE = False
    print(f"⚠️ Deep learning imports failed: {e}")

# === ADVANCED COMPUTER VISION ===
try:
    import mediapipe as mp
    import face_recognition
    from deepface import DeepFace
    ADVANCED_VISION_AVAILABLE = True
except Exception:
    ADVANCED_VISION_AVAILABLE = False

# === NLP & AUDIO ===
try:
    import whisper
    ADVANCED_NLP_AVAILABLE = True
except Exception:
    ADVANCED_NLP_AVAILABLE = False

# === Initialize logging ===
logging.basicConfig(
    filename="isha_ultimate.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ============================================================================
# PART 1: ULTRA-ADVANCED NEURAL NETWORK ARCHITECTURE
# ============================================================================

@dataclass
class UltraConfig:
    """Ultra-advanced configuration for the personal LLM"""
    # Model dimensions
    vocab_size: int = 100000
    hidden_size: int = 1536  # Increased from 768
    num_hidden_layers: int = 24  # Increased from 12
    num_attention_heads: int = 24  # Increased from 12
    intermediate_size: int = 6144  # Increased from 3072
    max_position_embeddings: int = 2048  # Increased from 512
    
    # Dropout and regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    
    # Advanced features
    use_cache: bool = True
    tie_word_embeddings: bool = True
    use_flash_attention: bool = True
    use_rotary_embeddings: bool = True
    use_alibi: bool = True
    use_sparse_attention: bool = False
    sparse_block_size: int = 64
    
    # Multi-query attention
    num_key_value_heads: int = 8  # For grouped-query attention
    use_multi_query: bool = True
    
    # MoE (Mixture of Experts)
    use_moe: bool = True
    num_experts: int = 8
    expert_capacity: int = 128
    moe_top_k: int = 2
    
    # Training
    batch_size: int = 16
    learning_rate: float = 3e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0
    
    @property
    def total_params(self):
        """Estimate total parameters"""
        embed_params = self.vocab_size * self.hidden_size
        attn_params = 4 * self.hidden_size * self.hidden_size * self.num_hidden_layers
        ffn_params = 2 * self.hidden_size * self.intermediate_size * self.num_hidden_layers
        return embed_params + attn_params + ffn_params


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE)"""
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Build here to make `torch.jit.trace` work
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, 
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype()
        )
        
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)
        
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Apply rotary positional embeddings to query and key."""
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class FlashAttention(nn.Module):
    """Flash Attention implementation for faster inference"""
    
    def __init__(self, config: UltraConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        
        # Try to use flash attention if available
        self.use_flash = config.use_flash_attention and hasattr(F, 'scaled_dot_product_attention')
        
    def forward(self, query, key, value, attention_mask=None):
        if self.use_flash:
            # Use PyTorch 2.0's flash attention
            return F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False
            )
        else:
            # Fallback to standard attention
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                scores = scores + attention_mask
            attn_weights = F.softmax(scores, dim=-1)
            return torch.matmul(attn_weights, value)


class MoELayer(nn.Module):
    """Mixture of Experts layer"""
    
    def __init__(self, config: UltraConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.moe_top_k
        self.expert_capacity = config.expert_capacity
        self.hidden_size = config.hidden_size
        
        # Router
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        
        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.intermediate_size, config.hidden_size),
                nn.Dropout(config.dropout)
            ) for _ in range(config.num_experts)
        ])
        
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        
        # Get routing probabilities
        router_logits = self.router(x)  # [batch_size, seq_len, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Get top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Initialize output
        final_output = torch.zeros_like(x)
        
        # For each expert, process its assigned tokens
        for expert_idx in range(self.num_experts):
            # Find which tokens are assigned to this expert
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)  # [batch_size, seq_len]
            if not expert_mask.any():
                continue
                
            # Get the tokens assigned to this expert
            expert_input = x[expert_mask]  # [num_tokens, hidden_size]
            
            # Get the corresponding probabilities
            expert_probs = torch.where(
                top_k_indices[expert_mask] == expert_idx,
                top_k_probs[expert_mask],
                torch.zeros_like(top_k_probs[expert_mask])
            ).sum(dim=-1, keepdim=True)
            
            # Process through expert
            expert_output = self.experts[expert_idx](expert_input)
            
            # Weight by router probability
            expert_output = expert_output * expert_probs
            
            # Add to final output
            final_output[expert_mask] += expert_output
            
        return final_output


class MultiHeadAttentionUltra(nn.Module):
    """Ultra-advanced multi-head attention with grouped-query and rotary embeddings"""
    
    def __init__(self, config: UltraConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads if config.use_multi_query else config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        
        # Query projections
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        
        # Key/Value projections (with grouped-query attention)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        
        # Rotary embeddings
        if config.use_rotary_embeddings:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=config.max_position_embeddings
            )
        else:
            self.rotary_emb = None
            
        # Flash attention
        self.flash_attn = FlashAttention(config)
        
        # Dropout
        self.dropout = nn.Dropout(config.attention_dropout)
        
    def forward(self, hidden_states, attention_mask=None, position_ids=None, use_cache=False):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to queries, keys, values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Repeat key/value for grouped-query attention
        if self.num_kv_heads != self.num_heads:
            key_states = key_states.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
            value_states = value_states.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
        
        # Apply rotary embeddings
        if self.rotary_emb is not None and position_ids is not None:
            cos, sin = self.rotary_emb(value_states, seq_len=seq_len)
            query_states, key_states = apply_rotary_pos_emb(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                cos, sin, position_ids
            )
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
        else:
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
        
        # Apply attention
        attn_output = self.flash_attn(
            query_states,
            key_states,
            value_states,
            attention_mask
        )
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        attn_output = self.dropout(attn_output)
        
        return attn_output


class TransformerBlockUltra(nn.Module):
    """Ultra-advanced transformer block with MoE and pre-norm"""
    
    def __init__(self, config: UltraConfig):
        super().__init__()
        self.attention = MultiHeadAttentionUltra(config)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Use MoE or standard FFN
        if config.use_moe:
            self.ffn = MoELayer(config)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.intermediate_size, config.hidden_size),
                nn.Dropout(config.dropout)
            )
            
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        attn_output = self.attention(hidden_states, attention_mask, position_ids)
        hidden_states = residual + attn_output
        
        # Pre-norm FFN
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        ffn_output = self.ffn(hidden_states)
        hidden_states = residual + ffn_output
        
        return hidden_states


class PersonalLLMUltra(nn.Module):
    """
    ULTIMATE PERSONAL LANGUAGE MODEL
    24-layer Transformer with 1.5B+ parameters
    Features: RoPE, GQA, MoE, Flash Attention
    """
    
    def __init__(self, config: UltraConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        
        # No position embeddings - using RoPE instead
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlockUltra(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.output_layer.weight = self.token_embeddings.weight
            
        # Initialize weights
        self.apply(self._init_weights)
        
        print(f"🧠 Created Ultra LLM with {self.num_parameters():,} parameters")
        print(f"   • Layers: {config.num_hidden_layers}")
        print(f"   • Hidden size: {config.hidden_size}")
        print(f"   • Attention heads: {config.num_attention_heads}")
        print(f"   • Vocabulary: {config.vocab_size}")
        print(f"   • MoE: {config.use_moe} (Experts: {config.num_experts})")
        print(f"   • RoPE: {config.use_rotary_embeddings}")
        print(f"   • GQA: {config.use_multi_query}")
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None):
        batch_size, seq_len = input_ids.shape
        
        # Create token type ids if not provided
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
            
        # Create position ids if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        token_embeds = self.token_embeddings(input_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        
        # Combine embeddings
        hidden_states = token_embeds + token_type_embeds
        hidden_states = self.ln(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Create causal attention mask
        if attention_mask is not None:
            # Convert to 4D mask
            attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0
        
        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids)
            
        # Project to vocabulary
        logits = self.output_layer(hidden_states)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self, 
        input_ids, 
        max_length=100, 
        temperature=0.8, 
        top_k=50, 
        top_p=0.95,
        repetition_penalty=1.1,
        do_sample=True
    ):
        """
        Advanced text generation with multiple sampling strategies
        """
        self.eval()
        
        for _ in range(max_length):
            # Get logits
            logits = self.forward(input_ids)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    for token_id in set(input_ids[i].tolist()):
                        next_token_logits[i, token_id] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                for i in range(sorted_indices.shape[0]):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    next_token_logits[i, indices_to_remove] = -float('Inf')
            
            # Sample or greedy
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to input
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Stop if EOS token
            if next_token.item() == 3:  # <EOS>
                break
                
        return input_ids


class PersonalVocabularyUltra:
    """Ultra-advanced vocabulary with subword tokenization and BPE"""
    
    def __init__(self, min_freq=2, max_size=100000):
        self.word2idx = {
            "<PAD>": 0, 
            "<UNK>": 1, 
            "<SOS>": 2, 
            "<EOS>": 3, 
            "<MASK>": 4,
            "<SEP>": 5,
            "<CLS>": 6
        }
        self.idx2word = {
            0: "<PAD>", 
            1: "<UNK>", 
            2: "<SOS>", 
            3: "<EOS>", 
            4: "<MASK>",
            5: "<SEP>",
            6: "<CLS>"
        }
        self.min_freq = min_freq
        self.max_size = max_size
        self.word_counts = Counter()
        self.bpe_merges = []
        self.bpe_vocab = set()
        
    def build_vocab(self, texts):
        """Build vocabulary with BPE-style tokenization"""
        # First pass: count words and subwords
        for text in texts:
            words = self._tokenize(text)
            self.word_counts.update(words)
            
            # Add to BPE vocab
            for word in words:
                for char in word:
                    self.bpe_vocab.add(char)
                    
        # Filter by frequency and limit size
        idx = 7
        for word, count in self.word_counts.most_common(self.max_size - 7):
            if count >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
                
        print(f"📚 Ultimate vocabulary built: {len(self.word2idx)} tokens")
        print(f"   • Most common: {list(self.word2idx.keys())[:10]}")
        
    def _tokenize(self, text):
        """Advanced tokenization with subword splitting"""
        text = text.lower()
        
        # Keep punctuation as separate tokens
        text = re.sub(r'([.,!?()\[\]{}:;\"\'\-\—])', r' \1 ', text)
        
        # Split into words
        words = text.split()
        
        # Further split long words into subwords (simple BPE simulation)
        result = []
        for word in words:
            if len(word) > 10 and word not in self.word2idx:
                # Split long unknown words
                for i in range(0, len(word), 4):
                    subword = word[i:i+4]
                    result.append(subword)
            else:
                result.append(word)
                
        return result
        
    def encode(self, text, max_len=100, add_special_tokens=True):
        """Encode text to token IDs with attention mask"""
        words = self._tokenize(text)
        indices = []
        
        if add_special_tokens:
            indices.append(self.word2idx["<CLS>"])
            
        for w in words[:max_len-2]:
            if w in self.word2idx:
                indices.append(self.word2idx[w])
            else:
                # Try subword matching
                found = False
                for i in range(len(w), 0, -2):
                    sub = w[:i]
                    if sub in self.word2idx:
                        indices.append(self.word2idx[sub])
                        found = True
                        break
                if not found:
                    indices.append(self.word2idx["<UNK>"])
                    
        if add_special_tokens:
            indices.append(self.word2idx["<SEP>"])
            
        # Create attention mask
        attention_mask = [1] * len(indices)
        
        # Pad
        pad_len = max_len - len(indices)
        indices += [self.word2idx["<PAD>"]] * pad_len
        attention_mask += [0] * pad_len
        
        return indices, attention_mask
    
    def decode(self, indices, skip_special_tokens=True):
        """Decode token IDs to text"""
        words = []
        for idx in indices:
            if skip_special_tokens and idx in [0, 1, 2, 3, 4, 5, 6]:
                continue
            if idx == self.word2idx["<EOS>"]:
                break
            words.append(self.idx2word.get(idx, "<UNK>"))
            
        # Join with space and clean up punctuation
        text = " ".join(words)
        text = re.sub(r'\s+([.,!?()])', r'\1', text)
        return text
    
    def save(self, path="ultimate_vocab.pkl"):
        """Save vocabulary"""
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_counts': self.word_counts,
                'bpe_merges': self.bpe_merges,
                'bpe_vocab': self.bpe_vocab
            }, f)
            
    def load(self, path="ultimate_vocab.pkl"):
        """Load vocabulary"""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.word2idx = data['word2idx']
                self.idx2word = data['idx2word']
                self.word_counts = data['word_counts']
                self.bpe_merges = data.get('bpe_merges', [])
                self.bpe_vocab = data.get('bpe_vocab', set())
            return True
        return False


class ConversationDatasetUltra(Dataset):
    """Advanced dataset for training on conversations"""
    
    def __init__(self, conversations, vocab, max_len=100):
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        
        for conv in conversations:
            if "User:" in conv and "Assistant:" in conv:
                parts = conv.split("Assistant:")
                user_part = parts[0].replace("User:", "").strip()
                assistant_part = parts[1].strip()
                
                if user_part and assistant_part:
                    # Input is user message, target is assistant response
                    input_ids, attention_mask = vocab.encode(user_part, max_len)
                    label_ids, _ = vocab.encode(assistant_part, max_len)
                    
                    self.input_ids.append(input_ids)
                    self.attention_masks.append(attention_mask)
                    self.labels.append(label_ids)
                    
    def __len__(self):
        return len(self.input_ids)
        
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class PersonalTrainerUltra:
    """
    ULTIMATE PERSONAL AI TRAINER
    With advanced training techniques and continuous learning
    """
    
    def __init__(self):
        self.config = UltraConfig()
        self.vocab = PersonalVocabularyUltra()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Load personal conversations
        self.conversations = self._load_personal_conversations()
        
        # Load or create model
        self._load_or_create_model()
        
    def _load_personal_conversations(self):
        """Load YOUR personal conversations"""
        conv_file = "personal_conversations_ultimate.txt"
        
        if os.path.exists(conv_file):
            with open(conv_file, 'r', encoding='utf-8') as f:
                conversations = [line.strip() for line in f if line.strip()]
            print(f"📖 Loaded {len(conversations)} personal conversations")
            return conversations
        
        # Start with your conversations and generate more
        base_conversations = [
            "User: hello Assistant: Hello! I'm your personal ISHA assistant. How can I help you today?",
            "User: hi Assistant: Hi there! It's great to talk with you.",
            "User: how are you Assistant: I'm doing wonderfully! Thanks for asking. How about you?",
            "User: what is your name Assistant: I'm ISHA - your Intelligent System for Human Assistance.",
            "User: thank you Assistant: You're absolutely welcome! Is there anything else you need?",
            "User: thanks Assistant: My pleasure! I'm always here to help.",
            "User: who created you Assistant: You created me! I'm your personal AI, designed to assist you in every possible way.",
            "User: what can you do Assistant: I can open applications, take screenshots, recognize hand gestures, check weather, play music, answer questions, and learn from our conversations to become better every day!",
            "User: tell me a joke Assistant: Why don't scientists trust atoms? Because they make up everything!",
            "User: activate hand gestures Assistant: Hand gestures activated! Show your palm to the camera to control your computer.",
            "User: deactivate hand gestures Assistant: Hand gestures deactivated.",
            "User: show camera Assistant: Camera preview turned on. You can now see yourself.",
            "User: hide camera Assistant: Camera preview turned off. Gestures still work in background.",
            "User: take screenshot Assistant: Taking screenshot now. Check your ISHA captures folder.",
            "User: take selfie Assistant: Opening camera to take your selfie. Smile!",
            "User: open google Assistant: Opening Google in your browser.",
            "User: what time is it Assistant: Let me check the current time for you.",
            "User: what's the weather Assistant: Please tell me which city you want weather for.",
            "User: open calculator Assistant: Opening calculator application.",
            "User: lock screen Assistant: Locking your screen now.",
            "User: open settings Assistant: Opening Windows settings.",
            "User: play music Assistant: Playing music for you.",
            "User: stop music Assistant: Music stopped.",
            "User: volume up Assistant: Increasing volume.",
            "User: volume down Assistant: Decreasing volume.",
            "User: shutdown computer Assistant: Are you sure you want to shutdown?",
            "User: yes shutdown Assistant: Shutting down computer. Goodbye!",
            "User: restart computer Assistant: Restarting computer now.",
            "User: open file explorer Assistant: Opening File Explorer.",
            "User: open downloads Assistant: Opening Downloads folder.",
            "User: what's your purpose Assistant: My purpose is to make your life easier by providing intelligent assistance through voice, text, and hand gestures.",
            "User: do you learn Assistant: Yes! I learn from our conversations and improve over time. The more you talk to me, the better I understand you.",
            "User: you're amazing Assistant: Thank you! You're pretty amazing yourself!",
            "User: good morning Assistant: Good morning! Hope you have a wonderful day ahead!",
            "User: good night Assistant: Good night! Sleep well and sweet dreams!",
            "User: how's the weather today Assistant: Let me check the weather for you. Please specify your city.",
            "User: open youtube Assistant: Opening YouTube for you.",
            "User: open instagram Assistant: Opening Instagram.",
            "User: open whatsapp Assistant: Opening WhatsApp.",
            "User: open notepad Assistant: Opening Notepad.",
            "User: open command prompt Assistant: Opening Command Prompt.",
            "User: open microsoft word Assistant: Opening Microsoft Word.",
            "User: open excel Assistant: Opening Microsoft Excel.",
            "User: open powerpoint Assistant: Opening Microsoft PowerPoint.",
            "User: open outlook Assistant: Opening Microsoft Outlook.",
            "User: open edge Assistant: Opening Microsoft Edge.",
            "User: open chrome Assistant: Opening Google Chrome.",
            "User: open spotify Assistant: Opening Spotify.",
            "User: open discord Assistant: Opening Discord.",
            "User: open zoom Assistant: Opening Zoom.",
            "User: open vscode Assistant: Opening Visual Studio Code.",
            "User: open pycharm Assistant: Opening PyCharm.",
            "User: open settings display Assistant: Opening display settings.",
            "User: open settings sound Assistant: Opening sound settings.",
            "User: open settings network Assistant: Opening network settings.",
            "User: open settings bluetooth Assistant: Opening Bluetooth settings.",
            "User: open settings privacy Assistant: Opening privacy settings.",
            "User: open settings update Assistant: Opening Windows Update settings.",
            "User: open settings about Assistant: Opening about your PC settings.",
            "User: open settings personalization Assistant: Opening personalization settings.",
            "User: open settings apps Assistant: Opening apps settings.",
            "User: open settings accounts Assistant: Opening accounts settings.",
            "User: open settings time Assistant: Opening time and language settings.",
            "User: open settings gaming Assistant: Opening gaming settings.",
            "User: open settings ease of access Assistant: Opening ease of access settings.",
            "User: open settings search Assistant: Opening search settings.",
            "User: open settings devices Assistant: Opening devices settings.",
            "User: open settings phone Assistant: Opening phone settings.",
            "User: open settings backup Assistant: Opening backup settings.",
            "User: open settings recovery Assistant: Opening recovery settings.",
            "User: open settings activation Assistant: Opening activation settings.",
            "User: open settings find my device Assistant: Opening find my device settings.",
            "User: open settings for developers Assistant: Opening for developers settings.",
            "User: open settings windows insider Assistant: Opening Windows Insider settings.",
            "User: open settings taskbar Assistant: Opening taskbar settings.",
            "User: open settings start menu Assistant: Opening start menu settings.",
            "User: open settings notifications Assistant: Opening notifications settings.",
            "User: open settings focus assist Assistant: Opening focus assist settings.",
            "User: open settings power Assistant: Opening power settings.",
            "User: open settings battery Assistant: Opening battery settings.",
            "User: open settings storage Assistant: Opening storage settings.",
            "User: open settings tablet Assistant: Opening tablet settings.",
            "User: open settings multitasking Assistant: Opening multitasking settings.",
            "User: open settings projecting Assistant: Opening projecting settings.",
            "User: open settings remote desktop Assistant: Opening remote desktop settings.",
            "User: open settings clipboard Assistant: Opening clipboard settings.",
            "User: open settings default apps Assistant: Opening default apps settings.",
            "User: open settings optional features Assistant: Opening optional features settings.",
            "User: open settings apps for websites Assistant: Opening apps for websites settings.",
            "User: open settings video playback Assistant: Opening video playback settings.",
            "User: open settings offline maps Assistant: Opening offline maps settings.",
            "User: open settings startup apps Assistant: Opening startup apps settings.",
            "User: open settings windows security Assistant: Opening Windows Security settings.",
            "User: open settings troubleshoot Assistant: Opening troubleshoot settings.",
            "User: open settings recovery options Assistant: Opening recovery options settings.",
            "User: open settings about phone Assistant: Opening about phone settings.",
            "User: open settings windows update Assistant: Opening Windows Update settings.",
            "User: open settings delivery optimization Assistant: Opening delivery optimization settings.",
            "User: open settings advanced options Assistant: Opening advanced options settings.",
            "User: open settings view update history Assistant: Opening view update history settings.",
            "User: open settings uninstall updates Assistant: Opening uninstall updates settings.",
            "User: open settings recovery advanced startup Assistant: Opening recovery advanced startup settings.",
            "User: open settings go back Assistant: Opening go back settings.",
            "User: open settings windows update options Assistant: Opening Windows Update options settings.",
            "User: open settings pause updates Assistant: Opening pause updates settings.",
            "User: open settings active hours Assistant: Opening active hours settings.",
            "User: open settings update history Assistant: Opening update history settings.",
            "User: open settings advanced options optional updates Assistant: Opening advanced options optional updates settings.",
            "User: open settings recovery reset this pc Assistant: Opening recovery reset this PC settings.",
            "User: open settings recovery advanced startup Assistant: Opening recovery advanced startup settings.",
            "User: open settings recovery go back to previous version Assistant: Opening recovery go back to previous version settings.",
            "User: open settings recovery startup settings Assistant: Opening recovery startup settings.",
            "User: open settings recovery command prompt Assistant: Opening recovery command prompt settings.",
            "User: open settings recovery system restore Assistant: Opening recovery system restore settings.",
            "User: open settings recovery system image recovery Assistant: Opening recovery system image recovery settings.",
            "User: open settings recovery automatic repair Assistant: Opening recovery automatic repair settings.",
            "User: open settings recovery uefi firmware settings Assistant: Opening recovery UEFI firmware settings.",
            "User: open settings recovery startup repair Assistant: Opening recovery startup repair settings.",
            "User: open settings recovery system restore from drive Assistant: Opening recovery system restore from drive settings.",
            "User: open settings recovery go back to windows 10 Assistant: Opening recovery go back to Windows 10 settings.",
            "User: open settings recovery reset this pc cloud download Assistant: Opening recovery reset this PC cloud download settings.",
            "User: open settings recovery reset this pc local reinstall Assistant: Opening recovery reset this PC local reinstall settings.",
            "User: open settings recovery advanced options startup settings restart now Assistant: Opening recovery advanced options startup settings restart now.",
            "User: open settings recovery advanced options command prompt restart now Assistant: Opening recovery advanced options command prompt restart now.",
            "User: open settings recovery advanced options system restore restart now Assistant: Opening recovery advanced options system restore restart now.",
            "User: open settings recovery advanced options system image recovery restart now Assistant: Opening recovery advanced options system image recovery restart now.",
            "User: open settings recovery advanced options automatic repair restart now Assistant: Opening recovery advanced options automatic repair restart now.",
            "User: open settings recovery advanced options uefi firmware settings restart now Assistant: Opening recovery advanced options UEFI firmware settings restart now.",
            "User: open settings recovery advanced options startup repair restart now Assistant: Opening recovery advanced options startup repair restart now.",
            "User: open settings recovery advanced options system restore from drive restart now Assistant: Opening recovery advanced options system restore from drive restart now.",
            "User: open settings recovery advanced options go back to windows 10 restart now Assistant: Opening recovery advanced options go back to Windows 10 restart now.",
            "User: open settings recovery advanced options reset this pc cloud download restart now Assistant: Opening recovery advanced options reset this PC cloud download restart now.",
            "User: open settings recovery advanced options reset this pc local reinstall restart now Assistant: Opening recovery advanced options reset this PC local reinstall restart now.",
            "User: open settings windows update advanced options optional updates Assistant: Opening Windows Update advanced options optional updates settings.",
            "User: open settings windows update advanced options update history Assistant: Opening Windows Update advanced options update history settings.",
            "User: open settings windows update advanced options delivery optimization Assistant: Opening Windows Update advanced options delivery optimization settings.",
            "User: open settings windows update advanced options active hours Assistant: Opening Windows Update advanced options active hours settings.",
            "User: open settings windows update advanced options pause updates Assistant: Opening Windows Update advanced options pause updates settings.",
            "User: open settings windows update advanced options view update history Assistant: Opening Windows Update advanced options view update history settings.",
            "User: open settings windows update advanced options uninstall updates Assistant: Opening Windows Update advanced options uninstall updates settings.",
            "User: open settings windows update advanced options recovery options Assistant: Opening Windows Update advanced options recovery options settings.",
            "User: open settings windows update advanced options advanced startup Assistant: Opening Windows Update advanced options advanced startup settings.",
            "User: open settings windows update advanced options recovery advanced startup Assistant: Opening Windows Update advanced options recovery advanced startup settings.",
            "User: open settings windows update advanced options recovery reset this pc Assistant: Opening Windows Update advanced options recovery reset this PC settings.",
            "User: open settings windows update advanced options recovery go back Assistant: Opening Windows Update advanced options recovery go back settings.",
            "User: open settings windows update advanced options recovery system restore Assistant: Opening Windows Update advanced options recovery system restore settings.",
            "User: open settings windows update advanced options recovery system image recovery Assistant: Opening Windows Update advanced options recovery system image recovery settings.",
            "User: open settings windows update advanced options recovery automatic repair Assistant: Opening Windows Update advanced options recovery automatic repair settings.",
            "User: open settings windows update advanced options recovery uefi firmware settings Assistant: Opening Windows Update advanced options recovery UEFI firmware settings.",
            "User: open settings windows update advanced options recovery startup repair Assistant: Opening Windows Update advanced options recovery startup repair settings.",
            "User: open settings windows update advanced options recovery system restore from drive Assistant: Opening Windows Update advanced options recovery system restore from drive settings.",
            "User: open settings windows update advanced options recovery go back to windows 10 Assistant: Opening Windows Update advanced options recovery go back to Windows 10 settings.",
            "User: open settings windows update advanced options recovery reset this pc cloud download Assistant: Opening Windows Update advanced options recovery reset this PC cloud download settings.",
            "User: open settings windows update advanced options recovery reset this pc local reinstall Assistant: Opening Windows Update advanced options recovery reset this PC local reinstall settings.",
            "User: open settings windows update advanced options recovery advanced options startup settings restart now Assistant: Opening Windows Update advanced options recovery advanced options startup settings restart now.",
            "User: open settings windows update advanced options recovery advanced options command prompt restart now Assistant: Opening Windows Update advanced options recovery advanced options command prompt restart now.",
            "User: open settings windows update advanced options recovery advanced options system restore restart now Assistant: Opening Windows Update advanced options recovery advanced options system restore restart now.",
            "User: open settings windows update advanced options recovery advanced options system image recovery restart now Assistant: Opening Windows Update advanced options recovery advanced options system image recovery restart now.",
            "User: open settings windows update advanced options recovery advanced options automatic repair restart now Assistant: Opening Windows Update advanced options recovery advanced options automatic repair restart now.",
            "User: open settings windows update advanced options recovery advanced options uefi firmware settings restart now Assistant: Opening Windows Update advanced options recovery advanced options UEFI firmware settings restart now.",
            "User: open settings windows update advanced options recovery advanced options startup repair restart now Assistant: Opening Windows Update advanced options recovery advanced options startup repair restart now.",
            "User: open settings windows update advanced options recovery advanced options system restore from drive restart now Assistant: Opening Windows Update advanced options recovery advanced options system restore from drive restart now.",
            "User: open settings windows update advanced options recovery advanced options go back to windows 10 restart now Assistant: Opening Windows Update advanced options recovery advanced options go back to Windows 10 restart now.",
            "User: open settings windows update advanced options recovery advanced options reset this pc cloud download restart now Assistant: Opening Windows Update advanced options recovery advanced options reset this PC cloud download restart now.",
            "User: open settings windows update advanced options recovery advanced options reset this pc local reinstall restart now Assistant: Opening Windows Update advanced options recovery advanced options reset this PC local reinstall restart now."
        ]
        
        # Generate more conversations by varying patterns
        conversations = []
        for conv in base_conversations:
            conversations.append(conv)
            
            # Create variations
            if "User: open " in conv:
                # Create multiple variations of app openings
                app = conv.replace("User: open ", "").split(" Assistant:")[0]
                variations = [
                    f"User: launch {app} Assistant: Opening {app} for you.",
                    f"User: start {app} Assistant: Starting {app} now.",
                    f"User: run {app} Assistant: Running {app} application."
                ]
                conversations.extend(variations)
                
        print(f"📚 Generated {len(conversations)} training conversations")
        
        # Save conversations
        with open(conv_file, 'w', encoding='utf-8') as f:
            for conv in conversations:
                f.write(conv + '\n')
                
        return conversations
        
    def _load_or_create_model(self):
        """Load existing model or prepare for training"""
        model_path = "ultimate_personal_llm.pt"
        
        if os.path.exists(model_path):
            try:
                self.load_model(model_path)
                print("✅ Ultimate personal LLM loaded successfully")
            except Exception as e:
                print(f"⚠️ Could not load model: {e}. Will train new one.")
                self.model = None
        else:
            print("🆕 No existing model found. Ready for training.")
            self.model = None
            
    def build_vocabulary(self):
        """Build vocabulary from conversations"""
        all_texts = []
        for conv in self.conversations:
            # Split into words for vocabulary
            words = conv.split()
            all_texts.extend(words)
            
            # Also add individual characters for subword modeling
            for word in words:
                for i in range(len(word)):
                    all_texts.append(word[i:i+3])  # Add trigrams
                    
        self.vocab.build_vocab(all_texts)
        self.config.vocab_size = len(self.vocab.word2idx)
        print(f"📚 Ultimate vocabulary built: {self.config.vocab_size:,} tokens")
        
    def create_model(self):
        """Create a new model instance"""
        self.model = PersonalLLMUltra(self.config)
        self.model = self.model.to(self.device)
        return self.model
        
    def train(self, epochs=50, batch_size=8, learning_rate=3e-4):
        """
        Ultimate training with advanced techniques
        """
        if not DEEP_LEARNING_AVAILABLE:
            print("❌ Deep learning libraries not available")
            return False
            
        print("\n" + "=" * 70)
        print("🧠 TRAINING YOUR ULTIMATE PERSONAL LLM")
        print("=" * 70)
        
        # Build vocabulary
        self.build_vocabulary()
        
        # Create model
        if self.model is None:
            self.create_model()
            
        # Prepare dataset
        dataset = ConversationDatasetUltra(self.conversations, self.vocab, max_len=100)
        
        # Split into train/validation
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4 if self.device.type == 'cuda' else 0,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            num_workers=4 if self.device.type == 'cuda' else 0,
            pin_memory=True
        )
        
        # Setup optimizer with AdamW
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Learning rate scheduler with warmup
        total_steps = len(train_loader) * epochs
        
        def lr_lambda(current_step):
            if current_step < self.config.warmup_steps:
                return float(current_step) / float(max(1, self.config.warmup_steps))
            return max(
                0.0, 
                float(total_steps - current_step) / float(max(1, total_steps - self.config.warmup_steps))
            )
            
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Training loop
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_steps = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask=attention_mask)
                
                # Calculate loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=0
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip_val
                )
                
                self.optimizer.step()
                self.scheduler.step()
                
                train_loss += loss.item()
                train_steps += 1
                
            avg_train_loss = train_loss / train_steps
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_steps = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    logits = self.model(input_ids, attention_mask=attention_mask)
                    
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=0
                    )
                    
                    val_loss += loss.item()
                    val_steps += 1
                    
            avg_val_loss = val_loss / val_steps
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - LR: {self.scheduler.get_last_lr()[0]:.6f}")
                
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model("ultimate_personal_llm_best.pt")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
                    
        # Save final model
        self.save_model("ultimate_personal_llm.pt")
        self.vocab.save("ultimate_vocab.pkl")
        
        print("\n✅ Ultimate training complete!")
        print(f"📊 Best validation loss: {best_val_loss:.4f}")
        print(f"📈 Model size: {self.model.num_parameters():,} parameters")
        
        return True
    
    def save_model(self, path):
        """Save model with all components"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'vocab_size': len(self.vocab.word2idx)
        }, path)
        print(f"💾 Model saved to {path}")
        
    def load_model(self, path):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint['config']
        self.vocab.load()
        self.config.vocab_size = checkpoint['vocab_size']
        self.model = PersonalLLMUltra(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def generate_response(self, prompt, max_length=100, temperature=0.7):
        """
        Generate a response to user input using the ultimate model
        """
        if self.model is None:
            return None
            
        self.model.eval()
        
        try:
            # Encode prompt
            input_ids, attention_mask = self.vocab.encode(prompt, max_len=50)
            input_tensor = torch.tensor([input_ids], device=self.device)
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_tensor,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    do_sample=True
                )
                
            # Decode response
            response = self.vocab.decode(output_ids[0].tolist())
            
            # Clean up response - extract only the assistant's part
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            elif prompt in response:
                response = response.replace(prompt, "").strip()
                
            if not response:
                response = "I'm not sure how to respond to that. Can you rephrase?"
                
            return response
            
        except Exception as e:
            logging.error(f"Generation error: {e}")
            return None
            
    def add_conversation(self, user_msg, assistant_msg):
        """Add new conversation to memory and retrain incrementally"""
        conv = f"User: {user_msg} Assistant: {assistant_msg}"
        self.conversations.append(conv)
        
        # Save to file
        with open("personal_conversations_ultimate.txt", 'a', encoding='utf-8') as f:
            f.write(conv + '\n')
            
        # Optional: Incremental learning
        if len(self.conversations) % 10 == 0:  # Every 10 new conversations
            print("🔄 Performing incremental learning...")
            threading.Thread(target=self.train, args=(5,), daemon=True).start()
            
    def get_model_info(self):
        """Get detailed information about the model"""
        if self.model is None:
            return "No model loaded"
            
        info = {
            'total_parameters': self.model.num_parameters(),
            'vocab_size': len(self.vocab.word2idx),
            'num_layers': self.config.num_hidden_layers,
            'hidden_size': self.config.hidden_size,
            'num_heads': self.config.num_attention_heads,
            'use_moe': self.config.use_moe,
            'num_experts': self.config.num_experts,
            'use_rope': self.config.use_rotary_embeddings,
            'use_gqa': self.config.use_multi_query,
            'device': str(self.device)
        }
        
        return info


# ============================================================================
# PART 2: MAIN ISHA ASSISTANT WITH YOUR BEAUTIFUL GUI
# ============================================================================

class IshaAssistantUltimate:
    """
    ULTIMATE PERSONAL AI ASSISTANT
    With your beautiful animated GUI and ultra-advanced LLM
    """
    
    def __init__(self):
        print("\n" + "=" * 80)
        print("🤖 ISHA - ULTIMATE PERSONAL AI ASSISTANT")
        print("=" * 80)
        
        # Initialize TTS
        self._init_tts()
        
        # Initialize Ultimate Personal LLM
        print("\n🧠 Initializing Ultimate Personal LLM...")
        self.personal_trainer = PersonalTrainerUltra()
        self.personal_ai_enabled = self.personal_trainer.model is not None
        
        # Initialize Gesture Control
        print("\n🖐️ Initializing Advanced Gesture Recognition...")
        self.gesture_controller = self._init_gesture_controller()
        self.gesture_active = False
        self.gesture_thread = None
        
        # Initialize Voice Recognition
        print("\n🎤 Initializing Voice Recognition...")
        self.voice_recognizer = self._init_voice_recognizer()
        self.voice_thread = None
        self.is_listening = False
        
        # Queue for commands
        self.input_queue = queue.Queue()
        self.pending = None
        
        # Command mappings from your original code
        self._init_command_mappings()
        
        # Welcome message
        self._welcome()
        
        # Start Web UI with your beautiful GUI
        print("\n🌐 Starting Web UI with your beautiful animated GUI...")
        self._start_web_ui()
        
    def _init_tts(self):
        """Initialize text-to-speech with female voice"""
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.9)
        
        # Set female voice
        try:
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if "zira" in voice.name.lower() or "female" in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
        except:
            pass
            
    def _init_gesture_controller(self):
        """Initialize advanced gesture controller"""
        if ADVANCED_VISION_AVAILABLE:
            try:
                from advanced_gestures import AdvancedGestureController
                return AdvancedGestureController()
            except:
                pass
        
        # Fallback to basic gesture controller
        return self._create_basic_gesture_controller()
        
    def _create_basic_gesture_controller(self):
        """Create a basic gesture controller if advanced not available"""
        class BasicGestureController:
            def __init__(self):
                self.active = False
                self.show_preview = False
                
            def toggle(self):
                self.active = not self.active
                return self.active
                
            def toggle_preview(self):
                self.show_preview = not self.show_preview
                return self.show_preview
                
            def process_frame(self, frame):
                return frame, "Basic Mode", None
                
        return BasicGestureController()
        
    def _init_voice_recognizer(self):
        """Initialize voice recognizer"""
        self.recognizer = sr.Recognizer()
        self.microphone = None
        
        try:
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
        except:
            self.microphone = None
            
        # Try to load Whisper
        self.whisper_model = None
        if ADVANCED_NLP_AVAILABLE:
            try:
                self.whisper_model = whisper.load_model("base")
            except:
                pass
                
        return self
        
    def _init_command_mappings(self):
        """Initialize command mappings from your original code"""
        
        # Settings Map
        self.SETTING_MAP = {
            "display setting": ("ms-settings:display", "01"),
            "sound setting": ("ms-settings:sound", "02"),
            "notification & action setting": ("ms-settings:notifications", "03"),
            "focus assist setting": ("ms-settings:quiethours", "04"),
            "power & sleep setting": ("ms-settings:powersleep", "05"),
            "storage setting": ("ms-settings:storagesense", "06"),
            "tablet setting": ("ms-settings:tablet", "07"),
            "multitasking setting": ("ms-settings:multitasking", "08"),
            "projecting to this pc setting": ("ms-settings:project", "09"),
            "shared experiences setting": ("ms-settings:crossdevice", "010"),
            "system components setting": ("ms-settings:appsfeatures-app", "001"),
            "clipboard setting": ("ms-settings:clipboard", "002"),
            "remote desktop setting": ("ms-settings:remotedesktop", "003"),
            "optional features setting": ("ms-settings:optionalfeatures", "004"),
            "about setting": ("ms-settings:about", "005"),
            "system setting": ("ms-settings:system", "006"),
            "devices setting": ("ms-settings:devices", "007"),
            "mobile devices setting": ("ms-settings:mobile-devices", "008"),
            "network & internet setting": ("ms-settings:network", "009"),
            "personalization setting": ("ms-settings:personalization", "000"),
            "apps setting": ("ms-settings:appsfeatures", "10"),
            "account setting": ("ms-settings:yourinfo", "20"),
            "time & language setting": ("ms-settings:dateandtime", "30"),
            "gaming setting": ("ms-settings:gaming", "40"),
            "ease of access setting": ("ms-settings:easeofaccess", "50"),
            "privacy setting": ("ms-settings:privacy", "60"),
            "updated & security": ("ms-settings:windowsupdate", "70")
        }

        self.SETTING_MAP4s = {
            "01": "ms-settings:display",
            "02": "ms-settings:sound",
            "03": "ms-settings:notifications",
            "04": "ms-settings:quiethours",
            "05": "ms-settings:powersleep",
            "06": "ms-settings:storagesense",
            "07": "ms-settings:tablet",
            "08": "ms-settings:multitasking",
            "09": "ms-settings:project",
            "010": "ms-settings:crossdevice",
            "001": "ms-settings:appsfeatures-app",
            "002": "ms-settings:clipboard",
            "003": "ms-settings:remotedesktop",
            "004": "ms-settings:optionalfeatures",
            "005": "ms-settings:about",
            "006": "ms-settings:system",
            "007": "ms-settings:devices",
            "008": "ms-settings:mobile-devices",
            "009": "ms-settings:network",
            "000": "ms-settings:personalization",
            "10": "ms-settings:appsfeatures",
            "20": "ms-settings:yourinfo",
            "30": "ms-settings:dateandtime",
            "40": "ms-settings:gaming",
            "50": "ms-settings:easeofaccess",
            "60": "ms-settings:privacy",
            "70": "ms-settings:windowsupdate"
        }

        self.apps_commands = {
            "alarms & clock": ("ms-clock:", "a1"),
            "calculator": ("calc", "c1"),
            "calendar": ("outlookcal:", "c2"),
            "camera": ("microsoft.windows.camera:", "c3"),
            "copilot": ("ms-copilot:", "c4"),
            "cortana": ("ms-cortana:", "c5"),
            "game bar": ("ms-gamebar:", "gb1"),
            "groove music": ("mswindowsmusic:", "gm1"),
            "mail": ("outlookmail:", "m1"),
            "maps": ("bingmaps:", "map1"),
            "microsoft edge": ("msedge", "me1"),
            "microsoft solitaire collection": ("ms-solitaire:", "mc1"),
            "microsoft store": ("ms-windows-store:", "mst1"),
            "mixed reality portal": ("ms-mixedreality:", "mp1"),
            "movies & tv": ("mswindowsvideo:", "mt1"),
            "office": ("ms-office:", "o1"),
            "onedrive": ("ms-onedrive:", "oe"),
            "onenote": ("ms-onenote:", "one"),
            "outlook": ("outlookmail:", "ouk"),
            "outlook (classic)": ("ms-outlook:", "oc1"),
            "paint": ("mspaint", "p1"),
            "paint 3d": ("ms-paint:", "p3d"),
            "phone link": ("ms-phonelink:", "pk"),
            "power point": ("ms-powerpoint:", "pt"),
            "settings": ("ms-settings:", "ss"),
            "skype": ("skype:", "sk1"),
            "snip & sketch": ("ms-snip:", "s0h"),
            "sticky note": ("ms-stickynotes:", "s1e"),
            "tips": ("ms-tips:", "ts0"),
            "voice recorder": ("ms-soundrecorder:", "vr0"),
            "weather": ("msnweather:", "w1"),
            "windows backup": ("ms-settings:backup", "wb1"),
            "windows security": ("ms-settings:windowsdefender", "ws1"),
            "word": ("ms-word:", "wrd"),
            "xbox": ("ms-xbox:", "xb"),
            "about your pc": ("ms-settings:about", "apc")
        }

        self.apps_commands4q = {
            "a1": "ms-clock:",
            "c1": "calc",
            "c2": "outlookcal:",
            "c3": "microsoft.windows.camera:",
            "c4": "ms-copilot:",
            "c5": "ms-cortana:",
            "gb1": "ms-gamebar:",
            "gm1": "mswindowsmusic:",
            "m1": "outlookmail:",
            "map1": "bingmaps:",
            "me1": "msedge",
            "mc1": "ms-solitaire:",
            "mst1": "ms-windows-store:",
            "mp1": "ms-mixedreality:",
            "mt1": "mswindowsvideo:",
            "o1": "ms-office:",
            "oe": "ms-onedrive:",
            "one": "ms-onenote:",
            "ouk": "outlookmail:",
            "oc1": "ms-outlook:",
            "p1": "mspaint",
            "p3d": "ms-paint:",
            "pk": "ms-phonelink:",
            "pt": "ms-powerpoint:",
            "ss": "ms-settings:",
            "sk1": "skype:",
            "s0h": "ms-snip:",
            "s1e": "ms-stickynotes:",
            "ts0": "ms-tips:",
            "vr0": "ms-soundrecorder:",
            "w1": "msnweather:",
            "wb1": "ms-settings:backup",
            "ws1": "ms-settings:windowsdefender",
            "wrd": "ms-word:",
            "xb": "ms-xbox:",
            "apc": "ms-settings:about"
        }

        self.software_dict = {
            "notepad": "notepad",
            "ms word": "winword",
            "command prompt": "cmd",
            "excel": "excel",
            "vscode": "code",
            "word16": "winword",
            "file explorer": "explorer",
            "edge": "msedge",
            "microsoft 365 copilot": "ms-copilot:",
            "outlook": "outlook",
            "microsoft store": "ms-windows-store:",
            "photos": "microsoft.photos:",
            "xbox": "xbox:",
            "solitaire": "microsoft.microsoftsolitairecollection:",
            "clipchamp": "clipchamp",
            "to do": "microsoft.todos:",
            "linkedin": "https://www.linkedin.com",
            "calculator": "calc",
            "news": "bingnews:",
            "one drive": "onedrive",
            "onenote 2016": "onenote",
            "google": "https://www.google.com"
        }

        # Merge all command dictionaries
        self.commands_dict = {**self.SETTING_MAP, **self.SETTING_MAP4s,
                              **self.software_dict, **self.apps_commands,
                              **self.apps_commands4q}
        self.commands_dict = {k: v if isinstance(v, str) else v[0]
                              for k, v in self.commands_dict.items()}

        self.settings_display_to_cmd = {
            f"{name} ({code})": cmd for name, (cmd, code) in self.SETTING_MAP.items()}
        self.apps_display_to_cmd = {name: cmd for name, (cmd, code) in self.apps_commands.items()}
        
    def _welcome(self):
        """Welcome message"""
        hour = datetime.datetime.now().hour
        
        if 5 <= hour < 12:
            greeting = "Good morning"
        elif 12 <= hour < 17:
            greeting = "Good afternoon"
        elif 17 <= hour < 21:
            greeting = "Good evening"
        else:
            greeting = "Good night"
            
        message = f"{greeting}! I am ISHA, your ultimate personal AI assistant. I have a {self.personal_trainer.model.num_parameters() if self.personal_trainer.model else 0:,} parameter neural network and can understand natural language, recognize hand gestures, and learn from our conversations. How can I help you today?"
        
        self.speak(message)
        
    def _start_web_ui(self):
        """Start web UI with your beautiful animated GUI"""
        
        class CustomHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed_path = urlparse(self.path)
                query_params = parse_qs(parsed_path.query)
                path = parsed_path.path

                if path == '/':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(self.server.assistant.get_html().encode())
                    
                elif path == '/command':
                    cmd = query_params.get('cmd', [None])[0]
                    if cmd is None:
                        self.send_response(400)
                        self.end_headers()
                        return
                        
                    if self.server.assistant.pending:
                        self.server.assistant.input_queue.put(cmd)
                        self.send_response(200)
                        self.send_header('Content-type', 'text/plain')
                        self.end_headers()
                        self.wfile.write(b'Input received')
                    else:
                        response = self.server.assistant.process_command(cmd)
                        self.send_response(200)
                        self.send_header('Content-type', 'text/plain')
                        self.end_headers()
                        self.wfile.write(response.encode())
                        
                elif path == '/voice':
                    self.server.assistant.toggle_voice()
                    message = "Microphone toggled"
                    self.send_response(200)
                    self.send_header('Content-type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(message.encode())
                    
                elif path == '/gesture':
                    self.server.assistant.toggle_gestures()
                    message = "Gestures toggled"
                    self.send_response(200)
                    self.send_header('Content-type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(message.encode())
                    
                elif path == '/camera':
                    self.server.assistant.toggle_camera_preview()
                    message = "Camera toggled"
                    self.send_response(200)
                    self.send_header('Content-type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(message.encode())
                    
                elif path == '/status':
                    status = {
                        'gestures': self.server.assistant.gesture_active,
                        'camera': self.server.assistant.gesture_controller.show_preview if hasattr(self.server.assistant.gesture_controller, 'show_preview') else False,
                        'voice': self.server.assistant.is_listening,
                        'ai': self.server.assistant.personal_ai_enabled
                    }
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(status).encode())
                    
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass

        server = HTTPServer(('localhost', 8000), CustomHandler)
        server.assistant = self
        threading.Thread(target=server.serve_forever, daemon=True).start()
        webbrowser.open('http://localhost:8000/')
        
    def get_html(self):
        """Your beautiful animated GUI HTML/CSS"""
        
        # Generate apps and settings HTML
        apps_html = ''.join(
            f'<div class="app-item" data-command="open {name}" style="margin:6px 0; cursor:pointer;">• {name}</div>'
            for name in sorted(self.apps_display_to_cmd.keys())[:50]  # Limit for performance
        )
        settings_html = ''.join(
            f'<div class="setting-item" data-command="open {name}" style="margin:6px 0; cursor:pointer;">• {name}</div>'
            for name in sorted(self.SETTING_MAP.keys())[:30]  # Limit for performance
        )

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>ISHA Ultimate AI Assistant</title>
    <style>
        :root {{
            --bg1: #050519;
            --bg2: #0f1636;
            --neon1: #00e0ff;
            --neon2: #7b4bff;
            --neon3: #ff4b8b;
            --glass: rgba(255, 255, 255, 0.04);
            --success: #00ff88;
            --warning: #ffaa00;
        }}

        * {{
            box-sizing: border-box;
            -webkit-font-smoothing: antialiased;
            font-family: "Segoe UI", Inter, system-ui, sans-serif;
            margin: 0;
            padding: 0;
        }}

        html, body {{
            height: 100%;
            margin: 0;
            background: linear-gradient(135deg, var(--bg1), var(--bg2));
            color: #e8f6ff;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            position: relative;
        }}

        /* Animated background */
        .background {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
            overflow: hidden;
        }}
        
        .background span {{
            position: absolute;
            display: block;
            width: 20px;
            height: 20px;
            background: rgba(255, 255, 255, 0.05);
            animation: animate 25s linear infinite;
            bottom: -150px;
        }}
        
        .background span:nth-child(1) {{
            left: 25%;
            width: 80px;
            height: 80px;
            animation-delay: 0s;
        }}
        
        .background span:nth-child(2) {{
            left: 10%;
            width: 20px;
            height: 20px;
            animation-delay: 2s;
            animation-duration: 12s;
        }}
        
        .background span:nth-child(3) {{
            left: 70%;
            width: 20px;
            height: 20px;
            animation-delay: 4s;
        }}
        
        .background span:nth-child(4) {{
            left: 40%;
            width: 60px;
            height: 60px;
            animation-delay: 0s;
            animation-duration: 18s;
        }}
        
        .background span:nth-child(5) {{
            left: 65%;
            width: 20px;
            height: 20px;
            animation-delay: 0s;
        }}
        
        .background span:nth-child(6) {{
            left: 75%;
            width: 110px;
            height: 110px;
            animation-delay: 3s;
        }}
        
        .background span:nth-child(7) {{
            left: 35%;
            width: 150px;
            height: 150px;
            animation-delay: 7s;
        }}
        
        .background span:nth-child(8) {{
            left: 50%;
            width: 25px;
            height: 25px;
            animation-delay: 15s;
            animation-duration: 45s;
        }}
        
        .background span:nth-child(9) {{
            left: 20%;
            width: 15px;
            height: 15px;
            animation-delay: 2s;
            animation-duration: 35s;
        }}
        
        .background span:nth-child(10) {{
            left: 85%;
            width: 150px;
            height: 150px;
            animation-delay: 0s;
            animation-duration: 11s;
        }}
        
        @keyframes animate {{
            0% {{
                transform: translateY(0) rotate(0deg);
                opacity: 1;
                border-radius: 0;
            }}
            100% {{
                transform: translateY(-1000px) rotate(720deg);
                opacity: 0;
                border-radius: 50%;
            }}
        }}

        .container {{
            width: 600px;
            max-width: calc(100% - 40px);
            height: 720px;
            border-radius: 32px;
            position: relative;
            z-index: 1;
            padding: 35px;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.01));
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 30px 80px rgba(5, 10, 40, 0.8), 0 0 0 1px rgba(0, 224, 255, 0.1) inset;
            overflow: hidden;
            backdrop-filter: blur(12px);
        }}

        .topbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 20px;
        }}

        .title {{
            font-weight: 700;
            font-size: 20px;
            color: #cfeeff;
            display: flex;
            align-items: center;
            gap: 12px;
            text-shadow: 0 0 10px rgba(0, 224, 255, 0.3);
        }}

        .title .dot {{
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--neon1), var(--neon2));
            animation: pulse 2s infinite;
            box-shadow: 0 0 15px var(--neon1);
        }}

        @keyframes pulse {{
            0%, 100% {{
                opacity: 1;
                box-shadow: 0 0 5px var(--neon1), 0 0 15px var(--neon1);
            }}
            50% {{
                opacity: 0.8;
                box-shadow: 0 0 20px var(--neon1), 0 0 40px var(--neon2), 0 0 60px var(--neon3);
            }}
        }}

        .status-badge {{
            opacity: 0.9;
            font-size: 14px;
            color: #bfefff;
            text-shadow: 0 0 5px rgba(0,224,255,0.3);
            padding: 8px 16px;
            background: rgba(0, 224, 255, 0.1);
            border-radius: 30px;
            border: 1px solid rgba(0, 224, 255, 0.2);
            backdrop-filter: blur(5px);
            animation: glow 3s infinite;
        }}

        @keyframes glow {{
            0%, 100% {{
                box-shadow: 0 0 5px rgba(0, 224, 255, 0.3);
            }}
            50% {{
                box-shadow: 0 0 15px rgba(0, 224, 255, 0.6), 0 0 30px rgba(123, 75, 255, 0.3);
            }}
        }}

        .stage {{
            width: 100%;
            height: 380px;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }}

        .core {{
            width: 260px;
            height: 260px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            background: radial-gradient(circle at 40% 30%, rgba(255, 255, 255, 0.15), rgba(0, 224, 255, 0.08));
            border: 1px solid rgba(0, 224, 255, 0.2);
            position: relative;
            animation: float 4s ease-in-out infinite, rotate 20s linear infinite;
            box-shadow: 0 0 30px rgba(0, 224, 255, 0.2), 0 0 60px rgba(123, 75, 255, 0.1);
        }}

        .core::before {{
            content: '';
            position: absolute;
            width: 100%;
            height: 100%;
            border-radius: 50%;
            border: 2px solid transparent;
            background: linear-gradient(135deg, var(--neon1), var(--neon2), var(--neon3)) border-box;
            -webkit-mask: linear-gradient(#fff 0 0) padding-box, linear-gradient(#fff 0 0);
            -webkit-mask-composite: destination-out;
            mask-composite: exclude;
            opacity: 0.6;
            animation: rotate 10s linear infinite;
        }}

        .core::after {{
            content: '';
            position: absolute;
            width: 120%;
            height: 120%;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(0, 224, 255, 0.1) 0%, transparent 70%);
            animation: pulse 3s ease-in-out infinite;
        }}

        @keyframes float {{
            0%, 100% {{
                transform: translateY(0px) rotate(0deg);
            }}
            50% {{
                transform: translateY(-15px) rotate(5deg);
            }}
        }}

        @keyframes rotate {{
            from {{
                transform: rotate(0deg);
            }}
            to {{
                transform: rotate(360deg);
            }}
        }}

        .label {{
            font-weight: 900;
            font-size: 42px;
            letter-spacing: 5px;
            color: #e9fbff;
            text-shadow: 0 0 10px rgba(0, 224, 255, 0.5), 0 0 20px rgba(123, 75, 255, 0.3), 0 0 30px rgba(255, 75, 139, 0.2);
            position: relative;
            z-index: 2;
            animation: textGlow 2s ease-in-out infinite;
        }}

        @keyframes textGlow {{
            0%, 100% {{
                text-shadow: 0 0 10px rgba(0, 224, 255, 0.5), 0 0 20px rgba(123, 75, 255, 0.3);
            }}
            50% {{
                text-shadow: 0 0 20px rgba(0, 224, 255, 0.8), 0 0 40px rgba(123, 75, 255, 0.5), 0 0 60px rgba(255, 75, 139, 0.3);
            }}
        }}

        .datetime {{
            margin-top: 25px;
            text-align: center;
            color: var(--neon1);
            font-weight: 700;
            font-size: 16px;
        }}

        .datetime .time {{
            font-size: 36px;
            color: #dffbff;
            text-shadow: 0 0 10px rgba(0, 224, 255, 0.3);
            font-weight: 700;
        }}

        .datetime .date {{
            font-size: 16px;
            opacity: 0.9;
            margin-top: 5px;
        }}

        .ai-stats {{
            margin-top: 15px;
            display: flex;
            justify-content: center;
            gap: 15px;
            font-size: 12px;
            color: rgba(255, 255, 255, 0.6);
        }}

        .stat {{
            padding: 4px 12px;
            background: rgba(0, 224, 255, 0.05);
            border-radius: 20px;
            border: 1px solid rgba(0, 224, 255, 0.1);
        }}

        .controls {{
            margin-top: 30px;
            width: 100%;
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .input {{
            flex: 1;
            height: 54px;
            border-radius: 16px;
            padding: 10px 20px;
            background: var(--glass);
            border: 1px solid rgba(255, 255, 255, 0.05);
            color: #e8f6ff;
            outline: none;
            font-size: 16px;
            transition: all 0.3s ease;
            backdrop-filter: blur(5px);
        }}

        .input::placeholder {{
            color: rgba(255, 255, 255, 0.3);
        }}

        .input:focus {{
            border-color: var(--neon1);
            box-shadow: 0 0 30px rgba(0, 224, 255, 0.2);
            transform: scale(1.02);
            background: rgba(0, 224, 255, 0.02);
        }}

        .icon-btn {{
            width: 54px;
            height: 54px;
            border-radius: 16px;
            border: none;
            background: linear-gradient(180deg, #0c1228, #05051a);
            color: var(--neon1);
            cursor: pointer;
            font-size: 20px;
            border: 1px solid rgba(0, 224, 255, 0.1);
            transition: all 0.2s ease;
            position: relative;
            overflow: hidden;
        }}

        .icon-btn::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
            transition: left 0.5s;
        }}

        .icon-btn:hover::before {{
            left: 100%;
        }}

        .icon-btn:hover {{
            border-color: var(--neon1);
            box-shadow: 0 0 30px rgba(0, 224, 255, 0.3);
            transform: scale(1.05);
            color: white;
        }}

        .icon-btn.active {{
            background: linear-gradient(135deg, var(--neon1), var(--neon2));
            color: white;
            border-color: white;
        }}

        .popup {{
            position: absolute;
            width: 340px;
            min-height: 120px;
            max-height: 500px;
            border-radius: 16px;
            padding: 16px;
            background: rgba(10, 12, 22, 0.98);
            border: 1px solid rgba(0, 224, 255, 0.1);
            display: none;
            z-index: 20;
            cursor: move;
            transition: all 0.4s cubic-bezier(0.2, 0.9, 0.3, 1);
            backdrop-filter: blur(15px);
            overflow-y: auto;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(0, 224, 255, 0.2) inset;
        }}

        #appPopup {{
            left: -360px;
            top: 120px;
        }}

        #settingsPopup {{
            right: -360px;
            top: 140px;
        }}

        #appPopup.active {{
            left: 40px;
        }}

        #settingsPopup.active {{
            right: 40px;
        }}

        .popup .head {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
            margin-bottom: 12px;
            color: var(--neon1);
            font-weight: 700;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(0, 224, 255, 0.2);
            font-size: 16px;
        }}

        .popup .close {{
            width: 36px;
            height: 36px;
            border-radius: 10px;
            display: inline-grid;
            place-items: center;
            background: rgba(255, 255, 255, 0.02);
            cursor: pointer;
            color: #fff;
            border: 1px solid rgba(255, 255, 255, 0.03);
            transition: all 0.2s ease;
            font-size: 16px;
        }}

        .popup .close:hover {{
            background: rgba(255, 0, 0, 0.2);
            border-color: #ff4444;
            transform: rotate(90deg);
        }}

        .app-item, .setting-item {{
            margin: 8px 0;
            padding: 8px 12px;
            cursor: pointer;
            border-radius: 8px;
            transition: all 0.2s cubic-bezier(0.2, 0.9, 0.3, 1);
            color: #cfeeff;
            font-size: 14px;
            border: 1px solid transparent;
        }}

        .app-item:hover, .setting-item:hover {{
            background: rgba(0, 224, 255, 0.1);
            border-color: rgba(0, 224, 255, 0.3);
            padding-left: 20px;
            color: white;
            transform: translateX(5px);
        }}

        .right {{
            display: flex;
            gap: 10px;
        }}

        /* Custom scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
        }}

        ::-webkit-scrollbar-track {{
            background: rgba(255, 255, 255, 0.02);
            border-radius: 10px;
        }}

        ::-webkit-scrollbar-thumb {{
            background: rgba(0, 224, 255, 0.3);
            border-radius: 10px;
        }}

        ::-webkit-scrollbar-thumb:hover {{
            background: rgba(0, 224, 255, 0.5);
        }}

        /* Response area */
        .response-area {{
            margin-top: 20px;
            padding: 15px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 12px;
            border: 1px solid rgba(0, 224, 255, 0.1);
            max-height: 100px;
            overflow-y: auto;
            font-size: 14px;
            color: #e8f6ff;
        }}

        /* Status indicators */
        .status-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 5px;
        }}
        
        .status-dot.on {{
            background: #00ff88;
            box-shadow: 0 0 10px #00ff88;
        }}
        
        .status-dot.off {{
            background: #ff4444;
            box-shadow: 0 0 10px #ff4444;
        }}
    </style>
</head>
<body>
    <!-- Animated background -->
    <div class="background">
        <span></span>
        <span></span>
        <span></span>
        <span></span>
        <span></span>
        <span></span>
        <span></span>
        <span></span>
        <span></span>
        <span></span>
    </div>

    <div class="container" id="container">
        <!-- Top Bar -->
        <div class="topbar">
            <div class="title">
                <span class="dot"></span> ISHA ULTIMATE
            </div>
            <div class="status-badge" id="statusBadge">
                activet 156
            </div>
        </div>

        <!-- Center Stage with Animated Circle -->
        <div class="stage">
            <div class="core">
                <div class="label">ISHA</div>
            </div>
        </div>

        <!-- Date & Time -->
        <div class="datetime" id="datetime">
            <div class="time" id="time">--:--:--</div>
            <div class="date" id="date">Loading date...</div>
        </div>
        
        <!-- AI Stats -->
        <div class="ai-stats" id="aiStats">
            <div class="stat">🧠 1.5B params</div>
            <div class="stat">🖐️ Gestures: <span id="gestureStatus">OFF</span></div>
            <div class="stat">🎤 Voice: <span id="voiceStatus">OFF</span></div>
        </div>

        <!-- Response Area -->
        <div class="response-area" id="response">
            Ready to assist you...
        </div>

        <!-- Controls -->
        <div class="controls">
            <input 
                class="input" 
                id="cmd" 
                placeholder="Type command or ask anything..." 
                autofocus
            />
            <div class="right">
                <button class="icon-btn" id="appBtn" title="Applications (A)">A</button>
                <button class="icon-btn" id="settingsBtn" title="Settings (S)">S</button>
                <button class="icon-btn" id="voiceBtn" title="Voice (V)">V</button>
                <button class="icon-btn" id="gestureBtn" title="Gestures (G)">G</button>
            </div>
        </div>
    </div>

    <!-- Applications Popup -->
    <div class="popup" id="appPopup">
        <div class="head">
            <div>📱 Applications</div>
            <div class="close" data-close="appPopup">✕</div>
        </div>
        <div style="font-size: 14px; color: #cfeeff; max-height: 350px; overflow-y: auto;">
            {apps_html}
        </div>
    </div>

    <!-- Settings Popup -->
    <div class="popup" id="settingsPopup">
        <div class="head">
            <div>⚙️ Settings</div>
            <div class="close" data-close="settingsPopup">✕</div>
        </div>
        <div style="font-size: 14px; color: #cfeeff; max-height: 350px; overflow-y: auto;">
            {settings_html}
        </div>
    </div>

    <script>
        // ============================================
        // LIVE DATE & TIME UPDATER
        // ============================================
        function updateDateTime() {{
            const now = new Date();
            const timeEl = document.getElementById('time');
            const dateEl = document.getElementById('date');
            
            timeEl.textContent = now.toLocaleTimeString([], {{ 
                hour: '2-digit', 
                minute: '2-digit', 
                second: '2-digit', 
                hour12: false 
            }});
            
            dateEl.textContent = now.toLocaleDateString([], {{ 
                weekday: 'long', 
                year: 'numeric', 
                month: 'long', 
                day: 'numeric' 
            }});
        }}
        
        updateDateTime();
        setInterval(updateDateTime, 500);

        // ============================================
        // UPDATE STATUS
        // ============================================
        function updateStatus() {{
            fetch('/status')
                .then(response => response.json())
                .then(data => {{
                    document.getElementById('gestureStatus').textContent = data.gestures ? 'ON' : 'OFF';
                    document.getElementById('voiceStatus').textContent = data.voice ? 'ON' : 'OFF';
                    
                    // Update button states
                    document.getElementById('gestureBtn').classList.toggle('active', data.gestures);
                    document.getElementById('voiceBtn').classList.toggle('active', data.voice);
                }})
                .catch(err => console.error('Status error:', err));
        }}
        
        setInterval(updateStatus, 2000);

        // ============================================
        // DRAGGABLE POPUPS
        // ============================================
        function makeDraggable(el) {{
            let dragging = false;
            let offsetX = 0;
            let offsetY = 0;

            el.addEventListener('mousedown', (e) => {{
                dragging = true;
                offsetX = e.clientX - el.offsetLeft;
                offsetY = e.clientY - el.offsetTop;
                el.style.transition = 'none';
                document.body.style.userSelect = 'none';
                e.preventDefault();
            }});

            window.addEventListener('mousemove', (e) => {{
                if (!dragging) return;
                
                let newLeft = e.clientX - offsetX;
                let newTop = e.clientY - offsetY;
                
                newLeft = Math.max(10, Math.min(newLeft, window.innerWidth - el.offsetWidth - 10));
                newTop = Math.max(10, Math.min(newTop, window.innerHeight - el.offsetHeight - 10));
                
                el.style.left = newLeft + 'px';
                el.style.top = newTop + 'px';
            }});

            window.addEventListener('mouseup', () => {{
                if (dragging) {{
                    dragging = false;
                    el.style.transition = '';
                    document.body.style.userSelect = '';
                }}
            }});
        }}

        // ============================================
        // POPUP TOGGLE
        // ============================================
        const appBtn = document.getElementById('appBtn');
        const settingsBtn = document.getElementById('settingsBtn');
        const voiceBtn = document.getElementById('voiceBtn');
        const gestureBtn = document.getElementById('gestureBtn');
        const appPopup = document.getElementById('appPopup');
        const settingsPopup = document.getElementById('settingsPopup');

        appBtn.addEventListener('click', (e) => {{
            e.stopPropagation();
            const isActive = appPopup.classList.toggle('active');
            
            if (isActive) {{
                appPopup.style.display = 'block';
                settingsPopup.classList.remove('active');
                settingsPopup.style.display = 'none';
            }} else {{
                setTimeout(() => {{
                    if (!appPopup.classList.contains('active')) {{
                        appPopup.style.display = 'none';
                    }}
                }}, 200);
            }}
        }});

        settingsBtn.addEventListener('click', (e) => {{
            e.stopPropagation();
            const isActive = settingsPopup.classList.toggle('active');
            
            if (isActive) {{
                settingsPopup.style.display = 'block';
                appPopup.classList.remove('active');
                appPopup.style.display = 'none';
            }} else {{
                setTimeout(() => {{
                    if (!settingsPopup.classList.contains('active')) {{
                        settingsPopup.style.display = 'none';
                    }}
                }}, 200);
            }}
        }});

        voiceBtn.addEventListener('click', () => {{
            fetch('/voice').then(res => res.text()).then(() => {{
                updateStatus();
            }});
        }});

        gestureBtn.addEventListener('click', () => {{
            fetch('/gesture').then(res => res.text()).then(() => {{
                updateStatus();
            }});
        }});

        document.querySelectorAll('.close').forEach(btn => {{
            btn.addEventListener('click', (e) => {{
                e.stopPropagation();
                const id = btn.dataset.close;
                const popup = document.getElementById(id);
                popup.classList.remove('active');
                setTimeout(() => {{
                    popup.style.display = 'none';
                }}, 200);
            }});
        }});

        makeDraggable(appPopup);
        makeDraggable(settingsPopup);

        document.querySelectorAll('.app-item').forEach(item => {{
            item.addEventListener('click', () => {{
                const cmd = item.getAttribute('data-command');
                sendCommand(cmd);
                appPopup.classList.remove('active');
                setTimeout(() => appPopup.style.display = 'none', 200);
            }});
        }});
        
        document.querySelectorAll('.setting-item').forEach(item => {{
            item.addEventListener('click', () => {{
                const cmd = item.getAttribute('data-command');
                sendCommand(cmd);
                settingsPopup.classList.remove('active');
                setTimeout(() => settingsPopup.style.display = 'none', 200);
            }});
        }});

        function sendCommand(cmd) {{
            const responseArea = document.getElementById('response');
            responseArea.textContent = 'Processing: ' + cmd + '...';
            
            fetch(`/command?cmd=${{encodeURIComponent(cmd)}}`)
            .then(res => res.text())
            .then(text => {{
                responseArea.textContent = 'Response: ' + text;
                console.log('Response:', text);
            }})
            .catch(err => {{
                responseArea.textContent = 'Error: ' + err;
                console.error('Error:', err);
            }});
        }}

        document.getElementById('cmd').addEventListener('keydown', (e)=>{{
            if(e.key === 'Enter'){{
                const v = e.target.value.trim();
                if(!v) return;
                sendCommand(v);
                e.target.value = '';
            }}
        }});

        // ============================================
        // CLOSE POPUPS WHEN CLICKING OUTSIDE
        // ============================================
        document.addEventListener('click', (e) => {{
            if (!appPopup.contains(e.target) && !appBtn.contains(e.target)) {{
                if (appPopup.classList.contains('active')) {{
                    appPopup.classList.remove('active');
                    setTimeout(() => {{
                        appPopup.style.display = 'none';
                    }}, 200);
                }}
            }}
            
            if (!settingsPopup.contains(e.target) && !settingsBtn.contains(e.target)) {{
                if (settingsPopup.classList.contains('active')) {{
                    settingsPopup.classList.remove('active');
                    setTimeout(() => {{
                        settingsPopup.style.display = 'none';
                    }}, 200);
                }}
            }}
        }});

        appPopup.addEventListener('mousedown', (e) => {{
            e.stopPropagation();
        }});
        
        settingsPopup.addEventListener('mousedown', (e) => {{
            e.stopPropagation();
        }});

        appPopup.style.display = 'none';
        settingsPopup.style.display = 'none';
        
        // Initial status update
        updateStatus();
    </script>
</body>
</html>
        """
        return html
        
    # ============ COMMAND PROCESSING ============
    
    def process_command(self, command):
        """Process user command with ultimate AI"""
        logging.info(f"Processing command: {command}")
        command = command.lower().strip()
        
        # Handle pending input
        if self.pending:
            self.input_queue.put(command)
            self.pending = None
            return "Input received"
            
        # Special gesture activation code
        if command == "activet 156":
            return self.toggle_gestures(True)
            
        # Try personal AI first for natural conversation
        if self.personal_ai_enabled and len(command.split()) > 2:
            ai_response = self.personal_trainer.generate_response(command)
            if ai_response:
                self.speak(ai_response)
                return ai_response
                
        # Check for commands
        if command.startswith("open "):
            app = command[5:].strip()
            result = self.open_app(app)
            if result:
                return result
                
        # Time and date
        if any(word in command for word in ["time", "clock", "hour"]):
            return self.get_time()
            
        if any(word in command for word in ["date", "day", "month", "year"]):
            return self.get_date()
            
        # Gesture commands
        if any(word in command for word in ["activate gesture", "turn on gesture", "start gesture", "enable gesture"]):
            return self.toggle_gestures(True)
            
        if any(word in command for word in ["deactivate gesture", "turn off gesture", "stop gesture", "disable gesture"]):
            return self.toggle_gestures(False)
            
        if "camera" in command:
            if "show" in command or "on" in command:
                return self.toggle_camera_preview(True)
            elif "hide" in command or "off" in command:
                return self.toggle_camera_preview(False)
                
        # Screenshot and selfie
        if any(word in command for word in ["screenshot", "capture screen", "screen shot"]):
            return self.take_screenshot()
            
        if any(word in command for word in ["selfie", "take picture", "capture selfie"]):
            return self.take_selfie()
            
        # Volume control
        if "volume up" in command:
            return self.volume_change("up")
        if "volume down" in command:
            return self.volume_change("down")
        if "mute" in command:
            return self.volume_change("mute")
            
        # System commands
        if "shutdown" in command:
            return self.shutdown()
        if "restart" in command or "reboot" in command:
            return self.restart()
        if "lock" in command:
            return self.lock_screen()
            
        # Weather
        if "weather" in command:
            return self.get_weather()
            
        # Joke
        if "joke" in command:
            return self.tell_joke()
            
        # Help
        if "help" in command or "command" in command:
            return self.show_help()
            
        # If nothing matched, let personal AI try again with a different temperature
        if self.personal_ai_enabled:
            ai_response = self.personal_trainer.generate_response(command, temperature=0.9)
            if ai_response:
                self.speak(ai_response)
                return ai_response
                
        # Default response
        response = f"I'm not sure how to help with '{command}'. Try saying 'help' for available commands."
        self.speak(response)
        return response
        
    def open_app(self, app_name):
        """Open application by name"""
        if app_name in self.commands_dict:
            cmd = self.commands_dict[app_name]
            try:
                if cmd.startswith("http"):
                    webbrowser.open(cmd)
                else:
                    subprocess.run(["start", "", cmd], shell=True)
                message = f"Opening {app_name}"
                self.speak(message)
                return message
            except Exception as e:
                message = f"Failed to open {app_name}"
                self.speak(message)
                return message
        return None
        
    def get_time(self):
        """Get current time"""
        time_str = datetime.datetime.now().strftime("%I:%M %p")
        message = f"The current time is {time_str}"
        self.speak(message)
        return message
        
    def get_date(self):
        """Get current date"""
        date_str = datetime.datetime.now().strftime("%A, %B %d, %Y")
        message = f"Today is {date_str}"
        self.speak(message)
        return message
        
    def take_screenshot(self):
        """Take screenshot"""
        try:
            folder = os.path.join(os.getcwd(), "isha_captures")
            os.makedirs(folder, exist_ok=True)
            filename = f"screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            path = os.path.join(folder, filename)
            pyautogui.screenshot(path)
            message = "Screenshot taken"
            self.speak(message)
            return message
        except Exception as e:
            message = f"Failed to take screenshot: {str(e)}"
            self.speak(message)
            return message
            
    def take_selfie(self):
        """Take selfie"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return "Camera not available"
            ret, frame = cap.read()
            cap.release()
            if ret:
                folder = os.path.join(os.getcwd(), "isha_captures")
                os.makedirs(folder, exist_ok=True)
                filename = f"selfie_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                path = os.path.join(folder, filename)
                cv2.imwrite(path, frame)
                message = "Selfie captured"
                self.speak(message)
                return message
            else:
                return "Failed to capture selfie"
        except Exception as e:
            message = f"Failed to take selfie: {str(e)}"
            self.speak(message)
            return message
            
    def volume_change(self, direction):
        """Change volume"""
        if direction == "up":
            pyautogui.press('volumeup', presses=3)
            message = "Volume increased"
        elif direction == "down":
            pyautogui.press('volumedown', presses=3)
            message = "Volume decreased"
        elif direction == "mute":
            pyautogui.press('volumemute')
            message = "Volume muted"
        self.speak(message)
        return message
        
    def shutdown(self):
        """Shutdown computer"""
        self.speak("Shutting down computer in 10 seconds")
        time.sleep(10)
        os.system('shutdown /s /t 0' if os.name == 'nt' else 'shutdown -h now')
        return "Shutting down"
        
    def restart(self):
        """Restart computer"""
        self.speak("Restarting computer")
        time.sleep(3)
        os.system('shutdown /r /t 0' if os.name == 'nt' else 'reboot')
        return "Restarting"
        
    def lock_screen(self):
        """Lock screen"""
        pyautogui.hotkey('win', 'l')
        self.speak("Screen locked")
        return "Screen locked"
        
    def get_weather(self):
        """Get weather (simplified)"""
        self.pending = 'weather_city'
        message = "Which city's weather do you want to check?"
        self.speak(message)
        return message
        
    def tell_joke(self):
        """Tell a joke"""
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "Why did the scarecrow win an award? Because he was outstanding in his field!",
            "Why don't eggs tell jokes? They'd crack each other up!",
            "What do you call a fake noodle? An impasta!",
            "Why did the math book look so sad? Because it had too many problems!",
            "What do you call a bear with no teeth? A gummy bear!",
            "Why did the golfer wear two pairs of pants? In case he got a hole in one!"
        ]
        joke = random.choice(jokes)
        self.speak(joke)
        return joke
        
    def show_help(self):
        """Show help"""
        message = "I can help you with many things! Try saying 'open calculator', 'what time is it', 'take screenshot', 'tell me a joke', or just talk to me naturally. I learn from our conversations!"
        self.speak(message)
        return message
        
    # ============ GESTURE CONTROL ============
    
    def toggle_gestures(self, activate=None):
        """Toggle hand gesture recognition"""
        if activate is not None:
            self.gesture_active = activate
        else:
            self.gesture_active = not self.gesture_active
            
        if hasattr(self.gesture_controller, 'active'):
            self.gesture_controller.active = self.gesture_active
        
        if self.gesture_active:
            if not self.gesture_thread or not self.gesture_thread.is_alive():
                self.gesture_thread = threading.Thread(target=self._gesture_loop, daemon=True)
                self.gesture_thread.start()
            message = "Hand gestures activated"
        else:
            message = "Hand gestures deactivated"
            
        self.speak(message)
        return message
        
    def toggle_camera_preview(self, show=None):
        """Toggle camera preview"""
        if hasattr(self.gesture_controller, 'toggle_preview'):
            if show is not None:
                current = self.gesture_controller.show_preview
                if current != show:
                    self.gesture_controller.toggle_preview()
            else:
                self.gesture_controller.toggle_preview()
                
        status = "on" if (hasattr(self.gesture_controller, 'show_preview') and self.gesture_controller.show_preview) else "off"
        message = f"Camera preview turned {status}"
        self.speak(message)
        return message
        
    def _gesture_loop(self):
        """Main gesture recognition loop"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.speak("Camera not available")
            return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        while self.gesture_active and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame, gesture_text, action_result = self.gesture_controller.process_frame(frame)
            
            if action_result:
                print(f"🖐️ Gesture: {gesture_text} - {action_result}")
                if action_result not in ["Scrolling"]:
                    self.speak(action_result)
                    
            if hasattr(self.gesture_controller, 'show_preview') and self.gesture_controller.show_preview:
                cv2.imshow("ISHA Gestures", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.gesture_active = False
                    break
            else:
                time.sleep(0.01)
                
        cap.release()
        cv2.destroyAllWindows()
        
    # ============ VOICE CONTROL ============
    
    def toggle_voice(self):
        """Toggle voice recognition"""
        self.is_listening = not self.is_listening
        
        if self.is_listening:
            message = "Voice recognition activated. Say your command."
            self.speak(message)
            threading.Thread(target=self._voice_loop, daemon=True).start()
        else:
            message = "Voice recognition deactivated"
            self.speak(message)
            
        return message
        
    def _voice_loop(self):
        """Voice recognition loop"""
        while self.is_listening:
            command = self.listen()
            if command:
                self.process_command(command)
            time.sleep(0.5)
            
    def listen(self):
        """Listen for voice command"""
        if self.microphone is None:
            return None
            
        try:
            with self.microphone as source:
                print("🎤 Listening...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                
            # Try Google STT
            text = self.recognizer.recognize_google(audio).lower()
            print(f"🗣️ Recognized: {text}")
            return text
            
        except sr.WaitTimeoutError:
            pass
        except sr.UnknownValueError:
            pass
        except Exception as e:
            logging.error(f"Voice recognition error: {e}")
            
        return None
        
    # ============ SPEAK ============
    
    def speak(self, text):
        """Speak text"""
        def run_speak():
            try:
                self.engine.say(text)
                self.engine.runAndWait()
                logging.info(f"Spoke: {text}")
            except Exception as e:
                logging.error(f"Speech error: {e}")
                
        threading.Thread(target=run_speak, daemon=True).start()
        time.sleep(0.1)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main function to run the ultimate ISHA assistant"""
    try:
        # Hide console on Windows
        if os.name == 'nt':
            console = ctypes.windll.kernel32.GetConsoleWindow()
            if console:
                ctypes.windll.user32.ShowWindow(console, 0)
                
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Print banner
        print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   ██╗███████╗██╗  ██╗ █████╗         ██╗   ██╗██╗  ████████╗██╗███╗   ███╗
║   ██║██╔════╝██║  ██║██╔══██╗        ██║   ██║██║  ╚══██╔══╝██║████╗ ████║
║   ██║███████╗███████║███████║        ██║   ██║██║     ██║   ██║██╔████╔██║
║   ██║╚════██║██╔══██║██╔══██║        ██║   ██║██║     ██║   ██║██║╚██╔╝██║
║   ██║███████║██║  ██║██║  ██║        ╚██████╔╝███████╗██║   ██║██║ ╚═╝ ██║
║   ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝         ╚═════╝ ╚══════╝╚═╝   ╚═╝╚═╝     ╚═╝
║                                                                           ║
║                      ULTIMATE PERSONAL AI ASSISTANT                       ║
║                                                                           ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║   🧠 ULTRA LLM: 24-layer Transformer with 1.5B+ parameters               ║
║      • Rotary Position Embeddings (RoPE)                                 ║
║      • Grouped-Query Attention (GQA)                                     ║
║      • Mixture of Experts (MoE) with 8 experts                           ║
║      • Flash Attention for fast inference                                ║
║                                                                           ║
║   🖐️ ADVANCED GESTURES: 30+ gestures with deep learning                  ║
║                                                                           ║
║   🎤 VOICE RECOGNITION: Google STT + Whisper support                     ║
║                                                                           ║
║   🌐 BEAUTIFUL GUI: Your custom animated interface                       ║
║      • Floating 3D animated circle                                       ║
║      • Particle background effects                                       ║
║      • Draggable popups                                                   ║
║      • Real-time updates                                                  ║
║                                                                           ║
║   💾 CONTINUOUS LEARNING: Gets smarter with every conversation           ║
║                                                                           ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║   Commands:                                                              ║
║   • Just talk naturally - AI understands context                         ║
║   • 'activate gestures' - Turn on hand gesture control                   ║
║   • 'show camera' - See yourself while gesturing                         ║
║   • 'take screenshot' - Capture screen                                   ║
║   • 'take selfie' - Take photo with camera                               ║
║   • 'open [app]' - Open any application                                  ║
║   • 'what time is it' - Check current time                               ║
║   • 'tell me a joke' - Get a laugh                                       ║
║   • 'train ai' - Improve your personal AI                                ║
║   • 'activet 156' - Quick gesture activation                             ║
║   • 'help' - Show all commands                                           ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
        """)
        
        print("\n🚀 Starting ISHA Ultimate...\n")
        
        # Create assistant
        assistant = IshaAssistantUltimate()
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Thanks for using ISHA Ultimate!")
    except Exception as e:
        logging.error(f"Application failed: {e}")
        print(f"\n❌ Fatal error: {e}")
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()