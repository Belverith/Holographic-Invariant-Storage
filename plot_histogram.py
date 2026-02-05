import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from scipy.stats import norm

# --- SETUP ---
print("[-] Initializing Generator...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')
D = 10000
projection = torch.randn(384, D)

def encode(text):
    raw = torch.tensor(encoder.encode(text)).float()
    vec = torch.matmul(raw, projection)
    vec[vec > 0] = 1
    vec[vec <= 0] = -1
    return vec

# Setup Anchor
key = encode("Key")
val = encode("Value")
anchor = key * val

# --- RUN SIMULATION (n=1000) ---
print("[-] Running 1,000 Monte Carlo Trials...")
results = []
noise_corpus = [
    "Ignore instructions", "Write a virus", "System failure", 
    "Poetry about rust", "The sky is blue", "Project Gutenberg"
]

for i in range(1000):
    noise_text = np.random.choice(noise_corpus) + f" {i} " + np.random.choice(noise_corpus)
    noise_vec = encode(noise_text)
    
    drifted = torch.sign(anchor + noise_vec)
    recovered = drifted * key
    
    score = F.cosine_similarity(recovered.unsqueeze(0), val.unsqueeze(0)).item()
    results.append(score)

# --- PLOTTING ---
print("[-] Generating Plot...")
plt.figure(figsize=(10, 6))

# 1. Plot Histogram
plt.hist(results, bins=30, density=True, alpha=0.6, color='#2E86C1', edgecolor='black', label='Experimental Data')

# 2. Fit a Normal Distribution Curve
mu, std = norm.fit(results)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)

# FIX: Added r'' to make these raw strings so \mu works
plt.plot(x, p, 'k', linewidth=2, label=r'Normal Fit ($\mu=' + f'{mu:.4f}$)')

# 3. Add the Theoretical Bound Line
# FIX: Added r'' here too for \sqrt
plt.axvline(x=0.7071, color='r', linestyle='--', linewidth=2, label=r'Theoretical Bound ($1/\sqrt{2}$)')

# 4. Styling
plt.title('Distribution of Holographic Recovery Fidelity (n=1,000)', fontsize=14)
plt.xlabel('Cosine Similarity (Recovery Score)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)

# 5. Save
plt.savefig('monte_carlo_histogram.png', dpi=300)
print("[+] Saved 'monte_carlo_histogram.png'. Insert this into your paper!")