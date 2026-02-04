"""
Quick test script to debug KV-cache implementation
Run this to see debug output without full benchmark
"""

import torch
from backbone_mastery_debug import DecoderOnlyTransformer

print("="*70)
print("KV-CACHE DEBUG TEST")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Load trained model
print("\nLoading model...")
checkpoint = torch.load('checkpoints/best_model.pt', map_location=device)
model = DecoderOnlyTransformer(
    vocab_size=256,
    dim=256,
    num_layers=4,
    num_heads=8,
    ff_hidden_dim=1024,
    max_seq_len=50,
    dropout=0.1
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Test with cache
print("\n" + "="*70)
print("TESTING WITH CACHE (should show cache growing)")
print("="*70)

input_ids = torch.randint(0, 256, (1, 5), device=device)
print(f"\nInitial input shape: {input_ids.shape}")
print(f"Generating 10 new tokens...\n")

output, gen_time = model.generate_with_cache(input_ids, max_new_tokens=10)

print(f"\n{'='*70}")
print(f"Final output shape: {output.shape}")
print(f"Generation time: {gen_time*1000:.2f} ms")
print("="*70)

print("\n🔍 WHAT TO LOOK FOR:")
print("  1. 'past_kvs is None' should be True at step 0, False after")
print("  2. Cache K/V shapes should GROW each step (e.g., [1,8,1,32] → [1,8,2,32] → [1,8,3,32])")
print("  3. 'Input shape to forward' should be [1,1] after step 0 (only new token)")
print("  4. ATTENTION debug should show concatenation happening")
print("\n❌ IF CACHE NOT WORKING:")
print("  - Cache shapes stay at [1,8,1,32] (not growing)")
print("  - past_kvs stays None")
print("  - Input shape stays [1,N] (processing all tokens)")
