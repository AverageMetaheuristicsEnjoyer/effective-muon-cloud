import os

from .slim_adam import SlimAdamW

# Absolute path to the bundled layer map for this project's Llama architecture.
# Covers: wte (token_embedding), lm_head, attn.c_proj (attention_output),
#         mlp.w12 (mlp_up/gate fused), mlp.c_proj (mlp_down), layer norms.
# Note: fused c_attn (combined Q/K/V) is intentionally left uncompressed
#       because separate per-head compression cannot be applied to a fused matrix.
DEFAULT_LAYER_MAP_PATH = os.path.join(os.path.dirname(__file__), "layer_map.json")
