import unittest
from types import SimpleNamespace

import torch

from scripts.benchmark_layerwise_scaling import PROFILES
from scripts.benchmark_layerwise_tucker import make_config
from models.llama import Llama


class LayerwiseScalingTest(unittest.TestCase):
    def test_geometry_and_dense_parameter_counts(self):
        for profile, (layers, width, heads, ff) in PROFILES.items():
            for variant in ("dense", "A", "B"):
                for fraction in (0.25, 0.5):
                    with self.subTest(profile=profile, variant=variant, fraction=fraction):
                        args = SimpleNamespace(
                            tiny=False, width=width, heads=heads, layers=layers,
                            ffn_hidden_size=ff, rank_fraction=fraction,
                            sequence_length=1024, device="cuda", liger=False,
                            execution="reordered", variant=variant,
                        )
                        config = make_config(args)
                        with torch.device("meta"):
                            model = Llama(config)
                        self.assertEqual(len(model.transformer.h), layers)
                        self.assertEqual(config.n_embd // config.n_head, 128)
                        if variant == "dense":
                            expected = 2 * 50304 * width + layers * (4 * width**2 + 3 * width * ff + 2 * width) + width
                            self.assertEqual(sum(p.numel() for p in model.parameters()), expected)
                        else:
                            self.assertEqual(model.transformer.h[0].mlp.core.shape[:2],
                                             (int(ff * fraction), int(width * fraction)))


if __name__ == "__main__":
    unittest.main()
