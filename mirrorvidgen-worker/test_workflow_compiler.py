"""No-GPU regressions for exported, promoted ComfyUI subgraph widgets."""
import unittest

from workflow_compiler import WorkflowCompiler
from ltx25_workflow import CapabilityUnavailable, validate_workflow_inputs


class PromotedWidgetTests(unittest.TestCase):
    def graph(self, values):
        return {
            "nodes": [{"id": 5516, "type": "sampler-group", "inputs": [
                {"name": "noise_seed", "type": "INT", "widget": {"name": "noise_seed"}},
                {"name": "cfg", "type": "FLOAT", "widget": {"name": "cfg"}},
            ], "widgets_values": values}],
            "definitions": {"subgraphs": [{"id": "sampler-group", "inputs": [
                {"name": "noise_seed", "type": "INT", "linkIds": [1]},
                {"name": "cfg", "type": "FLOAT", "linkIds": [2]},
                {"name": "sampler_name", "type": "COMBO", "linkIds": [3]},
            ], "nodes": [{"id": 4831, "type": "KSamplerSelect", "inputs": [
                {"name": "sampler_name", "type": "COMBO", "widget": {"name": "sampler_name"}, "link": 3}
            ], "widgets_values": ["euler_ancestral"]}], "links": [
                [3, -10, 2, 4831, 0, "COMBO"]
            ]}]}
        }

    def test_saved_promoted_sampler_is_preserved(self):
        graph = WorkflowCompiler(self.graph([42, 1, "euler_ancestral_cfg_pp"])).compile()
        self.assertEqual(graph["r:5516:4831"]["inputs"]["sampler_name"], "euler_ancestral_cfg_pp")
        self.assertTrue(validate_workflow_inputs(graph))

    def test_missing_saved_sampler_is_blocked_not_guessed(self):
        graph = WorkflowCompiler(self.graph([42, 1])).compile()
        with self.assertRaises(CapabilityUnavailable):
            validate_workflow_inputs(graph)

    def test_hidden_widgets_survive_official_export_socket_arrays(self):
        # Exact exported shapes from the failed Ingredients graph. These
        # nodes expose only some (or none) of their saved widgets as sockets.
        nodes = [
            {"id": 5602, "type": "UNETLoader", "inputs": [{"name": "unet_name", "widget": {"name": "unet_name"}}], "widgets_values": ["transformer.safetensors", "default"]},
            {"id": 5605, "type": "CLIPLoader", "inputs": [{"name": "clip_name", "widget": {"name": "clip_name"}}], "widgets_values": ["encoder.safetensors", "ltxv", "default"]},
            {"id": 5561, "type": "ComfyMathExpression", "inputs": [], "widgets_values": ["1 + floor(a*b/8)*8"]},
            {"id": 4984, "type": "ManualSigmas", "inputs": [], "widgets_values": ["1.0, 0.9, 0.5, 0.0"]},
            {"id": 3059, "type": "EmptyLTXVLatentVideo", "inputs": [], "widgets_values": [960, 544, 121, 1]},
            {"id": 9009, "type": "LTXVEmptyLatentAudio", "inputs": [], "widgets_values": [121, 25, 1]},
        ]
        graph = WorkflowCompiler({"nodes": nodes}).compile()
        self.assertTrue(validate_workflow_inputs(graph))
        self.assertEqual(graph["r:5602"]["inputs"]["weight_dtype"], "default")
        self.assertEqual(graph["r:5605"]["inputs"]["type"], "ltxv")
        self.assertEqual(graph["r:5561"]["inputs"]["expression"], "1 + floor(a*b/8)*8")
        self.assertEqual(graph["r:3059"]["inputs"]["batch_size"], 1)

    def test_partial_widget_socket_does_not_shift_saved_positions(self):
        graph = WorkflowCompiler({"nodes": [{"id": 1, "type": "LTXVImgToVideoInplace", "inputs": [{"name": "bypass", "widget": {"name": "bypass"}}], "widgets_values": [0.7, False]}]}).compile()
        self.assertIs(graph["r:1"]["inputs"]["bypass"], False)
        self.assertEqual(graph["r:1"]["inputs"]["strength"], 0.7)

    def test_resize_dynamic_branch_keeps_selector_and_argument(self):
        graph = WorkflowCompiler({"nodes": [{"id": 1, "type": "ResizeImageMaskNode", "inputs": [], "widgets_values": ["scale shorter dimension", 544, "lanczos"]}]}).compile()
        self.assertEqual(graph["r:1"]["inputs"], {"resize_type": "scale shorter dimension", "resize_type.shorter_size": 544, "scale_method": "lanczos"})

    def test_incomplete_loader_fails_before_paid_queue(self):
        graph = WorkflowCompiler({"nodes": [{"id": 1, "type": "UNETLoader", "widgets_values": ["transformer.safetensors"]}]}).compile()
        with self.assertRaises(CapabilityUnavailable):
            validate_workflow_inputs(graph)


if __name__ == "__main__":
    unittest.main()
