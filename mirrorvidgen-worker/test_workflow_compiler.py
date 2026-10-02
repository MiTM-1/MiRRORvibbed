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


if __name__ == "__main__":
    unittest.main()
