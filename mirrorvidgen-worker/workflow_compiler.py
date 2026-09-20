"""Small, dependency-free compiler for official ComfyUI workflow graphs.

The LTX example workflows are distributed as editable UI graphs (including
subgraphs), while the ComfyUI API accepts an executable node map.  This module
performs the same graph-to-prompt conversion used by the ComfyUI frontend and
keeps the conversion local to the worker.  It deliberately does not contain
any prompt-generation logic; the caller supplies the user's prompt verbatim.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


VIRTUAL_TYPES = {
    "MarkdownNote",
    "Note",
    "PreviewAny",
    "Reroute",
    "PrimitiveStringMultiline",
    "PrimitiveFloat",
    "PrimitiveInt",
    "PrimitiveBoolean",
}


def _link(value: Any) -> Dict[str, Any]:
    """Normalise both legacy array links and current object links."""
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)) and len(value) >= 5:
        return {
            "id": value[0],
            "origin_id": value[1],
            "origin_slot": value[2],
            "target_id": value[3],
            "target_slot": value[4],
            "type": value[5] if len(value) > 5 else "*",
        }
    raise ValueError(f"Invalid ComfyUI link: {value!r}")


def _is_virtual(node: Mapping[str, Any]) -> bool:
    node_type = str(node.get("type") or "")
    return node_type in VIRTUAL_TYPES or node_type.startswith("Primitive")


class WorkflowCompiler:
    """Flatten a ComfyUI UI graph into an API prompt map."""

    def __init__(self, document: Mapping[str, Any]):
        self.document = document
        definitions = document.get("definitions") or {}
        self.subgraphs = {
            str(item.get("id")): item
            for item in definitions.get("subgraphs") or []
            if item.get("id")
        }
        self.nodes: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _widget_values(node: Mapping[str, Any]) -> Dict[str, Any]:
        values = list(node.get("widgets_values") or [])
        result: Dict[str, Any] = {}
        value_index = 0
        for input_item in node.get("inputs") or []:
            widget = input_item.get("widget") or {}
            if not widget:
                continue
            name = str(widget.get("name") or input_item.get("name") or "")
            if not name:
                continue
            result[name] = values[value_index] if value_index < len(values) else None
            value_index += 1
        return result

    @classmethod
    def _literal_widget(cls, node: Mapping[str, Any], name: str) -> Any:
        widgets = cls._widget_values(node)
        if name in widgets:
            return widgets[name]
        node_type = str(node.get("type") or "")
        values = list(node.get("widgets_values") or [])
        # Nodes without a serialised input array still have stable widgets.
        if node_type == "LoadImage":
            return values[1] if len(values) > 1 else (values[0] if values else "")
        if node_type == "LoadVideo":
            return values[0] if values else ""
        if node_type == "SaveVideo":
            return values[0] if values else "output"
        if node_type == "CLIPTextEncode" and name == "text":
            return values[0] if values else ""
        if node_type.startswith("Primitive"):
            return values[0] if values else None
        return None

    @staticmethod
    def _find_input_link(links: Mapping[Any, Mapping[str, Any]], node_id: Any, slot: int) -> Optional[Mapping[str, Any]]:
        for item in links.values():
            if item.get("target_id") == node_id and item.get("target_slot") == slot:
                return item
        return None

    def _flatten(
        self,
        graph: Mapping[str, Any],
        prefix: str,
        boundary_inputs: Mapping[int, Any],
    ) -> Dict[int, Any]:
        links = {_link(item)["id"]: _link(item) for item in graph.get("links") or []}
        nodes = {str(item.get("id")): item for item in graph.get("nodes") or []}
        subgraph_outputs: Dict[str, Dict[int, Any]] = {}
        expanding = set()

        def expand_child(node: Mapping[str, Any]) -> Dict[int, Any]:
            child_id = str(node.get("id"))
            if child_id in subgraph_outputs:
                return subgraph_outputs[child_id]
            if child_id in expanding:
                raise ValueError(f"Cyclic ComfyUI subgraph reference at {child_id}")
            child = self.subgraphs.get(str(node.get("type") or ""))
            if child is None:
                return {}
            expanding.add(child_id)
            child_boundary: Dict[int, Any] = {}
            child_links = [_link(item) for item in child.get("links") or []]
            for child_link in child_links:
                if child_link.get("origin_id") != -10:
                    continue
                slot = int(child_link.get("origin_slot", 0))
                input_name = None
                for definition in child.get("inputs") or []:
                    if child_link.get("id") in (definition.get("linkIds") or []):
                        input_name = definition.get("name")
                        break
                parent_input = next(
                    (item for item in node.get("inputs") or [] if item.get("name") == input_name),
                    None,
                )
                parent_link_id = parent_input.get("link") if parent_input else None
                if parent_link_id in links:
                    parent_link = links[parent_link_id]
                    child_boundary[slot] = resolve_source(
                        parent_link["origin_id"], parent_link["origin_slot"]
                    )
                elif input_name is not None:
                    child_boundary[slot] = self._literal_widget(node, str(input_name))
            subgraph_outputs[child_id] = self._flatten(
                child,
                f"{prefix}:{child_id}",
                child_boundary,
            )
            expanding.remove(child_id)
            return subgraph_outputs[child_id]

        def resolve_source(origin_id: Any, origin_slot: int) -> Any:
            if origin_id == -10:
                return boundary_inputs.get(origin_slot)
            if origin_id == -20:
                return None
            child = subgraph_outputs.get(str(origin_id))
            if child is None:
                source_node = nodes.get(str(origin_id))
                if source_node is not None and str(source_node.get("type") or "") in self.subgraphs:
                    child = expand_child(source_node)
            if child is not None:
                return child.get(origin_slot)
            source = nodes.get(str(origin_id))
            if source is None:
                return None
            source_type = str(source.get("type") or "")
            if _is_virtual(source):
                link = self._find_input_link(links, source.get("id"), 0)
                if link:
                    return resolve_source(link["origin_id"], link["origin_slot"])
                if source_type.startswith("Primitive"):
                    return self._literal_widget(source, "value")
                return None
            return [f"{prefix}:{origin_id}", origin_slot]

        # Expand child subgraphs first.  Their output boundary refs can then
        # be used by normal nodes in this graph regardless of node ordering.
        for node in graph.get("nodes") or []:
            if str(node.get("type") or "") in self.subgraphs:
                expand_child(node)

        for node in graph.get("nodes") or []:
            node_id = str(node.get("id"))
            node_type = str(node.get("type") or "")
            if node_type in self.subgraphs or _is_virtual(node):
                continue

            inputs: Dict[str, Any] = {}
            for input_item in node.get("inputs") or []:
                name = str(input_item.get("name") or "")
                if not name:
                    continue
                link_id = input_item.get("link")
                if link_id in links:
                    link = links[link_id]
                    value = resolve_source(link["origin_id"], link["origin_slot"])
                    if value is not None:
                        inputs[name] = value
                    continue
                value = self._literal_widget(node, name)
                if value is not None:
                    inputs[name] = value

            if node_type == "LoadImage":
                inputs["image"] = self._literal_widget(node, "image") or ""
            elif node_type == "LoadVideo":
                inputs["video"] = self._literal_widget(node, "video") or ""
            elif node_type == "SaveVideo":
                inputs["filename_prefix"] = self._literal_widget(node, "filename_prefix") or "output"

            api_id = f"{prefix}:{node_id}"
            self.nodes[api_id] = {
                "inputs": inputs,
                "class_type": node_type,
                "_meta": {"title": node.get("title") or node_type},
            }

        outputs: Dict[int, Any] = {}
        for raw_link in graph.get("links") or []:
            link = _link(raw_link)
            if link.get("target_id") == -20:
                outputs[int(link.get("target_slot", 0))] = resolve_source(
                    link["origin_id"], link["origin_slot"]
                )
        return outputs

    def compile(self) -> Dict[str, Dict[str, Any]]:
        self._flatten(self.document, "r", {})
        return copy.deepcopy(self.nodes)


def compile_workflow(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists() or path.stat().st_size < 100:
        raise FileNotFoundError(f"Official workflow is not available: {path}")
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    return WorkflowCompiler(document).compile()


def iter_nodes(workflow: Mapping[str, Mapping[str, Any]], class_type: Optional[str] = None) -> Iterable[Tuple[str, Mapping[str, Any]]]:
    for node_id, node in workflow.items():
        if class_type is None or node.get("class_type") == class_type:
            yield node_id, node


def set_all_inputs(workflow: Dict[str, Dict[str, Any]], class_type: str, values: Mapping[str, Any]) -> int:
    count = 0
    for _, node in iter_nodes(workflow, class_type):
        node.setdefault("inputs", {}).update(values)
        count += 1
    return count


def set_first_input(workflow: Dict[str, Dict[str, Any]], class_type: str, values: Mapping[str, Any], title_contains: str = "") -> bool:
    for _, node in iter_nodes(workflow, class_type):
        title = str((node.get("_meta") or {}).get("title") or "").lower()
        if title_contains and title_contains.lower() not in title:
            continue
        node.setdefault("inputs", {}).update(values)
        return True
    return False


def disable_prompt_enhancer(workflow: Dict[str, Dict[str, Any]]) -> None:
    """Remove optional enhancer branches so they can never execute.

    Official graphs contain enhancer nodes even when the UI switch is off.  A
    server-side API prompt must not load or execute those nodes because the
    MiRRORvidgen contract explicitly excludes Prompt Enhancer.
    """
    aliases: Dict[str, Any] = {}
    remove = set()
    for node_id, node in workflow.items():
        class_type = str(node.get("class_type") or "")
        title = str((node.get("_meta") or {}).get("title") or "").lower()
        if class_type in {"GemmaAPITextEncode", "TextGenerateLTX2Prompt", "StringContains"}:
            remove.add(node_id)
        elif class_type == "ComfyNotNode":
            value = (node.get("inputs") or {}).get("value")
            if value is not None:
                aliases[node_id] = value
            remove.add(node_id)
        elif class_type == "ComfySwitchNode" and ("conditioning source" in title or "prompt" in title or "switch" in title):
            false_value = (node.get("inputs") or {}).get("on_false")
            if false_value is not None:
                aliases[node_id] = false_value
                remove.add(node_id)

    def resolve(value: Any) -> Any:
        seen = set()
        while isinstance(value, list) and len(value) == 2 and value[0] in aliases and value[0] not in seen:
            seen.add(value[0])
            value = aliases[value[0]]
        return value

    for node in workflow.values():
        inputs = node.get("inputs") or {}
        for name, value in list(inputs.items()):
            inputs[name] = resolve(value)
    for node_id in remove:
        workflow.pop(node_id, None)
