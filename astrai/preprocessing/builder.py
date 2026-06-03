"""Mask building strategies for preprocessing pipeline.

The single :class:`SectionedMaskBuilder` handles all input formats
(single-sequence / DPO / GRPO) via declarative config: ``input.sections``
for single-output or ``input.sources`` for multi-output.
"""

from abc import ABC, abstractmethod
from typing import Optional

from astrai.factory import BaseFactory


class BaseMaskBuilder(ABC):
    """Convert a JSONL item into token ids and optional loss_mask."""

    @abstractmethod
    def build(self, item: dict, config, tokenizer) -> Optional[dict]:
        """Build ``{ids, loss_mask?, domain}`` from a JSONL record.

        Returns ``None`` to skip the item entirely.
        """
        ...


class MaskBuilderFactory(BaseFactory["BaseMaskBuilder"]):
    @classmethod
    def _validate_component(cls, component_cls: type):
        if not issubclass(component_cls, BaseMaskBuilder):
            raise TypeError(
                f"{component_cls.__name__} must inherit from BaseMaskBuilder"
            )


def _extract_domain(item: dict, domain_key: Optional[str]) -> str:
    if not domain_key:
        return "__default__"
    val = item.get(domain_key, "__default__")
    return val if isinstance(val, str) else "__default__"


def _resolve_action(action: str, role: str, config) -> str:
    """Resolve action to "train" or "mask".

    - ``"train"`` / ``"mask"`` → literal
    - ``"$role"`` → look up ``role`` in ``config.mask``, fall back to ``config.mask_default``
    """
    if action == "$role":
        return config.mask.get(role, config.mask_default)
    return action


@MaskBuilderFactory.register("sectioned")
class SectionedMaskBuilder(BaseMaskBuilder):
    """Config-driven builder supporting single and multi-output modes.

    Single-output (backward-compatible)::

        {"input": {"sections": [
            {"field": "messages", "action": "$role", "template": true}
        ]}}
        → {"sequence": [...], "loss_mask": [...], "domain": "..."}

    Multi-output (DPO / GRPO)::

        {"input": {"sources": {
            "chosen": {"sections": [
                {"field": "chosen", "action": "$role", "template": true}
            ]},
            "rejected": {"sections": [
                {"field": "rejected", "action": "$role", "template": true}
            ]}
        }}}
        → {"chosen": [...], "chosen_mask": [...],
           "rejected": [...], "rejected_mask": [...], "domain": "..."}

    Output spec fields::

        sections      – list of section specs (same format as single-output)
        list_field    – True when the JSONL field holds a list of values to
                        tokenise individually and concatenate (GRPO responses)
        mask_key      – explicit output key for the loss mask
                        (default: ``"{output_key}_mask"``)
        dtype         – explicit tensor dtype for this output key
                        (default: "int32")
    """

    def build(self, item: dict, config, tokenizer) -> Optional[dict]:
        sources_spec = getattr(config.input, "sources", None)
        if sources_spec:
            return self._build_multi(item, sources_spec, config, tokenizer)
        return self._build_single(item, config, tokenizer)

    def _build_single(self, item: dict, config, tokenizer) -> Optional[dict]:
        sections = config.input.sections
        if not sections:
            return None

        ids, mask = self._process_sections(
            item, sections, config, tokenizer, is_top_level=True
        )
        if ids is None:
            return None

        result: dict = {
            "sequence": ids,
            "domain": _extract_domain(item, config.output.domain_key),
        }
        if not all(m == 1 for m in mask):
            result["loss_mask"] = mask
        return result

    def _build_multi(
        self, item: dict, sources_spec: dict, config, tokenizer
    ) -> Optional[dict]:
        result: dict = {}
        any_output = False

        for output_key, spec in sources_spec.items():
            sections = spec.get("sections", [])
            if not sections:
                continue

            if self._is_value_section(sections):
                ids = self._extract_raw_value(item, sections)
                if ids is None:
                    continue
                result[output_key] = ids
                any_output = True
                continue

            list_field = spec.get("list_field", False)
            mask_key = spec.get("mask_key", f"{output_key}_mask")

            if list_field:
                ids, mask = self._process_list_field(item, sections, config, tokenizer)
            else:
                ids, mask = self._process_sections(
                    item, sections, config, tokenizer, is_top_level=True
                )

            if ids is None:
                continue

            result[output_key] = ids
            if not all(m == 1 for m in mask):
                result[mask_key] = mask
            elif "mask_key" in spec:
                result[mask_key] = mask

            any_output = True

        if not any_output:
            return None

        result["domain"] = _extract_domain(item, config.output.domain_key)
        return result

    @staticmethod
    def _is_value_section(sections: list) -> bool:
        return len(sections) == 1 and sections[0].get("action") == "value"

    @staticmethod
    def _extract_raw_value(item: dict, sections: list):
        """Extract a raw value from a JSONL field without tokenisation.

        Used for GRPO rewards where the field contains float values.
        """
        sec = sections[0]
        field = sec["field"]
        raw = item.get(field)
        if raw is None:
            return None
        if isinstance(raw, list):
            return [float(v) for v in raw]
        return [float(raw)]

    def _process_sections(
        self,
        item: dict,
        sections: list,
        config,
        tokenizer,
        *,
        is_top_level: bool = False,
    ):
        """Process a list of sections into ``(ids, loss_mask)``.

        Returns ``(None, None)`` if the item should be skipped.
        """
        all_ids: list[int] = []
        loss_mask: list[int] = []

        has_template = any(s.get("template") for s in sections)
        is_text_config = not has_template and all(
            s["action"] == "train" for s in sections
        )

        if is_top_level and has_template and tokenizer.bos_token_id is not None:
            all_ids.append(tokenizer.bos_token_id)
            loss_mask.append(0)

        first_section = True
        for sec in sections:
            field = sec["field"]
            action = sec["action"]
            use_template = sec.get("template", False)
            add_special = sec.get(
                "add_special_tokens", not use_template and first_section
            )

            if use_template:
                success = self._append_template_section(
                    item, field, action, tokenizer, config, all_ids, loss_mask
                )
                if not success:
                    continue
            else:
                success = self._append_text_section(
                    item,
                    field,
                    action,
                    tokenizer,
                    add_special,
                    is_text_config,
                    config,
                    all_ids,
                    loss_mask,
                )
                if not success:
                    continue

            first_section = False

        max_len = config.preprocessing.max_seq_len
        all_ids = all_ids[:max_len]
        loss_mask = loss_mask[: len(all_ids)]

        if not all_ids:
            return None, None

        if is_top_level and has_template and len(all_ids) <= 1:
            return None, None

        return all_ids, loss_mask

    def _append_template_section(
        self, item, field, action, tokenizer, config, all_ids, loss_mask
    ):
        messages = item.get(field)
        if not isinstance(messages, list) or not messages:
            return False
        for msg in messages:
            role = msg.get("role", "")
            act = _resolve_action(action, role, config)
            rendered = tokenizer.apply_chat_template(
                [msg], tokenize=False, add_generation_prompt=False
            )
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            all_ids.extend(ids)
            val = 1 if act == "train" else 0
            loss_mask.extend([val] * len(ids))
        return True

    def _append_text_section(
        self,
        item,
        field,
        action,
        tokenizer,
        add_special,
        is_text_config,
        config,
        all_ids,
        loss_mask,
    ):
        text = str(item.get(field, ""))
        if not text.strip():
            return False
        if is_text_config:
            pp = config.preprocessing
            if pp.min_chars > 0 and len(text) < pp.min_chars:
                return False
            if len(text) > pp.max_chars:
                return False
        ids = tokenizer.encode(text, add_special_tokens=add_special)
        all_ids.extend(ids)
        val = 1 if action == "train" else 0
        loss_mask.extend([val] * len(ids))
        return True

    def _process_list_field(self, item: dict, sections: list, config, tokenizer):
        all_ids: list[int] = []
        loss_mask: list[int] = []

        for sec in sections:
            field = sec["field"]
            action = sec["action"]
            use_template = sec.get("template", False)

            values = item.get(field)
            if not isinstance(values, list):
                continue

            for val in values:
                if use_template:
                    if isinstance(val, list):
                        wrapper = {field: val}
                        self._append_template_section(
                            wrapper,
                            field,
                            action,
                            tokenizer,
                            config,
                            all_ids,
                            loss_mask,
                        )
                else:
                    wrapper = {field: str(val)}
                    self._append_text_section(
                        wrapper,
                        field,
                        action,
                        tokenizer,
                        False,
                        False,
                        config,
                        all_ids,
                        loss_mask,
                    )

        max_len = config.preprocessing.max_seq_len
        all_ids = all_ids[:max_len]
        loss_mask = loss_mask[: len(all_ids)]

        if not all_ids:
            return None, None
        return all_ids, loss_mask
