"""Bridge: turn a LangGraph program built on the prompt registry into a DSPy module.

For each optimizable PromptSpec we create a dspy.Predict whose signature is generated from the spec
(instructions = current prompt text, fields = spec inputs + output schema fields). forward() runs the
*unchanged* LangGraph graph, passing those predictors through the run config; `app.llm.call_llm`
routes matching calls to them, so DSPy records each call in its trace and GEPA can rewrite the
predictors' instructions. Non-optimizable prompts (e.g. the judge) keep using the LangChain path.
"""
import copy

import numpy  # noqa: F401  (import before dspy)
import dspy

from app.prompts import SPECS, PromptSpec


def spec_to_signature(spec: PromptSpec, instructions: str) -> type[dspy.Signature]:
    fields = {name: (str, dspy.InputField(desc=desc)) for name, desc in spec.inputs.items()}
    for name, f in spec.output.model_fields.items():
        fields[name] = (f.annotation, dspy.OutputField(desc=f.description or ""))
    return dspy.make_signature(fields, instructions, signature_name=f"{spec.name}_signature")


class LangGraphProgram(dspy.Module):
    """A dspy.Module whose forward() is a LangGraph invocation."""

    def __init__(self, graph, prompts: dict[str, str], optimizable: list[str], input_keys=("question",),
                 output_keys=("answer",), extra_config: dict | None = None):
        super().__init__()
        self.graph = graph
        self.input_keys = tuple(input_keys)
        self.output_keys = tuple(output_keys)
        self.extra_config = extra_config or {}
        self.optimizable = list(optimizable)
        # Attribute names become GEPA's predictor names (pred_name in the metric).
        for name in self.optimizable:
            setattr(self, name, dspy.Predict(spec_to_signature(SPECS[name], prompts[name])))

    def __deepcopy__(self, memo):
        # Optimizers deep-copy candidate programs. A compiled LangGraph graph must NOT be deep-copied
        # (it compares internal sentinels by identity); it is stateless, so all copies share it.
        memo[id(self.graph)] = self.graph
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            setattr(new, k, copy.deepcopy(v, memo))
        return new

    def forward(self, **inputs):
        predictors = {name: getattr(self, name) for name in self.optimizable}
        state = self.graph.invoke(
            {k: inputs[k] for k in self.input_keys},
            config={"configurable": {**self.extra_config, "dspy_predictors": predictors}},
        )
        return dspy.Prediction(**{k: state.get(k) for k in self.output_keys})

    def export_prompts(self) -> dict[str, str]:
        """Current (possibly optimized) instructions, ready for the LangGraph prompt registry."""
        return {name: getattr(self, name).signature.instructions for name in self.optimizable}


def langgraph_to_dspy(graph, prompts: dict[str, str], optimizable: list[str], **kw) -> LangGraphProgram:
    return LangGraphProgram(graph, prompts, optimizable, **kw)
