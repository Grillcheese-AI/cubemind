"""Unit tests for cubemind.brain.llm_interface.LLMInterface."""

from __future__ import annotations


from cubemind.brain.llm_interface import LLMInterface


def test_init_no_model():
    llm = LLMInterface(model_path=None)
    assert not llm.available
    assert llm._mode == "none"


def test_init_api_mode():
    llm = LLMInterface(api_url="http://localhost:8080", api_key="test")
    assert llm._mode == "api"


def test_generate_unavailable():
    llm = LLMInterface()
    result = llm.generate("test")
    assert "not available" in result.lower()


def test_build_prompt_empty():
    llm = LLMInterface()
    prompt = llm._build_prompt("hello", None, None)
    assert "hello" in prompt


def test_build_prompt_with_context():
    llm = LLMInterface()
    ctx = {
        "spatial_context": {"current_location": [1.0, 2.0], "n_memories": 5},
        "neurochemistry": {"dopamine": 0.8, "serotonin": 0.5, "cortisol": 0.2},
        "neurogenesis": {"neuron_count": 64},
    }
    prompt = llm._build_prompt("test", ctx, None)
    assert "Brain State" in prompt
    assert "DA=0.80" in prompt
    assert "Neurons: 64" in prompt


def test_build_prompt_with_memories():
    llm = LLMInterface()
    memories = ["memory_a", "memory_b", "memory_c"]
    prompt = llm._build_prompt("query", None, memories)
    assert "Episodic Memories" in prompt
    assert "memory_a" in prompt


def test_system_prompt_default():
    llm = LLMInterface()
    assert "CubeMind" in llm.system_prompt


def test_system_prompt_custom():
    llm = LLMInterface(system_prompt="Custom persona")
    assert llm.system_prompt == "Custom persona"


def test_get_logits_no_model():
    llm = LLMInterface()
    assert llm.get_logits("test") is None


def test_attach_injector():
    llm = LLMInterface()
    llm.attach_injector("fake_injector")
    assert llm._injector == "fake_injector"
