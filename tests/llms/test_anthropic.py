import guidance
from ..utils import get_llm


def test_nostream():
    guidance.llm = get_llm('anthropic:claude-1')
    a = guidance('''Hello,  my name is{{gen 'name' stream=False max_tokens=5}}''', stream=False)
    a = a()
    assert len(a['name']) > 0


def test_stream():  # anthropic doesn't stream per token but per section
    guidance.llm = get_llm('anthropic:claude-1')
    a = guidance('''Hello,  my name is{{gen 'name' stream=True max_tokens=5}}''', stream=False)
    a = a()
    assert len(a['name']) > 0


def test_chat_no_stream():
    guidance.llm = get_llm("anthropic:claude-1")
    prompt = guidance(
        '''{{#system~}}
        You are a helpful assistant.
        {{~/system}}
        {{#user~}}
        {{conversation_question}}
        {{~/user}}
        {{#assistant~}}
        {{gen "answer" max_tokens=5 stream=False}}
        {{~/assistant}}''')
    prompt = prompt(conversation_question='Whats is the meaning of life??')
    assert len(prompt['answer']) > 0
