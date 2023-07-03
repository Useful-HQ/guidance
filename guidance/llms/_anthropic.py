import random

import aiohttp
import anthropic
import os
import copy
import time
import asyncio
import types
import collections
import json
import re
import regex

from ._llm import LLM, LLMSession, SyncSession


class MalformedPromptException(Exception):
    pass

async def add_text_to_chat_mode_generator(chat_mode):
    async for resp in chat_mode:
        if "choices" in resp:
            for c in resp['choices']:
                if "content" in c['delta']:
                    c['text'] = c['delta']['content']
                else:
                    break  # the role markers are outside the generation in chat mode right now TODO: consider how this changes for uncontrained generation
            else:
                yield resp
        else:
            yield resp


def add_text_to_chat_mode(chat_mode):
    if isinstance(chat_mode, (types.AsyncGeneratorType, types.GeneratorType)):
        return add_text_to_chat_mode_generator(chat_mode)
    else:
        for c in chat_mode['choices']:
            c['text'] = c['message']['content']
        return chat_mode


# model that need to use the completion API
chat_models = [
    "claude-1",
    "claude-1-100k",
    "claude-instant-1"
    "claude-instant-1-100k"
]


class Anthropic(LLM):
    llm_name: str = "anthropic"

    def __init__(self, model=None, caching=True, max_retries=5, max_calls_per_min=60,
                 api_key=None, api_type="anthropic", temperature=0.0, chat_mode=True):
        super().__init__()

        # fill in default model value
        if model is None:
            model = os.environ.get("ANTHROPIC_MODEL", None)
        if model is None:
            try:
                with open(os.path.expanduser('~/.anthropic_model'), 'r') as file:
                    model = file.read().replace('\n', '')
            except:
                pass

        # fill in default API key value
        if api_key is None:  # get from environment variable
            api_key = os.environ.get("ANTHROPIC_API_KEY", None)
        if api_key is not None and not api_key.startswith("sk-") and os.path.exists(
                os.path.expanduser(api_key)):  # get from file
            with open(os.path.expanduser(api_key), 'r') as file:
                api_key = file.read().replace('\n', '')
        if api_key is None:  # get from default file location
            try:
                with open(os.path.expanduser('~/.openai_api_key'), 'r') as file:
                    api_key = file.read().replace('\n', '')
            except:
                pass

        self._tokenizer = anthropic.Client.get_tokenizer(None)
        self.model_name = model
        self.caching = caching
        self.max_retries = max_retries
        self.max_calls_per_min = max_calls_per_min
        if isinstance(api_key, str):
            api_key = api_key.replace("Bearer ", "")
        self.api_key = api_key
        self.api_type = api_type
        self.current_time = time.time()
        self.call_history = collections.deque()
        self.temperature = temperature
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        self.caller = self._library_call
        self.chat_mode = chat_mode

    def session(self, asynchronous=False):
        if asynchronous:
            return AnthropicSession(self)
        else:
            return SyncSession(AnthropicSession(self))

    def role_start(self, role):
        assert self.chat_mode, "role_start() can only be used in chat mode"
        if role != "assistant":
            return anthropic.HUMAN_PROMPT
        else:
            return anthropic.AI_PROMPT

    def role_end(self, role=None):
        assert self.chat_mode, "role_end() can only be used in chat mode"
        return ""

    def end_of_text(self):
        return ""

    @classmethod
    async def stream_then_save(cls, gen, key, stop_regex, n):
        list_out = []
        cached_out = None

        # init stop_regex variables
        if stop_regex is not None:
            if isinstance(stop_regex, str):
                stop_patterns = [regex.compile(stop_regex)]
            else:
                stop_patterns = [regex.compile(pattern) for pattern in stop_regex]

            current_strings = ["" for _ in range(n)]
            # last_out_pos = ["" for _ in range(n)]

        # iterate through the stream
        all_done = False
        async for curr_out in gen:

            # if we have a cached output, extend it with the current output
            if cached_out is not None:
                out = merge_stream_chunks(cached_out, curr_out)
            else:
                out = curr_out

            # check if we have stop_regex matches
            found_partial = False
            if stop_regex is not None:

                # keep track of the generated text so far
                for i, choice in enumerate(curr_out['choices']):
                    current_strings[i] += choice['text']

                # check if all of the strings match a stop string (and hence we can stop the batch inference)
                all_done = True
                for i in range(len(current_strings)):
                    found = False
                    for s in stop_patterns:
                        if s.search(current_strings[i]):
                            found = True
                    if not found:
                        all_done = False
                        break

                # find where trim off the stop regex matches if needed (and look for partial matches)
                stop_pos = [1e10 for _ in range(n)]
                stop_text = [None for _ in range(n)]
                for i in range(len(current_strings)):
                    for s in stop_patterns:
                        m = s.search(current_strings[i], partial=True)
                        if m:
                            span = m.span()
                            if span[1] > span[0]:
                                if m.partial:  # we might be starting a stop sequence, so we can't emit anything yet
                                    found_partial = True
                                    break
                                else:
                                    stop_text[i] = current_strings[i][span[0]:span[1]]
                                    stop_pos[i] = min(span[0], stop_pos[i])
                    if stop_pos != 1e10:
                        stop_pos[i] = stop_pos[i] - len(current_strings[i])  # convert to relative position from the end

            # if we might be starting a stop sequence, we need to cache the output and continue to wait and see
            if found_partial:
                cached_out = out
                continue

            # if we get here, we are not starting a stop sequence, so we can emit the output
            else:
                cached_out = None

                if stop_regex is not None:
                    for i in range(len(out['choices'])):
                        if stop_pos[i] < len(out['choices'][i]['text']):
                            out['choices'][i] = out['choices'][
                                i].to_dict()  # because sometimes we might need to set the text to the empty string (and OpenAI's object does not like that)
                            out['choices'][i]['text'] = out['choices'][i]['text'][:stop_pos[i]]
                            out['choices'][i]['stop_text'] = stop_text[i]
                            out['choices'][i]['finish_reason'] = "stop"

                list_out.append(out)
                yield out
                if all_done:
                    gen.aclose()
                    break

        # if we have a cached output, emit it
        if cached_out is not None:
            list_out.append(cached_out)
            yield out

        cls.cache[key] = list_out

    def _stream_completion(self):
        pass

    # Define a function to add a call to the deque
    def add_call(self):
        # Get the current timestamp in seconds
        now = time.time()
        # Append the timestamp to the right of the deque
        self.call_history.append(now)

    # Define a function to count the calls in the last 60 seconds
    def count_calls(self):
        # Get the current timestamp in seconds
        now = time.time()
        # Remove the timestamps that are older than 60 seconds from the left of the deque
        while self.call_history and self.call_history[0] < now - 60:
            self.call_history.popleft()
        # Return the length of the deque as the number of calls
        return len(self.call_history)

    async def _library_call(self, **kwargs):
        """ Call the Anthropic API using the python package.
        """
        assert self.api_key is not None, "You must provide an Anthropic API key to use the Antrhopic LLM. Either pass it in the constructor, set the OPENAI_API_KEY environment variable, or create the file ~/.openai_api_key with your key in it."

        kwargs["prompt"] = kwargs["prompt"].rstrip()

        if not kwargs["prompt"].startswith(anthropic.HUMAN_PROMPT):
            kwargs["prompt"] = anthropic.HUMAN_PROMPT + kwargs["prompt"]

        if not kwargs["prompt"].endswith(anthropic.AI_PROMPT):
            kwargs["prompt"] = kwargs["prompt"] + anthropic.AI_PROMPT

        if kwargs["stream"]:
            session = aiohttp.ClientSession()
            response = self._rest_stream_handler(await self.client.completions.create(**kwargs), session)
            return response
        else:
            response = await self.client.completions.create(**kwargs)
            return {"choices": [{"text": response.completion}]}

    async def _rest_stream_handler(self, responses, session):
        async for response in responses:
            yield {"choices": [{"text": response.completion}]}
    async def _close_response_and_session(self, response, session):
        await response.release()
        await session.close()

    def encode(self, string):
        # note that is_fragment is not used used for this tokenizer
        return self._tokenizer.encode(string, allowed_special=self.allowed_special_tokens)

    def decode(self, tokens):
        return self._tokenizer.decode(tokens)


def merge_stream_chunks(first_chunk, second_chunk):
    """ This merges two stream responses together.
    """

    out = copy.deepcopy(first_chunk)

    # merge the choices
    for i in range(len(out['choices'])):
        out_choice = out['choices'][i]
        second_choice = second_chunk['choices'][i]
        out_choice['text'] += second_choice['text']
        if 'index' in second_choice:
            out_choice['index'] = second_choice['index']
        if 'finish_reason' in second_choice:
            out_choice['finish_reason'] = second_choice['finish_reason']
        if out_choice.get('logprobs', None) is not None:
            out_choice['logprobs']['token_logprobs'] += second_choice['logprobs']['token_logprobs']
            out_choice['logprobs']['top_logprobs'] += second_choice['logprobs']['top_logprobs']
            out_choice['logprobs']['text_offset'] = second_choice['logprobs']['text_offset']

    return out


# Define a deque to store the timestamps of the calls
class AnthropicSession(LLMSession):
    async def __call__(self, prompt, stop=None, stop_regex=None, temperature=None, n=1, max_tokens=1000, logprobs=None,
                       top_p=1.0, echo=False, logit_bias=None, token_healing=None, pattern=None, stream=None,
                       cache_seed=0, caching=None, **completion_kwargs):
        """ Generate a completion of the given prompt.
        """

        # we need to stream in order to support stop_regex
        if stream is None:
            stream = stop_regex is not None
        assert stop_regex is None or stream, "We can only support stop_regex for the Anthropic API when stream=True!"
        assert stop_regex is None or n == 1, "We don't yet support stop_regex combined with n > 1 with the Anthropic API!"

        assert token_healing is None or token_healing is False, "The Anthropic API does not yet support token healing! Please either switch to an endpoint that does, or don't use the `token_healing` argument to `gen`."

        # set defaults
        if temperature is None:
            temperature = self.llm.temperature

        # get the arguments as dictionary for cache key generation
        args = locals().copy()

        assert not pattern, "The Anthropic API does not support Guidance pattern controls! Please either switch to an endpoint that does, or don't use the `pattern` argument to `gen`."
        # assert not stop_regex, "The OpenAI API does not support Guidance stop_regex controls! Please either switch to an endpoint that does, or don't use the `stop_regex` argument to `gen`."

        # define the key for the cache
        cache_params = self._cache_params(args)
        llm_cache = self.llm.cache
        key = llm_cache.create_key(self.llm.llm_name, **cache_params)

        # allow streaming to use non-streaming cache (the reverse is not true)
        if key not in llm_cache and stream:
            cache_params["stream"] = False
            key1 = llm_cache.create_key(self.llm.llm_name, **cache_params)
            if key1 in llm_cache:
                key = key1

        # check the cache
        if key not in llm_cache or caching is False or (caching is not True and not self.llm.caching):

            # ensure we don't exceed the rate limit
            while self.llm.count_calls() > self.llm.max_calls_per_min:
                await asyncio.sleep(1)

            fail_count = 0
            while True:
                try_again = False
                try:
                    self.llm.add_call()
                    call_args = {
                        "model": self.llm.model_name,
                        "prompt": prompt,
                        "max_tokens_to_sample": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "stop_sequences": stop,
                        "stream": stream,
                        **completion_kwargs
                    }
                    call_args = {k: v for k, v in call_args.items() if v is not None}
                    if logit_bias is not None:
                        call_args["logit_bias"] = {str(k): v for k, v in
                                                   logit_bias.items()}  # convert keys to strings since that's the open ai api's format
                    out = await self.llm.caller(**call_args)

                except anthropic.APIError as e:
                    print(e) #TODO @daniel review the expcetion types
                    await asyncio.sleep(3 + 3*fail_count + random.randint(0, 3*fail_count + 5))
                    try_again = True
                    fail_count += 1

                if not try_again:
                    break

                if fail_count > self.llm.max_retries:
                    raise Exception(
                        f"Too many (more than {self.llm.max_retries}) Anthropic API RateLimitError's in a row!")

            if stream:
                return self.llm.stream_then_save(out, key, stop_regex, n)
            else:
                llm_cache[key] = out

        # wrap as a list if needed
        if stream:
            if isinstance(llm_cache[key], list):
                return llm_cache[key]
            return [llm_cache[key]]

        return llm_cache[key]
