# smoke_test.py - run on your laptop, through the SSH tunnel that `./serve.sh tunnel` prints:
#   ssh -N -L 30000:127.0.0.1:30000 -L 8000:127.0.0.1:8000 ubuntu@<gpu-host-ip>
#
#   python smoke_test.py                 # smoke tests, printing every input and output
#   python smoke_test.py chat            # interactive streaming chat over stdin/stdout
#   python smoke_test.py chat --tools --system "You are a terse assistant."
# The API address defaults to http://127.0.0.1:30000/v1 (override with --base-url or API_PORT=...).
import argparse, json, os, sys, time, urllib.request
from openai import OpenAI

API_PORT = os.environ.get("API_PORT", "30000")
DASH_PORT = os.environ.get("DASH_PORT", "8000")

p = argparse.ArgumentParser()
p.add_argument("mode", nargs="?", default="test", choices=["test", "chat"])
p.add_argument("--base-url", default=f"http://127.0.0.1:{API_PORT}/v1")
p.add_argument("--model", default="qwen3-4b")
p.add_argument("--system", default="You are a helpful assistant.")
p.add_argument("--tools", action="store_true", help="chat mode: enable the demo get_weather tool")
p.add_argument("--temperature", type=float, default=0.7)
p.add_argument("--max-tokens", type=int, default=1024)
args = p.parse_args()

try:  # fail fast, with the fix, if the tunnel isn't up
    urllib.request.urlopen(args.base_url.rstrip("/") + "/models", timeout=5).read()
except Exception as e:
    sys.exit(f"Can't reach the model API at {args.base_url} ({e}).\n"
             f"Open the tunnel in another terminal and leave it running:\n"
             f"  ssh -N -L {API_PORT}:127.0.0.1:{API_PORT} -L {DASH_PORT}:127.0.0.1:{DASH_PORT} ubuntu@<gpu-host-ip>\n"
             f"(on the GPU host, `./serve.sh tunnel` prints the exact command)")
client = OpenAI(base_url=args.base_url, api_key="none")

TOOLS = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {"type": "object",
        "properties": {"city": {"type": "string"},
                       "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city"]}}}]

def run_tool(name, arguments):
    # Fake tool so the agent loop can be exercised end to end.
    if name == "get_weather":
        a = json.loads(arguments or "{}")
        return {"city": a.get("city"), "temperature": 21, "unit": a.get("unit", "celsius"),
                "conditions": "sunny (demo data)"}
    return {"error": f"unknown tool {name}"}

def short(text, n=200):
    text = str(text)
    return text if len(text) <= n else f"{text[:n]}... [{len(text)} chars total]"

def call(label, messages, **kw):
    print(f"\n=== {label} ===\n  INPUT:")
    for m in messages:
        print(f"    [{m['role']}] {short(m['content'])}")
    if kw.get("tools"):
        print(f"    [tools] {[t['function']['name'] for t in kw['tools']]}")
    t = time.perf_counter()
    r = client.chat.completions.create(model=args.model, messages=messages, **kw)
    ms = 1000 * (time.perf_counter() - t)
    c = r.choices[0]
    print("  OUTPUT:")
    print(f"    [assistant] {c.message.content!r}")
    for tc in c.message.tool_calls or []:
        print(f"    [tool_call] {tc.function.name}({tc.function.arguments})")
    print(f"    finish_reason={c.finish_reason}  latency={ms:.0f} ms")
    print(f"    usage={r.usage}")
    return r

def smoke_tests():
    call("1. Plain chat", [{"role": "user", "content": "Say hello in five words."}],
         temperature=0, max_tokens=32)
    r = call("2. Tool call (expect finish_reason=tool_calls)",
             [{"role": "user", "content": "What's the weather in Mountain View, in celsius?"}],
             tools=TOOLS, temperature=0)
    for tc in r.choices[0].message.tool_calls or []:
        json.loads(tc.function.arguments)  # raises if the arguments are not valid JSON
    system = "You are a support agent. " + "Policy text. " * 1500
    for i in (1, 2):
        call(f"3.{i} Prefix cache (2nd should be faster, with cached tokens in usage)",
             [{"role": "system", "content": system}, {"role": "user", "content": "Hi"}],
             max_tokens=1)

def stream_turn(history):
    """One streamed assistant turn; returns the tool calls the model made, if any."""
    kw = {"tools": TOOLS} if args.tools else {}
    t0, ttft, text, calls = time.perf_counter(), None, "", {}
    stream = client.chat.completions.create(model=args.model, messages=history, stream=True,
        temperature=args.temperature, max_tokens=args.max_tokens, **kw)
    print("bot> ", end="", flush=True)
    for chunk in stream:
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta
        if ttft is None and (d.content or d.tool_calls):
            ttft = time.perf_counter() - t0
        if d.content:
            text += d.content
            print(d.content, end="", flush=True)
        for tc in d.tool_calls or []:
            c = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
            if tc.id:
                c["id"] = tc.id
            if tc.function and tc.function.name:
                c["name"] += tc.function.name
            if tc.function and tc.function.arguments:
                c["args"] += tc.function.arguments
    total = time.perf_counter() - t0
    print(f"\n      (TTFT {1000 * (ttft or total):.0f} ms, total {total:.1f} s)")
    msg = {"role": "assistant", "content": text}
    if calls:
        msg["tool_calls"] = [{"id": c["id"], "type": "function",
            "function": {"name": c["name"], "arguments": c["args"]}} for c in calls.values()]
    history.append(msg)
    return list(calls.values())

def chat():
    history = [{"role": "system", "content": args.system}]
    print("Commands: /reset  /system <text>  /history  /quit")
    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/reset":
            history = history[:1]
            print("(history cleared)")
            continue
        if line.startswith("/system "):
            history = [{"role": "system", "content": line[len("/system "):]}]
            print("(new system prompt, history cleared)")
            continue
        if line == "/history":
            for m in history:
                print(json.dumps(m, ensure_ascii=False))
            continue
        history.append({"role": "user", "content": line})
        for _ in range(5):  # at most 5 tool rounds per user turn
            calls = stream_turn(history)
            if not calls:
                break
            for c in calls:
                result = run_tool(c["name"], c["args"])
                print(f"tool> {c['name']}({c['args']}) -> {json.dumps(result)}")
                history.append({"role": "tool", "tool_call_id": c["id"],
                                "content": json.dumps(result)})

if __name__ == "__main__":
    smoke_tests() if args.mode == "test" else chat()
