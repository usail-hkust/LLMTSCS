from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import uvicorn, json, datetime
import torch
import transformers


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

app = FastAPI()
B_INST, E_INST = "[INST]", "[/INST]"

model = None
tokenizer = None
last_model_name = None
gpu_count = torch.cuda.device_count()

@app.post("/")
async def create_item(request: Request):
    global model, tokenizer, last_model_name, gpu_count
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    model_name = json_post_list.get('model')
    system_message = json_post_list.get('system_message')
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')

    # init model
    if model is None or tokenizer is None or last_model_name != model_name:
        last_model_name = model_name
        model = None
        tokenizer = None
        model_path = f"./llm_models/{model_name}"
        print("====================================")
        print(f"Loading model {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        config.pretraining_tp = 1
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        model.eval()
        print("====================================")
        print(f"Load success!")

    if system_message:
        SYSTEM = f"""<<SYS>>\n{system_message}\n<</SYS>>\n\n"""
        prompt = SYSTEM + prompt.strip()

    input_ids = []
    for q, a in history:
        input_ids += tokenizer.encode(f"{B_INST} {q} {E_INST} {a}") + [tokenizer.eos_token_id]
    input_ids += tokenizer.encode(f"{B_INST} {prompt} {E_INST}")

    response = model.generate(torch.tensor([input_ids]).cuda(),
                              do_sample=True,
                              max_length=max_length if max_length else 4096,
                              top_p=top_p if top_p else 1.0,
                              temperature=temperature if temperature else 0.1,
                              top_k=50)
    response = tokenizer.decode(response[0, len(input_ids):], skip_special_tokens=True)
    history.append([prompt, response])

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()
    return answer
