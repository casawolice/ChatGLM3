from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from typing import Union

global_model = None
global_tokenizer = None


def resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def init():
    global global_model
    global global_tokenizer

    if global_model is None:
        global_tokenizer = AutoTokenizer.from_pretrained(
            resolve_path("./chatglm3-6b-01"), trust_remote_code=True)
        model = AutoModel.from_pretrained(resolve_path("./chatglm3-6b-01"),
                                          trust_remote_code=True,
                                          device='cuda')
        global_model = model.eval()
    return (global_model, global_tokenizer)


def chat():
    init()
    history = []  # 用于存储对话历史
    while True:
        user_input = input("Human：").encode('utf-8', errors='ignore').decode('utf-8')
        if user_input.lower() == 'exit':
            break
        history.append(user_input)  # 将用户输入添加到历史中
        response, history = global_model.chat(global_tokenizer,
                                              user_input,
                                              history=[])  # 处理用户输入并生成响应
        print(f"AI:{response}")
        history.append(response)  # 将响应添加到历史中


chat()
