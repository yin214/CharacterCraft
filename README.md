# CharacterCraft: Bridging the Literature-Reality Dialogue Gap for Practical Role-Playing Agents

<p align="center"> <strong>
📄 <a href="https://github.com/yin214/CharacterCraft">Paper</a> ·
🗂️ <a href="https://huggingface.co/datasets/alice3214/CharacterCraft-Data">Dataset</a> ·
🤗 <a href="https://huggingface.co/alice3214/Qwen2-1.5B-DE">Model</a> 
</strong>
</p>

## Dialogue Extraction

**Qwen2-1.5B-DE** is a specialized language model fine-tuned from `Qwen/Qwen2-1.5B-Instruct` for the specific task of **extracting character dialogues from Chinese literary texts**. The model takes a segment of novel text as input and outputs a structured list of dialogues, complete with speaker identification.

### How to use

You can easily use this model with the `transformers` library.

``` python
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda"

model = AutoModelForCausalLM.from_pretrained(
    'alice3214/Qwen2-1.5B-DE',
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "你是一个小说文本分析领域的专家，你需要从给定的文本中提取人物对话，包括说话人物名称和对话内容"
user_prompt = '''小说文本:宝玉看罢，因笑道：“这个妹妹我曾见过的。”贾母笑道：“可又是胡说！你又何曾见过他？”宝玉笑道：“虽然未曾见过她，然我看着面善，心里就算是旧相识，今日只作远别重逢，未为不可。”贾母笑道：“更好，更好，若如此，更相和睦了！”宝玉便走近黛玉身边坐下，又细细打量一番，因问：“妹妹可曾读书？”黛玉道：“不曾读书，只上了一年学，些须认得几个字。”宝玉又道：“妹妹尊名是那两个字？”黛玉便说了名字。宝玉又问表字。黛玉道：“无字。”'''
messages = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": user_prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1024
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

Result:
``` text
宝玉|这个妹妹我曾见过的。
贾母|可又是胡说！你又何曾见过他？
宝玉|虽然未曾见过她，然我看着面善，心里就算是旧相识，今日只作远别重逢，未为不可。
贾母|更好，更好，若如此，更相和睦了！
宝玉|妹妹可曾读书？
黛玉|不曾读书，只上了一年学，些须认得几个字。
宝玉|妹妹尊名是那两个字？
黛玉|无字。
```
