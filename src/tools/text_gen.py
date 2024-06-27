import re

def clean_text(text):
  text_rmv = re.sub('[-=+,#/\?^@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]', '', text)
  return text_rmv

def gen(x, model, tokenizer):
    gened = model.generate(
        **tokenizer(
            x,
            return_tensors='pt',
            return_token_type_ids=False
        ),
        max_new_tokens=1024,
        early_stopping=True,
        do_sample=True,
        eos_token_id=2,
    )
    gened = tokenizer.decode(gened[0])
    split_gened = gened.split('\n')
    
    output_text = ""
    idx = 0
    for gen_text in split_gened:
        if gen_text.find('요약:')>=0:
            gen_text = clean_text(gen_text)
            output_text +=f"{idx+1}. {gen_text}\n"

    return output_text

def preprocessing_function(examples, tokenizer, MAX_LEN=1024):
    texts = examples['text']

    descriptions = ""

    for text in texts:
        if len(descriptions) != 0:
            descriptions = descriptions + '\n'
        for text_line in text:
            if len(descriptions) == 0:
                descriptions = text_line['sentence']
            else:
                descriptions = descriptions + ' ' + text_line['sentence']
        
    prompt = f"### 요청 : 다음 제목이랑 본문을 보고 내용을 요약 해줘 [제목 : {examples['title']}\n본문: {descriptions}]\n ### 답변 : [요약: {examples['abstractive']}]"
    tokenized = tokenizer(prompt, truncation=True, padding="max_length", max_length=MAX_LEN)


    return tokenized

def preprocessing_function_test(examples, tokenizer, MAX_LEN=1024):
    texts = examples['text']

    descriptions = ""

    for text in texts:
        if len(descriptions) != 0:
            descriptions = descriptions + '\n'
        for text_line in text:
            if len(descriptions) == 0:
                descriptions = text_line['sentence']
            else:
                descriptions = descriptions + ' ' + text_line['sentence']
        
    prompt = f"### 요청 : 제목이랑 본문의 내용 요약해줘 [제목 : {examples['title']}\n본문: {descriptions}]\n ### 답변 : [요약: ]"
    tokenized = tokenizer(prompt, truncation=True, max_length=MAX_LEN)


    return tokenized