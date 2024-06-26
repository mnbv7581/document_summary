import re

remove_enter = ["'\'", '\n','\n\n','\n\n\n','\n\n\n\n','\n\n\n\n\n','\n\n\n\n\n\n',\
                '\n\n\n\n\n\n\n','\n\n\n\n\n\n\n\n','\n\n\n\n\n\n\n\n\n','\n\n\n\n\n\n\n\n\n\n', u'\xa0',]
remove_space = ['  ','   ','    ','     ','       ','        ','         ','          ',]

def url_encode(text: str):
    "url 검출"
    
    # URL 추출할 정규표현식 생성
    url_regex = r"(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&\/\/=]*)"

    reg = re.compile(url_regex)
    
    res = reg.search(text)
    
    if res == None:
        return text
    
    else:
        indexes = res.span()
        
        url_txt = text[indexes[0]:indexes[1]]
        
        return url_txt

def clean_text(text):
    text_rmv = re.sub('[-=+,#/\?^@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]', '', text)
    text_rmv = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", text_rmv)
    text_rmv = text_rmv.replace('\n', '')       
    for rm_spc in remove_space:
        text_rmv = text_rmv.replace(rm_spc, ' ')
    return text_rmv

def gen(x, model, tokenizer):
    gened = model.generate(
        **tokenizer(
            x,
            return_tensors='pt',
            return_token_type_ids=False
        ),
        max_new_tokens=256,
        early_stopping=True,
        do_sample=True,
        num_beams=3, #Beam Search 수행, 1보다 큰 값을 지정
        num_return_sequences=2, # 두 개의 문장을 리턴
        top_k=50, # 확률 순위가 50위 밖인 토큰은 샘플링에서 제외
        top_p=0.95, # 누적 확률이 95%인 후보집합에서만 생성
        temperature=0.7, # 확률분포를 sharp하게 만들고 확률값이 높은 토큰이 더 잘 나오도록 설정.
        repetition_penalty=1.5,
        eos_token_id=2,
        pad_token_id=0
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
