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
    summary = split_gened[-1]

    summary_list = summary.split('요약: ')
    output_text = ""
    for idx, summary in  enumerate(summary_list):
        output_text += f"{idx+1}. {summary}\n"

    print(output_text)

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
        
        

    prompt = f"제목 : {examples['title']}\n본문: {descriptions}\n요약: {examples['abstractive']}"
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
        
        

    prompt = f"제목 : {examples['title']}\n본문: {descriptions}\n요약:"
    tokenized = tokenizer(prompt, truncation=True, max_length=MAX_LEN)


    return tokenized
