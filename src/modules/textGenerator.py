def textGenerator(seedText, textLength):
    for _ in range(textLength):
        token_list = tokenizer.texts_to_sequences([seedText])[0]
        token_list = pad_sequences([token_list], maxlen=MAX_SEQ_LEN-1,padding='pre')
        pred = model.predict_classes(token_list,verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == pred:
                output_word = word
                break
        seedText += " " + output_word
    return print(seedText)
