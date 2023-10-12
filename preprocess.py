def build_inputs_with_special_tokens(vocab, token_ids_0, token_ids_1 = None, ):
    """
    make [cls] + sent1 + [sep] (+ sent2 + [sep])
    input>
    token_ids_0 = 1st sentence, |token_ids_0| = (batch_size, max_sent)
    token_ids_1 = 2nd sentence
    """
    cls_token_id, sep_token_id = vocab['[CLS]'], vocab['[SEP]']
    if token_ids_1 is None:
        return [cls_token_id] + token_ids_0 + [sep_token_id]
    return [cls_token_id] + token_ids_0 + [sep_token_id] + token_ids_1 + [sep_token_id]
