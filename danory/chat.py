ROLES={'system':'<|system|>','user':'<|user|>','assistant':'<|assistant|>','tool':'<|tool|>'}
END='<|end|>'

def normalize_messages(messages):
    if not isinstance(messages,list) or not messages: raise ValueError('messages must be a non-empty list')
    out=[]
    for m in messages:
        if not isinstance(m,dict): continue
        role=str(m.get('role','')).lower().strip(); content=m.get('content','')
        if role in ROLES and isinstance(content,str) and content.strip(): out.append({'role':role,'content':content.strip()})
    if not out: raise ValueError('No valid messages')
    return out

def encode_chat(messages,tokenizer,max_length):
    ids=[]; mask=[]
    for m in normalize_messages(messages):
        rid=tokenizer.token_to_id(ROLES[m['role']]); eid=tokenizer.token_to_id(END)
        if rid is None or eid is None: raise ValueError('Tokenizer is missing chat special tokens')
        content=tokenizer.encode(m['content'],add_special_tokens=False).ids
        ids.append(rid); mask.append(0); ids.extend(content); mask.extend([1 if m['role']=='assistant' else 0]*len(content))
        ids.append(eid); mask.append(1 if m['role']=='assistant' else 0)
    bos=tokenizer.token_to_id('<|bos|>')
    if bos is not None: ids.insert(0,bos); mask.insert(0,0)
    if len(ids)>max_length:
        ids=ids[-max_length:]; mask=mask[-max_length:]
    return ids,mask

def render_prompt(messages,tokenizer):
    ids=[]; bos=tokenizer.token_to_id('<|bos|>')
    if bos is not None: ids.append(bos)
    for m in normalize_messages(messages):
        rid=tokenizer.token_to_id(ROLES[m['role']]); eid=tokenizer.token_to_id(END)
        if rid is None or eid is None: raise ValueError('Tokenizer missing chat tokens')
        ids.append(rid); ids.extend(tokenizer.encode(m['content'],add_special_tokens=False).ids); ids.append(eid)
    aid=tokenizer.token_to_id(ROLES['assistant'])
    if aid is None: raise ValueError('Tokenizer missing assistant token')
    ids.append(aid); return ids
