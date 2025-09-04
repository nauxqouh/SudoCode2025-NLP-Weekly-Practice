def chunk_text(tokens, chunk_size=15):
    for i in range(0, len(tokens), chunk_size):
        yield tokens[i:i + chunk_size]