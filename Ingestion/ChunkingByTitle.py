from unstructured.chunking.title import chunk_by_title


## Chucking the extracted sturctured elements using title
def chunking_by_title(elements):
    
    print(f"\n\nChunking the elements using title method.\n")

    chunks = chunk_by_title(
        elements= elements,
        max_characters = 3000,
        new_after_n_chars = 2500,
        combine_text_under_n_chars = 500
    )
    
    print(f"Loading the chunk size : {len(chunks)}\n")

    return chunks
