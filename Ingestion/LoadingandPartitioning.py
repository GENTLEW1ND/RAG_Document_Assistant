from unstructured.partition.pdf import partition_pdf


# Extract the contents from the documents
def document_partitioning(file_path:str):
    """Extrating Structured data from the Unstructured Documents"""

    print(f"Partitioning the contents of the file from {file_path}")

    elements = partition_pdf(
        filename = file_path,
        strategy = "hi_res",
        infer_table_structure=True,
        extract_image_block_types=['Image'],
        extract_image_block_to_payload=True
    )
    
    print(f"Partitioned elements : \n {elements}")
    
    return elements