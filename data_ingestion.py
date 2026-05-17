from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import json
from langchain_core.documents import Document



load_dotenv()


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


def seperate_using_types(chunk):
    
    content_data={
        'text': chunk.texts,
        'tables': [],
        'images': [],
        'types': ['text']
    }
    
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for elements in chunk.metadata.orig_elements:
            elements_type = type(elements).__name__
            
            if elements_type == 'Table':
                content_data['types'].append('table')
                table_content = getattr(chunk.metadata, 'text_as_html', elements.text)
                content_data['tables'].append(table_content)
                
            elif elements_type == 'Image':
                if hasattr(chunk, 'metadata') and  hasattr(chunk.metadata, 'image'):
                    content_data['types'].append('image')
                    content_data['images'].append(chunk.metadata.image_base64)

    content_data['types'] = list(set(content_data['types']))
       
    return content_data



def summarize_content(text:str, tables:list[str], images:list[str])->str:
    """
    Summarize the contents of the table and images.
    """
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        #Creating a prompt for the llm
        prompt_text = f"""You are to create a searchable description for document content retrival
        
            CONTENT TO ANALYZE:
            TEXT:
            {text}
            
        """

        if tables:
            prompt_text += "Tables \n"
            
            for i, table in enumerate(tables):
               prompt_text += f"Table {i+1}: \n {table}\n\n" 
               
               prompt_text += """
               YOUR TASK:
               Generate a comprehensive, searchable description that covers:
               
               1. Key facts, numbers, and data points from text and tables
               2. Main topics and concepts discussed
               3. Questions this content could answer
               4. Visual content analysis (charts, diagrams, patterns in images)
               5. Alternative search terms users might use
               
               Make it detailed and serachable - prioritize findability over brevity
               
               SEARCHABLE DESCRIPTION:
               """

        # Build message content starting with text
        message_content = [{"type": "text", "text": prompt_text}]

        # Add images to the message
        for image_base64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
            
        message = HumanMessage(content=message_content)
            
        response = llm.invoke([message])
        
        return response.content
        
        
    except Exception as e:
        print(f" AI summary failed: {e}")
        #Fallback to simple summary
        summary = f"{text[:300]}"
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary    
 

def summarize_chunks(chunks):
    """
    Summarize the chunks which contain text, images, tables 
    """
    
    langchain_document = []
    
    # total_chunks = len(chunks)
    
    for chunk in enumerate(chunks):
        # current_chunk = i+1
        
        #Analyze the chunk 
        analyze_chunk = seperate_using_types(chunk)
            
        if analyze_chunk['tables'] or analyze_chunk['images']:
            
            print(f"Creating summary for the tables and images")
            
            try:
                enhanced_summary = summarize_content(
                    analyze_chunk['text'],
                    analyze_chunk['tables'],
                    analyze_chunk['images']
                )
                print("AI summary was completed successfully.")

            except Exception as e:
                print(f"Summarizing failed {e}")
        
                enhanced_summary = analyze_chunk['text']

        else:
            print("No tables or image are found.")
            enhanced_summary = analyze_chunk['text']


        # Creating Langchain document
        doc = Document(
            page_content = enhanced_summary,
            metadata = {
                "original_content": json.dumps({
                    "raw_text": analyze_chunk['text'],
                    "tables_html": analyze_chunk['tables'],
                    "images_base64": analyze_chunk['images']
                })
            }
        )
        
        langchain_document.append(doc)
    
    print(f"Created langchain document of chunks size of {len(langchain_document)}.")

    return langchain_document

def main():
    elements = document_partitioning("Document/attention-is-all-you-need-Paper.pdf")
    chunks = chunking_by_title(elements)
    doc = summarize_chunks(chunks)


if __name__ == "__main__":
    main()
    
