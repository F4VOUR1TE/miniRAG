from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("docs/sample.pdf")
pages = loader.load()

print(f"Страниц: {len(pages)}")
print(f"Символов на первой странице: {len(pages[7].page_content)}")
print("--- первые 300 символов ---")
print(pages[7].page_content[:300])