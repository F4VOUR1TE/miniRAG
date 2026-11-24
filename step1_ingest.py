from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------- Шаг 1: загрузка PDF ----------
# PyPDFLoader читает PDF и возвращает по одному Document на каждую страницу
loader = PyPDFLoader("docs/sample.pdf")  # замените на имя вашего PDF
pages = loader.load_and_split()
print(f"Загружено страниц: {len(pages)}")
print(f"Метаданные первой страницы: {pages[0].metadata}")

# ---------- Шаг 2: разрезание на фрагменты ----------
# RecursiveCharacterTextSplitter сначала режет по абзацам, затем по предложениям, словам, символам
# chunk_size=500: каждый фрагмент около 500 символов
# chunk_overlap=50: соседние фрагменты перекрываются на 50 символов, чтобы избежать разрыва контекста
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(pages)
print(f"Получено фрагментов: {len(chunks)}")

# ---------- Шаг 3: модель эмбеддингов ----------
# Локальная бесплатная модель, преобразует текст в векторы для сравнения семантического сходства
embeddings = HuggingFaceEmbeddings(
    # model_name="sentence-transformers/all-MiniLM-L6-v2"
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" #для русского языка
)

# ---------- Шаг 4: сохранение в Chroma ----------
# from_documents автоматически: генерирует векторы для каждого фрагмента → сохраняет в локальную базу данных
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("Векторная база данных сохранена в ./chroma_db")
print(f"Количество векторов в базе: {vectorstore._collection.count()}")