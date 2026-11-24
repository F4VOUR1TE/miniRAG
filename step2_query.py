import os
from dotenv import load_dotenv

# Импорты для работы с базой и эмбеддингами
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Импорты для LLM и формирования цепочки
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Загружаем ключ из .env
load_dotenv()

# 2. Подключаемся к векторной базе
# Важно использовать ту же модель эмбеддингов, что и при индексации!
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 3. Создаем "Retriever" (Искатель)
# search_kwargs={"k": 3} означает: найди и верни 3 самых релевантных чанка
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. Настраиваем LLM (Подключаемся к OpenRouter)
# ВАЖНО: Впиши сюда модель, которую ты выберешь на OpenRouter (с суффиксом :free)
llm = ChatOpenAI(
    # model="deepseek/deepseek-r1:free", 
    model="deepseek/deepseek-v4.1-flash-1017",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0 # 0 означает "минимум креатива", чтобы модель не фантазировала
)

# 5. Создаем шаблон промпта
template = """Ты — ассистент, который отвечает на вопросы строго по предоставленному контексту.

Правила:
1. Отвечай ТОЛЬКО на основе контекста ниже. 
2. Если в контексте нет ответа, напиши "В предоставленных документах нет информации об этом".
3. В конце ответа укажи страницы, откуда взята информация (например: [Источник: стр. 12]).

Контекст из документов:
{context}

Вопрос: {question}

Ответ:"""

prompt = ChatPromptTemplate.from_template(template)

# 6. Функция для склеивания найденных кусков в один текст
def format_docs(docs):
    # Проходимся по каждому найденному документу и добавляем номер страницы в начало
    formatted = []
    for d in docs:
        # В PyPDFLoader страницы нумеруются с 0, поэтому прибавляем 1 для красоты
        page_num = d.metadata.get("page", 0) + 1
        formatted.append(f"[Страница {page_num}]\n{d.page_content}")
    return "\n\n---\n\n".join(formatted)

# 7. Собираем цепочку (Chain)
# Логика: Вопрос -> Ищем контекст в базе -> Подставляем в промпт -> Отправляем в LLM -> Парсим строку
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 8. Запускаем!
if __name__ == "__main__":
    print("Ассистент готов. Введите вопрос (или 'выход' для завершения):")
    
    # Делаем бесконечный цикл, чтобы можно было задавать несколько вопросов подряд
    while True:
        question = input("\nВаш вопрос: ")
        if question.lower() in ["выход", "exit", "quit"]:
            break
            
        print("Ищу ответ...")
        try:
            answer = chain.invoke(question)
            print("\nОтвет ассистента:")
            print(answer)
        except Exception as e:
            print(f"Ошибка: {e}")