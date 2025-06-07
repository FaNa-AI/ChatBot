import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- 1. تنظیمات مدل‌ها و توکنایزرها ---
# مدل برای تولید متن (Generator)
# شما می‌توانید مدل‌های فارسی مانند 'HooshvareLab/gpt2-fa' یا 'distilgpt2' را امتحان کنید
# برای این مثال از یک مدل کوچک و سریع استفاده می‌کنیم.
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# افزودن پد توکن به توکنایزر برای جلوگیری از ارور
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# مدل برای تبدیل متن به بردار (Retriever - برای محاسبه شباهت)
# این مدل برای ایجاد بردارهای معنی‌دار از جملات استفاده می‌شود.
sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# --- 2. پایگاه دانش (Knowledge Base) ---
# این یک پایگاه دانش ساده است. در کاربردهای واقعی، این می‌تواند از فایل‌ها، دیتابیس‌ها و... خوانده شود.
knowledge_base = [
    "پایتون یک زبان برنامه‌نویسی سطح بالا و تفسیر شده است.",
    "یادگیری ماشین زیرمجموعه‌ای از هوش مصنوعی است که به کامپیوترها اجازه می‌دهد بدون برنامه‌ریزی صریح، از داده‌ها یاد بگیرند.",
    "یادگیری عمیق نوعی از یادگیری ماشین است که از شبکه‌های عصبی عمیق استفاده می‌کند.",
    "RAG (Retrieval-Augmented Generation) یک معماری برای مدل‌های زبانی است که بازیابی اطلاعات را با تولید متن ترکیب می‌کند.",
    "RAG باعث می‌شود مدل‌های زبانی به اطلاعات خارجی دسترسی داشته باشند و پاسخ‌های دقیق‌تری ارائه دهند.",
    "تنسورفلو و پای‌تورچ دو فریم‌ورک محبوب برای یادگیری عمیق هستند.",
    "چت‌بات‌ها برنامه‌های کامپیوتری هستند که مکالمه انسانی را شبیه‌سازی می‌کنند.",
    "مدل‌های زبانی بزرگ (LLMs) مدل‌هایی هستند که بر روی حجم عظیمی از داده‌های متنی آموزش دیده‌اند و می‌توانند متن تولید کنند، ترجمه کنند و به سوالات پاسخ دهند.",
    "هوش مصنوعی رشته‌ای در علوم کامپیوتر است که به ساخت ماشین‌های هوشمند می‌پردازد."
]

# ایجاد بردارهای (Embeddings) برای پایگاه دانش
# این بردارهای متنی برای یافتن شباهت بین سوال کاربر و دانش موجود استفاده می‌شوند.
knowledge_embeddings = sentence_model.encode(knowledge_base, convert_to_tensor=True)

# --- 3. تابع بازیابی (Retriever Function) ---
def retrieve_relevant_docs(query, top_k=2):
    """
    اسناد مرتبط را از پایگاه دانش بر اساس شباهت معنایی بازیابی می‌کند.
    """
    query_embedding = sentence_model.encode(query, convert_to_tensor=True)
    # محاسبه شباهت کسینوسی بین سوال و تمام اسناد پایگاه دانش
    similarities = cosine_similarity(query_embedding.cpu().numpy().reshape(1, -1), knowledge_embeddings.cpu().numpy())
    # مرتب‌سازی و بازیابی اسناد با بالاترین شباهت
    top_indices = np.argsort(similarities[0])[-top_k:][::-1]
    relevant_docs = [knowledge_base[i] for i in top_indices]
    return relevant_docs

# --- 4. تابع تولید پاسخ (Generator Function) ---
def generate_response(prompt, max_length=150):
    """
    با استفاده از مدل زبانی، پاسخ تولید می‌کند.
    """
    # افزودن توکن [PAD] به عنوان پد برای توکنایزر
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id # استفاده از EOS token به عنوان pad token
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# --- 5. تابع اصلی چت‌بات (RAG Chatbot) ---
def rag_chatbot(query):
    """
    فرایند کامل RAG را برای پاسخ به یک پرس و جو اجرا می‌کند.
    """
    # گام 1: بازیابی اطلاعات مرتبط
    relevant_docs = retrieve_relevant_docs(query)
    print(f"اسناد بازیابی شده: {relevant_docs}")

    # گام 2: آماده‌سازی پرامپت برای تولید متن
    # ما اطلاعات بازیابی شده را به عنوان "context" به مدل می‌دهیم.
    context = "\n".join(relevant_docs)
    prompt = f"متن زیر را در نظر بگیرید:\n{context}\n\nبر اساس این اطلاعات و دانش خود، به سوال زیر پاسخ دهید: {query}\nپاسخ:"

    # گام 3: تولید پاسخ
    response = generate_response(prompt)
    return response

# --- 6. مثال‌های استفاده ---
if __name__ == "__main__":
    print("به چت‌بات RAG خوش آمدید! برای خروج، 'خروج' را تایپ کنید.")
    while True:
        user_query = input("شما: ")
        if user_query.lower() == 'خروج':
            print("چت‌بات: خدانگهدار!")
            break
        
        response = rag_chatbot(user_query)
        print(f"چت‌بات: {response}")