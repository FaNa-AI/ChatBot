from transformers import pipeline

# بارگذاری مدل Flant-T5-base برای task "text2text-generation"

try:
    chatbot = pipeline("text2text-generation", model="google/flan-t5-base")
except Exception as e:
    print(f"خطا در بارگذاری مدل: {e}")
    print("لطفاً از اتصال به اینترنت اطمینان حاصل کنید یا دوباره تلاش کنید.")
    exit()

print("چت‌بات آماده است. برای خروج، 'خروج' یا 'بای' را تایپ کنید.")

while True:
    user_input = input("شما: ")

    if user_input.lower() in ["خروج", "بای", "exit", "bye"]:
        print("چت‌بات: خداحافظ!")
        break

    # تولید پاسخ با استفاده از مدل
   
    response = chatbot(user_input, max_length=50, num_return_sequences=1)

    # نمایش پاسخ
    if response:
        print(f"چت‌بات: {response[0]['generated_text']}")
    else:
        print("چت‌بات: متوجه نشدم، می‌توانید دوباره بپرسید؟")