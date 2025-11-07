"""
⚖️ Conflit Analyse – محلل تنازع القوانين الدولي الخاص
مشروع أكاديمي موجه لطلبة السنة الثالثة قانون خاص
إعداد الطالب: ت. نورالدين – جامعة النعامة، الجزائر
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 🔹 النموذج العربي المفتوح المصدر
MODEL_NAME = "FreedomIntelligence/AceGPT-7B"

print("⏳ جارٍ تحميل النموذج... قد يستغرق بضع دقائق.")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("✅ النموذج جاهز!")

PROMPT_TEMPLATE = """
أنت أستاذ في القانون الدولي الخاص موجه لطلبة السنة الثالثة قانون خاص.
حلّل الوقائع التالية تحليلاً أكاديميًا ومنهجيًا وفق المواد من 9 إلى 24 من القانون المدني الجزائري.

1) التكييف القانوني (م9)
2) العنصر الأجنبي
3) ضابط الإسناد
4) القانون الواجب التطبيق
5) الإحالة (م23 مكرر 1)
6) النظام العام (م24)
7) النتيجة القانونية النهائية.

الوقائع: {}
"""

def analyser(faits: str):
    prompt = PROMPT_TEMPLATE.format(faits)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=350)
    texte = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n📘 التحليل القانوني:\n")
    print(texte)

if __name__ == "__main__":
    print("⚖️ Conflit Analyse – محلل تنازع القوانين الدولي الخاص")
    faits = input("\n📝 أدخل الوقائع القانونية:\n> ")
    analyser(faits)
