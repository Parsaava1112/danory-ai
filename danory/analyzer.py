"""Lightweight Persian request analysis used to shape prompts, not a reasoning model."""
import re

RULES={
 'summarization':('خلاصه','تلخیص','چکیده','خلاصه کن'),
 'comparison':('مقایسه','تفاوت','فرق','مقایسه کن'),
 'explanation':('توضیح بده','شرح بده','چیست','چگونه','چطور','چرا'),
 'math':('حساب کن','محاسبه','چقدر می‌شود','معادله','ریاضی'),
 'translation':('ترجمه کن','به انگلیسی','به فارسی','ترجمه'),
 'writing':('بنویس','نگارش','متن تولید کن','داستان'),
 'code':('کد بنویس','برنامه‌نویسی','پایتون','خطا','باگ','کد'),
 'analysis':('تحلیل کن','بررسی کن','نقد کن','استدلال'),
}

def analyze_request(text):
    value=re.sub(r'\s+',' ',str(text or '')).strip()
    lowered=value.lower()
    matches=[name for name,terms in RULES.items() if any(term in lowered for term in terms)]
    intent=matches[0] if matches else 'general'
    multi=bool(re.search(r'\b(و|همچنین|به‌علاوه|علاوه بر این|همزمان)\b',value))
    return {'language':'fa','intent':intent,'multi_part':multi,'length':'detailed' if len(value)>180 or multi else 'concise'}

def response_guidance(analysis):
    intent=analysis['intent']
    guidance={
      'summarization':'متن را وفادارانه و بدون افزودن اطلاعات تازه خلاصه کن.',
      'comparison':'معیارهای مقایسه را روشن کن و شباهت‌ها و تفاوت‌ها را جدا کن.',
      'explanation':'مفهوم را مرحله‌به‌مرحله و با مثال روشن توضیح بده.',
      'math':'محاسبات را دقیق انجام بده و مراحل کوتاه و قابل بررسی را بنویس.',
      'translation':'معنا و لحن متن مبدأ را حفظ کن و فقط ترجمه را ارائه بده.',
      'writing':'متنی طبیعی، منسجم و بدون تکرار عبارت‌های کلیشه‌ای بنویس.',
      'code':'ابتدا مسئله را تشخیص بده؛ کد را کامل و همراه با روش اجرا و تست ارائه کن.',
      'analysis':'ادعاها، شواهد و نتیجه‌گیری را از هم تفکیک کن و عدم قطعیت را مشخص کن.',
      'general':'مستقیم، دقیق و متناسب با سؤال پاسخ بده؛ اگر داده کافی نیست، شفاف بگو.'}
    return guidance[intent]
