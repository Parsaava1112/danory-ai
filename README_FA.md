# Danory GPT v8 — مدل زبانی فارسی حدود ۱۰۰ میلیون پارامتر

این نسخه یک بازطراحی مستقل از v7 است. ماژول تحلیل ورودی فعلی heuristic است و نباید آن را توانایی استدلال مستقل یا تضمین صحت پاسخ تلقی کرد؛ توانایی‌های زبانی و دستورپذیری باید با SFT و آزمون‌های جداگانه سنجیده شوند. معماری جدید با وزن‌های تصادفی تازه شروع می‌شود؛ وزن‌های v7 مستقیماً قابل بارگذاری نیستند، چون ابعاد مدل و واژگان تغییر کرده‌اند.

## مشخصات معماری

| ویژگی | مقدار |
|---|---:|
| پارامترهای دقیق | 99,922,560 (حدود 99.923M) |
| Vocabulary | 24,000 |
| Embedding | 640 |
| Transformer Layers | 16 |
| Attention Heads | 10 |
| KV Heads | 2 |
| FFN | 2,240 (SwiGLU) |
| Context | 2,048 توکن |
| Attention | Causal GQA + SDPA |
| Position | RoPE |
| Normalization | RMSNorm |
| Embeddings | وزن‌های ورودی/خروجی مشترک |

پارامترها با `python scripts/model_info.py` از خود مدل محاسبه شده‌اند. وزن FP32 به‌تنهایی تقریباً 381 MiB است؛ حافظه واقعی آموزش به‌مراتب بیشتر است.

## قابلیت‌های پیاده‌سازی‌شده

- مدل decoder-only با GQA، RoPE، RMSNorm و SwiGLU
- محاسبه Loss با ماسک توکن و نادیده‌گرفتن Padding
- بسته‌بندی پیوسته متن‌های Pretraining در بلوک‌های 2049 توکنی (برای کاهش Padding)
- تحلیل اولیه درخواست برای هدایت سبک پاسخ (قواعد سبک و قابل توسعه؛ نه یک شبکه عصبی مستقل)
- پشتیبانی از تاریخچه مکالمه از طریق فایل JSON
- Chat Template دارای role tokenهای system/user/assistant/tool
- SFT با Loss فقط روی پاسخ‌های assistant
- بارگذاری JSONL به‌صورت streaming از دیسک، بدون خواندن کل دیتاست در RAM
- Checkpoint اتمیک، نسخه‌دار، شامل Optimizer، RNG، تنظیمات و fingerprint توکنایزر
- بررسی ناسازگاری مدل/توکنایزر/مرحله و طول برنامه LR هنگام Resume
- ذخیره جداگانه `*_last.pt` و `*_best.pt`
- Workflowهای جداگانه تست، Pretraining و SFT با Artifact سی‌روزه
- Benchmark ثابت فارسی برای مقایسه خروجی‌ها

## محیط پیشنهادی

- Python 3.11 (پیشنهادی) یا 3.12
- PyTorch 2.5 یا جدیدتر
- برای آموزش 100M، GPU با حافظه مناسب به‌شدت توصیه می‌شود.
- کامپیوتر محلی قبلی شما (Celeron N4000 و حدود 3.8GB RAM) برای آموزش این مدل مناسب نیست. حتی inference محلی ممکن است کند و از نظر RAM محدود باشد.
- GitHub-hosted runner در Workflow حاضر CPU است؛ می‌توان با Stepهای کوتاه امکان‌پذیری را سنجید، اما آموزش کامل 100M روی CPU احتمالاً بسیار زمان‌بر است. زمان هر Step را در اجرای آزمایشی اندازه بگیرید؛ عدد ثابتی را نمی‌توان تضمین کرد.

## نصب محلی در Windows PowerShell

از پوشه پروژه اجرا کنید:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

اگر نصب PyTorch روی سیستم شما با خطا روبه‌رو شد، نسخه سازگار CPU را از راهنمای رسمی PyTorch نصب کنید.

## آماده‌سازی داده عمومی

این اسکریپت فقط دیتاست‌های عمومی را از Hugging Face stream می‌کند و به پوشه داده خصوصی شما دست نمی‌زند:

```powershell
python scripts/prepare_public_data.py --naab-train-rows 150000 --fineweb-rows 100000 --naab-val-rows 12000 --include-sft --sft-rows 20000
```

فایل‌ها در `data/public/` ساخته می‌شوند:

- `train.jsonl`
- `validation.jsonl`
- `sft.jsonl`
- `sft_validation.jsonl`
- `data_manifest.json`

مقادیر بالا سقف نمونه هستند، نه تعداد تضمین‌شده. فیلتر زبان، طول، و Deduplication ممکن است تعداد خروجی را کمتر کند.

### منابع و ملاحظات مجوز

| منبع | استفاده در پروژه | مجوز/ملاحظه درج‌شده در کارت دیتاست |
|---|---|---|
| [SLPL/naab](https://huggingface.co/datasets/SLPL/naab) | متن عمومی Pretraining و validation | کارت فعلی Hugging Face برچسب MIT دارد؛ corpus از چند منبع تشکیل شده و کارت درباره اطلاعات شخصی و منشأ داده هشدار می‌دهد. منشأ و شرایط استفاده را برای انتشار/استفاده تجاری جداگانه بررسی کنید. |
| [FineWeb2-HQ Persian](https://huggingface.co/datasets/mohammadkrb/fineweb2-HQ-persian) | منبع تکمیلی Pretraining | کارت فعلی ODC-BY را اعلام می‌کند؛ attribution را حفظ کنید. |
| [ParsiAI/FarsInstruct](https://huggingface.co/datasets/ParsiAI/FarsInstruct) | دستورپذیری و چندوظیفگی SFT | کارت فعلی Apache-2.0 را اعلام می‌کند؛ اطلاعات منبع و attribution را نگه دارید. |
| [PersianML/persian-general-knowledge](https://huggingface.co/datasets/PersianML/persian-general-knowledge) | مکالمه چندمرحله‌ای و دانش عمومی SFT | کارت فعلی CC-BY-4.0 را اعلام می‌کند؛ attribution لازم است. |
| [Kamyar-zeinalipour/CS_Farsi](https://huggingface.co/datasets/Kamyar-zeinalipour/CS_Farsi) | استدلال مبتنی بر متن و commonsense SFT | کارت فعلی CC-BY-SA-4.0 را اعلام می‌کند؛ شرایط ShareAlike و attribution را بررسی کنید. |

**Persian-SFT-102K عمداً به‌صورت خودکار وارد نمی‌شود**؛ کارت آن می‌گوید مجموعه ترکیبی است و مجوز نمونه‌ها وابسته به منبع اولیه است. فقط پس از بررسی مجوز هر منبع می‌توان زیرمجموعه مجاز را اضافه کرد.

این توضیحات مشاوره حقوقی نیستند. قبل از انتشار وزن‌ها یا استفاده تجاری، مجوز و منشأ داده‌های هر نمونه را بررسی کنید. `data_manifest.json` منشأهای استفاده‌شده را ثبت می‌کند.

## ساخت Tokenizer

Tokenizer روی متن عمومی Pretraining ساخته می‌شود و role tokenهای مکالمه را هم در واژگان رزرو می‌کند:

```powershell
python build_tokenizer.py --input data/public/train.jsonl --out data/tokenizer --vocab-size 24000 --max-chars 300000000
```

اگر اسکریپت اعلام کرد که Vocabulary واقعی با 24,000 تفاوت دارد، آموزش را شروع نکنید؛ مقدار واقعی را در `configs/model.json` وارد کنید و تست‌ها را دوباره اجرا کنید.

## تست قبل از آموزش

```powershell
python scripts/model_info.py
python scripts/check_params.py
python scripts/smoke_test.py
python -m compileall -q danory train.py generate.py scripts
python scripts/preflight.py
```

## آموزش Pretraining محلی

```powershell
python train.py --config configs/pretrain.json --stage pretrain --steps-this-run 20
```

`max_steps` کل برنامه زمان‌بندی LR است و در Resume نباید تغییر کند. `steps_this_run` فقط تعیین می‌کند این اجرا چند Step جلو برود.

## آموزش SFT محلی

پس از اتمام Pretraining و وجود `checkpoints/pretrain_best.pt`:

```powershell
python train.py --config configs/sft.json --stage sft --init-from checkpoints/pretrain_best.pt --steps-this-run 20
```

در SFT، Loss فقط روی توکن‌های پاسخ assistant محاسبه می‌شود؛ پیام‌های system و user نقش context دارند.

## تولید پاسخ

برای مدل پس از SFT. ماژول تحلیل درخواست، راهنمای کوتاهی به پیام system اضافه می‌کند و جایگزین توانایی آموخته‌شده مدل نیست:

```powershell
python generate.py --checkpoint checkpoints/sft_best.pt --config configs/pretrain.json --prompt "هوش مصنوعی چیست و چه کاربردهایی دارد؟" --tokens 120 --temperature 0.7
```

برای بررسی مدل صرفاً Pretrained، مسیر checkpoint را به `checkpoints/pretrain_best.pt` تغییر دهید. مدل Pretraining به‌تنهایی هنوز دستیار دستورپذیر محسوب نمی‌شود.

## Benchmark

```powershell
python scripts/benchmark.py --checkpoint checkpoints/sft_best.pt --config configs/pretrain.json
```

گزارش در `reports/benchmark.json` ذخیره می‌شود. خروجی‌ها باید انسانی ارزیابی شوند؛ معیارهای خودکار به‌تنهایی کیفیت استدلال یا صحت factual را ثابت نمی‌کنند.

## GitHub Actions

مخزن را در GitHub قرار دهید و از تب Actions استفاده کنید.

### 1) Pretraining

Workflow: **Danory GPT v8 pretraining**

- اجرای اول: `resume_run_id` خالی باشد.
- `steps_this_run` را ابتدا 20 تا 100 انتخاب کنید تا سرعت واقعی Runner مشخص شود.
- پس از اجرای موفق، در Summary بخش Artifacts، فایل `danory-v8-pretrain` را خواهید دید.
- برای ادامه، `resume_run_id` را با عدد Run ID اجرای قبلی پر کنید؛ این عدد در URL بعد از `/actions/runs/` است، نه شماره نمایشی `#N`.
- `max_steps=6000` ثابت می‌ماند. فقط `steps_this_run` در هر Job تغییر می‌کند.
- Artifact شامل `pretrain_last.pt`، `pretrain_best.pt` و Tokenizer است.

### 2) SFT

Workflow: **Danory GPT v8 instruction tuning**

- `pretrain_run_id`: شناسه اجرای Pretraining که Artifact معتبر `danory-v8-pretrain` دارد.
- اجرای اول SFT: `resume_run_id` خالی باشد.
- ادامه SFT: شناسه اجرای قبلی SFT را در `resume_run_id` قرار دهید.
- Artifact نهایی با نام `danory-v8-sft` شامل `sft_last.pt` و `sft_best.pt` است.

Workflowها CPU هستند و سقف زمان هر Job شش ساعت است. مقدار 100 Step صرفاً برای سنجش سرعت Runner است و به معنی مناسب‌بودن این تعداد برای آموزش نهایی نیست. تعداد Step قابل انجام در این مدت به سرعت Runner و تنظیمات بستگی دارد. قبل از اجرای طولانی، یک Job آزمایشی کوتاه انجام دهید.

## بازیابی Artifact روی Windows

Artifact را از صفحه اجرای GitHub Actions دانلود و Extract کنید. فایل‌ها را با همین ساختار در پروژه قرار دهید:

```text
checkpoints/pretrain_best.pt
checkpoints/pretrain_last.pt
data/tokenizer/tokenizer.json
data/tokenizer/tokenizer_report.json
```

برای SFT:

```text
checkpoints/sft_best.pt
checkpoints/sft_last.pt
```

فایل‌های `.pt` را در Git repository commit نکنید؛ آن‌ها را در GitHub Actions Artifacts یا فضای ذخیره‌سازی خصوصی نگه دارید.

## نکته مهم درباره v7

این نسخه معماری و Vocabulary جدید دارد. Checkpointهای v7 با v8 سازگار نیستند و عمداً بارگذاری نمی‌شوند. فایل‌های v7 را به‌عنوان نسخه پشتیبان نگه دارید؛ v8 با وزن‌های تازه شروع می‌شود.
