> **وضعیت:** فاز ۰ (زیرساخت آزاد از Gurobi)، فاز ۱ (Branch-and-Price کامل با ng-route pricing)، و از فاز ۲: هدف دوفازی lexicographic (اول تعداد وسیله، بعد فاصله — همون objective استاندارد مقاله‌های VRPTW) و dual stabilization (du Merle et al. 1999) پیاده‌سازی و با تست واحد + اوراکل brute-force مستقل اعتبارسنجی شده‌اند. کد در [`vrptw_cg/`](vrptw_cg/)، جزئیات فنی در [`docs/REPORT.md`](docs/REPORT.md)، و پیاده‌سازی اصلی (بدون تغییر) برای مقایسه در [`legacy/`](legacy/) نگه‌داری شده. باقی‌مانده‌ی سطح ۲/۳/۴ (EVRPTW، ML-guided pricing، دمو وب) هنوز پیاده‌سازی نشده‌اند.

# تحلیل گپ‌ها و نقشه‌راه توسعه پروژه VRPTW-Column-Generation

> این سند نتیجه‌ی خواندن کامل کد ریپازیتوری [SimoneRichetti/VRPTW-Column-Generation](https://github.com/SimoneRichetti/VRPTW-Column-Generation) (که در همین پوشه clone شده) به‌علاوه‌ی تحقیق روی جدیدترین تکنیک‌های علمی حوزه VRP (۲۰۲۴-۲۰۲۶) و بررسی دقیق اینکه «ORTEC» واقعاً چه چیزی به این پروژه اضافه می‌کند، است. هدف: فهرستی از ایده‌های واقعی و قابل‌اجرا برای تصمیم‌گیری، نه صرفاً یک لیست آرزو.

---

## ۱. چه چیزی الان واقعاً در کد هست (نه چیزی که README می‌گوید)

خواندن فایل‌به‌فایل کد نشان می‌دهد این پروژه از یک ایده‌ی درست شروع کرده ولی در نیمه‌راه متوقف شده:

| فایل | نقش واقعی | وضعیت |
|---|---|---|
| `col-gen-vrptw.py` | حلقه‌ی اصلی Column Generation | فقط **LP Relaxation** حل می‌شود؛ هیچ branch/bound برای integer شدن جواب وجود ندارد |
| `optimization.py` → `subProblem` | Pricing problem با DP روی resource دیسکرت‌شده (شبیه Desrochers 1992) | **غیرelementary** (اجازه‌ی تکرار گره را می‌دهد جز جلوگیری ساده از ۲-سیکل با ترفند `f_tk`)؛ خالص Python، حلقه‌های تودرتو روی (node × capacity × time) → برای n>25-50 عملاً می‌ترکد |
| `ESPmodel.py` | مدل MIP دقیق ESPPRC با Gurobi | طبق اعتراف خود نویسنده در `note.txt`: «بعد از ۱۶۰ دقیقه truncate شد، infeasible» → **در عمل مرده و استفاده نمی‌شود** |
| `impact.py` | هیوریستیک IMPACT (Ioannou et al. 2001) برای تولید ستون‌های اولیه | خوب پیاده شده، نقطه‌قوت واقعی کد |
| `coverCost.py` | راند کردن جواب کسری به جواب صحیح با یک هیوریستیک حریصانه‌ی set-cover | این یعنی جواب نهایی **تضمین بهینگی ندارد** — این دقیقاً همان نکته‌ای‌ست که در فایل تحلیلی شما هم به آن اشاره شده بود |
| `utilities.py` → `readInstanceN` | ورودی از طریق `input()` تعاملی | غیرقابل‌اسکریپت‌سازی، امکان batch benchmark وجود ندارد |
| کل پروژه | فقط `gurobipy` | وابسته به لایسنس تجاری Gurobi، هیچ fallback متن‌باز ندارد |
| کل پروژه | — | **صفر** تست، صفر CI، صفر `requirements.txt`/`pyproject.toml`، صفر visualization، صفر logging (فقط `print`) |

**جمع‌بندی گپ اصلی:** این کد یک **Column Generation ساده** است، نه یک **Branch-and-Price**. فایل تحلیلی شما همین را حدس زده بود («جواب‌های اعشاری می‌دهد») و کد این را ۱۰۰٪ تأیید می‌کند. علاوه بر آن، pricing problem هم از نظر علمی عقب است (نه elementary، نه ng-route، نه bucket-graph — همه‌ی این‌ها استاندارد صنعتی امروز هستند).

---

## ۲. ORTEC واقعاً چیست و چطور می‌شود «قاطی» پروژه کرد؟ (بررسی واقع‌گرایانه)

نکته‌ی مهم و صادقانه: **ORTEC یک شرکت نرم‌افزار Enterprise SaaS است** (مثل SAP یا Blue Yonder در حوزه لجستیک) و سالور اصلی‌اش کاملاً **closed-source و B2B** است؛ هیچ API عمومی یا SDK رایگان برای دولوپرهای مستقل منتشر نکرده که بشود مستقیم در یک پروژه‌ی دانشجویی «صدا زد». پس این گزینه که سالور تجاری ORTEC را داخل این ریپو فراخوانی کنیم، **عملاً وجود ندارد** (دقیقاً مثل مشکل لایسنس Gurobi، ولی حتی محدودتر).

اما وقتی عمیق‌تر رفتم، سه ارتباط **واقعی و رایگان** با ORTEC پیدا شد که خیلی باارزش‌ترند از یک API ساده:

1. **ORTEC خودش PyVRP را روی GitHub fork کرده** (`github.com/ortec/PyVRP`, MIT license). PyVRP یک سالور Hybrid Genetic Search است که رتبه‌ی اول چالش **DIMACS VRPTW 2021** و رتبه‌ی اول بخش static چالش **EURO meets NeurIPS 2022** را گرفته است. یعنی ORTEC عملاً استاندارد baseline متن‌باز صنعت را همین معرفی کرده.
2. **ORTEC برگزارکننده‌ی مسابقات آکادمیک واقعی VRP است**: VeRoLog Solver Challenge (۲۰۱۶-۲۰۱۷ و ۲۰۱۹، بر اساس داده‌ی مشتریان واقعی ORTEC با محدودیت‌های multi-period و technician routing) و **EURO Meets NeurIPS 2022 Vehicle Routing Competition** (اولین تلاش رسمی برای اتصال OR کلاسیک به Machine Learning در VRP). دیتاست‌ها و quickstart-code این مسابقات روی GitHub سازمان ORTEC عمومی و رایگان است.
3. این یعنی راه واقعی «استفاده از ORTEC» این است: **پروژه را با فرمت و معیار داده‌ای این مسابقات محک بزنیم و در مقابل baseline متن‌باز خودشان (PyVRP) بنچمارک کنیم** — نه فراخوانی یک API تجاری که وجود ندارد.

> **نتیجه برای رزومه:** جمله‌ی «این پروژه با benchmark instances چالش ORTEC × EURO/NeurIPS 2022 و در مقایسه با baseline متن‌باز خودِ ORTEC (PyVRP) اعتبارسنجی شده» خیلی قوی‌تر و صادقانه‌تر از ادعای «یکپارچه‌سازی با ORTEC» است که در عمل ممکن نیست.

---

## ۳. ایده‌های خفن (اولویت‌بندی‌شده با سطح افورت)

### سطح ۱ — فونداسیون (ضروری، افورت کم، بدون این‌ها بقیه بی‌معناست)
- **حذف `input()`**: آرگومان‌های CLI با `argparse` + قابلیت اجرای batch روی همه‌ی instance‌های Solomon پشت سر هم (لازم برای هر benchmark جدی).
- **Backend سالور قابل‌تعویض**: انتزاع لایه‌ی مدل‌سازی MIP/LP تا بشود بین `gurobipy`، `HiGHS` (رایگان، خیلی سریع، از طریق `highspy` یا `linopy`) و `OR-Tools` سوییچ کرد → دیگر وابسته به لایسنس Gurobi نیستیم.
- **Visualization**: رسم مسیرها با `matplotlib`/`plotly` (۲بعدی روی مختصات Solomon) و نسخه‌ی نقشه‌ی واقعی با `folium`/`kepler.gl` — دقیقاً همان چیزی که در TODO خود کد هم نوشته شده (`# Plot routes`) ولی هیچ‌وقت انجام نشده.
- **بسته‌بندی درست پروژه**: `pyproject.toml`، `requirements.txt` مجزا برای نسخه‌ی open-source و نسخه‌ی Gurobi، تست واحد با `pytest` (مثلاً تست `reduceTimeWindows`، `createDistanceMatrix`)، GitHub Actions CI.

### سطح ۲ — هسته‌ی علمی (بیشترین ارزش آکادمیک، افورت متوسط تا زیاد)
- **تکمیل Branch-and-Price واقعی**: اضافه‌کردن یک لایه‌ی branch-and-bound روی متغیرهای کسری خروجی master problem (branching روی arc-flow، نه روی خودِ ستون‌ها) تا جواب integer **تضمین‌شده بهینه** بگیریم — همان چیزی که فایل تحلیلی شما هم به‌عنوان مهم‌ترین ارتقا اشاره کرده بود.
- **ارتقای pricing problem به ng-route relaxation**: این تکنیک (Baldacci et al., و بعدها Sadykov/Uchoa) با یک تغییر نسبتاً کوچک در DP فعلی، هم مشکل «non-elementary path» کد فعلی را حل می‌کند و هم به‌شدت سریع‌تر می‌شود. این استاندارد امروز صنعت/آکادمی است.
- **Dual stabilization** (مثلاً smoothing به روش Wentges/du Merle) برای رفع پدیده‌ی معروف *tailing-off* در Column Generation — چیزی که در حلقه‌ی فعلی هیچ اثری از آن نیست.
- **بنچمارک در برابر solverهای متن‌باز state-of-the-art**:
  - [`PyVRP`](https://github.com/PyVRP/PyVRP) (MIT) به‌عنوان heuristic باکیفیت برای مقایسه‌ی upper bound.
  - [`VRPSolverEasy`](https://github.com/inria-UFF/VRPSolverEasy) (بر پایه‌ی BaPCod/VRPSolver، رایگان برای academic use) به‌عنوان یک Branch-Cut-and-Price *واقعی و آماده* برای اعتبارسنجی جواب دقیق خودمان با ng-route + bidirectional labeling + stabilization — می‌شود صادقانه گفت «جواب ما با state-of-the-art exact solver دنیا مو‌به‌مو یکی است».

### سطح ۳ — بخش تحقیقاتی/خفن (تمایز واقعی نسبت به پروژه‌های مشابه روی گیت‌هاب)
- **ML/GNN-guided pricing** (مقالات ۲۰۲۴-۲۰۲۵): Gerbaux et al. و Koutecká et al. نشان داده‌اند یک GNN آموزش‌دیده به‌صورت unsupervised می‌تواند گراف pricing را قبل از حل DP هرس کند و تا ~۹٪ بهبود objective روی instance‌های ۱۰۰۰-نودی بدهد. پیاده‌سازی نسخه‌ی ساده‌شده‌ی این ایده (حتی بدون GNN، فقط با یک heuristic arc-filtering مبتنی بر reduced-cost/فاصله) رزومه را از «برنامه‌نویس OR» به «محقق در تقاطع OR و ML» می‌برد — دقیقاً هدفی که فایل تحلیلی شما هم برای ریپوی قبلی مطرح کرده بود.
- **گسترش به Electric-VRPTW (EVRPTW)**: اضافه‌کردن ایستگاه شارژ و محدودیت باتری به مدل. این موضوع داغ‌ترین زیرشاخه‌ی فعلی VRP در صنعت لجستیک سبز است (چندین بنچمارک تازه در ۲۰۲۴-۲۰۲۶ منتشر شده مثل EVRP-TW-D و SynthCharge) و مستقیماً به دغدغه‌ی واقعی شرکت‌های حمل‌ونقل امروز مرتبط است.
- **داده‌ی واقعی به‌جای فاصله‌ی اقلیدسی**: جایگزینی ماتریس فاصله‌ی Solomon (که صرفاً فاصله‌ی خط‌مستقیم گرد‌شده است) با ماتریس فاصله/زمان واقعی از OSRM برای یک شهر واقعی (مثلاً تهران) — این دقیقاً چیزی‌ست که سالورهای صنعتی مثل ORTEC واقعاً روی آن کار می‌کنند (ترافیک، شبکه‌ی جاده‌ای واقعی) نه فاصله‌ی هندسی.
- **فرمت داده‌ی غنی‌تر شبیه چالش‌های ORTEC**: چندروزه (multi-day)، ناوگان ناهمگون (heterogeneous fleet)، استراحت راننده — این‌ها همان محدودیت‌هایی هستند که در VeRoLog Challenge خود ORTEC مطرح شده و این پروژه را از «تمرین دانشگاهی صرف» به «شبیه‌سازی واقعی صنعتی» تبدیل می‌کند.

### سطح ۴ — لایه‌ی محصول/دمو (برای نمایش در رزومه/گیت‌هاب)
- یک وب‌اپ سبک (Streamlit یا FastAPI + یک فرانت ساده) که کاربر instance را آپلود کند، الگوریتم زنده اجرا شود و مسیرهای بهینه روی نقشه real-time دیده شوند. برای اساتید و ریکروترها چیزی که «run» می‌شود و نتیجه‌ی بصری می‌دهد تأثیر بسیار بیشتری از کد خام دارد.

---

## ۴. جمع‌بندی و پیشنهاد فازبندی

| فاز | شامل | هدف |
|---|---|---|
| **فاز ۰ (Quick win)** | حذف `input()`، `requirements.txt`، پشتیبانی HiGHS رایگان، visualization پایه | پروژه را «قابل اجرا برای هرکسی بدون لایسنس» و «قابل نمایش» می‌کند |
| **فاز ۱ (علمی)** | Branch-and-Price کامل + ng-route pricing + benchmark در برابر PyVRP/VRPSolverEasy | همان چیزی که سطح آکادمیک را از «heuristic» به «exact method واقعی» می‌رساند |
| **فاز ۲ (تمایز/خفن)** | یکی یا هر دوی: EVRPTW، ML-guided arc pruning، داده‌ی واقعی OSRM | پروژه را از مشابه‌های زیاد روی گیت‌هاب متمایز می‌کند |
| **فاز ۳ (نمایش)** | دمو وب/نقشه‌ی زنده | برای معرفی در رزومه/LinkedIn/مصاحبه |

**وضعیت اجرا:** فاز ۰ و فاز ۱ کامل پیاده‌سازی شدند، به‌علاوه‌ی دو مورد از فاز ۲ (هدف دوفازی lexicographic و dual stabilization) — همگی با تست واحد و اوراکل brute-force مستقل اعتبارسنجی شده‌اند (جزئیات در [`docs/REPORT.md`](docs/REPORT.md)). باقی‌مانده‌ی فاز ۲ (EVRPTW، benchmark در برابر VRPSolverEasy/PyVRP)، فاز ۳ (ML-guided pricing)، و فاز ۴ (دمو وب) به‌عمد اجرا نشدند: هرکدام افورت و ریسک قابل‌توجهی دارند و پیاده‌سازی شتاب‌زده‌شان می‌توانست به قیمت صحت و کیفیت تست‌شده‌ی بقیه‌ی پروژه تمام شود. این‌ها future work مستندی هستند برای وقتی که واقعاً وقت گذاشته شود.

---

## منابع (Sources)

- [ORTEC – Route Optimization Software](https://ortec.com/products/apps-and-services/route-optimization)
- [ORTEC GitHub organization](https://github.com/ortec) (شامل fork رسمی PyVRP، quickstart کد EURO-NeurIPS 2022، آرشیو VeRoLog 2017/2019)
- [EURO Meets NeurIPS 2022 Vehicle Routing Competition](https://euro-neurips-vrp-2022.challenges.ortec.com/)
- [The VeRoLog Solver Challenge 2019](https://link.springer.com/article/10.1007/s41604-019-00011-8)
- [PyVRP: A High-Performance VRP Solver Package (INFORMS J. on Computing, 2024)](https://pubsonline.informs.org/doi/10.1287/ijoc.2023.0055)
- [VRPSolverEasy — Python interface for VRPSolver/BaPCod](https://github.com/inria-UFF/VRPSolverEasy)
- [A Bucket Graph Based Labeling Algorithm with Application to Vehicle Routing (Sadykov, Uchoa, Pessoa — Transportation Science, 2021)](https://pubsonline.informs.org/doi/10.1287/trsc.2020.0985)
- [Partial Column Generation with Graph Neural Networks (Koutecká et al., 2025)](https://www.arxiv.org/pdf/2509.15275)
- [Graph Reduction with Unsupervised Learning in Column Generation (Gerbaux et al., 2025)](https://arxiv.org/abs/2504.08401)
- [RouteOpt: An Open-Source Modular Exact Solver for VRPs (2025)](https://www.researchgate.net/publication/392873341_RouteOpt_An_Open-Source_Modular_Exact_Solver_for_Vehicle_Routing_Problems)
- [Electric Vehicle Routing with Time Windows and Heterogeneous Charging-Station Attribute Dataset](https://doi.org/10.3390/data11040083)
- [Google OR-Tools — VRPTW](https://developers.google.com/optimization/routing/vrptw)
