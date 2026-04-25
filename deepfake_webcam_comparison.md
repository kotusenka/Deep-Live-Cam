# Сравнение open-source real-time face-swap проектов (на 2026-04-25)

Все цифры по звёздам / коммитам — на 25 апреля 2026, проверены через GitHub API агентами.

## TL;DR

- **Deep-Live-Cam** — самый раскрученный (92k звёзд), активно развивается, но качество среднее и **встроенной виртуальной камеры у него нет** — все вирусные демки идут через OBS Virtual Camera. Это главная неочевидная вещь.
- **Roop** — родитель всех. **Архивирован** автором ("разочаровался во втором порядке последствий"). Всё, что ниже — потомки.
- **Roop-Unleashed** — **снесён GitHub'ом 22 января 2025** за нарушение AUP по synthetic media. Живёт на Codeberg как `Cognibuild/ROOP-FLOYD`, маленькое комьюнити.
- **Facefusion** — самый зрелый и активный мейнтримный (релизы ежемесячно, последний 3.6.1 от 19.04.2026), 11 процессоров в пайплайне, кросс-платформенный. Лучший выбор по совокупности.
- **DeepFaceLive** — **архивирован iperov'ом 13 ноября 2024**. Но его DFM-модели (тренируются под конкретного человека часами/днями) до сих пор дают **самое высокое качество в live-режиме**, если ты готов их тренировать. Только Windows + NVIDIA.
- **Rope** — фактически дормант с мая 2024, только оффлайн-видео.
- **VisoMaster** — духовный наследник Rope от тех же разработчиков (argenspin + Alucard24), сделанный в начале 2025. Сейчас это **SOTA open-source GUI** для лайва: TensorRT, DFM, LivePortrait, virtual camera. Windows+NVIDIA only, на RTX 50-й серии багует.

**Для live-вебки в 2026 реально стоит выбирать между**: VisoMaster (макс. качество если есть NVIDIA), Facefusion (универсальный), Deep-Live-Cam (если Mac или нужно "поставил-запустил"), DeepFaceLive (если готов тренировать DFM на конкретное лицо).

---

## Подробно по каждому проекту

### 1. Deep-Live-Cam ([github.com/hacksider/Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam))

| Параметр | Значение |
|---|---|
| Звёзды / форки | 92 288 / 13 402 |
| Лицензия | AGPL-3.0 |
| Последний коммит | 2026-04-23 (активен) |
| Модель свопа | `inswapper_128` / `inswapper_128_fp16` |
| Энхансеры | GFPGAN 1.4, GPEN-256/512 (с 2.7-beta) |
| Акселераторы | CUDA, CoreML, DirectML, ROCm |
| **Virtual camera** | **Нет в коде!** Через OBS как screen capture |
| ОС | Windows / macOS (лучший Mac-стори) / Linux |
| NSFW filter | Есть, но **opt-in** (`--nsfw-filter`), не by default |

**Почему такой хайп.** Вирусанул в августе 2024 (PetaPixel, Slashdot, Ars Technica), мем "стань Маском за один клик". Потом несколько раз попадал в топ trending GitHub. Часто упоминается в репортажах про KYC-фрод (Veriff, Trend Micro, Reality Defender) рядом с DeepFaceLive и Magicam.

**Что внутри по факту.** Тонкая обёртка вокруг того же `inswapper_128`, что и у Roop. Уникальные фичи: **Mouth Mask** (сохраняет настоящий рот таргета под свапнутым лицом, чтобы убрать рассинхрон) и **Face Mapping** (мульти-лицо с разными источниками). 2.7-beta добавила Realtime Face Enhancer, GPU Changer, Resolution Changer.

**Главный подвох — нет встроенной виртуальной камеры.** README прямо говорит: *"Use a screen capture tool like OBS to stream"*. PR #329 в августе 2024 добавлял `pyvirtualcam`, но был удалён. PR #1666 в марте 2026 пытался вернуть — закрыт без мержа 19.04.2026, при том что родительский issue #1662 помечен "completed". `pyvirtualcam` не в `requirements.txt`. Workflow остался: окно DLC → window capture в OBS → OBS Virtual Camera → Zoom/Teams.

**FPS (анекдотические данные из issues, не бенчмарки).**
- RTX 4070+: 24-30 fps swap-only, ~12-15 с GFPGAN.
- RTX 3060: 15-20 fps swap-only.
- RTX 50-я серия: ломается, [issue #1303](https://github.com/hacksider/Deep-Live-Cam/issues/1303).
- M3/M4 Pro: жалобы на **0.1-1.5 fps** в [#1289](https://github.com/hacksider/Deep-Live-Cam/issues/1289), GFPGAN падает на CPU. Свежий CoreML-фикс (commit 890a6d4 от 22.04.2026) дал ~12 fps только GFPGAN на M-series.
- Apple Silicon после PR #295: face-swap-only ~20+ fps на M3 Max, ~10 с GFPGAN.

**Топ багов из трекера:**
1. CUDA не загружается на RTX 5070 Ti ([#1303](https://github.com/hacksider/Deep-Live-Cam/issues/1303))
2. Камера не инициализируется на macOS 15.4+ из-за Continuity Camera ([#1238](https://github.com/hacksider/Deep-Live-Cam/issues/1238))
3. Apple Silicon GPU underutilization ([#1289](https://github.com/hacksider/Deep-Live-Cam/issues/1289), [#1273](https://github.com/hacksider/Deep-Live-Cam/issues/1273), [#1190](https://github.com/hacksider/Deep-Live-Cam/issues/1190))
4. `onnxruntime-silicon==1.13.1` ломается на Python 3.12+ ([#1160](https://github.com/hacksider/Deep-Live-Cam/issues/1160), [#1564](https://github.com/hacksider/Deep-Live-Cam/issues/1564))
5. Convex hull crash в face-mask ([#1593](https://github.com/hacksider/Deep-Live-Cam/issues/1593))
6. App падает на "Go Live" / preview ([#1690](https://github.com/hacksider/Deep-Live-Cam/issues/1690))
7. Чёрная рамка вокруг свапнутого лица ([#1781](https://github.com/hacksider/Deep-Live-Cam/issues/1781))
8. `insightface` wheel build fail на новых Python ([#1148](https://github.com/hacksider/Deep-Live-Cam/issues/1148))

**Установка** — болезненно для не-инженеров. Нужен Python 3.11 ровно, ровные версии CUDA/cuDNN, ручная скачка моделей с HuggingFace. Авторы продают платный установщик через `deeplivecam.net` ("30+ extra features") — это часть бизнес-модели.

---

### 2. Roop ([github.com/s0md3v/roop](https://github.com/s0md3v/roop))

| Параметр | Значение |
|---|---|
| Звёзды / форки | 30 491 / 6 898 |
| Лицензия | GPL-3.0 |
| Последний релиз | 1.3.2 от 2023-08-03 |
| **Статус** | **Permanently discontinued** (архивирован) |

Создан s0md3v (известный security-исследователь — XSStrike, Photon). Сделал `inswapper_128` стандартом де-факто. Архивирован в октябре 2023, в марте 2026 добавлен финальный disclaimer о "втором порядке последствий технологии". Issues отключены.

**Триггер заброса**: контрибьютор с write-доступом запушил "проблемное видео" в документацию пока s0md3v был в отпуске летом 2023 ([HN #37274331](https://news.ycombinator.com/item?id=37274331)). Точное содержимое спорное в треде.

Только batch-обработка изображений/видео — никакой webcam-функциональности. NSFW-фильтр включён по умолчанию (в отличие от Deep-Live-Cam). Технический потолок — то же качество, что и потомки. Все, кому нужна была лайв-вебка, ушли в форки.

---

### 3. Roop-Unleashed (`C0untFloyd/roop-unleashed`)

| Параметр | Значение |
|---|---|
| Статус GitHub | **DISABLED 22.01.2025** (HTTP 403, "tos") |
| Активный преемник | [`Cognibuild/ROOP-FLOYD`](https://codeberg.org/Cognibuild/ROOP-FLOYD) на Codeberg |
| Лицензия (была) | AGPL-3.0 |
| Последний релиз | 4.4.1 (январь 2025) |

**Самое громкое enforcement-событие в этой нише.** GitHub снёс репо без DMCA, без предупреждения — со ссылкой на обновлённый AUP по synthetic media. Это объясняет, почему Deep-Live-Cam держит NSFW-фильтр в коде хоть и opt-in: хеджируют риск.

Сейчас живёт на Codeberg, основной мейнтейн — `Cognibuild/ROOP-FLOYD` (46 звёзд, 32 форка). Распространение в основном через Patreon (SECourses).

**Почему он был интереснее Deep-Live-Cam технически:**
- Реальная **встроенная виртуальная камера** — `pyvirtualcam` в зависимостях, маршрутит в OBS Virtual Camera (Win/Mac), v4l2loopback (Linux), unitycapture (Win).
- **ReSwapper-128/256** в дополнение к inswapper_128 — единственный из тройки с >128px свапом.
- 4 энхансера: GFPGAN, CodeFormer, RestoreFormer++, DMDNet.
- **Текстовая окклюзионная маска** через `mask_clip2seg` — единственный, который реально умеет защищать руки/очки/микрофон перед лицом.
- Gradio-веб-UI вместо tkinter — работает headless, по сети, в Colab.
- Один-кликовый installer (`windows_run.bat`).
- Никакого NSFW-фильтра (часть причины бана).

Топ багов в Codeberg-преемнике: `IndexError: list index out of range` в детекции, `Failed to build insightface`, отсутствие выбора DirectML provider у AMD, Live Cam ломается после апдейтов зависимостей.

---

### 4. Facefusion ([github.com/facefusion/facefusion](https://github.com/facefusion/facefusion))

| Параметр | Значение |
|---|---|
| Звёзды / форки | 28 042 / 4 547 |
| Лицензия | **OpenRAIL-AS** (responsible-AI лицензия с use-restrictions) |
| Последний релиз | **3.6.1 от 19.04.2026** (релизы каждый месяц) |
| Issues tab | Отключён, поддержка через [Discord (~43k)](https://docs.facefusion.io/) |

**Самый зрелый и универсальный.** Не одна модель, а целый набор процессоров (11 штук, проверено агентом по `facefusion/processors/modules/`):

`age_modifier, background_remover, deep_swapper, expression_restorer, face_debugger, face_editor, face_enhancer, face_swapper, frame_colorizer, frame_enhancer, lip_syncer`

**Модели свопа:** inswapper_128_fp16 (legacy), **hyperswap_1a/1b/1c_256** (новый дефолт, собственная модель FaceFusion, ResearchRAIL-лицензия), simswap_256/512, ghost, blendswap_256, uniface_256. **`deep_swapper`** грузит DFM-модели DeepFaceLab — то есть здесь они тоже работают.

**Энхансеры:** face — GFPGAN 1.4 / CodeFormer / GPEN / RestoreFormer; frame — SPAN, Real-ESRGAN, Ultra Sharp + ~13 апскейлеров.

**Акселераторы — самая широкая матрица в нише:** CPU, CUDA 12.9, **TensorRT 10.x**, **CoreML**, **DirectML**, **ROCm**, **OpenVINO 2025.3**, **MIGraphX**, экспериментально **QNN**.

**Live webcam.** Запуск: `python facefusion.py run --ui-layouts webcam`. Выходы: inline preview, **UDP-стрим на `udp://localhost:27000`** (OBS подхватывает как Media Source), V4L2 `/dev/video*` на Linux. **Нативного virtual cam драйвера на Win/Mac нет** — мост через OBS. Доки заявляют "solid 25 fps at 1080p"; комьюнити подтверждает ~30 fps на 720p у RTX 3060+ с hyperswap_256, меньше с face_enhancer.

**Топ багов (из Discord и changelog):**
1. Norton/Avast ломают модель-даунлоады (форсированный `--ssl-no-revoke` в 3.6.1)
2. Conda env PATH-конфликты (переделывали в 3.6.0)
3. OOM на VRAM при стэке процессоров выше 720p
4. Protobuf parse errors от обрывов скачивания
5. **NSFW-фильтр + анти-тампер**: 3.5.4 добавил `git pull` `core.py` и `content_analyser.py` на старте — обвинения в anti-tampering от пользователей-форкеров
6. **License confusion**: лицензия проекта permissive, но веса (inswapper, hyperswap, simswap, gpen, retinaface, scrfd) — non-commercial / research-only. Юридический винегрет.
7. TensorRT engine rebuilds после каждого апдейта драйвера

**Установка:** Pinokio one-click, официальные установщики Windows/macOS, conda env с Python 3.12. Без Docker.

**Железо:** мин 8 GB VRAM, 12+ рекомендуется для стэка процессоров.

**Этика:** автор Henry Ruhs занимает публичную no-NSFW позицию, в доках прямо "we will report illegal use". Параллельно есть Patreon-форки с "uncensored" модами.

---

### 5. DeepFaceLive ([github.com/iperov/DeepFaceLive](https://github.com/iperov/DeepFaceLive))

| Параметр | Значение |
|---|---|
| Звёзды / форки | 30 772 / 1 212 |
| Лицензия | GPL-3.0 |
| **Статус** | **Архивирован 13.11.2024** |
| Последний коммит кода | 2023-07-28 |

**Создан iperov, автором DeepFaceLab.** Был эталоном live-deepfake'а 2021-2023. Iperov ушёл в тишину по всем своим проектам (DeepFaceLab, DeepFaceLive, MachineVideoEditor) — никакого преемника не назначил. В трекере осталось ровно 1 открытый issue: [#41 "Stop developing this technology"](https://github.com/iperov/DeepFaceLive/issues/41), 507 апвоутов, открыт с 2022 года.

**Главная техническая идея — DFM-модели.** В отличие от one-shot подходов (inswapper берёт одно фото и переносит лицо), DFM = DeepFaceLive Model — это персонально обученная модель на конкретного человека, экспортированная из DeepFaceLab SAEHD/AMP. Тренировка: ~1 день на RTX 3090 на pretrain'е, days-to-weeks с нуля. Итог — **identity locked**, стабильно держит идентичность через повороты, мимику, освещение. Качество принципиально выше one-shot.

Плюс с июля 2023 (последняя реальная фича) добавили **Insight (inswapper_128)** — one-shot путь для тех, у кого нет тренированной DFM.

**Ecosystem DFM:** ~30 официальных публичных моделей (Keanu, Margot Robbie, Jackie Chan, Mr. Bean...) + тысячи комьюнити-моделей на [HuggingFace](https://huggingface.co/datasets/dimanchkek/Deepfacelive-DFM-Models), [DeepfakeVFX](https://www.deepfakevfx.com/deepfacelive-models-dfm/), Discord-дропах. **Этот каталог — главное наследие проекта**, его до сих пор используют через Facefusion (`deep_swapper`) и VisoMaster.

**Virtual camera — нативная, в самом приложении.** Stream output модуль. Это до сих пор самый чистый прямой путь к Zoom/Discord. Сабж старый, но работает.

**Топ багов (всё уже не починят):**
1. DLL load failed / kernel32 на Windows ([#28](https://github.com/iperov/DeepFaceLive/issues/28), [#75](https://github.com/iperov/DeepFaceLive/issues/75), [#120](https://github.com/iperov/DeepFaceLive/issues/120))
2. Mega.nz таймауты на скачку дистрибутива
3. **OBS 28+ сломал handoff виртуалки**, который работал на OBS 27 ([#95](https://github.com/iperov/DeepFaceLive/issues/95), [#151](https://github.com/iperov/DeepFaceLive/issues/151))
4. AMD DirectX12 Face Merger чёрные точки — fallback на CPU
5. `cudaGetDeviceCount() failed` на новых драйверах — встроенные CUDA-либы застряли в 2022 году, страдают RTX 4xxx/5xxx
6. Docker под Linux перманентно сломан

**Железо:** мин AVX CPU + DX12 GPU + 4 GB RAM + 32 GB pagefile. Для live комфортно — RTX 2070+ / RX 5700 XT+. Win-only официально.

**Notoriety.** Регулярно упоминается в репортажах про deepfake-фрод (Hong Kong $25M, Binance impersonation, Veriff/Sumsub/Bloomberg). Прямо в суд не попадал, но репутационная нагрузка почти точно повлияла на архивацию.

---

### 6. Rope ([github.com/Hillobar/Rope](https://github.com/Hillobar/Rope))

| Параметр | Значение |
|---|---|
| Звёзды / форки | ~5 300 / 953 |
| Лицензия | GPL-3.0 |
| Последний релиз | Rope-Pearl-00, **27 мая 2024** |
| Статус | **Дормант** (~2 года), не архивирован формально |

Только `inswapper_128`, селектор разрешения 128/256/512 (256/512 — это апсемплы 128, не настоящий 512). Энхансеры GFPGAN/CodeFormer. CUDA only.

**Без webcam** в мейнлайне. Только batch-видео.

**Был** легендой энтузиастов 2023-2024 за гранулярность параметров: отдельные occluder, mouth parser, CLIP-text-mask, restorer blend, gamma, mean/median merge, embed averaging, multi-pass swap strength. На RTX 3090 Ti — 0.4-2 fps на 1080p с restorer. Максимальное качество в офлайне для своей эпохи, но всё забытое.

В мейнлайне сидит 9+ открытых PR, никто не мержит. Hillobar засветился последний раз в середине 2024 года в X (тизерил inswapper-256 вариант) и пропал.

---

### 7. VisoMaster ([github.com/visomaster/VisoMaster](https://github.com/visomaster/VisoMaster))

| Параметр | Значение |
|---|---|
| Звёзды / форки | ~1 800 / 313 |
| Лицензия | GPL-3.0 |
| Создан | 2025-01-27 |
| Последний релиз | v0.1.6, март 2025 (нужно перепроверить, не появилось ли нового с тех пор) |

**Прямой наследник Rope** от тех же контрибьюторов: **argenspin** (автор Rope-Live, который добавил вебку и virtual cam) и **Alucard24** (автор Alucard24/Rope с TensorRT и LivePortrait). Hillobar не участвует. В коде есть `tools/convert_old_rope_embeddings.py` — миграция эмбеддингов со старого Rope. Третьи стороны (SECourses, YouTube-туториалы) прямо называют VisoMaster "переименованный Rope-Live".

**Что внутри:**
- Свопы: inswapper_128, inswapper_128_fp16, **DFM** (формат DeepFaceLive — это даёт качество DFL прямо здесь)
- Селектор разрешения 128/256/384/512
- **TensorRT first-class** — собирает engines в `tensorrt-engines/` при первом запуске
- **LivePortrait** как Face Editor для ретаргетинга мимики/позы
- **Expression Restorer** — свапнутое лицо наследует микро-мимику оригинала (бьёт frame-flicker)
- Энхансеры: GFPGAN, CodeFormer, GPEN-1024/2048, RestoreFormer++, VQFR v2
- XSeg маска + face parser + occluder — лучшая окклюзия среди open-source
- Multi-source identity averaging
- Video Markers для per-frame параметров
- **Live webcam + Virtual camera output** для OBS/Zoom/Teams/Twitch
- CUDA 11.8 / 12.4.1, **только NVIDIA**, Windows-таргет (Linux unofficial, Mac не работает)

**Это сейчас open-source SOTA для live-вебки на NVIDIA.** Все, кто сравнивал в 2025-2026 (SECourses, профильные YouTube-каналы), ставят его выше Facefusion и Deep-Live-Cam по чистому качеству свопа, при том что он умеет ещё и live + virtual cam.

**Топ багов (issue tracker очень активный):**
1. [#2 "Does it work on linux or mac?"](https://github.com/visomaster/VisoMaster/issues/2) — 38 комментов, главный мета-issue про платформы
2. [#95 RTX 50-series sm_120 несовместимость](https://github.com/visomaster/VisoMaster/issues/95) — встроенный PyTorch 2.4.1+cu124 не имеет sm_120 ядер, RTX 5070/5080/5090 не запускается
3. [#130 RTX 5060 16GB не работает](https://github.com/visomaster/VisoMaster/issues/130)
4. [#78 v0.1.6 BSOD'ы](https://github.com/visomaster/VisoMaster/issues/78) — TensorRT engine build instability
5. [#30 ONNX Runtime CUDA EP issue](https://github.com/visomaster/VisoMaster/issues/30) с конкретными версиями cuDNN
6. [#46 TensorRT crashes на Arch Linux](https://github.com/visomaster/VisoMaster/issues/46)
7. `dependencies_cu124.7z` периодически 404
8. cuDNN 9.1 vs 9.4 mismatch с TensorRT 10.6
9. [#144 Mouth not moving smoothly](https://github.com/visomaster/VisoMaster/issues/144) в live-режиме
10. Слабая обработка профильных углов (общая болезнь inswapper)

**Дисклеймер по статусу:** активные релизы были фев-март 2025 (v0.1.1 → v0.1.6). Между мартом 2025 и сейчас (апрель 2026) агент не смог надёжно подтвердить свежие коммиты в main — возможно, активность ушла в форки `VisoMasterFusion/VisoMaster-Fusion` и SECourses' "Ultimate Mod" ([issue #96](https://github.com/visomaster/VisoMaster/issues/96)). Стоит проверить текущее состояние commit log перед использованием.

---

### Краткий обзор остальных

- **InsightFace / inswapper** — апстрим. Веса `inswapper_128.onnx` имеют двусмысленный статус редистрибуции (формально только по запросу). В 2025 InsightFace выкатили **`inswapper_512_live`** ([github.com/deepinsight/inswapper-512-live](https://github.com/deepinsight/inswapper-512-live)) и собственные iOS (август 2025) и macOS (ноябрь 2025) приложения. **Веса 512-live — commercial-only**, поэтому в open-source GUI не интегрируется. Это главный апстрим-эвент 2025-го и причина, почему "настоящего 512" в open-source ещё нет.
- **SimSwap** — старая (2020) академическая база, референс качества. Не для live.
- **ReActor** — нода для ComfyUI/Stable Diffusion WebUI, обёртка над inswapper_128 + hyperswap. Не таргетит live-вебку.
- **HyperSwap** (FaceFusion Labs) — новая модель 2024-2025, используется в FaceFusion 3.x и ReActor. Альтернатива inswapper'у с лучшим identity preservation.
- **LivePortrait** (KwaiVGI/Kuaishou, июль 2024) — не свопер, а ретаргетинг мимики на статичный портрет. Real-time на RTX 4090. **Уже интегрирован в VisoMaster** как Face Editor / Expression Restorer. MIT-style лицензия, лучший вклад от китайских big-tech лабораторий.
- **InstantID / PhotoMaker / IP-Adapter-FaceID / PuLID** — diffusion-based identity preservation (SD/SDXL/Flux). Это **другая ветка**: они **генерируют** изображения с нужной идентичностью, а не делают пиксельный своп. Слишком медленно для live (секунды на кадр даже на 4090). Для live-вебки нерелевантно.

---

## Виртуальная камера: как они вообще "мимикрируют" под вебку

Самое важное место, где проекты различаются на практике. Архитектура у всех общая:

```
физическая камера → захват кадра → инференс свопа → виртуальный драйвер камеры → Zoom/Teams читают как обычную вебку
```

Боль — в последнем шаге, и она зависит от ОС.

### Windows — единственная по-настоящему здоровая платформа

Стандарт де-факто — **OBS Virtual Camera** (встроена в OBS Studio 26.0+) или **pyvirtualcam** (под Windows тоже использует OBS-драйвер). DeepFaceLive имеет свой DirectShow virtual cam встроенный.

**Главный gotcha:** OBS Virtual Camera — **DirectShow only**. Media Foundation API для регистрации виртуалок появился только в Windows 11. Практический эффект: всё на DirectShow (Zoom desktop, Teams classic, Discord, Chrome/Meet) — работает; некоторые UWP-приложения и Teams "new" на WebView2 — могут не видеть. Это объясняет периодические "у меня не работает" в issues всех проектов.

**Single-instance constraint**: один процесс владеет OBS Virtual Camera. Чтобы скомбинировать с OBS-оверлеями — нужна вторая виртуалка типа Unity Capture, или цепочка DLC → OBS source → OBS Virtual Camera.

Альтернативы: **e2eSoft VCam** (платная, выручает когда OBS не работает у конкретного приложения), ManyCam, SplitCam.

### macOS — самая болезненная история

Apple за последние 5 лет три раза переделал модель виртуалок: kext → CoreMediaIO DAL → System Extension (Catalina/Big Sur) → ужесточили в Sonoma (14) → дальше в Sequoia (15).

**Реальность апреля 2026:**
- **Только OBS 30.0+** работает. OBS 29.1 и ранее — полностью сломаны на Sonoma/Sequoia.
- Нужно вручную включить расширение в **System Settings → General → Login Items & Extensions → Camera Extensions**, тогл OBS, ребут. Это частая боль в issues Mac-пользователей DLC.
- Sequoia добавил еженедельный re-prompt на разрешение камеры — не ломает, но шумит.
- **CamTwist мёртв** для современных macOS. Open-source альтернатив активно поддерживаемых нет.
- **DeepFaceLive, Rope, VisoMaster — НЕ работают на Mac вообще.** Мак-юзеры заперты на Deep-Live-Cam и нескольких форках типа iRoopDeepFaceCam, MacFaceSwap.

### Linux

**v4l2loopback** — единственный путь, зрелый. `modprobe v4l2loopback`, кадры в `/dev/video10`. Zoom Linux client, Discord (с `--enable-features=WebRTCPipeWireCapturer`), Chrome-based Meet — видят. Главная заминка: snap/flatpak версии Zoom иногда не видят `/dev/video*` из-за confinement. DeepFaceLive под Linux официально не работает.

### Latency budget

Сложение типичных задержек: capture (5-15 мс USB) + face-detect/align (10-30 мс) + inswapper-128 (8-20 мс на RTX 3060-4090; 30-60 мс на M-series; ~100 мс на iGPU) + GFPGAN/CodeFormer (15-40 мс — самый дорогой шаг) + virtual cam copy (1-5 мс) + Zoom encode (~30-60 мс).

**Итого ~90-200 мс на хорошем GPU**, **300-500 мс с энхансером на mid-range карте**. Всё, что выше ~250 мс — заметно как "слегка опаздывающие губы" на стороне приёмника, и это один из самых явных tells.

DeepFaceLive целит 25 fps = 40 мс на кадр; Deep-Live-Cam — 60 fps = ~16 мс внутренне.

---

## "Палево" — по чему ловят дипфейк-вебку

**Высокая уверенность** (подтверждено академическими работами):

- **Граничные швы по овалу лица.** Inswapper и почти все one-shot модели делают alpha-feathered переход в районе челюсти. Видно как чуть размытую "маску" по контуру, особенно при поворотах. Главная мишень всех детекторов 2024-2025 годов.
- **Линия волос.** Inswapper'овская маска не включает волосы. Когда прическа источника и таргета не совпадают — частичное наложение волос источника или несоответствующая чёлка. Документировано в [DLC issue #1418](https://github.com/hacksider/Deep-Live-Cam/issues/1418).
- **Профиль / поворот головы >45-60°.** inswapper_128 тренировался в основном на фронталях. На сильных поворотах identity transfer ломается, лицо "schlap"-ом возвращается во фронталь, иногда мелькает оригинальное лицо. **Самый надёжный ручной тест**: попроси подозрительного собеседника повернуть голову в профиль.
- **Окклюзия рукой.** One-shot модели не умеют динамическую окклюзию. Рука перед лицом — размывается, "исчезает за лицом", или лицо проступает сквозь руку. Только Roop-Unleashed с `mask_clip2seg` и VisoMaster с XSeg более-менее справляются.
- **Lip-sync drift на согласных.** [LipFD (NeurIPS 2024)](https://github.com/AaronComo/LipFD) и [AVSFF](https://link.springer.com/article/10.1007/s44196-025-00911-7) ловят >95% точности на рассинхроне губ с фонемами (особенно плозивные /p/, /b/, /m/). Real-time свопы варпят рот таргета на движения говорящего, варп неидеальный — миллисекундный drift детектируется и ушами тоже.
- **Несовпадение освещения.** Свапнутое лицо с подразумеваемым светом из исходного фото на телесную часть с другим освещением — тень на подбородке не совпадает с тенью на шее.
- **Очки, серьги, борода.** Серьги клипаются, оправа очков мерцает, борода флипает кадр-к-кадру (каждый кадр сегментируется независимо).
- **Идентификационный дрейф ("плавающее лицо").** Per-frame свопы без temporal anchoring (Roop, ванильный DLC, ванильный Facefusion) — лицо "дышит" или мерцает. Тулзы с identity averaging (DeepFaceLive DFM, VisoMaster Identity Anchor) заметно стабильнее.
- **Frame-to-frame flicker.** Дрожание границы маски, цветовое мерцание скул, искры на бровях. Универсальная болезнь frame-independent пайплайнов.
- **Сдвиг тона кожи / WB.** Свап подкрашивается под таргет, шея и уши — нет. На warm tungsten или blue daylight — несовпадение очевидно.

**Средняя уверенность:**
- **Аномалии моргания** — старые DeepFakes-автоэнкодеры моргали редко. inswapper_128 наследует моргания таргета прилично, поэтому это менее надёжно, чем принято считать. Но статистически детекторы всё ещё ловят.
- **Halo от GFPGAN/CodeFormer** — энхансер работает на crop'е и имеет резкую границу re-blend. На zoom-in видно "фарфоровость" кожи и мягкий halo.

---

## Состояние детекции (важно знать)

**Платформы Zoom/Teams/Meet нативной детекции в 2026 не имеют.** Стратегия — интегрировать сторонние решения:

- **Pindrop Pulse for Meetings** — GA на Zoom App Marketplace с **ноября 2025**, теперь и в Webex/Teams. TIME's Best Inventions 2025. Заявляют до 99% accuracy на синтетическом аудио, <1% FP rate. Топ-1 коммерческое в ACM MM Deepfake Detection Challenge 2025.
- **Reality Defender (RealityCheck)** — Zoom-интеграция, заявленный 91% accuracy на free tier.
- **Truly, Beyond Identity** — другой подход: вместо детекции бьёт по device fingerprint / signed attestation. Даже идеальный дипфейк фейлится на проверке устройства.

**KYC/банки:** Sumsub, Microblink, Sardine, Facia продают injection-attack detection (IAD). **CEN/TS 18099** (европейский стандарт) и грядущий **ISO 25456** формализуют требования. Deepfake injection attacks выросли на **783% YoY** (kyc-chain). Out-of-the-box face-swap из этих репо уже регулярно проходит single-factor liveness.

**Open-source детекторы:**
- [Deepfake-o-Meter](https://tattle.co.in/blog/2025-03-12-deepfake-o-meter/) (Univ of Buffalo) — 18 моделей в одном
- [DeepSafe](https://github.com/siddharthksah/DeepSafe) — ансамбль
- [LipFD](https://github.com/AaronComo/LipFD) — топовый по lip-sync
- `prithivMLmods/deepfake-detector-model-v1` на HF — SigLIP image classifier

**Реальные кейсы фрода:**
- **Arup, Гонконг, январь 2024 — $25.6M.** Сотрудник финдепа сделал 15 переводов на HK$200M после видеоконференции, где CFO и коллеги все были дипфейками из публичных корпоративных записей. Канонический пример.
- **WPP CEO** (середина 2024) — voice clone в Teams, попытка отбита.
- **Сингапур, март 2025 — $499K.** Финдиректор multinational авторизовал перевод после "Zoom" с дипфейкнутым руководством.
- Q4 2025: Gen Threat Labs зафиксировал 159 378 уникальных дипфейк-скам инстансов, ~700% YoY surge у ScamWatch HQ, 30% impersonation incidents у Cyble.

---

## Рейтинг по качеству для **live webcam** (ключевое — именно live, не оффлайн)

**S-tier:**
- **DeepFaceLive с тренированной DFM** (224×224+). DFM специально обучена на конкретного человека — identity locked, держит повороты, мимику, освещение. Цена: часы-дни тренировки. **Минус: репо архивирован**, новых моделей не будет. Но существующий каталог DFM огромный.

**A-tier (актуальные):**
- **VisoMaster** — Rope-качество + DFM-поддержка + LivePortrait + TensorRT + virtual cam. Лучший open GUI 2025-2026 если у тебя NVIDIA.
- **Facefusion 3.x** — broader matrix, hyperswap_256, кросс-платформенный. Чуть менее тонко настраиваемый, чем VisoMaster, но универсальнее.

**B-tier:**
- **Deep-Live-Cam** — Mac-friendly, viral, но базовый inswapper_128 + GFPGAN, очевидные tells на профиле/окклюзии. И **нет встроенной virtual cam — через OBS bridge**.
- **Roop-Unleashed/Roop-Floyd** — единственный с реально нативным virtual cam в этом тире, ReSwapper-256, окклюзионная маска. Но снят с GitHub, маленькое комьюнити.

**C-tier:**
- **Roop** — мёртв, бесполезен для live.

**Для оффлайн** ranking сдвигается: **DeepFaceLab > VisoMaster ≈ Facefusion ≈ Rope > live-tools**, потому что оффлайн можно тренировать на 384-512 и итерировать.

---

## Железо: что реально потянет

| Карта | Качество live | Замечания |
|---|---|---|
| **RTX 4090 (24 GB)** | Любой пайплайн, 60 fps+, 512-live влезает | Без потолка |
| **RTX 4070 / 4070 Ti (12 GB)** | 30 fps стабильно с энхансером | Практический пол "убедительной" вебки на Win |
| **RTX 3060 (12 GB)** | DeepFaceLive 25-30 fps на 224 DFM без энхансера, 15-20 с GFPGAN | Sweet spot цена/качество |
| **RTX 2070 / 3050 / 3060 8GB** | Маргинально, 480p или без энхансера | Некоторые DFM не лезут в VRAM |
| **Apple Silicon M3 Max / M4** | Только Deep-Live-Cam: 20+ fps swap-only, ~10 с GFPGAN | DeepFaceLive/VisoMaster/Rope не работают |
| **M1 / M2 base** | 5-10 fps без энхансера | Демо-уровень, не для разговоров |
| **Integrated Intel Iris / AMD Vega** | <5 fps, фризы | Не вариант |

**RTX 50-я серия (sm_120)** — сейчас зона риска у всех проектов. У VisoMaster главный класс багов, у DLC ([#1303](https://github.com/hacksider/Deep-Live-Cam/issues/1303)) тоже, у DeepFaceLive — дохлый CUDA stack. Facefusion починил быстрее всех.

GFPGAN/CodeFormer — самое узкое место. Везде ниже RTX 3060 / M3 ты выбираешь между размытым 128px (без энхансера) и тормозами (с энхансером).

---

## Сводная таблица

| | Deep-Live-Cam | Roop | Roop-Unleashed | Facefusion | DeepFaceLive | Rope | VisoMaster |
|---|---|---|---|---|---|---|---|
| Звёзды | 92.3k | 30.5k | DISABLED | 28.0k | 30.8k | 5.3k | 1.8k |
| Лицензия | AGPL-3.0 | GPL-3.0 | AGPL-3.0 | OpenRAIL-AS | GPL-3.0 | GPL-3.0 | GPL-3.0 |
| Статус | Активен | **Архив** | **Снят GitHub'ом** | Активен | **Архив** | Дормант | Релизы 2025, тек. неясно |
| Последний коммит | 2026-04-23 | 2026-03-13 (cleanup) | 2025-04-24 (mirror) | 2026-04-22 | 2023-07-28 | 2024-05 | 2025-03 |
| Real-time webcam | Да | Нет | Да | Да | **Да (нативно)** | Нет | **Да** |
| **Встроенный virtual cam** | **Нет**, через OBS | — | **Да** (pyvirtualcam) | Нет, UDP→OBS | **Да** (нативно) | — | **Да** |
| One-shot своп | inswapper_128 | inswapper_128 | inswapper_128 + ReSwapper-256 | hyperswap_256 + 6 других | inswapper_128 | inswapper_128 | inswapper + DFM |
| Per-identity модели | — | — | — | DFM через deep_swapper | **DFM (нативно)** | — | DFM |
| Энхансеры | GFPGAN, GPEN | GFPGAN | GFPGAN/CF/RestoreFormer++/DMDNet | GFPGAN/CF/GPEN/RF + 13 frame upscalers | — | GFPGAN/CF | GFPGAN/CF/GPEN-1024-2048/RF++/VQFR2 |
| Окклюзия (руки) | Mouth mask only | — | **clip2seg text-prompt** | mouth mask, region tools | DFM-обученная | Occluder + parser | **XSeg + parser + occluder** |
| TensorRT | Нет | Нет | Нет | Да | Нет | Нет | **Да (first-class)** |
| Win | Да | Да | Да | Да | Да | Да | Да |
| Mac | Да (лучший) | Да | Через bat | Да (CoreML) | **Нет** | **Нет** | **Нет** |
| Linux | Да | Да | Да | Да | Через WSL/хак | Через хак | Unofficial |
| Min VRAM | 4 GB | 4 GB | 4 GB | 8 GB | 4 GB | 6-8 GB | 6 GB |
| One-click | Платный | Нет | Да | Pinokio + офиц. Win/Mac | ZIP | Нет | Win-installer |
| NSFW filter | Opt-in | Default-on | Нет | Default + anti-tamper | Нет | Нет | Нет |
| Качество live (общее) | B | C (нет live) | B+ | A | S (с DFM) | C (нет live) | A+ |

---

## Честный итог и рекомендации

1. **Если у тебя NVIDIA на Windows и хочешь max качество → VisoMaster.** Но готовься к боли с RTX 50-й серией и неочевидной установке. Если активность апстрима подохла, смотри `VisoMasterFusion` форки.
2. **Если нужна универсальность, кросс-платформенность, активный апстрим, "взрослый" продукт → Facefusion.** Лучший release discipline в нише, 11 процессоров, поддержка всего железа. Минус — UDP→OBS bridge для virtual cam, не нативно.
3. **Если нужно лицо конкретного человека и ты готов тренировать модель → DeepFaceLive с DFM.** Качество всё ещё непревзойдённое для live, но проект мёртв — на новых GPU/ОС будет потихоньку отваливаться.
4. **Если ты на Mac → Deep-Live-Cam.** Единственный реальный выбор. Готовься к OBS-bridge'у для virtual cam и FPS не выше 20-25.
5. **Если хочется минимум возни и есть бюджет → платный пакет** Deep-Live-Cam (deeplivecam.net) или SECourses' VisoMaster/Roop-Floyd на Patreon. Они продают именно за решение установочной боли.
6. **Roop, Rope в 2026 не используй.** Они оба интересны исторически, но устарели и не развиваются.
7. **Roop-Unleashed** — если очень нужны его уникальные фичи (text-prompt окклюзия, ReSwapper-256), бери `Cognibuild/ROOP-FLOYD` на Codeberg, понимая, что это caretaker-mode.

**Ключевые риски, о которых стоит знать заранее:**
- Все эти проекты в 2025-2026 находятся под усилившимся давлением: GitHub снёс Roop-Unleashed без предупреждения, iperov закрыл DeepFaceLive, hacksider держит NSFW-фильтр в коде хоть и opt-in. Любой из оставшихся может уйти в архив.
- Pindrop Pulse, Reality Defender и т.д. интегрируются в Zoom/Teams. Эра "слепых платформ" заканчивается.
- KYC-системы тоже подтягиваются (CEN/TS 18099, ISO 25456). Но пока что out-of-the-box тулзы реально проходят single-factor liveness — это документированная индустриальная боль.
- Реальный фрод на этой технологии (Arup HK $25M, Сингапур $499K) — уже не футурология, и юридически риск использования "не для развлечения" вырос радикально.

---

## Sources

Агрегировано из 4 параллельных агентских ресерчей.

**Репозитории:**
- [hacksider/Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam)
- [s0md3v/roop](https://github.com/s0md3v/roop)
- [Cognibuild/ROOP-FLOYD (Codeberg)](https://codeberg.org/Cognibuild/ROOP-FLOYD)
- [facefusion/facefusion](https://github.com/facefusion/facefusion)
- [iperov/DeepFaceLive (archived)](https://github.com/iperov/DeepFaceLive)
- [Hillobar/Rope](https://github.com/Hillobar/Rope)
- [visomaster/VisoMaster](https://github.com/visomaster/VisoMaster)
- [deepinsight/inswapper-512-live](https://github.com/deepinsight/inswapper-512-live)
- [KwaiVGI/LivePortrait](https://github.com/KwaiVGI/LivePortrait)

**Виртуальная камера / OS:**
- [OBS Virtual Camera Guide](https://obsproject.com/kb/virtual-camera-guide)
- [OBS Sequoia compatibility](https://obsproject.com/forum/threads/sequoia-15-0-and-obs.179883/)
- [pyvirtualcam](https://pypi.org/project/pyvirtualcam/)

**Детекция / фрод:**
- [LipFD (NeurIPS 2024)](https://github.com/AaronComo/LipFD)
- [Pindrop Pulse for Meetings](https://www.pindrop.com/article/pindrop-pulse-app-zoom-meetings-defend-ai-deepfakes/)
- [Arup HK $25M Deepfake — CNN](https://www.cnn.com/2024/05/16/tech/arup-deepfake-scam-loss-hong-kong-intl-hnk)
- [Singapore $499K case](https://www.tookitaki.com/blog/deepfake-ceo-scam-singapore-2025)
- [Trend Micro: AI vs AI Deepfakes & eKYC](https://www.trendmicro.com/vinfo/us/security/news/cyber-attacks/ai-vs-ai-deepfakes-and-ekyc)

**Покрытие в прессе:**
- [PetaPixel: DLC viral coverage](https://petapixel.com/2024/08/14/deep-live-cam-deepfake-ai-tool-lets-you-become-anyone-in-a-video-call-with-single-photo-mark-zuckerberg-jd-vance-elon-musk/)
- [Hacker News: Roop discontinuation](https://news.ycombinator.com/item?id=37274331)
- [VisoMaster vs Facefusion (vendor)](https://visomaster.com/posts/visomaster-vs-facefusion)
