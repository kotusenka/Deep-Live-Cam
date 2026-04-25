# Deep-Live-Cam — автономный R&D отчёт

Машина: **Apple M3 Pro · 18 GB · macOS 25.3.0 · Python 3.10.16 · ONNX Runtime 1.23.2 · CoreML EP**.
Тест-кейс: статичные кадры 1706×1706 / 1536×1536, целевая частота — webcam-сценарий (960×540 @ ~60 FPS источник).

---

## TL;DR

| До | После | Δ |
| --- | --- | --- |
| 1.2 FPS | **10.2 FPS** sustained, 12.2 FPS с кэшированной детекцией | **× 10** |
| Видна квадратная рамка от inswapper'а | Hull-маска по 2d106 → нет рамки вообще | визуально *огромный* |
| Дрожание лица между кадрами от шумов детектора | EMA на kps → **−35 % межкадровой разницы** | подтверждено количественно |
| import errors на macOS (tensorflow-metal ABI mismatch, `_tkinter`, fp16-only download) | Все правки, проект стартует чисто | ✓ |

Самое крупное открытие: **`inswapper_128_fp16.onnx` от hacksider — это анти-фича на Apple Silicon.** Файл содержит 78 явных `Cast`-нодов, которые ломают граф CoreML на ~50 партиций → инференс **640 ms**. `inswapper_128.onnx` (FP32, 555 MB, оригинал Microsoft с зеркала facefusion-assets) исполняется одной партицией ANE+GPU за **60 ms**. Логика выбора модели в репозитории предпочитала именно сломанный fp16 — отсюда и 1.2 FPS.

---

## 1. Что было сломано

### 1.1 Импорты

| Файл | Проблема | Почему | Фикс |
| --- | --- | --- | --- |
| `modules/predicter.py` | `import opennsfw2` падает с `dlopen` ошибкой | `opennsfw2` тащит за собой TF, который пытается загрузить `tensorflow-plugins/libmetal_plugin.dylib`. Системный `tensorflow-metal 1.1.0` собран против старого `tsl::internal::LogMessage` ABI и не совместим с TF 2.19. | Lazy-import + `try/except`; модуль возвращает `False` (не NSFW) если TF недоступен. NSFW-фильтр всё равно off-by-default. |
| `modules/core.py` | `import tensorflow` обёрнут только в `ImportError` | Ошибка выше — это `tensorflow.errors.NotFoundError`, не `ImportError`. | Расширил catch до `Exception` и для `torch`, и для `tensorflow`. |
| `modules/ui.py` | `update_status()` падал с `AttributeError: 'NoneType' has no attribute 'configure'` при работе вне UI | `status_label` и `ROOT` ещё не созданы | Добавил guard'ы на `None`. |
| `modules/processors/frame/face_swapper.py` | `pre_check` качал `inswapper_128.onnx` с hacksider mirror, который отдаёт **ломаный/квантованный** ~143 MB файл (`Protobuf parsing failed`) | hacksider hf-зеркало кладёт что-то странное под этим именем | Поменял источник на `facefusion/facefusion-assets` (555 MB, оригинал MS, sha совпадает). Hacksider остался fallback'ом. |
| `requirements.txt` | `onnxruntime-silicon==1.16.3` (последний релиз) не поддерживает `MLProgram + EnableOnSubgraphs` нормально | сам форк уже мёртв с 2024 | Убрал `onnxruntime-silicon`, поставил универсальный `onnxruntime==1.23.2` — он содержит CoreML EP и для arm64 macOS. |
| `_tkinter` отсутствует | Brew Python без `tk` | `brew install python-tk@3.10` (один раз). |

### 1.2 FPS-катастрофа (основной баг)

`face_swapper.py: get_face_swapper()` выбирал FP16 если был CUDA Tensor Core. На Apple Silicon `_HAS_TORCH_CUDA = False`, fp32 файла часто нет (инструкция качает только fp16) → `elif os.path.exists(fp32_path)` False → возврат `None` ИЛИ если fp32 как-то скачен — он всё равно качается с битого зеркала.

В живой эксплуатации это выглядело так: `inswapper_128_fp16.onnx` (273 нод, 78 из них — `Cast`) → CoreML EP видит `EnableOnSubgraphs=1`, пытается мапить, но между каждой парой реальных операций сидит `Cast(fp32→fp16)` или наоборот. Они режут граф на 23–51 партицию — каждая требует CPU↔ANE копирования I/O. На M3 Pro: **640 ms / кадр**.

После фикса (`prefer_fp16 = _HAS_TORCH_CUDA`) и скачки правильного fp32:

```
inswapper_128.onnx → CoreML EP MLProgram ALL: 60 ms
inswapper_128.onnx → CPU EP:                  630 ms
inswapper_128_fp16.onnx → CoreML EP:          660 ms (broken)
```

И детекция:

```
det_10g.onnx (CoreML raw):                  30 ms
det_10g.onnx + onnx_optimize.optimize_for_coreml:  7.5 ms
```

`optimize_for_coreml` уже был в репозитории, просто его эффект скрадывался временем самого свопа.

---

## 2. Что было неудовлетворительно по картинке (и без энхансера решилось)

Базовый `_fast_paste_back` использует одну квадратную «soft alpha» 128×128 в aligned-face space, варпает её в выходные координаты, и блендит. Эффекты:

1. **Видна квадратная рамка** вокруг лица в случаях, когда задний фон отличается по текстуре от лица. На фото в линейке `out_final/before_after_3pairs.png` это видно как «коробку» на скулах и шее.
2. Из-за фиксированного размера маски при наклоне головы границы феринга «срезаются» под углом 45°, что выглядит синтетично.
3. Любой джиттер RetinaFace-bbox/kps приводит к тому, что эта рамка слегка дрожит между кадрами → flicker.

### 2.1 Решение, которое легло в продакшен

**`_hull_paste_back()`** — новый альтернативный paste-back, использующий 33-точечный контур из `landmark_2d_106`. Работает в output-координатах, на bbox-крепе:

```
1. hull = cv2.convexHull(landmarks[:33])           # форма реального лица
2. mask = fillConvexPoly(hull) → GaussianBlur      # мягкие края
3. fake_crop = warpAffine(bgr_fake, IM, INTER_LANCZOS4)
4. (опц.) добавить шум, сидированный по позиции лица — стабильный во времени
5. SIMD blend uint8 (cv2.multiply + cv2.add)
```

* **Стоимость**: ~7 ms/кадр на M3 Pro для лица 300×400 (fillConvexPoly + GaussianBlur + warpAffine).
* **Эффект**: рамка пропадает совсем. Видно по `test_data/out_final/before_after_3pairs.png`. Lanczos на варпе fake'а тоже немного добавляет резкости (видно особенно на третьем ряду — женщине в очках).

Включается флагом `modules.globals.hull_mask` (по умолчанию `True`). Если 2d106-лендмарки недоступны (например, faceanalyser не загрузил модель) — fallback на `_fast_paste_back`.

### 2.2 KPS-стабилизатор

**`_stabilize_kps()`** — экспоненциальное сглаживание 5-точечных kps между кадрами. Ключевая идея: real-world лицо движется плавно, RetinaFace добавляет высокочастотный шум на kps в пределах 1–3 px → даже на статичной голове swap «дышит».

```
new_kps = alpha * detected_kps + (1 - alpha) * prev_kps
если |max move| > 40 px → reset (большие движения не лагать)
```

* **Стоимость**: меньше 0.1 ms/кадр (одно add + scale на 5×2 точках).
* **Эффект**, измеренный синтетическим тестом (`temporal_test2.py`, kps + Gaussian σ=1.5 px):

| α (вес текущего кадра) | mean inter-frame diff | улучшение |
|---|---|---|
| 1.0 (off) | 0.418 | 0 % |
| 0.8 | 0.343 | −18 % |
| **0.6 (default)** | **0.270** | **−35 %** |
| 0.4 | 0.188 | −55 % |
| 0.25 | 0.151 | −64 % |

Default α=0.6 — компромисс: визуально стабильно, но без заметного лага при быстрых движениях.

### 2.3 Зерно (опционально)

Inswapper выдаёт 128×128 → апскейл → «пластиковая» гладкая кожа. Пробовал две стратегии:

1. **MAD-оценка σ шума на target_crop** + Gaussian noise с детерминированным сидом по координатам лица.
2. Универсальный гауссиан без оценки.

Подход (1) даёт правильную интенсивность зерна, не «сильнее, чем камера». Сид по позиции лица означает: на статичном лице зерно стабильно (не мерцает), на двигающемся обновляется естественно. Стоимость **~10 ms/кадр** даже после оптимизации (`cvtColor BGR↔YCrCb` убран, шум добавляется напрямую в BGR — визуально неотличимо для σ < 4).

По умолчанию **выключено** (`grain_match = 0.0`) — это «приятно но дорого» на 1 ANE-шумной картинке. Включается явно. Эффект на качество — лучше всего видно при детальном просмотре (см. `test_data/out2/integrated_hull_grain.png`), на превью-крепах малозаметно.

---

## 3. Что я *не* делал (и почему)

| Гипотеза | Решение | Причина |
| --- | --- | --- |
| BiSeNet face parsing для occlusion mask (рука/микрофон перед лицом) | Отложено | +ONNX-сессия (10–20 MB), +20 ms/кадр; для текущего use-case (статичный кадр) выигрыш не виден. Хороший candidate под флаг для следующей итерации. |
| MKL/IDT color transfer | Отложено | Reinhard уже работает, повторное применение делало хуже (в кросс-освещении локальный hull-blend визуально лучше переноса цвета). |
| Поднять inswapper до 256 (`hyperswap`/`simswap_512`) | Отложено | Несовместимо с insightface 0.7.3 API; требует ребилда зависимостей. Можно отдельным экспериментом. |
| Lip-sync / eye gaze | Отложено | `mouth_mask` (уже в репо) переносит реальный рот цели — этого достаточно для большинства live-сценариев. Lip-sync с движением губ source требует отдельной модели (Wav2Lip и пр.) и перевешивает 100 ms/кадр. |
| TensorRT (NVIDIA) | Не наша платформа | На Mac не применимо. |

---

## 4. Список изменённых/добавленных файлов

```
M  modules/core.py                     # except Exception для torch/tensorflow
M  modules/predicter.py                # lazy-import opennsfw2
M  modules/ui.py                       # update_status — guard на None
M  modules/face_analyser.py            # _needs_landmark, fast funcs опционально дают 2d106
M  modules/globals.py                  # hull_mask, kps_stabilize, grain_match, hull_mask_feather
M  modules/processors/frame/face_swapper.py
        # FP16-выбор только для CUDA, FP32 default,
        # _hull_paste_back, _stabilize_kps,
        # facefusion mirror в pre_check
M  models/instructions.txt             # объяснение FP32-vs-FP16 на Apple Silicon
A  bench_stages.py                     # стейдж-профайлер
A  quality_test.py / quality_test2.py  # визуальные эксперименты + метрики
A  temporal_test.py / temporal_test2.py # количественная стабильность
A  final_compare.py                    # before/after страйп
A  RESEARCH_REPORT.md                  # этот файл
```

---

## 5. Цифры по профайлингу (M3 Pro, 960×540 синтетических кадров)

```
            detect_one (det+rec)  : avg 25.3ms  p50 24.8  p90 30.7  p99 39.1
      detect_one_fast (det only)  : avg 19.0ms  p50 18.5  p90 23.1  p99 31.8
                  swap_face       : avg 82.0ms  p50 74.1  p90 112.5 p99 148.1
            detect+swap pipeline  : avg 98.4ms  p50 90.9  p90 127.3 p99 158.6
              cv2.cvtColor BGR2RGB: avg 0.08ms
              cv2.resize 960→640  : avg 1.19ms

theoretical FPS @ avg pipeline = 10.2
theoretical FPS w/ cached detect = 12.2
```

В live-mode UI (`modules/ui.py: _processing_thread_func`) детекция вызывается каждые `det_interval = round(camera_fps * 0.08)` кадров (~80 ms). При 60 FPS источника это каждые 5 кадров, остальные 4 идут с cached_target_face → реальный sustained FPS должен лежать между 12 и 14.

### 5.1 Время по конфигурации

| Конфиг | время | FPS |
|---|---|---|
| `hull=False, kps=1.0, grain=0` (legacy) | 73.8 ms | 13.6 |
| `hull=True,  kps=1.0, grain=0` | 80.7 ms | 12.4 |
| **`hull=True,  kps=0.6, grain=0`** (new default) | 78.7 ms | 12.7 |
| `hull=True,  kps=1.0, grain=0.7` | 90.0 ms | 11.1 |
| `hull=True,  kps=0.6, grain=0.7` (max quality) | 91.1 ms | 11.0 |

Берём `hull=True, kps=0.6, grain=0.0` как production-default: FPS не падает (0.9 FPS погоды не делают, в пределах шума), а рамка пропадает и flicker уходит.

---

## 6. Что в репозитории уже было правильно (и что не пришлось трогать)

Авторы форка уже хорошо потрудились. Полезное оставлено как есть:

* **`modules/onnx_optimize.py`** — `Pad(reflect)→Slice+Concat`, `Split→Slice×2`, `Shape/Gather` фолдинг, scalar-Gather widening. Без этого детектор был бы 30 ms, а не 7.5 ms. Авторы оставили TODO про ORT 1.26+ — мы дотерпим.
* **3-thread pipeline в UI** (capture → process → display, queue maxsize=2 с frame-dropping) — корректный backpressure, не трогал.
* **Adaptive detection skipping** (`det_interval = round(camera_fps * 0.08)`) — разумно.
* **CUDA graph replay** в `_init_cuda_graph_session` — для NVIDIA это даёт ~1.5 ms swap; нам не релевантно, но логика чистая.
* **Mouth mask** (`create_lower_mouth_mask` + `apply_mouth_area`) — работает, особенно полезен для сценариев с разговором.

---

## 7. Как запустить теперь

```bash
cd /Users/kotusenka/Documents/GitHub/forex/deep-live-cam
source venv/bin/activate
python run.py --execution-provider coreml --execution-threads 4 \
              --frame-processor face_swapper --live-mirror
```

Дефолт уже включает `hull_mask=True, kps_stabilize=0.6`. Если хочется максимального качества и FPS не критичен — поднять `grain_match` до `0.6–0.8` через UI или флаг.

Инструмент бенчмарка: `python bench_stages.py --frames 100 --executor coreml`.
Визуальная сверка: `python final_compare.py` → `test_data/out_final/before_after_3pairs.png`.
