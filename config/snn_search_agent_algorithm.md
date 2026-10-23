# Алгоритм поиска `snn_search_agent.py`

Этот документ описывает поиск архитектур SNN через псевдокод. Каждая строка псевдокода содержит короткий комментарий.

## Общая схема

```text
seed_config + history -> proposal -> validation -> deduplication -> codegen -> training -> metrics -> history
```

Агент строит новую архитектуру, проверяет ее, обучает модель и пишет результат в историю. Следующая попытка использует эту историю.

## `run_search`

```python
function run_search(args):  # Запускает серию попыток поиска.
    loaded_env = load_env_file(args.env_file)  # Загружает API-ключи и настройки из .env.
    run_dir = resolve(args.run_dir)  # Генрируется полный путь к папке эксперимента на основе переданных в скрипт аргументов
    create_directory(run_dir)  # Создает папку эксперимента.
    create_directory(run_dir / "attempts")  # Создает папку для прогона (текущего запуска).

    seed_config = load_json(args.seed_config)  # Загружает стартовую архитектуру (baseline), которая заранее определена.
    seed_errors = validate_config(seed_config)  # Проверяет baseline архитектуру.
    if seed_errors is not empty:  # Если есть ошибки в baseline.
        raise error(seed_errors)  # Останавливает прогон.

    manifest = ensure_manifest(run_dir, args, args.seed_config)  # Создает или читает <run_dir>/run_manifest.json, который фиксирует, с какими настройками был начат эксперимент. Если ты потом посмотришь history.jsonl, attempts/, checkpoint.pt, то run_manifest.json объясняет контекст: какой seed использовали, каким LLM-провайдером генерировали архитектуры, сколько эпох учили, каким snn_coderun.py (на случай если их будет несколько) запускали.
    run_id = manifest["run_id"]  # Берет id прогона (запуска).

    memory = ExperimentMemory(run_dir)  # Подключает историю попыток.
    provider = create_provider(args, seed_config)  # Выбирает LLM.

    start_idx = memory.next_attempt_index()  # Находит номер следующей попытки.
    completed = []  # Готовит список результатов текущей команды.

    print_run_header(run_dir, run_id, loaded_env, provider)  # Печатает параметры запуска.

    for offset in range(args.attempts):  # Повторяет цикл нужное число раз.
        idx = start_idx + offset  # Считает номер попытки.
        record = execute_attempt(idx, run_id, run_dir, memory, provider, seed_config, args)  # Запускает одну попытку.
        append(completed, record)  # Сохраняет результат в памяти процесса.
        print_attempt_result(record)  # Печатает статус и метрику.

    best_success = filter_successful_attempts(completed)  # Оставляем только успешные попытки.
    if best_success is not empty:  # Проверяем наличие успешных запусков.
        champion = max_by_best_test_accuracy(best_success)  # Выбираем лучший результат.
        print_champion(champion)  # Печатаем лучшую попытку.
    else:
        print_no_successful_attempts()  # Печатаем, что НЕТ успешных попыток.

    return 0
```

## `execute_attempt`

```python
function execute_attempt(attempt_index, run_id, run_dir, memory, provider, seed_config, args):  # Выполняет одну попытку.
    attempt_id = make_attempt_id(attempt_index)  # Создает имя вида attempt_0001.
    attempt_dir = run_dir / "attempts" / attempt_id  # Генерирует путь к папке попытки.
    create_directory(attempt_dir)  # Создает папку попытки.

    started_at = now_utc_iso()  # Записывает время старта.
    context_text = memory.build_context(args.context_attempts, args.max_context_chars)  # Читает history.jsonl и делает summary для prompt.
    write_text(attempt_dir / "context.txt", context_text)  # Сохраняет контекст.

    history = memory.load()  # Загружает полную историю для dedup, не для prompt.
    feedback = []  # Готовит список ошибок для retry.
    proposal = None  # Держит принятый кандидат.
    proposal_errors = []  # Держит ошибки proposal.

    for retry in range(1, args.proposal_retries + 1):
        system_prompt, user_prompt = build_prompts(seed_config, context_text, feedback)  # Готовит prompt.
        write_prompt(attempt_dir, retry, system_prompt, user_prompt)  # Сохраняет prompt.

        raw_text = provider.propose(system_prompt, user_prompt, attempt_index)  # Запрашивает архитектуру.
        write_text(attempt_dir / f"proposal_raw_retry{retry}.txt", raw_text)  # Сохраняет ответ LLM.

        try:  # Начинает парсинг ответа.
            candidate = parse_proposal(raw_text, retry)  # Достает JSON-конфиг.
        except error as exc:  # Ловит ошибку парсинга.
            proposal_errors = ["Proposal parsing error: " + str(exc)]  # Записывает причину.
            feedback = proposal_errors  # Передает причину в следующий retry.
            write_validation_errors(attempt_dir, retry, proposal_errors)  # Пишет ошибку в файл.
            continue  # Запрашивает новый ответ.

        proposal_errors = validate_config(candidate.config)  # Проверяет конфиг.
        if proposal_errors is not empty:  # Находит ошибки конфига.
            feedback = proposal_errors  # Передает ошибки в следующий retry.
            write_validation_errors(attempt_dir, retry, proposal_errors)  # Пишет ошибки в файл.
            continue  # Запрашивает новый конфиг.

        duplicate_error = find_seen_config_duplicate(candidate.config, history, run_dir)  # Ищет повторы.
        if duplicate_error exists:  # Находит уже проверенную схему.
            proposal_errors = [duplicate_error]  # Записывает ошибку повтора.
            feedback = proposal_errors  # Передает ошибку в следующий retry.
            write_validation_errors(attempt_dir, retry, proposal_errors)  # Пишет ошибку в файл.
            continue  # Запрашивает новую схему.

        proposal = candidate  # Принимает кандидата.
        break  # Завершает retry-цикл.

    if proposal is None:  # Проверяет провал всех retry.
        record = build_failed_proposal_record(run_id, attempt_index, attempt_id, started_at, args, proposal_errors)  # Создает запись failed_proposal.
        return persist_attempt_record(memory, attempt_dir, record)  # Пишет провал в историю.

    config_path = attempt_dir / "config.json"  # Выбирает путь для конфига.
    write_json(config_path, proposal.config)  # Сохраняет конфиг.
    write_text(attempt_dir / "reasoning_summary.txt", proposal.reasoning_summary)  # Сохраняет reasoning токены LLM.

    config_signature = config_signature(proposal.config)  # Создает схему конфига (график).
    config_sha = json_sha256(proposal.config)  # Считает точный хэш конфига.
    architecture_fingerprint = architecture_fingerprint(proposal.config)  # Считает хэш структуры.
    param_count = estimate_param_count(proposal.config)  # Оценивает число параметров.

    model_path = attempt_dir / "model.py"  # Выбирает путь для модели.
    codegen_cmd = build_codegen_command(args.codegen_path, config_path, model_path, args.save_diagram)  # Собирает команду codegen.
    codegen_res = run_command(codegen_cmd, codegen_stdout_log, codegen_stderr_log, args.timeout_seconds)  # Генерирует model.py.

    if codegen_res.returncode != 0:  # Проверяет ошибку codegen.
        record = build_failed_codegen_record(run_id, attempt_index, attempt_id, started_at, proposal, config_metadata, codegen_res)  # Создает запись failed_codegen.
        return persist_attempt_record(memory, attempt_dir, record)  # Пишет ошибку в историю.

    checkpoint_path = attempt_dir / "checkpoint.pt"  # Выбирает путь checkpoint.
    coderun_cmd = build_coderun_command(args.coderun_path, model_path, checkpoint_path, args, attempt_index)  # Собирает команду обучения.
    coderun_res = run_command(coderun_cmd, coderun_stdout_log, coderun_stderr_log, args.timeout_seconds)  # Обучает и оценивает модель.

    test_acc, best_acc, ckpt_meta = read_checkpoint_metrics(checkpoint_path)  # Читает метрики.
    ended_at = now_utc_iso()  # Записывает время конца.

    if coderun_res.returncode == 0:  # Проверяет успешность обучения и тестирования.
        status = "success"  # Ставит статус успеха.
        error_message = ""  # Оставляет ошибку пустой.
    else:  # Обрабатывает ошибку обучения/тестирования.
        status = "failed_train"  # Ставит статус обучения.
        error_message = training_error_message(coderun_res)  # Формирует причину.

    record = build_final_attempt_record(run_id, attempt_index, attempt_id, started_at, ended_at, args, proposal, config_metadata, checkpoint_path, test_acc, best_acc, ckpt_meta, status, error_message)  # Собирает итог.
    return persist_attempt_record(memory, attempt_dir, record)  # Пишет итог в историю.
```

## `build_context`

```python
function build_context(max_attempts, max_chars):  # Делает короткий текст истории для prompt.
    history = load()  # Читает строки из history.jsonl.
    if history is empty:  # Проверяет первый запуск.
        return "No previous attempts yet."  # Возвращает простой контекст без истории.

    successes = []  # Готовит список успешных попыток.
    for item in history:  # Обходит всю историю.
        if item.status == "success" and item.best_test_accuracy is number:  # Берет попытки с метрикой.
            append(successes, item)

    top = sort_by_best_test_accuracy_desc(successes)[0:3]  # Берет три лучших результата.
    if max_attempts <= 0:  # Проверяет режим полной истории.
        recent = history  # Берет всю историю.
    else:  # Ограничивает число записей.
        recent = last_items(history, max_attempts)  # Берет последние max_attempts попыток.

    failures = []  # Готовит список недавних ошибок.
    for item in recent:  # Обходит последние попытки.
        if item.status != "success":  # Берет неуспешные попытки.
            append(failures, item)  # Добавляет ошибку.
    failures = last_items(failures, 3)  # Оставляет три последние ошибки.

    repeated_by_sha = {}  # Группирует точные повторы JSON.
    repeated_by_signature = {}  # Группирует похожие схемы по signature.
    for item in history:  # Обходит всю историю.
        if item.config_sha256 exists:  # Проверяет точный хэш.
            repeated_by_sha[item.config_sha256].append(item)  # Добавляет попытку в группу SHA.
        if item.config_signature exists:  # Проверяет краткую сигнатуру.
            repeated_by_signature[item.config_signature].append(item)  # Добавляет попытку в группу signature.

    exact_repeats = top_repeated_groups(repeated_by_sha, min_count=2, limit=5)  # Находит частые точные повторы.
    signature_repeats = top_repeated_groups(repeated_by_signature, min_count=5, limit=5)  # Находит частые повторные схемы.

    lines = []  # Готовит строки итогового контекста.
    append(lines, "Total attempts so far: " + count(history))  # Добавляет число попыток.

    if top is not empty:  # Проверяет наличие успешных попыток.
        append(lines, "Top successful attempts:")  # Добавляет заголовок топа.
        for item in top:  # Обходит лучшие попытки.
            append(lines, format_top_attempt(item))  # Добавляет id, accuracy, signature и sha.

    if exact_repeats is not empty:  # Проверяет точные повторы.
        append(lines, "Repeated exact configs to avoid:")  # Добавляет заголовок повторов.
        for group in exact_repeats:  # Обходит группы SHA.
            append(lines, format_exact_repeat(group))  # Добавляет sha, count, best и attempts.

    if signature_repeats is not empty:  # Проверяет повторные signature.
        append(lines, "Exhausted/repeated signatures to avoid unless structurally changed:")  # Добавляет заголовок signature.
        for group in signature_repeats:  # Обходит группы signature.
            append(lines, format_signature_repeat(group))  # Добавляет count, best, signature и attempts.

    append(lines, "Most recent attempts:")  # Добавляет заголовок последних попыток.
    for item in recent:  # Обходит последние попытки.
        append(lines, format_recent_attempt(item))  # Добавляет id, status, best_acc и reasoning.

    if failures is not empty:  # Проверяет недавние ошибки.
        append(lines, "Recent failure diagnostics:")  # Добавляет заголовок ошибок.
        for item in failures:  # Обходит ошибки.
            append(lines, format_failure(item))  # Добавляет id и error_message.

    context = join_lines(lines)  # Склеивает строки в один текст.
    if length(context) > max_chars:  # Проверяет лимит длины prompt-контекста.
        context = context[0:max_chars - 3] + "..."  # Обрезает слишком длинный контекст.

    return context  # Возвращает summary истории для build_prompts.
```

## Генерация proposal

```python
function build_prompts(seed_config, context_text, feedback):  # Готовит prompt.
    system_prompt = "Ты ассистент по поиску архитектур SNN..."  # Задает роль и формат ответа.
    feedback_block = format_feedback(feedback)  # Форматирует ошибки retry.
    user_prompt = template(seed_config, context_text, feedback_block)  # Добавляет seed, историю и правила.
    return system_prompt, user_prompt  # Возвращает промпты для LLM.

function provider.propose(system_prompt, user_prompt, attempt_index):  # Возвращает новый кандидат.
    if provider is OpenAIProposalProvider:  # Использует LLM.
        response = call_openai_compatible_api(system_prompt, user_prompt, model, temperature)  # Отправляет prompt.
        return response.text  # Возвращает текст ответа.
    if provider is MockProposalProvider:  # (для дебага остальной части кода)
        cfg = deepcopy(seed_config)  # Копирует seed.
        mutate_reservoir_sizes(cfg)  # Меняет размеры reservoir.
        mutate_weight_initializers(cfg)  # Меняет init весов.
        return json_with_rationale_and_config(cfg)  # Возвращает JSON proposal.
```

## Парсинг proposal

```python
function parse_proposal(raw_text, retry_index):  # Нормализует ответ provider.
    payload = extract_first_json_object(raw_text)  # Достает первый JSON object.
    if "config" in payload:  # Проверяет формат с оберткой.
        cfg = payload["config"]  # Берет архитектуру.
        rationale = payload.get("rationale", "")  # Берет объяснение.
        reasoning_summary = payload.get("reasoning_summary", "")  # Берет reasoning.
    else:  # Поддерживает ответ без обертки.
        cfg = payload
        rationale = ""  # Оставляет объяснение пустым.
        reasoning_summary = ""  # Оставляет reasoning пустым.

    if cfg is not dict:  # Проверяет тип конфига.
        raise error("Proposal must include config object")  # Отклоняет неверный тип.
    if rationale is empty:  # Проверяет объяснение.
        rationale = "Модель не вернула rationale."  # Подставляет текст по умолчанию.
    if reasoning_summary is empty:  # Проверяет reasoning.
        reasoning_summary = rationale  # Берет explanation как reasoning (костыль).

    return Proposal(cfg, rationale, reasoning_summary, raw_text, retry_index)  # Возвращает proposal.
```

## Извлечение JSON

```python
function extract_first_json_object(text):  # Ищет JSON object в тексте.
    stripped = trim(text)  # Убирает пробелы по краям.
    if stripped is empty:  # Проверяет пустой ответ.
        raise error("Empty LLM response")  # Отклоняет пустой текст.

    try direct_json_parse(stripped)  # Парсит весь ответ.
    if direct_parse_success and result is dict:  # Проверяет JSON object.
        return result  # Возвращает object.

    for fenced_block in find_markdown_json_blocks(text):  # Ищет блоки ```json.
        try parse fenced_block as JSON  # Парсит содержимое блока.
        if parse_success and result is dict:  # Проверяет object.
            return result  # Возвращает object из блока.

    for each "{" position in text:  # Ищет начало JSON object.
        candidate = read_balanced_braces_from_position(text, position)  # ищет закрывающую фигурную скобку "}" с учетом вложенных при их появлении.
        try parse candidate as JSON  # Парсит фрагмент.
        if parse_success and result is dict:  # Проверяет object.
            return result  # Возвращает object.

    raise error("Could not extract a JSON object")  # Сообщает о провале.
```

## Валидация конфига

```python
function validate_config(cfg):  # Проверяет конфиг перед запуском.
    errors = []  # Собирает все ошибки.

    unknown_top = keys(cfg) - supported_top_level_keys  # Ищет лишние верхние поля.
    if unknown_top is not empty:  # Проверяет лишние поля.
        add_error(errors, unknown_top)  # Записывает ошибку.

    validate_model_name(cfg["model_name"], errors)  # Проверяет имя класса.

    nodes = cfg["nodes"]  # Берет узлы.
    if nodes is not non_empty_list:  # Проверяет список узлов.
        add_error(errors, "`nodes` must be a non-empty list")  # Записывает ошибку.
        return errors

    node_map = {}  # Готовит индекс узлов.
    input_nodes = []  # Готовит список input узлов.
    output_nodes = []  # Готовит список output узлов.
    reservoir_nodes = []  # Готовит список reservoir (не input и не output) узлов.

    for node in nodes:  # Обходит узлы.
        validate_node_schema(node, errors)  # Проверяет id, type и size.
        validate_node_tau_and_fraction(node, errors)  # Проверяет значения tau и fraction_long.
        register_node(node_map, input_nodes, output_nodes, reservoir_nodes, node, errors)  # Добавляет узел в индексы.

    if count(input_nodes) != 1:  # Проверяет один input.
        add_error(errors, "Exactly one input node is required")
    if count(output_nodes) != 1:  # Проверяет один output.
        add_error(errors, "Exactly one output node is required")
    if count(reservoir_nodes) == 0:  # Проверяет наличие reservoir.
        add_error(errors, "At least one reservoir node is required")

    if input_size != 4096:  # Проверяет размер входа.
        add_error(errors, "Input node size must be 4096")
    if output_size != 30:  # Проверяет размер выхода.
        add_error(errors, "Output node size must be 30")

    edges = cfg["edges"]  # Берет ребра.
    if edges is not non_empty_list:  # Проверяет список ребер.
        add_error(errors, "`edges` must be a non-empty list")  # Записывает ошибку.
        return errors  # Завершает проверку.

    adjacency = create_empty_adjacency(node_map)  # Готовит граф достижимости.
    seen_edge_pairs = set()  # Готовит набор ребер.
    out_incoming = 0  # Считает входы в output.

    for edge in edges:  # Обходит ребра.
        validate_edge_schema(edge, node_map, errors)  # Проверяет from и to.
        reject_duplicate_edge(edge, seen_edge_pairs, errors)  # Ищет дубликаты ребер.
        add_to_adjacency(adjacency, edge)  # Добавляет ребро в граф.
        validate_edge_direction(edge, node_map, errors)  # Проверяет направление ребра.
        validate_edge_bias_and_weight_init(edge, errors)  # Проверяет bias и weight_init.
        if edge.to is output:  # Проверяет ведет ли ребро в output.
            out_incoming += 1  # Увеличивает счетчик output-ребер.

    if out_incoming == 0:  # Проверяет наличие ребер в output.
        add_error(errors, "Output node must have at least one incoming edge")

    if no_directed_path_from_input_to_output(adjacency):  # Проверяет налчиие пути input to output.
        add_error(errors, "No directed path from input to output")  # Записывает ошибку пути.

    validate_neuron_params(cfg.get("neuron_params"), errors)  # Проверяет neuron_params.
    validate_surrogate_params(cfg.get("surrogate_params"), errors)  # Проверяет surrogate_params.
    validate_conv_frontend(cfg.get("conv_frontend"), input_size, errors)  # Проверяет conv_frontend.

    if errors is empty:  # Продолжает только для валидной структуры.
        pcount = estimate_param_count(cfg)  # Считает параметры.
        if pcount["total"] > 10_000_000:  # Проверяет лимит параметров.
            add_error(errors, "Model exceeds parameter limit")

    return errors  # Возвращает список ошибок.
```

## Анти-повторы

```python
function find_seen_config_duplicate(candidate_config, history, run_dir):  # Ищет старую версию кандидата.
    candidate_sha = json_sha256(candidate_config)  # Считает точный хэш.
    for row in history:  # Обходит историю.
        if row["config_sha256"] == candidate_sha:  # Сравнивает точный хэш.
            return exact_duplicate_message(row)  # Возвращает ошибку точного повтора.

    candidate_fingerprint = architecture_fingerprint(candidate_config)  # Считает хэш структуры.
    for row in history:  # Обходит историю снова.
        fingerprint = row["architecture_fingerprint"]  # Берет старый хэш структуры.
        if fingerprint is missing:  # Проверяет старую запись.
            previous_config = load_history_config(run_dir, row)  # Загружает старый конфиг.
            fingerprint = architecture_fingerprint(previous_config)  # Считает старый хэш структуры.
        if fingerprint == candidate_fingerprint:  # Сравнивает структуры.
            return structural_duplicate_message(row)  # Возвращает ошибку структурного повтора.

    return None  # Разрешает новый кандидат.
```

## Structural fingerprint

```python
function architecture_fingerprint(cfg):  # Хэширует структуру архитектуры.
    id_to_role = {}  # Готовит канонические имена.
    reservoir_sizes = []  # Готовит размеры внутренних резервуаров.
    input_count = 0  # Считает количество input.
    output_count = 0  # Считает output.
    reservoir_count = 0  # Считает число резервуаров.

    for node in cfg["nodes"]:  # Обходит узлы.
        if node.type == "input":  # Находит input.
            role = canonical_input_role(input_count)  # Назначает роль input.
        elif node.type == "output":  # Находит output.
            role = canonical_output_role(output_count)  # Назначает роль.
        elif node.type == "reservoir":  # Находит reservoir.
            role = canonical_reservoir_role(reservoir_count)  # Назначает роль.
            append(reservoir_sizes, node.size)  # Сохраняет размер.
        else:  # Обрабатывает неизвестный тип.
            role = fallback_role(node)  # Назначает fallback-роль.
        id_to_role[node.id] = role  # Связывает id узла с ролью.

    canonical_edges = []  # Готовит список ребер.
    for edge in cfg["edges"]:  # Обходит ребра.
        src = id_to_role[edge.from]  # Берет роль источника.
        dst = id_to_role[edge.to]  # Берет роль приемника.
        append(canonical_edges, src + "->" + dst)  # Сохраняет ребро.

    payload = {}  # Готовит данные для хэша.
    payload["reservoir_sizes"] = reservoir_sizes  # Добавляет размеры.
    payload["edges"] = sort(canonical_edges)  # Добавляет ребра.
    payload["conv_frontend"] = cfg.get("conv_frontend", {})  # Добавляет CNN frontend.

    return json_sha256(payload)  # Возвращает хэш структуры.
```

## Оценка числа параметров

```python
function estimate_param_count(cfg):  # Считает число обучаемых параметров.
    nodes = cfg["nodes"]  # Берет узлы.
    edges = cfg["edges"]  # Берет ребра.
    node_map = map_by_id(nodes)  # Индексирует узлы.

    incoming = {}  # Готовит входящие ребра без self-loop.
    recurrent = {}  # Готовит self-loop ребра.
    for edge in edges:  # Обходит все ребра.
        if edge.from == edge.to:  # Находит recurrent self-loop.
            recurrent[edge.from] = edge  # Сохраняет self-loop по id узла.
        else:  # Обрабатывает обычное directed edge.
            incoming[edge.to].append((edge.from, edge))  # Сохраняет вход в destination.

    conv_total = estimate_conv_frontend_params(cfg["conv_frontend"])  # Считает параметры CNN.
    cnn_out_size = estimate_conv_frontend_output_size(cfg["conv_frontend"])  # Считает выход CNN.
    reservoir_tau = sum_sizes_of_reservoir_nodes(nodes)  # Считает tau_m reservoir.

    linear_total = 0  # Хранит число весов Linear.
    linear_bias_total = 0  # Хранит число bias-параметров Linear.

    function add_linear(src_size, dst_size, bias):  # Добавляет параметры одного Linear.
        linear_total += src_size * dst_size  # Увеличивает счетчик весов.
        if bias:  # Проверяет наличие bias.
            linear_bias_total += dst_size  # Увеличивает счетчик bias.

    for node in nodes:  # Обходит узлы.
        if node.type == "output":  # Находит output.
            in_edges = incoming[node.id]  # Берет входы output.
            if count(in_edges) == 1:  # Проверяет один источник.
                src_id, edge = first(in_edges)  # Берет source и edge.
                add_linear(node_map[src_id].size, node.size, edge.bias)  # Добавляет readout Linear.
            elif count(in_edges) > 1:  # Проверяет несколько источников.
                src_total = sum(node_map[src_id].size for src_id, edge in in_edges)  # Складывает размеры источников.
                add_linear(src_total, node.size, first(in_edges).edge.bias)  # Добавляет merged readout Linear.
        elif node.type == "reservoir":  # Находит reservoir.
            in_edges = incoming[node.id]  # Берет входы reservoir.
            reservoir_sources = filter_reservoir_sources(in_edges, node_map)  # Отбирает входы из reservoir.
            other_sources = filter_non_reservoir_sources(in_edges, node_map)  # Отбирает входы не из reservoir.

            for src_id, edge in other_sources:  # Обходит input to reservoir связи.
                src_size = cnn_out_size if conv_frontend_enabled_and_src_is_input(src_id) else node_map[src_id].size  # Берет размер входа.
                add_linear(src_size, node.size, edge.bias)  # Добавляет входную проекцию.

            if count(reservoir_sources) > 1:  # Проверяет несколько reservoir-входов.
                src_total = sum(node_map[src_id].size for src_id, edge in reservoir_sources)  # Складывает размеры reservoir-источников.
                add_linear(src_total, node.size, first(reservoir_sources).edge.bias)  # Добавляет merged reservoir projection.
            elif count(reservoir_sources) == 1:  # Проверяет один reservoir-вход.
                src_id, edge = first(reservoir_sources)  # Берет source и edge.
                add_linear(node_map[src_id].size, node.size, edge.bias)  # Добавляет reservoir projection.

            if node.id in recurrent:  # Проверяет self-loop.
                edge = recurrent[node.id]  # Берет recurrent edge.
                add_linear(node.size, node.size, edge.bias)  # Добавляет recurrent Linear.

    total = linear_total + linear_bias_total + reservoir_tau + conv_total  # Суммирует параметры.
    return parameter_breakdown(total, linear_total, linear_bias_total, reservoir_tau, conv_total)  # Возвращает breakdown.
```

## Codegen и обучение

```python
function build_codegen_command(codegen_path, config_path, model_path, save_diagram):  # Готовит команду codegen.
    cmd = [python_executable, codegen_path, config_path, model_path]  # Задает Python, вход и выход.
    if save_diagram:  # Проверяет флаг диаграммы.
        append(cmd, diagram_path)  # Добавляет путь diagram.png.
    return cmd  # Возвращает команду.

function build_coderun_command(coderun_path, model_path, checkpoint_path, args, attempt_index):  # Готовит команду обучения.
    cmd = [python_executable, coderun_path, model_path]  # Задает code_runner и model.py.
    append(cmd, ["--epochs", args.epochs])  # Передает эпохи.
    append(cmd, ["--timesteps", args.timesteps])  # Передает timesteps.
    append(cmd, ["--batch-size", args.batch_size])  # Передает batch size.
    append(cmd, ["--lr", args.lr])  # Передает learning rate.
    append(cmd, ["--num-workers", args.num_workers])  # Передает workers для dataloader.
    append(cmd, ["--seed", args.base_seed + attempt_index])  # Передает seed попытки.
    append(cmd, ["--output", checkpoint_path])  # Передает путь checkpoint.
    if args.data_path is not None:  # Проверяет путь к данным.
        append(cmd, ["--data-path", resolve(args.data_path)])  # Передает dataset path.
    if args.device is not None:  # Проверяет устройство.
        append(cmd, ["--device", args.device])  # Передает device для PyTorch.
    return cmd  # Возвращает команду.

function run_command(command, stdout_path, stderr_path, timeout_seconds):  # Запускает процесс.
    open stdout_path for writing  # Открывает stdout log.
    open stderr_path for writing  # Открывает stderr log.
    proc = subprocess.Popen(command)  # Стартует процесс.
    start_thread_to_tee_stdout(proc.stdout, console, stdout_file)  # Дублирует stdout.
    start_thread_to_tee_stderr(proc.stderr, console, stderr_file)  # Дублирует stderr.
    returncode = wait_for_process_or_kill_on_timeout(proc, timeout_seconds)  # Ждет процесс.
    join_output_threads()  # Дожидается потоков логирования.
    return CommandResult(returncode, stdout, stderr, timed_out, command)  # Возвращает результат.
```

## Чтение метрик

```python
function read_checkpoint_metrics(checkpoint_path):  # Читает checkpoint.
    if checkpoint_path does not exist:  # Проверяет файл.
        return None, None, {}  # Возвращает пустые метрики.

    try import torch  # Импортирует PyTorch.
    if torch_import_failed:  # Проверяет импорт.
        return None, None, warning_metadata  # Возвращает warning.

    checkpoint = torch.load(checkpoint_path, map_location="cpu")  # Загружает checkpoint.
    if checkpoint is not dict:  # Проверяет формат.
        return None, None, warning_metadata  # Возвращает warning.

    test_acc = to_float_or_none(checkpoint["test_accuracy"])  # Читает финальную accuracy.
    best_acc = to_float_or_none(checkpoint["best_test_accuracy"])  # Читает лучшую accuracy.

    if best_acc is None and checkpoint["history"]["test_acc"] exists:  # Проверяет fallback.
        best_acc = max(checkpoint["history"]["test_acc"])  # Берет максимум истории.
        if test_acc is None:  # Проверяет финальную accuracy.
            test_acc = last(checkpoint["history"]["test_acc"])  # Берет последнюю accuracy.

    subset = extract_checkpoint_metadata(checkpoint)  # Берет metadata.
    return test_acc, best_acc, subset  # Возвращает метрики.
```

## Что оптимизирует агент

```python
candidate_architecture = provider.propose(history_context)  # Выбирает новую архитектуру по истории.
score = train_and_evaluate(candidate_architecture)  # Получает best_test_accuracy после обучения.
history.append(candidate_architecture, score)  # Сохраняет результат для следующей попытки.
```

Агент оптимизирует `best_test_accuracy`. Он не меняет архитектуру градиентом. Он запускает внешний black-box поиск с памятью, валидацией и реальным train/eval.
