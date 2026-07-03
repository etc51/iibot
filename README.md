# MOEX AI Trader

Этот каталог собран на базе найденного на диске `C:\Users\HONOR\Documents\самосбор`
каркаса. Внутренний Python-пакет пока называется `samosbor`, чтобы не ломать
готовую архитектуру, но установленная команда для этого проекта — `moex-ai-trader`.

Текущая рабочая цель проекта:

- paper-бюджет: `300 000 RUB`;
- дневной ориентир: `2 000 RUB`;
- месячный эквивалент при 20 торговых днях: `40 000 RUB`;
- основной конфиг: `configs/server_tbank_stocks_intraday_300k_focused.toml`;
- режим по умолчанию: `local-paper`;
- реальный live-режим заблокирован кодом и конфигом.

Быстрая проверка после установки:

```powershell
moex-ai-trader --config configs/server_tbank_stocks_intraday_300k_focused.toml paper-cycle
```

Или тем же модулем без установленного entrypoint:

```powershell
python -m samosbor.cli --config configs/server_tbank_stocks_intraday_300k_focused.toml paper-cycle
```

---

# samosbor upstream notes

`samosbor` — Python-прототип системы automated paper-trading для Московской биржи
с интеграцией с T-Invest API Т-Банка.

Проект по умолчанию работает только в безопасных режимах:

- `local-paper` — локальный виртуальный брокер для paper-trading и бэктестов
- `tbank-sandbox` — песочница Т-Банка для виртуальных заявок
- `live` — намеренно заблокирован в текущем прототипе
- для intraday stock-профилей доступны `forced_flat_hours` / `forced_flat_weekdays`, чтобы принудительно закрывать позиции в конце сессии и не переносить их через ночь

Цели текущей версии:

- получать рыночные данные по акциям 1 и 2 эшелона Мосбиржи через T-Bank API и локальный архив на `D:`
- строить трендовые intraday-сигналы на базе `pandas-ta`, волатильности и ликвидности
- ограничивать риск на сделку и на портфель
- вести полный журнал действий
- считать базовые метрики эффективности: доходность, просадка, Sharpe, win rate, profit factor
- запускать backtest, research и paper-cycle из CLI с целевым виртуальным бюджетом `300 000 RUB`

## Структура

- `src/samosbor/` — код системы
- `configs/paper.toml` — базовый stock-trend paper-профиль на `300 000 RUB`
- `configs/local_pack_research.toml` — research-конфиг для локального архива свечей на `D:`
- `configs/local_pack_ta_research.toml` — сфокусированный TA-search по сильным MOEX futures
- `configs/local_pack_server_multi_300k.toml` — общий research-профиль под server-runtime, budget `300 000 RUB` и target `2000 RUB/день`
- `configs/local_pack_stocks_intraday_300k.toml` — локальный research-профиль под расширенный universe акций 1–2 эшелона из архива на `D:`
- `configs/local_pack_stocks_intraday_300k_focused.toml` — ускоренный локальный профиль для nightly-search на компактном shortlist с переключением между трендовыми и лёгкими hybrid-режимами
- `configs/local_pack_server_pair_cny_usd_candidate.toml` — evidence-backed candidate для пары `CNYRUBF + USDRUBF`
- `configs/local_pack_usdrubf_candidate.toml` — конфиг лучшего кандидата из локальной оптимизации
- `configs/local_pack_cnyrubf_ta_candidate.toml` — TA-кандидат на `CNYRUBF`
- `configs/local_pack_cnyrubf_ta_walk_forward.toml` — walk-forward валидация для `CNYRUBF` TA-кандидата
- `configs/local_pack_cnyrubf_ta_aggressive.toml` — усиленный риск-профиль для лучшего `CNYRUBF`-кандидата
- `configs/local_pack_fx_index_ta_aggressive.toml` — более агрессивный TA-портфель `USDRUBF + CNYRUBF + IMOEXF`
- `configs/server_tbank_cnyrubf_premium.toml` — legacy server-config для multi-futures paper-runtime, оставленный для архивного research
- `configs/server_tbank_stocks_intraday_300k.toml` — server paper-runtime под stock-trend контур через T-Bank API
- `configs/server_tbank_stocks_intraday_300k_focused.toml` — более лёгкий server runtime для ночной автономии на shortlist ликвидных бумаг с двумя быстрыми трендовыми TA-стилями и одним regime-hybrid
- `docs/architecture.md` — архитектура и логика работы
- `requirements-tbank.txt` — установка актуального SDK Т-Банка
- `tests/` — smoke/unit tests

## Быстрый старт

Создайте виртуальное окружение и установите актуальный SDK Т-Банка:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-tbank.txt
.\.venv\Scripts\python -m pip install -e .
```

`pip install -e .` теперь ставит локальные research-зависимости:

- `pandas`
- `pyarrow`
- `pandas-ta`

Создайте локальный `.env` c токеном и SSL-настройкой:

```dotenv
TBANK_ACCOUNT_ID=...
TBANK_ACCOUNT_NAME=Акции
SSL_TBANK_VERIFY=True
```

Основной runtime-конфиг уже умеет брать T-Bank токен из локального файла:
`%USERPROFILE%/Desktop/жрт новый про.txt`. Сам токен не хранится в репозитории.

Проверьте доступ к аккаунту:

```powershell
.\.venv\Scripts\python -m samosbor.cli --config configs/paper.toml accounts
```

Запустите бэктест на исторических данных T-Bank API:

```powershell
.\.venv\Scripts\python -m samosbor.cli --config configs/paper.toml backtest
```

Запустите один paper-cycle на актуальных данных:

```powershell
.\.venv\Scripts\python -m samosbor.cli --config configs/paper.toml paper-cycle
```

Соберите краткую сводку по фактическим paper-сделкам:

```powershell
.\.venv\Scripts\python -m samosbor.cli --config configs/server_tbank_stocks_intraday_300k_focused.toml paper-report --days 1
```

Постройте безопасную рекомендацию по часам входа из последних результатов:

```powershell
.\.venv\Scripts\python -m samosbor.cli --config configs/server_tbank_stocks_intraday_300k_focused.toml tune-entry-hours --days 45 --min-trades-per-hour 3
```

Постройте безопасную рекомендацию по параметрам входа/выхода из walk-forward:

```powershell
.\.venv\Scripts\python -m samosbor.cli --config configs/server_tbank_stocks_intraday_300k_focused.toml tune-strategy
```

Постройте отдельную рекомендацию по качеству выходов:

```powershell
.\.venv\Scripts\python -m samosbor.cli --config configs/server_tbank_stocks_intraday_300k_focused.toml tune-exits
```

Постройте рекомендацию по качеству входов уже из реальных paper-сделок:

```powershell
.\.venv\Scripts\python -m samosbor.cli --config configs/server_tbank_stocks_intraday_300k_focused.toml tune-entry-quality --lookback-trades 40 --min-trades 8
```

Постройте рекомендацию по временному отключению самых слабых тикеров из paper-статистики:

```powershell
.\.venv\Scripts\python -m samosbor.cli --config configs/server_tbank_stocks_intraday_300k_focused.toml tune-entry-symbols --days 45 --min-trades-per-symbol 4
```

Тот же autotune теперь умеет отдельно блокировать только слабый `long` или только слабый `short` по инструменту:

```powershell
.\.venv\Scripts\python -m samosbor.cli --config configs/server_tbank_stocks_intraday_300k_focused.toml tune-entry-symbols --days 45 --min-trades-per-direction-symbol 4
```

Если нужно сразу наполнить shadow feedback из недавней истории, а не ждать новые сделки:

```powershell
.\.venv\Scripts\python -m samosbor.cli --config configs/server_tbank_stocks_intraday_300k_focused.toml bootstrap-entry-feedback
```

Чтобы собрать производный runtime-конфиг из последних autotune-артефактов:

```powershell
.\.venv\Scripts\python -m samosbor.cli --config configs/server_tbank_stocks_intraday_300k_focused.toml refresh-effective-config
```

Чтобы принудительно прогнать весь ночной автономный цикл обучения:

```powershell
.\.venv\Scripts\python -m samosbor.cli --config configs/server_tbank_stocks_intraday_300k_focused.effective.toml nightly-autonomy --base-config configs/server_tbank_stocks_intraday_300k_focused.toml --effective-output configs/server_tbank_stocks_intraday_300k_focused.effective.toml
```

Чтобы локально поднять отдельный `samosbor` dashboard именно для этого paper-runtime:

```powershell
.\.venv\Scripts\python -m samosbor.dashboard --config configs/server_tbank_stocks_intraday_300k_focused.toml --effective-config configs/server_tbank_stocks_intraday_300k_focused.effective.toml --host 127.0.0.1 --port 8790
```

## Server Runtime

Для 24/7 серверного paper-режима подготовлены:

- runtime-конфиг [configs/server_tbank_stocks_intraday_300k_focused.toml](/D:/projects/samosbor/configs/server_tbank_stocks_intraday_300k_focused.toml)
- server scripts в [scripts/server](/D:/projects/samosbor/scripts/server)
- systemd units в [deploy/systemd](/D:/projects/samosbor/deploy/systemd)
- отдельный `samosbor-dashboard.service` для своей web-морды на порту `8790`

Логика расписания:

- systemd timer запускает `paper-cycle` каждые `5` минут в основную MOEX stock-сессию
- канонический focused runtime сейчас ориентирован на shortlist `SBER + GAZP + LKOH + TATN`, а широкий stock-universe оставлен для offline research
- входы по умолчанию открыты на основную intraday-сессию `10:00-17:59 MSK`, а nightly-autonomy уже может позже сузить часы по фактической статистике paper-сделок

Автообновление:

- `samosbor-updater.timer` проверяет GitHub каждые `15` минут
- при новом коммите делает `git pull --ff-only`, обновляет окружение, прогоняет unit tests и затем сам синхронизирует `systemd` units через `scripts/server/install-server.sh`
- `samosbor-dashboard.service` показывает только текущий `samosbor` paper-runtime, active overrides, open positions и autonomy artifacts, не смешивая их с legacy dashboards на сервере
- для stock paper-runtime через T-Bank API sizing идёт через обычный equity-based risk manager без реальных ордеров; futures-ветка остаётся в проекте как legacy-совместимость
- stock/paper-профили дополнительно ограничивают одну позицию через `max_position_exposure_ratio`, чтобы одна бумага не съедала весь виртуальный бюджет и оставляла место под другие акции
- `samosbor-daily-review.timer` после торговой сессии запускает единый `nightly-autonomy` цикл: daily analyze, entry restrictions, signal-feedback bootstrap, optimizer, walk-forward research, active-universe selection, Monte Carlo, strategy/exit tuning и финальную пересборку effective config
- daily review не меняет боевой TOML автоматически: он пишет артефакты в `runs/paper-reports`, `runs/autotune/entry-schedule`, `runs/autotune/entry-symbols`, `runs/autotune/entry-quality`, `runs/autotune/strategy` и `runs/autotune/exits`
- тот же nightly cycle теперь дополнительно пишет агрегированный summary в `runs/autotune/nightly-autonomy`
- `paper-cycle` теперь работает через производный `configs/server_tbank_stocks_intraday_300k_focused.effective.toml`, который каждый раз пересобирается именно из базового server TOML плюс последних autotune-артефактов и сохраняет `local-paper` / `allow_live_trading = false`
- autotune-артефакты теперь дополнительно изолируются по runtime-профилю на основе `state_path`, чтобы stock runtime не подхватывал stale candidates и ограничения от других paper-конфигов

Активная целевая функция autotune:

- рабочий target теперь привязан к прибыли `2000 RUB/день`
- в активном server-runtime и nightly-autonomy используется конкретная цель `2000 RUB/день`
- при виртуальном paper-капитале `300 000 RUB` это соответствует эквиваленту `40 000 RUB/мес` при `20` торговых днях, то есть целевым `13.333%` среднего месячного дохода
- последний stock-backtest на широком 10-бумажном universe оказался слабым, поэтому server runtime сейчас смещён к компактному shortlist, а не к широкой stock-сетке
- для exit autotune серверный research-grid теперь перебирает несколько соседних значений `atr_stop_multiple` и `reward_to_risk`, но применяет только candidate patch под guardrails
- для entry-quality autotune сделки теперь сохраняют `signal_strength`, а отдельный paper-feedback контур предлагает `min_signal_strength` только когда накоплено достаточно закрытых paper-сделок
- `tune-entry-symbols` использует `symbol_breakdown` из paper-report и может временно добавить слабые тикеры в `blocked_symbols`, не закрывая уже открытые позиции
- paper-cycle теперь ведёт отдельный shadow signal journal рядом со state-файлом и постепенно размечает сигналы как `take-profit`, `stop-loss` или `expired`
- `tune-entry-quality` сначала пытается учиться на resolved signal feedback, а если его ещё нет, честно откатывается к обычным closed trades
- `bootstrap-entry-feedback` позволяет безопасно прогреть этот journal на исторических свечах без отправки ордеров и без вмешательства в paper-позиции
- `tune-entry-symbols` теперь умеет не только полностью блокировать слабый тикер, но и отдельно добавлять `blocked_long_symbols` / `blocked_short_symbols`, если слабым оказался только один direction
- `tune-universe` теперь умеет сужать активный runtime-universe через `allowed_symbols`, если optimizer и последний walk-forward fold сходятся на более сильном поднаборе символов
- `refresh-effective-config` собирает следующую runtime-конфигурацию из последних entry/exit/strategy/schedule autotune результатов, не переписывая базовый серверный TOML
- новые strategy/exit/entry candidate changes теперь не применяются по одному сигналу: effective-config ждет как минимум `2` подряд подтверждающих daily-review артефакта, прежде чем включить их в active paper runtime
- тот же `refresh-effective-config` теперь пишет `rollback_guardrail` и при активных autotune-overrides может автоматически вернуться к базовому runtime-профилю, если недавнее paper-окно стало отрицательным или сработал drawdown halt
- nightly-autonomy теперь сначала делает `bootstrap-entry-feedback`, а уже потом считает entry-hours / entry-symbols / entry-quality, чтобы ночной цикл учился на максимально полном signal-feedback в ту же ночь

## Research На Данных С D:

Для локального архива свечей используется конфиг
[configs/local_pack_research.toml](/D:/projects/samosbor/configs/local_pack_research.toml).

Источник:

- `D:\3_pips_x2_project\moex_futures_data_pack`

Команды:

```powershell
.\.venv\Scripts\python -m samosbor.cli --config configs/local_pack_research.toml backtest
.\.venv\Scripts\python -m samosbor.cli --config configs/local_pack_research.toml optimize
.\.venv\Scripts\python -m samosbor.cli --config configs/local_pack_research.toml monte-carlo
.\.venv\Scripts\python -m samosbor.cli --config configs/local_pack_ta_research.toml optimize
.\\.venv\\Scripts\\python -m samosbor.cli --config configs/local_pack_cnyrubf_ta_walk_forward.toml walk-forward
```

Поддерживаемые стили стратегии:

- `sma_breakout` — исходный трендовый режим на SMA + breakout + ATR
- `ema_adx_macd` — режим на `pandas-ta` с EMA, ADX, RSI и MACD-фильтрами
- `ema_adx_donchian` — трендовый breakout-режим на `pandas-ta` с EMA, ADX, RSI и Donchian-каналами
- `adx_regime_hybrid` — гибридный режим: при сильном тренде работает как momentum/trend-following, а при слабом ADX переключается в mean-reversion
- `rsi_mean_reversion` — контртрендовый режим на возврат к средней через SMA + RSI-экстремумы; остаётся в широких stock research-профилях и может снова включаться в более глубокий nightly search
- в exit-логике доступны `trailing_profit_trigger_rub` и `trailing_profit_lock_ratio`: после заданной открытой прибыли по позиции стоп начинает подтягиваться и защищать часть уже заработанных рублей

Поля research-grid для TA-оптимизации:

- `strategy_styles`
- `require_breakout_values`
- `adx_min_values`
- `rsi_long_max_values`
- `rsi_short_min_values`
- `walk_forward_train_months`
- `walk_forward_test_months`
- `walk_forward_step_months`

Что делает локальный провайдер:

- читает `parquet` из `data_pack/candles_1m`
- использует уже готовые root-series вроде `GAZPF`, `IMOEXF`, `USDRUBF`
- агрегирует 1-минутные свечи в `hour/day/...`

## Последний Research-Результат

Ниже в этом разделе часть исторических Monte Carlo и walk-forward цифр относится к старому research-target `5%/мес`. Это архивные результаты прошлых прогонов. Активный autotune-контур server-runtime теперь ориентируется на `2000 RUB/день` (эквивалент `40 000 RUB/мес` при `20` торговых днях), а не на `5%`.

На локальном архиве с `D:` baseline-портфель `GAZPF + IMOEXF + USDRUBF` дал
отрицательный результат, но оптимизация нашла более устойчивый кандидат:

- инструмент: `USDRUBF`
- параметры: `fast=10`, `slow=40`, `atr_stop=2.0`, `rr=1.5`, `min_trend=0.004`
- backtest: `+10.017% total return`, `2.706% max drawdown`, `Sharpe 1.177`

Для этого кандидата есть отдельный конфиг:
[configs/local_pack_usdrubf_candidate.toml](/D:/projects/samosbor/configs/local_pack_usdrubf_candidate.toml)

И отдельные команды:

```powershell
.\.venv\Scripts\python -m samosbor.cli --config configs/local_pack_usdrubf_candidate.toml backtest
.\.venv\Scripts\python -m samosbor.cli --config configs/local_pack_usdrubf_candidate.toml monte-carlo
```

Новый TA-optimizer на локальном архиве дал более сильную версию кандидата на `CNYRUBF`:

- focused-search [configs/local_pack_ta_research.toml](/D:/projects/samosbor/configs/local_pack_ta_research.toml):
  `96` кандидатов за `~101` секунду, лучший результат у `CNYRUBF`
- базовый кандидат [configs/local_pack_cnyrubf_ta_candidate.toml](/D:/projects/samosbor/configs/local_pack_cnyrubf_ta_candidate.toml):
  `+23.073% total return`, `3.728% max drawdown`, `Sharpe 1.760`, `1.726% avg monthly return`
- Monte Carlo для базового кандидата:
  `99.4%` вероятность положительного результата за 12 месяцев, но `0.0%` вероятность достижения целевых `5%` среднего месячного дохода
- усиленный риск-профиль [configs/local_pack_cnyrubf_ta_aggressive.toml](/D:/projects/samosbor/configs/local_pack_cnyrubf_ta_aggressive.toml):
  `+37.889% total return`, `5.591% max drawdown`, `Sharpe 1.906`, `2.71% avg monthly return`
- Monte Carlo для усиленного профиля:
  `99.8%` вероятность положительного результата и `0.9%` вероятность выйти на целевые `5%` среднего месячного дохода
- ранее найденный агрессивный портфель [configs/local_pack_fx_index_ta_aggressive.toml](/D:/projects/samosbor/configs/local_pack_fx_index_ta_aggressive.toml):
  `USDRUBF + CNYRUBF + IMOEXF`, `+41.671% total return`, `10.77% max drawdown`, `3.09% avg monthly return`

Walk-forward для [configs/local_pack_cnyrubf_ta_walk_forward.toml](/D:/projects/samosbor/configs/local_pack_cnyrubf_ta_walk_forward.toml) показал более трезвую OOS-картину:

- базовый TA-кандидат: `7` fold-ов, `1.133% average OOS monthly return`, `57.143%` положительных месяцев, `7.97% compounded OOS return`
- aggressive risk-up профиль на маржинальном допущении: `0.841% average OOS monthly return`, `71.429%` положительных месяцев, `5.767% compounded OOS return`
- вывод: более агрессивное использование капитала выглядит лучше in-sample, но пока не приближает систему к устойчивым `5%` в месяц на OOS

## Профиль Брокера

- рабочее допущение проекта: счёт Т-Банка с тарифом `Premium`
- futures research-конфиги сейчас используют `commission_bps = 4.0` как paper-аппроксимацию под Premium-профиль
- использование маржинальных средств допустимо и уже отражается в risk-up конфигурациях через `max_gross_exposure > 1.0`
- серверный futures paper-runtime теперь подтягивает инструмент-специфичное ГО Т-Банка и использует его для sizing и проверки доступной маржи
- локальные backtest/research на parquet-архиве по-прежнему работают без онлайн-запроса к брокеру, поэтому для них нужна явная передача `initial_margin_*` в конфиг, если хочется симулировать ту же margin-модель
- стоимость переноса непокрытых позиций, overnight financing и прочие broker-specific carry costs пока не встроены
- официальные справочные страницы Т-Банка:
  [тарифы инвестора](https://www.tbank.ru/invest/help/brokerage/account/get-bs/tariff/)
  и [маржинальная торговля](https://www.tbank.ru/invest/help/brokerage/account/margin/advantages/)

Итог текущего этапа: исследовательский контур стал быстрее и честнее, потому что теперь включает walk-forward и безопасный autotune candidate flow. Старую цель `5%` в месяц система устойчиво не подтверждала; текущий активный target для runtime и autotune смещён к `2000 RUB/день`, а все research-артефакты теперь считают его месячный эквивалент автоматически.

## Безопасность

- Реальные заявки не отправляются.
- Режим `live` заблокирован кодом и требует отдельной явной доработки.
- Токены и локальное состояние исключены из git через `.gitignore`.

## Актуальный SDK

По состоянию на `2026-06-14` проект настроен под актуальный Python SDK
`t-tech-investments` из `opensource.tbank.ru`, а не под старый `tinkoff-investments`.
