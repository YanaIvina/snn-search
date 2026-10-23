# SNN Codegen — README

## Быстрый старт

```bash
# Сгенерировать код и сохранить в файл
python snn_codegen.py config.json model.py

# Сгенерировать код + диаграмму архитектуры
python snn_codegen.py config.json model.py arch.png

# Только диаграмма
python snn_codegen.py config.json --visualize arch.png

# Вывести код в консоль
python snn_codegen.py config.json
```

***

## Структура JSON конфига

```json
{
  "model_name": "MySNN",

  "nodes": [
    { "id": "inp",  "type": "input",     "size": 4096 },
    { "id": "res1", "type": "reservoir", "size": 512  },
    { "id": "res2", "type": "reservoir", "size": 256  },
    { "id": "out",  "type": "output",    "size": 30   }
  ],

  "edges": [
    { "from": "inp",  "to": "res1" },
    { "from": "res1", "to": "res1" },
    { "from": "res1", "to": "res2" },
    { "from": "res2", "to": "res2" },
    { "from": "res2", "to": "out"  }
  ],

  "neuron_params": {
    "vth":           1.0,
    "dt":            1.0,
    "fraction_long": 0.5,
    "tau_short":     [0,  20 ],
    "tau_long":      [80, 100]
  },

  "surrogate_params": {
    "h": 0.15,
    "l": 0.5,
    "s": 6
  },

  "conv_frontend": {
    "enabled":     true,
    "in_channels": 1,
    "channels":    [32, 64, 128],
    "use_bn":      true,
    "hw":          [128, 32],
    "pool":        "avg"
  }
}
```

***

## Поля конфига

### `nodes`
| Поле | Тип | Описание |
|---|---|---|
| `id` | string | Уникальное имя узла |
| `type` | `"input"` / `"reservoir"` / `"output"` | Тип узла |
| `size` | int | Размер (нейронов или признаков) |
| `tau_short` | `[min, max]` | Переопределить τ короткое для этого резервуара |
| `tau_long` | `[min, max]` | Переопределить τ длинное для этого резервуара |
| `fraction_long` | float | Доля нейронов с длинным τ |

### `edges`
| Поле | Описание |
|---|---|
| `"from" == "to"` | Рекуррентная петля `W_rec` |
| `"from": "inp", "to": "res"` | Проекция входа в резервуар |
| Несколько `"to": "out"` | Конкатенация → линейный выход |
| `weight_init` | Опционально: `{"type": "uniform", "low": -0.5, "high": 0.5}` |

Доступные инициализации: `uniform`, `normal`, `kaiming_uniform`, `kaiming_normal`, `xavier_uniform`, `xavier_normal`, `orthogonal`, `sparse`, `zeros`.

### `conv_frontend`
| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `enabled` | bool | `false` | Включить CNN front-end |
| `in_channels` | int | `1` | Каналов на входе (1 = моно мел-спектрограмма) |
| `channels` | `[int, ...]` | `[32,64,128]` | Каналы по блокам (количество блоков = длина списка) |
| `use_bn` | bool | `true` | BatchNorm после Conv2d |
| `hw` | `[H, W]` или int | `[64,64]` | Размер входного изображения до первого блока |
| `pool` | `"avg"` / `"max"` | `"avg"` | Тип пулинга в каждом блоке |

> Каждый блок уменьшает пространственный размер вдвое (`Pool2d(2)`).  
> Выходной размер CNN = `channels[-1] × (H / 2^N) × (W / 2^N)`, где N — число блоков.

***

## Зависимости

```bash
pip install torch scipy networkx matplotlib
```
