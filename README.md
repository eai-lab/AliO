# [NeurIPS 2025] AliO: Output Alignment Matters in Long-Term Time Series Forecasting
<img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white"/> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white"/> <img src="https://img.shields.io/badge/YouTube-FF0000?style=flat&logo=YouTube&logoColor=white"/> <img src="https://img.shields.io/badge/NeurIPS25:119426-3776AB?style=flat"/>

Official implementation of <a href="https://neurips.cc/virtual/2025/poster/119426">AliO</a>

# How to use
* Install AliO using
```
pip install -e .
```

* Import AliO
```
from alio import AliO

alio_loss = AliO(
    num_samples=2,
    lag=1,
    time_loss='mse',
    freq_loss='mse'
)
```

# Implementation repository
* <a href="examples/CycleNet">CycleNet</a>
* <a href="examples/TimesNet">TimesNet</a>
* <a href="examples/iTransformer">iTransformer</a>
* <a href="examples/GPT2">GPT4TS</a>
* <a href="examples/PatchTST">PatchTST</a>
* <a href="examples/DLinear">DLinear</a>
* <a href="examples/Autoformer">Autoformer</a>
