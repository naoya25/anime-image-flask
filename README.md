2つのモデルファイルを作成し、`models/`フォルダ内に入れておく

- https://huggingface.co/jomcs/NeverEnding_Dream-Feb19-2023/blob/652befe4e73f0356c1ece31c647b20c959742f01/Anime%20Pastel%20Dream/animePastelDream_softBakedVae.safetensors
- https://civitai.com/models/4468/counterfeit-v30

使うモデルはなんでもいい

```
./
├ models/
│ ├ animePastelDream_softBakedVae.safetensors
│ └ Counterfeit-V3.0_fix_fp16.safetensors
└ ...
```

```
pip install -r requirements.txt
python appn.py
```
