

# ДЗ_2 по курсу звука ВШЭ


## Отчёт 

[Тут](https://colab.research.google.com/drive/1M2ugG7FoTzGi58mjKDbD7IRn-FeO1u0T?usp=sharing) загружен отчёт о ходе работ с примером запуска окружения, инференса и тренировки

[Ссылка](https://wandb.ai/konstantinator/hw2_asr) на wandb


## Результаты

| Method | test-clean CER| test-clean WER | test-other CER | test-other WER |
|--------|---------------|----------------|----------------|----------------|
| Argmax |        5.90   |     16.40      |     18.93      |     40.04      |
| LM Beamsearch |  4.94  |     11.96      |     17.72      |     33.06      |




## Пререквизиты
Если хотите воспроизвести эксперименты где-то кроме гугл колаба
то вам потребуется python3.10 c venv и curl


## Установка окружения и загрузка весов модели

```
git clone https://github.com/konstantinator/hw2_sound.git
cd hw2_sound 
python -m venv my_venv 
. ./my_venv/bin/activate
pip install -q -r requirements.txt
curl -L $(yadisk-direct https://disk.yandex.ru/d/UwUd74Qzzl6H-g) -o ./pretrained_lm/model_best.pth
```

## Инференс

Инференс на чистом тесте (всё работает из коробки)
```
python inference.py -cn=inference_clean
```
Инференс на шумном тесте (всё работает из коробки)
```
python inference.py -cn=inference 
```

## Трейн

Воспроизведение обучения (всё работает из коробки, нужно будет войти в свой wandb аккаунт по ходу работы)

Запускаем обучение на чистом сете
```
python train.py -cn=finetune_clean_wo_bpe
```
Лучшую модель переложить по пути

```/pretrained_lm/wo_bpe/model_best.pth```

Запускаем обучение на шумном сете на основе лучшей модели с предыдщуего шага
```
python train.py -cn=finetune_clean_wo_bpe_other
```
Лучшую модель переложить по пути

```/pretrained_lm/model_best.pth```

Готово!
