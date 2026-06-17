# FER Challenge — Facial Expression Recognition CNN

### კონკურსის მიმოხილვა

FER Challenge-ის მიზანია 48×48 პიქსელის შავთეთრ სახის სურათებზე 7 ემოციის კლასიფიკაცია: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral. ამოცანა ფასდება accuracy მეტრიკით.

---

## რეპოზიტორიის სტრუქტურა

```
FER-Challenge/
│
├── model_experiment_CNN.ipynb   ← EDA, preprocessing, 3 არქიტექტურა, ექსპერიმენტები
└── README.md
```

---

## ფაილების აღწერა

| ფაილი | აღწერა |
|---|---|
| `model_experiment_CNN.ipynb` | CNN notebook — EDA, data preprocessing, TinyCNN / MediumCNN / DeepCNN არქიტექტურები, ტრენინგი, W&B-ზე დალოგვა და არქიტექტურების შედარება |

---

## მიდგომა

გამოვიყენე  სამი CNN არქიტექტურა - **Tiny → Medium → Deep** — რათა ვაჩვენო underfitting მოდელებიდან overfitting-მდე მისვლა და შემდეგ ოპტიმალური ბალანსი ვიპოვო.

ყველა ექსპერიმენტი დავლოგე Weights & Biases-ზე.

---

## EDA — მონაცემების ანალიზი

### 1. კლასების განაწილება

მონაცემები **არ არის ბალანსირებული** — `Disgust` კლასს სულ 436 სურათი აქვს, `Happy`-ს კი 7215.

<img width="989" height="390" alt="image" src="https://github.com/user-attachments/assets/e377912e-7d24-441d-a5c2-70faeb7b87bc" />


ამ პრობლემის გადასაჭრელად გამოვიყენე `compute_class_weight('balanced')`  ანუ CrossEntropyLoss-ს გადაეცა class weights, რაც იშვიათ კლასებს მეტ წონას ანიჭებს.

### 2. პიქსელების განაწილება

მთლიანი pixel value განაწილება skewed-ია, პიკი 150 თანაა სადღაც. ემოციების მიხედვით pixel-ის საშუალო მნიშვნელობები ძალიან ახლოსაა ერთმანეთთან, რაც ნიშნავს, რომ global brightness ემოციის კარგი indicator არ არის. მოდელმა სტრუქტურული ნიშნები უნდა ისწავლოს.

<img width="1289" height="390" alt="image" src="https://github.com/user-attachments/assets/34d3b4fe-c556-4029-8ed8-e0d3860e42a5" />


---

## Data Preprocessing

```
train/val split: 80% / 20% (stratified, random_seed=42)
```

- **Normalization** — `transforms.Normalize([0.5], [0.5])` პიქსელებს [-1, 1]-ში მოაქვს
- **Augmentation** — `RandomHorizontalFlip`, `RandomRotation(10°)`, `RandomAffine(translate=0.1)` 
- **WeightedRandomSampler** — class imbalance-ის გადასაჭრელად 

---

## Model Architectures

| არქიტექტურა | პარამეტრები | მიზანი |
|---|---|---|
| **TinyCNN** | ~148 K | Underfitting-ის დემონსტრაცია |
| **MediumCNN** | ~1.2 M | Overfitting-ის ჩვენება და Backward Checking-ით გამოსწორება |
| **DeepCNN** | ~2.8 M | საუკეთესო არქიტექურა + Dropout2d + Augmentation |

### TinyCNN
2 conv layer (1→8→16 channels), 64-unit FC head. არაა საკმარისი 7-კლასიანი ამოცანისთვის.

### MediumCNN
3 conv block (1→32→64→128), BatchNorm, ~1.2M params. საკმარისი capacity, მაგრამ რეგულარიზაციის გარეშე ოვერფიტდება.

### DeepCNN
double-conv blocks (32→64→128), **Dropout2d** feature map-ებზე, 512-unit FC head. ყველაზე ძლიერი არქიტექტურა.

---

## ტრენინგი და ექსპერიმენტები

- **Optimizer:** AdamW (`weight_decay=1e-4`), ასევე გაიტესტა SGD Nesterov
- **Scheduler:** `ReduceLROnPlateau(mode='max', factor=0.5, patience=5)`
- **Loss:** `CrossEntropyLoss(weight=class_weights, label_smoothing=...)`
- **Tracking:** W&B-ზე ყველა epoch-ზე ილოგება `train_acc`, `val_acc`, `overfit_gap`, `lr`

---

## Cross-Architecture შედარება

<img width="1489" height="523" alt="image" src="https://github.com/user-attachments/assets/eec64abb-5875-4a0a-82d5-ae00d761384f" />


**საუკეთესო შედეგები:**

| Run | Val Accuracy | Overfit Gap |
|---|---|---|
| `DeepCNN_lr0.001_drop0.25_bs64_augTrue_ep80` | **0.659** | ~0.03 |
| `DeepCNN_lr0.001_drop0.25_bs64_augFalse` | 0.628 | ~0.30 |
| `DeepCNN_drop0.25_augTrue_seed123` | 0.628 | ~0.03 |
| `MediumCNN_lr0.001_drop0.4_augTrue` | 0.624 | ~0.04 |
| `DeepCNN_drop0.25_augTrue_seed999` | 0.620 | ~0.03 |

**მთავარი დაკვირვება:** Augmentation-ის გარეშე DeepCNN-ს overfit gap 30%-ს აღწევს ამიტომ augmentation ყველაზე ეფექტური რეგულარიზაციაა ამ დატაზე. Epoch-ების 80-მდე გაზრდამ ვალიდაციის accuracy 65.9%-მდე გაზარდა.

---

## Confusion Matrix

<img width="641" height="389" alt="image" src="https://github.com/user-attachments/assets/d73c5ce8-59fe-4eaa-8fe7-2e12b3972506" />


Confusion Matrix-იდან ჩანს, რომ:
- **Happy** ყველაზე კარგად კლასიფიცირდება რადგან ყველაზე დიდი და ბალანსირებული კლასია
- **Disgust** ყველაზე ძნელია რადგან სულ 436 სურათია და ხშირად ერევა
- **Fear / Sad / Angry** ხშირად ერევა ერთმანეთში რადგან ემოციურად ეს ემოციები ძაან გვანან ერთმანეთს

---

## Overfitting ანალიზი

**Backward Checking სტრატეგია MediumCNN-ზე:**

| Config | Val Acc | Overfit Gap |
|---|---|---|
| `dropout=0.0` | ~0.588 | მაღალი |
| `dropout=0.3` | ~0.588 | გამოსწორდა |
| `dropout=0.5` | ~0.583 | ზედმეტი regularization |
| `dropout=0.4 + augment` | ~0.580 | ბალანსი |

Dropout-ის პროგრესიული გაზრდა overfit-ს ამცირებს, მაგრამ ძალიან მაღალი dropout (0.5+) accuracy-ს ასევე ამცირებს ამიტომ ოპტიმალური წერტილი dropout=0.25-0.3 + augmentation-ია.

### Seed Variance Check

გავტესტე DeepCNN სამ სხვადასხვა seed-ზე (42, 123, 999) ერთი და იმავე პარამეტრებით:

| Seed | Val Acc |
|---|---|
| 42 | 0.618 |
| 123 | 0.628 |
| 999 | 0.620 |

სტანდარტული დევიაცია მცირეა ანუ შედეგები სტაბილურია.

---

## W&B ექსპერიმენტები

ყველა run დარეგისტრირებულია: https://wandb.ai/sansi23_team/FER-Challenge

თითოეულ run-ში დაილოგა:
- ყველა ჰიპერპარამეტრი
- Train / Val Accuracy და Loss (ყოველ epoch-ზე)
- Overfit Gap
- Learning Rate 
- Confusion Matrix

# W&B რეპორტი

https://wandb.ai/sansi23_team/FER-Challenge/reports/FER2013---VmlldzoxNzI1NjkzOQ

---

## გამოცდილება

- **Class Imbalance** — weighted loss და WeightedRandomSampler ორივე ვცადე და weighted loss უფრო სტაბილური აღმოჩნდა
- **Dropout2d vs Dropout** — Dropout2d conv. ფენებისთვის უფრო ეფექტურია ვიდრე სტანდარტული Dropout
- **Augmentation > Regularization** — ამ ზომის dataset-ზე augmentation overfit-ს ყველაზე ეფექტურად ამცირებს
- **Forward/Backward Checking** — ჯერ underfitting-ის დემონსტრაცია TinyCNN-ით, შემდეგ overfitting-ის დემონსტრაცია MediumCNN, dropout=0-ით, შემდეგ backward checking regularization-ით
