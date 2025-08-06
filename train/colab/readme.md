
file structure:

├── colab/
│   ├── input/
│   │   └── examples.md                              # Your input file
│   ├── datasets/                                    # 👈 DATASETS STORED HERE
│   │   ├── sl2-query-api-dataset-chat/             # Hugging Face Dataset (Chat format)
│   │   │   ├── train/
│   │   │   ├── validation/
│   │   │   └── test/
│   │   ├── sl2-query-api-dataset-alpaca/           # Hugging Face Dataset (Alpaca format)  
│   │   │   ├── train/
│   │   │   ├── validation/
│   │   │   └── test/
│   │   ├── json_backups/                           # JSON backup files
│   │   │   ├── chat_train_sample.json
│   │   │   ├── chat_full_train.json
│   │   │   ├── alpaca_train_sample.json
│   │   │   └── alpaca_full_train.json
│   │   ├── dataset_statistics.json                # Dataset stats
│   │   └── README.md                               # Documentation 
│   ├── models/                            # Training outputs
│   └── logs/                              # Training logs
└── AI_Models/
    ├── SL2_QueryAPI/                       # Dedicated model folder
    │   ├── deepseek-coder-6.7b-sl2-query-api-20250105-143022/
    │   └── deepseek-coder-6.7b-sl2-query-api-info.json
    └── Backups/                            # Compressed archives
        └── deepseek-coder-6.7b-sl2-query-api-20250105-143022.zip



