from sklearn.model_selection import KFold
import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
def train_and_save_single_fold(
    model_name: str, 
    train_dataset, 
    eval_dataset, 
    fold_num: int,
    base_model_dir: str # Directorio base para el modelo, ej: ../modelos_final/model_dccuchile...
):
    """
    Entrena un único pliegue (fold), guarda el modelo resultante en su propia carpeta
    y devuelve su F1-score.
    """
    print(f"\n--- 🚀 Iniciando Entrenamiento FOLD {fold_num} ---")
    
    # --- Cargar Modelo y Tokenizer (se reinician en cada fold) ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        f1 = f1_score(labels, predictions, average="macro")
        return {"f1_macro": f1}

    # --- Directorios de Salida ---
    # Carpeta para los checkpoints temporales de este fold
    training_output_dir = f"./results/{base_model_dir.split('/')[-1]}/fold_{fold_num}"
    # Carpeta final donde se guardará el mejor modelo de este fold
    final_fold_path = os.path.join(base_model_dir, f"fold_{fold_num}")
    
    if not os.path.exists(final_fold_path):
        os.makedirs(final_fold_path)

    training_args = TrainingArguments(
        output_dir=training_output_dir,
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        eval_strategy="epoch",
        save_strategy="epoch",
        greater_is_better=True,
        report_to="none" # Desactiva el login a WandB/etc.
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    
    # --- Entrenar y Evaluar ---
    trainer.train()
    eval_results = trainer.evaluate()
    
    # --- Guardar el mejor modelo y el tokenizador en la carpeta final del fold ---
    print(f"Guardando el mejor modelo del Fold {fold_num} en: '{final_fold_path}'")
    trainer.save_model(final_fold_path)
    tokenizer.save_pretrained(final_fold_path)
    
    return eval_results['eval_f1_macro']
def run_kfold_training_workflow(
    df: pd.DataFrame,
    model_name: str,
    text_column: str = 'structured_text',
    label_column: str = 'Polarity',
    output_base_dir: str = '../modelos_final',
    n_splits: int = 5
):
    """
    Orquesta el entrenamiento K-Fold completo:
    1. Prepara los datos.
    2. Itera a través de K pliegues.
    3. Llama a la función de entrenamiento para cada pliegue.
    4. Guarda cada modelo de pliegue en una subcarpeta dedicada.
    5. Reporta el rendimiento promedio final.
    """
    print(f"🚀 Iniciando Proceso Completo de K-Fold (K={n_splits}) para el modelo: {model_name}")
    
    # --- Preparación de Datos (se hace una sola vez) ---
    df_processed = df.rename(columns={text_column: 'text', label_column: 'label'})
    df_processed['label'] = df_processed['label'].apply(lambda x: int(x) - 1)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
    
    full_dataset = Dataset.from_pandas(df_processed[['text', 'label']])
    tokenized_dataset = full_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    
    # --- Preparar K-Fold y Directorios ---
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_f1_scores = []
    
    model_output_dir_name = model_name.replace("/", "_")
    final_model_base_path = os.path.join(output_base_dir, model_output_dir_name)
    print(f"Todos los folds para este modelo se guardarán dentro de: '{final_model_base_path}'")

    # --- Bucle Principal de K-Fold ---
    for i, (train_idx, val_idx) in enumerate(kf.split(tokenized_dataset)):
        train_fold = tokenized_dataset.select(train_idx)
        val_fold = tokenized_dataset.select(val_idx)
        
        f1_score_fold = train_and_save_single_fold(
            model_name=model_name,
            train_dataset=train_fold,
            eval_dataset=val_fold,
            fold_num=i + 1,
            base_model_dir=final_model_base_path # Pasamos la ruta base específica del modelo
        )
        
        all_f1_scores.append(f1_score_fold)

    # --- Resultados Finales ---
    mean_f1 = np.mean(all_f1_scores)
    std_f1 = np.std(all_f1_scores)

    print("\n" + "="*60)
    print("--- RESUMEN FINAL DE LA VALIDACIÓN CRUZADA ---")
    print(f"Modelo: {model_name}")
    print(f"F1-Scores por Fold: {[round(f, 4) for f in all_f1_scores]}")
    print(f"Rendimiento Promedio (Mean F1-Macro): {mean_f1:.4f} ± {std_f1:.4f}")
    print("="*60)

    return {"mean_f1": mean_f1, "std_f1": std_f1}


def train_sentiment_model_basic_vanilla(
    df: pd.DataFrame,
    model_name: str,
    text_column: str = 'structured_text',
    label_column: str = 'Polarity',
    output_base_dir: str = '../modelos_final'
):
    """
    Entrena un modelo Transformer con una división simple de train/validation (80/20).
    No usa K-Fold, ni Optuna, ni pesos de clase.
    """
    print(f"🚀 Iniciando entrenamiento BÁSICO para el modelo: {model_name}")

    # --- 1. Preparación de Datos ---
    print("Paso 1/6: Preparando los datos...")
    df_processed = df.rename(columns={text_column: 'text', label_column: 'label'})
    df_processed['label'] = df_processed['label'].apply(lambda x: int(x) - 1)
    
    dataset = Dataset.from_pandas(df_processed[['text', 'label']])
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    print(f"Datos divididos: {len(train_dataset)} para entrenamiento, {len(eval_dataset)} para evaluación.")

    # --- 2. Cargar Tokenizador y Modelo ---
    print("Paso 2/6: Cargando tokenizador y modelo base...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    # --- 3. Definir Métricas ---
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        f1 = f1_score(labels, predictions, average="macro")
        return {"f1_macro": f1}

    # --- 4. Argumentos de Entrenamiento ---
    print("Paso 3/6: Configurando los argumentos de entrenamiento...")
    model_output_dir_name = model_name.replace("/", "_")
    training_output_dir = f"./results_{model_output_dir_name}"

    training_args = TrainingArguments(
        output_dir=training_output_dir,
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f'./logs_{model_output_dir_name}',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none"
    )

    # --- 5. Entrenar el Modelo ---
    print("Paso 4/6: ¡Iniciando el entrenamiento! 🏋️")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    trainer.train()

    # --- 6. Guardar el Modelo Final ---
    print("Paso 5/6: Entrenamiento completado. Guardando el mejor modelo...")
    final_model_path = os.path.join(output_base_dir, model_output_dir_name)
    
    if not os.path.exists(final_model_path):
        os.makedirs(final_model_path)

    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    print(f"Paso 6/6: ✅ ¡Modelo y tokenizador guardados exitosamente en '{final_model_path}'!")
    
    return trainer