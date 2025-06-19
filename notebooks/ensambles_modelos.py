import pandas as pd
import numpy as np
import os
import glob
import torch
from scipy.special import softmax
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
def get_predictions_from_model(df_test: pd.DataFrame, model_path: str) -> np.ndarray:
    """Carga un modelo guardado y devuelve sus predicciones (logits)."""
    print(f"🧠 Cargando y prediciendo con: {model_path}")
    if not os.path.exists(model_path):
        print(f"  -> ⚠️ ¡Atención! No se encontró. Saltando.")
        return None
    
    # Esta configuración es para evitar warnings de paralelismo que a veces saturan la salida
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Preparar el dataset de prueba
    test_dataset = Dataset.from_pandas(df_test.rename(columns={'structured_text': 'text'}))
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Usar el Trainer para predecir eficientemente
    trainer = Trainer(model=model, args=TrainingArguments(output_dir='./temp_results', per_device_eval_batch_size=64))
    predictions = trainer.predict(tokenized_test_dataset)
    
    os.environ["TOKENIZERS_PARALLELISM"] = "true" # Restaurar
    print(f"  -> ✅ Predicciones obtenidas.")
    return predictions.predictions

def prepare_test_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara el DataFrame de prueba creando la columna 'structured_text'."""
    print("Preparando DataFrame de prueba...")
    df['structured_text'] = df.apply(
        lambda row: f"tipo: {str(row['Type']).lower()}. pueblo: {str(row['Town']).lower()}. reseña: {str(row['Review']).lower()}",
        axis=1
    )
    print("Columna 'structured_text' creada.")
    return df

def predict_single_model(df_test: pd.DataFrame, model_path: str, output_filename: str = 'submission_single.csv'):
    """
    Genera un archivo de sumisión usando un único modelo.
    """
    print("\n" + "="*50)
    print(f"--- Iniciando Predicción con un Único Modelo ---")
    print(f"Modelo a usar: {model_path}")
    print("="*50)
    
    logits = get_predictions_from_model(df_test, model_path=model_path)
    
    if logits is not None:
        final_predictions = np.argmax(logits, axis=1)
        
        submission_df = pd.DataFrame({
            'ID': df_test['ID'],
            'label': final_predictions + 1
        })
        
        print(f"\n🎉 ¡Predicción completada! Archivo guardado como '{output_filename}'")
        return submission_df
    else:
        print("\n❌ Error: No se pudo cargar el modelo o realizar la predicción.")
        
def ensemble_single_models(df_test: pd.DataFrame, base_model_dir: str, model_folders: list, output_filename: str = 'submission_simple_ensemble.csv'):
    """
    Realiza un ensamble promediando las predicciones de varios modelos.
    Asume una estructura de un modelo por carpeta.
    """
    print("\n" + "="*50)
    print(f"--- Iniciando Ensamble Simple (Sin Folds) ---")
    print(f"Modelos a ensamblar: {model_folders}")
    print("="*50)

    model_paths = [os.path.join(base_model_dir, folder) for folder in model_folders]
    all_logits = [get_predictions_from_model(df_test, path) for path in model_paths]
    all_logits = [logits for logits in all_logits if logits is not None] # Filtrar modelos no encontrados

    if not all_logits:
        print("\n❌ Error: No se pudieron obtener predicciones de ningún modelo. Abortando.")
        return

    print(f"\n✨ Ensamblando los resultados de {len(all_logits)} modelos...")
    all_probs = [softmax(logits, axis=1) for logits in all_logits]
    ensembled_probs = np.mean(all_probs, axis=0)
    final_predictions = np.argmax(ensembled_probs, axis=1)

    submission_df = pd.DataFrame({'ID': df_test['ID'], 'label': final_predictions + 1})

    
    print(f"\n🎉 ¡Ensamble completado! Archivo guardado como '{output_filename}'")
    return submission_df
    
def ensemble_kfold_models(df_test: pd.DataFrame, base_model_dir: str, model_base_folders: list, output_filename: str = 'submission_kfold_ensemble.csv'):
    """
    Realiza un "ensamble de ensambles" a partir de modelos entrenados con K-Fold.
    """
    print("\n" + "="*60)
    print(f"--- Iniciando Ensamble K-Fold ---")
    print(f"Modelos base a ensamblar: {model_base_folders}")
    print("="*60)

    final_ensemble_logits = []

    for base_folder in model_base_folders:
        base_path = os.path.join(base_model_dir, base_folder)
        model_name_short = os.path.basename(base_path)
        print(f"\nProcesando modelo base: {model_name_short}")
        
        fold_paths = sorted(glob.glob(os.path.join(base_path, 'fold_*')))
        if not fold_paths:
            print(f"  -> ⚠️ No se encontraron folds. Saltando.")
            continue
            
        logits_from_folds = [get_predictions_from_model(df_test, path) for path in fold_paths]
        logits_from_folds = [l for l in logits_from_folds if l is not None]

        if logits_from_folds:
            print(f"  -> Ensamblando los {len(logits_from_folds)} folds para {model_name_short}...")
            avg_logits_for_model = np.mean(logits_from_folds, axis=0)
            final_ensemble_logits.append(avg_logits_for_model)
            print(f"  -> ✅ Predicción robusta para {model_name_short} generada.")

    if not final_ensemble_logits:
        print("\n❌ Error: No se pudieron obtener predicciones finales de ningún modelo base. Abortando.")
        return

    print(f"\n✨ Ensamblando las {len(final_ensemble_logits)} predicciones robustas de cada tipo de modelo...")
    all_final_probs = [softmax(logits, axis=1) for logits in final_ensemble_logits]
    ensembled_probs = np.mean(all_final_probs, axis=0)
    final_predictions = np.argmax(ensembled_probs, axis=1)

    submission_df = pd.DataFrame({'ID': df_test['ID'], 'label': final_predictions + 1})
    
    print(f"\n🎉 ¡Ensamble K-Fold completado! Archivo guardado como '{output_filename}'")

    return submission_df