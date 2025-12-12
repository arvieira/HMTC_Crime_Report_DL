import pandas as pd
import numpy as np
import gc
import re
import requests
import random
import torch
import time

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from IPython.display import Image, display, Markdown

from functools import partial
from typing import Literal, List
from pydantic import BaseModel, Field, ValidationError

from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from server.LLM_server import LLMServer
from server.generation_response import GenerationResponse

from transformers import logging
logging.set_verbosity_error()


tqdm.pandas()
MODEL = 'llama3.1'

# Meu servidor do llama3.1 com transformers
SERVER_URL = "http://localhost:8053/v1/completions"
SERVER_VERIFICATION_URL = "http://localhost:8053/ready"
SERVER_TIMEOUT = 1000
SERVER_INTERVAL = 1
MAX_TOKENS = 100
TEMPERATURE = 0.7
MAX_RETRIES = 5
SEED = 12345


# STATE DEFINITION
# Overall State
class OverallState(TypedDict):
    input_dataset: pd.DataFrame
    llm_predictions_dataset: pd.DataFrame
    ml_predictions_dataset: pd.DataFrame
    classification_dataset: pd.DataFrame
    service: LLMServer

# Input State
class InputState(TypedDict):
    input_dataset: pd.DataFrame

# Output State
class OutputState(TypedDict):
    classification_dataset: pd.DataFrame
# STATE DEFINITION


# PREPROCESSING
# Auxiliary function
def generate_embedding(row, embedding_model, embedding_columns):
    embedding = embedding_model.encode(row['DINAMICA'], convert_to_tensor=False, normalize_embeddings=True)
    return pd.concat([row, pd.Series(embedding, index=embedding_columns)])

# Node function
def preprocessing(state: InputState) -> OutputState:
    print("NODE Preprocessing:")
    input_dataset = state['input_dataset']

    embedding_model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')    
    embedding_columns = [f"Embedding_{i}" for i in range(1024)]
    input_dataset = input_dataset.progress_apply(generate_embedding, embedding_model=embedding_model, embedding_columns=embedding_columns, axis=1)
    
    ml_predictions_dataset = input_dataset[embedding_columns + ['N3']].copy()
    llm_predictions_dataset = input_dataset[['DINAMICA', 'N3']].copy()

    del input_dataset
    del embedding_model
    del embedding_columns
    gc.collect()
    torch.cuda.empty_cache()

    print("NODE Preprocessing: Concluded")
    
    return {
        'ml_predictions_dataset': ml_predictions_dataset,
        'llm_predictions_dataset': llm_predictions_dataset
    }
# PREPROCESSING


# LLM PREDICTION
def start_llm_server(state: OverallState) -> OverallState:
    service = LLMServer()
    service.start()

    return {
        'service': service
    }
    
def stop_llm_server(state: OverallState) -> OverallState:
    service = state['service']
    service.stop()

# Auxiliary function
def row_llm_prediction(row, model_name='llama', column_name='column', classes=None):
    # Se não passou a lista de classes é pq estou prevendo no N2 ou N3
    # Preciso pegar a lista a partir do que já foi predito antes
    prob_name = f'N1 {model_name} Pred Proba'
    if classes is None: 
        if column_name == f'N2 {model_name} Classification':
            prob_name = f'N2 {model_name} Pred Proba'
            title = f'N1 {model_name} Classification'
            classes = dm[dm['N1'] == row[title]]['N2'].unique()
        elif column_name == f'N3 {model_name} Classification':
            prob_name = f'N3 {model_name} Pred Proba'
            title = f'N2 {model_name} Classification'
            classes = dm[dm['N2'] == row[title]]['N3'].unique()
        

    counter = 0
    data = None
    while not data and counter <= MAX_RETRIES:
        counter += 1
        request_id = random.randint(0, 100_000_000)

        messages=[
            {
                'role': 'system',
                'content': (
                    f"Você é um inspetor de polícia da Polícia Civil do Estado do Rio de Janeiro.\n"
                    f"Você é especialista no código penal brasileiro e em classificação de narrativas em tipos de delitos.\n"
                    f"Será provido para você narrativas de crimes ocorridos e você deve classificá-las em uma das classes providas.\n"
                )
            },
            {
                'role': 'user',
                'content': (
                    f"Responda a instrução a seguir em formato JSON."
                    'Exemplo de resposta:\n{\n"delito": "..."\n}\n\n'
                    f"Classifique a seguinte narrativa em um dos tipos de delitos providos a seguir:\n\n"
                    f"NARRATIVA: {row['DINAMICA']}\n\n"
                    f"TIPOS DE DELITOS:\n{classes}\n\n"
                    f"A sua resposta deve ser em formato JSON com um campo chamado 'delito' com o valor da classificação."
                )
            }
        ]
        
        payload = {
            "request_id": request_id,
            "prompt": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "model": model_name
        }
        
        response = requests.post(SERVER_URL, json=payload)

        # Valida que o servidor respondeu com sucesso e que é o meu json correto da API
        if response.status_code == 200:
            try:
                data = GenerationResponse(**response.json())

                # Valida que essa é a resposta dessa requisição
                if request_id != data.response_id:
                    print("NODE: LLM Row Prediction - bad response id")
                    data = None

                # Se já veio assim do servidor é pq o servidor já tentou MAX_RETRIES lá e deu zebra e é melhor parar de tentar
                # Após sair desse if, ele entra no próximo pq a data.classification não estará entre uma das classes possíveis
                if data.classification == 'Sem Classificação':
                    count = MAX_RETRIES

                # Valida que o LLM respondeu uma das classes disponíveis para ele
                if data.classification not in classes:
                    print(f"NODE: LLM Row Prediction - bad classification - {model_name}")
                    data = None
                    row[column_name] = 'Sem Classificação'
                    row[prob_name] = 0

                # Já realizou todas as validações, logo posso atribuir aos valores da linha
                if data:
                    row[column_name] = data.classification
                    row[prob_name] = data.class_prob
                    
            except ValidationError as error:
                print("NODE: LLM Row Prediction - bad json")
                data = None
        else:
            print(f"{response.status_code} -> {response.text}")
            data = None        
            
    return row
    

# Node function
def llm_prediction(state: OverallState) -> OverallState:
    print("NODE LLM Prediction:")

    print("Aguardando servidor iniciar")
    server_ok = False
    start_time = time.time()
    while time.time() - start_time < SERVER_TIMEOUT and not server_ok:
        try:
            response = requests.get(SERVER_VERIFICATION_URL)
            if response.status_code == 200:
                print("Servidor está pronto.")
                server_ok = True
        except requests.exceptions.ConnectionError:
            print(".", end="")
            time.sleep(SERVER_INTERVAL)
    
    if server_ok:
        input_dataset = state['llm_predictions_dataset']
        input_dataset = input_dataset.progress_apply(row_llm_prediction, axis=1, model_name='llama', column_name='N1 llama Classification', classes=dm['N1'].unique())
        input_dataset = input_dataset.progress_apply(row_llm_prediction, axis=1, model_name='llama', column_name='N2 llama Classification')
        input_dataset = input_dataset.progress_apply(row_llm_prediction, axis=1, model_name='llama', column_name='N3 llama Classification')

        input_dataset = input_dataset.progress_apply(row_llm_prediction, axis=1, model_name='fine_llama', column_name='N1 fine_llama Classification', classes=dm['N1'].unique())
        input_dataset = input_dataset.progress_apply(row_llm_prediction, axis=1, model_name='fine_llama', column_name='N2 fine_llama Classification')
        input_dataset = input_dataset.progress_apply(row_llm_prediction, axis=1, model_name='fine_llama', column_name='N3 fine_llama Classification')
        print("NODE LLM Prediction: Concluded")
    else:
        print("NODE LLM Prediction: Skipping. Server not OK!")
    
    return {
        "llm_predictions_dataset": input_dataset,
    }
# LLM PREDICTION


# ML PREDICTION
# Auxiliary functions
def load_model(path):
    model = XGBClassifier()
    model.load_model(path)
    
    return model

def read_encoder(path):
    encoder = LabelEncoder()
    encoder.classes_ = np.load(path, allow_pickle=True)
    
    return encoder

xgboost_model_db = {
    "N0_root": {
        "model": load_model("./Models/N0/root/root.model"),
        "encoder": read_encoder('./Models/N0/root/root.npy'),
    },
    "N1_Crimes Contra Pessoa": {
        "model": load_model("./Models/N1/Crimes Contra Pessoa/Crimes Contra Pessoa.model"),
        "encoder": read_encoder('./Models/N1/Crimes Contra Pessoa/Crimes Contra Pessoa.npy'),
    },
    "N1_Crimes Contra Propriedade": {
        "model": load_model("./Models/N1/Crimes Contra Propriedade/Crimes Contra Propriedade.model"),
        "encoder": read_encoder('./Models/N1/Crimes Contra Propriedade/Crimes Contra Propriedade.npy'),
    },
    "N1_Crimes de Trânsito ou Meio Ambiente": {
        "model": load_model("./Models/N1/Crimes de Trânsito ou Meio Ambiente/Crimes de Trânsito ou Meio Ambiente.model"),
        "encoder": read_encoder('./Models/N1/Crimes de Trânsito ou Meio Ambiente/Crimes de Trânsito ou Meio Ambiente.npy'),
    },
    "N1_Recuperação de Veículo ou Atos Administrativos ou Registro de Acontecimentos": {
        "model": load_model("./Models/N1/Recuperação de Veículo ou Atos Administrativos ou Registro de Acontecimentos/Recuperação de Veículo ou Atos Administrativos ou Registro de Acontecimentos.model"),
        "encoder": read_encoder('./Models/N1/Recuperação de Veículo ou Atos Administrativos ou Registro de Acontecimentos/Recuperação de Veículo ou Atos Administrativos ou Registro de Acontecimentos.npy'),
    },
    "N1_Relacionados a Drogas, Entorpecentes e Porte de Armas": {
        "model": load_model("./Models/N1/Relacionados a Drogas, Entorpecentes e Porte de Armas/Relacionados a Drogas, Entorpecentes e Porte de Armas.model"),
        "encoder": read_encoder('./Models/N1/Relacionados a Drogas, Entorpecentes e Porte de Armas/Relacionados a Drogas, Entorpecentes e Porte de Armas.npy'),
    },
    "N1_Resistência, Desacato ou Desobediência": {
        "model": load_model("./Models/N1/Resistência, Desacato ou Desobediência/Resistência, Desacato ou Desobediência.model"),
        "encoder": read_encoder('./Models/N1/Resistência, Desacato ou Desobediência/Resistência, Desacato ou Desobediência.npy'),
    },
    "N1_Violação ou Perturbação ou Dano ou Exercício Arbitrário": {
        "model": load_model("./Models/N1/Violação ou Perturbação ou Dano ou Exercício Arbitrário/Violação ou Perturbação ou Dano ou Exercício Arbitrário.model"),
        "encoder": read_encoder('./Models/N1/Violação ou Perturbação ou Dano ou Exercício Arbitrário/Violação ou Perturbação ou Dano ou Exercício Arbitrário.npy'),
    },
    "N2_Ameaça ou Injúria ou Perseguição ou Dano ou Extorsão": {
        "model": load_model("./Models/N2/Ameaça ou Injúria ou Perseguição ou Dano ou Extorsão/Ameaça ou Injúria ou Perseguição ou Dano ou Extorsão.model"),
        "encoder": read_encoder('./Models/N2/Ameaça ou Injúria ou Perseguição ou Dano ou Extorsão/Ameaça ou Injúria ou Perseguição ou Dano ou Extorsão.npy'),
    },
    "N2_Atos Administrativos": {
        "model": load_model("./Models/N2/Atos Administrativos/Atos Administrativos.model"),
        "encoder": read_encoder('./Models/N2/Atos Administrativos/Atos Administrativos.npy'),
    },
    "N2_Estelionato": {
        "model": load_model("./Models/N2/Estelionato/Estelionato.model"),
        "encoder": read_encoder('./Models/N2/Estelionato/Estelionato.npy'),
    },
    "N2_Estupro": {
        "model": load_model("./Models/N2/Estupro/Estupro.model"),
        "encoder": read_encoder('./Models/N2/Estupro/Estupro.npy'),
    },
    "N2_Furto": {
        "model": load_model("./Models/N2/Furto/Furto.model"),
        "encoder": read_encoder('./Models/N2/Furto/Furto.npy'),
    },
    "N2_Homicídio": {
        "model": load_model("./Models/N2/Homicídio/Homicídio.model"),
        "encoder": read_encoder('./Models/N2/Homicídio/Homicídio.npy'),
    },
    "N2_Lesão Corporal": {
        "model": load_model("./Models/N2/Lesão Corporal/Lesão Corporal.model"),
        "encoder": read_encoder('./Models/N2/Lesão Corporal/Lesão Corporal.npy'),
    },
    "N2_Recuperação de Veículo": {
        "model": load_model("./Models/N2/Recuperação de Veículo/Recuperação de Veículo.model"),
        "encoder": read_encoder('./Models/N2/Recuperação de Veículo/Recuperação de Veículo.npy'),
    },
    "N2_Registro de Um Acontecimento": {
        "model": load_model("./Models/N2/Registro de Um Acontecimento/Registro de Um Acontecimento.model"),
        "encoder": read_encoder('./Models/N2/Registro de Um Acontecimento/Registro de Um Acontecimento.npy'),
    },
    "N2_Roubo": {
        "model": load_model("./Models/N2/Roubo/Roubo.model"),
        "encoder": read_encoder('./Models/N2/Roubo/Roubo.npy'),
    },
}

# Node function
def ml_prediction_N0(state: OverallState) -> OverallState:
    print("NODE ML Prediction N0:")
    input_dataset = state['ml_predictions_dataset']

    X = input_dataset.drop(columns=['N3'], axis=1)
    pred = xgboost_model_db["N0_root"]["model"].predict(X)
    pred_proba = xgboost_model_db["N0_root"]["model"].predict_proba(X)

    maximos = []
    for i in range(len(pred_proba)):
        maximos.append(pred_proba[i][pred[i]])

    ml_predictions_dataset = input_dataset
    ml_predictions_dataset["N1_pred"] = xgboost_model_db["N0_root"]["encoder"].inverse_transform(pred)
    ml_predictions_dataset["N1_pred_proba"] = maximos

    print("NODE ML Prediction N0: Concluded")

    return {
        "ml_predictions_dataset": ml_predictions_dataset,
    }

# Auxiliary function
def predict_row(row):
    model_name = "N1_" + row["N1_pred"]
    
    X = row.drop(labels=['N3', 'N1_pred', 'N1_pred_proba'])
    X = X.values.reshape(1, -1)
    
    pred = xgboost_model_db[model_name]["model"].predict(X)
    pred_proba = max(xgboost_model_db[model_name]["model"].predict_proba(X)[0])

    row["N2_pred"] = xgboost_model_db[model_name]["encoder"].inverse_transform(pred)[0]
    row["N2_pred_proba"] = pred_proba

    return row

# Node function
def ml_prediction_N1(state: OverallState) -> OverallState:
    print("NODE ML Prediction N1:")
    ml_predictions_dataset = state['ml_predictions_dataset']

    ml_predictions_dataset = ml_predictions_dataset.progress_apply(predict_row, axis=1)
    print("NODE ML Prediction N1: Concluded")

    return {
        "ml_predictions_dataset": ml_predictions_dataset
    }

# Aux function
def predict_row_n2(row):
    model_name = "N2_" + row["N2_pred"]
    if model_name in xgboost_model_db.keys():    
        X = row.drop(labels=['N3', 'N1_pred', 'N1_pred_proba', 'N2_pred', 'N2_pred_proba'])
        X = X.values.reshape(1, -1)
        
        pred = xgboost_model_db[model_name]["model"].predict(X)
        pred_proba = max(xgboost_model_db[model_name]["model"].predict_proba(X)[0])
    
        row["N3_pred"] = xgboost_model_db[model_name]["encoder"].inverse_transform(pred)[0]
        row["N3_pred_proba"] = pred_proba
    else:
        row["N3_pred"] = row["N2_pred"]
        row["N3_pred_proba"] = 1

    return row

# Node function
def ml_prediction_N2(state: OverallState) -> OverallState:
    print("NODE ML Prediction N2:")
    ml_predictions_dataset = state["ml_predictions_dataset"].progress_apply(predict_row_n2, axis=1)
    print("NODE ML Prediction N2: Concluded")

    return {
        "ml_predictions_dataset": ml_predictions_dataset
    }
# ML PREDICTION


# POSTPROCESSING
# Auxiliary function - add correct N1, N2, and N3 labels
def get_n_labels(row):
    row['Label_N1'], row['Label_N2'], row['Label_N3'] = dm[dm['N3'] == row['Label']].values[0]

    return row

# Node function
def postprocessing(state: OverallState) -> OutputState:
    print("NODE Postprocessing:")
    llm_predictions_dataset = state['llm_predictions_dataset']
    llm_predictions_dataset['Llama Final Prediction Proba'] = llm_predictions_dataset['N1 llama Pred Proba'] * llm_predictions_dataset['N2 llama Pred Proba'] * llm_predictions_dataset['N3 llama Pred Proba']
    llm_predictions_dataset['Fine Llama Final Prediction Proba'] = llm_predictions_dataset['N1 fine_llama Pred Proba'] * llm_predictions_dataset['N2 fine_llama Pred Proba'] * llm_predictions_dataset['N3 fine_llama Pred Proba']
    
    ml_predictions_dataset = state['ml_predictions_dataset']
    ml_predictions_dataset = ml_predictions_dataset.rename(columns={
        'N1_pred': 'N1 ML Classification',
        'N1_pred_proba': 'N1 Pred Proba',
        'N2_pred': 'N2 ML Classification',
        'N2_pred_proba': 'N2 Pred Proba',
        'N3_pred': 'N3 ML Classification',
        'N3_pred_proba': 'N3 Pred Proba',
    })
    ml_predictions_dataset['ML Final Prediction Proba'] = ml_predictions_dataset['N1 Pred Proba'] * ml_predictions_dataset['N2 Pred Proba'] * ml_predictions_dataset['N3 Pred Proba']

    classification_dataset = pd.concat(
        [
            llm_predictions_dataset[[
                'N3', 
                'DINAMICA', 
                'N1 llama Classification', 'N1 llama Pred Proba', 
                'N2 llama Classification', 'N2 llama Pred Proba', 
                'N3 llama Classification', 'N3 llama Pred Proba',
                'Llama Final Prediction Proba',
                'N1 fine_llama Classification', 'N1 fine_llama Pred Proba', 
                'N2 fine_llama Classification', 'N2 fine_llama Pred Proba', 
                'N3 fine_llama Classification', 'N3 fine_llama Pred Proba',
                'Fine Llama Final Prediction Proba'
            ]], 
            ml_predictions_dataset[[
                'N1 ML Classification', 'N1 Pred Proba', 
                'N2 ML Classification', 'N2 Pred Proba', 
                'N3 ML Classification', 'N3 Pred Proba', 
                'ML Final Prediction Proba'
            ]]
        ], 
        axis=1
    )
    
    classification_dataset = classification_dataset.rename(columns={'N3': 'Label'})
    classification_dataset = classification_dataset.progress_apply(get_n_labels, axis=1)
    
    print("NODE Postprocessing: Concluded")
    
    return {
        'classification_dataset': classification_dataset.copy()
    }
# POSTPROCESSING


# REVIEWER
# Auxiliary function
def joint_venture(row): 
    fine_prob = row['Fine Llama Final Prediction Proba']
    llama_prob = row['Llama Final Prediction Proba']
    ml_prob = row['ML Final Prediction Proba']

    # All models together
    if fine_prob >= llama_prob and proba1 >= proba3:
        row['Joint Classification'] = row['N3 fine_llama Classification']
    elif llama_prob > fine_prob and proba2 >= proba3:
        row['Joint Classification'] = row['N3 llama Classification']
    elif ml_prob > fine_prob and proba3 > proba2:
        row['Joint Classification'] = row['N3 ML Classification']

    # XGBoost + pre-trained Llama
    if llama_prob >= ml_prob:
        row['ML + Llama'] = row['N3 llama Classification']
    else:
        row['ML + Llama'] = row['N3 ML Classification']

    # XGBoost + fine-tuned Llama
    if fine_prob >= ml_prob:
        row['ML + Fine Llama'] = row['N3 fine_llama Classification']
    else:
        row['ML + Fine Llama'] = row['N3 ML Classification']

    # Pre-trained Llama + fine-tuned Llama
    if fine_prob >= llama_prob:        
        row['Llama + Fine Llama'] = row['N3 fine_llama Classification']
    else:
        row['Llama + Fine Llama'] = row['N3 llama Classification']
        
    return row
    

# Node function
def reviewer(state: OutputState) -> OutputState:
    print("NODE Reviewer:")
    classification_dataset = state['classification_dataset']
    classification_dataset = classification_dataset.progress_apply(joint_venture, axis=1)
    print("NODE Reviewer: Concluded")

    return {
        "classification_dataset": classification_dataset
    }
# REVIEWER


# GRAPH

# State
builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)

# Nodes
builder.add_node("Preprocessing", preprocessing)
builder.add_node("Start LLM Server", start_llm_server)
builder.add_node("LLM Prediction", llm_prediction)
builder.add_node("Stop LLM Server", stop_llm_server)
builder.add_node("ML Prediction N0", ml_prediction_N0)
builder.add_node("ML Prediction N1", ml_prediction_N1)
builder.add_node("ML Prediction N2", ml_prediction_N2)
builder.add_node("Post Processing", postprocessing)
builder.add_node("Reviewer", reviewer)

# Edges
builder.add_edge(START, "Preprocessing")
builder.add_edge("Preprocessing", "Start LLM Server")
builder.add_edge("Preprocessing", "ML Prediction N0")
builder.add_edge("Start LLM Server", "LLM Prediction")
builder.add_edge("LLM Prediction", "Stop LLM Server")
builder.add_edge("ML Prediction N0", "ML Prediction N1")
builder.add_edge("ML Prediction N1", "ML Prediction N2")
builder.add_edge(["Stop LLM Server", "ML Prediction N2"], "Post Processing")
builder.add_edge("Post Processing", "Reviewer")
builder.add_edge("Reviewer", END)

# Compile
graph = builder.compile()

input_dataset = pd.read_csv('../Datasets/08_FinalSampledDataset.csv')
input_message = {"input_dataset": input_dataset}

dm = pd.read_csv('../Datasets/04_DomainHierarchy.csv')

start = time.time()
answer = graph.invoke(input_message)
tempo = time.time() - start

answer['classification_dataset'].to_csv('../Datasets/09_Results.csv', index=False)
print(f"Finalizado com sucesso em {tempo} segundos!")

