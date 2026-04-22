import streamlit as st
import mlflow
from openai import OpenAI
import pandas as pd

st.set_page_config(page_title="Análise de Sentimento com LLM e MLflow", layout="wide")

st.title("🤖 Analisador de Sentimento com Rastreabilidade")
st.markdown("Esta aplicação usa **Streamlit** para interface, **OpenAI** para inferência e **MLflow** para controle de versão e rastreabilidade dos prompts.")

# Configuração da API na Barra Lateral
st.sidebar.header("⚙️ Configurações")
api_key = st.sidebar.text_input("Sua OpenAI API Key", type="password")

st.sidebar.markdown("---")
st.sidebar.subheader("Hiperparâmetros do Modelo")
model_name = st.sidebar.selectbox("Modelo", ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"])
temperature = st.sidebar.slider("Temperatura", 0.0, 1.0, 0.0)

# O Prompt é o principal hiperparâmetro e pode ser editado pelo usuário
prompt_template_default = """Você é um analista de sentimentos sênior.
Analise o texto abaixo e responda APENAS com uma destas palavras: [positivo, negativo, neutro].
Texto: {texto}"""

st.subheader("1. Defina o Prompt (Hiperparâmetro)")
prompt_template = st.text_area("Prompt Template (Altere para testar o versionamento no MLflow)", value=prompt_template_default, height=120)

# Golden Dataset de exemplo
dados = [
    {"texto": "O produto quebrou no primeiro dia de uso. Horrível!", "sentimento_real": "negativo"},
    {"texto": "Amei o suporte, me ajudaram muito rápido.", "sentimento_real": "positivo"},
    {"texto": "A entrega chegou no prazo normal.", "sentimento_real": "neutro"}
]
df_golden = pd.DataFrame(dados)

st.subheader("2. Base de Teste (Golden Dataset)")
st.dataframe(df_golden, use_container_width=True)

st.subheader("3. Execução e Rastreamento")
if st.button("Executar Análise e Rastrear no MLflow", type="primary"):
    if not api_key:
        st.error("⚠️ Por favor, insira a sua API Key na barra lateral para continuar.")
    else:
        client = OpenAI(api_key=api_key)
        
        # Inicia MLflow run localmente (o Streamlit Community cria a pasta mlruns na sessão)
        with st.spinner("Analisando sentimentos com a LLM e gravando logs no MLflow..."):
            # Usamos context manager do MLflow
            with mlflow.start_run() as run:
                # Logando parâmetros
                mlflow.log_param("modelo", model_name)
                mlflow.log_param("temperatura", temperature)
                mlflow.log_param("prompt_template", prompt_template)
                
                resultados = []
                for index, row in df_golden.iterrows():
                    texto_usuario = row['texto']
                    prompt_final = prompt_template.format(texto=texto_usuario)
                    
                    try:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": prompt_final}],
                            temperature=temperature
                        )
                        predicao = response.choices[0].message.content.strip().lower()
                    except Exception as e:
                        predicao = f"erro: {str(e)}"
                    
                    resultados.append({
                        "texto": texto_usuario,
                        "sentimento_real": row['sentimento_real'],
                        "predicao_llm": predicao,
                        "prompt_usado": prompt_final
                    })
                
                # Consolida resultados
                df_resultados = pd.DataFrame(resultados)
                
                # Validação Básica
                df_resultados['acertou'] = df_resultados.apply(
                    lambda x: x['sentimento_real'] in x['predicao_llm'] if not x['predicao_llm'].startswith('erro') else False, 
                    axis=1
                )
                acuracia = df_resultados['acertou'].mean()
                
                # Logando métricas e artefatos
                mlflow.log_metric("acuracia", acuracia)
                mlflow.log_table(data=df_resultados, artifact_file="resultados_analise.json")
                
                run_id = run.info.run_id
                
        st.success(f"✅ Análise concluída e registrada no MLflow! (Run ID: {run_id})")
        
        # Mostrando Métricas na Tela
        col1, col2 = st.columns(2)
        col1.metric("Acurácia do Prompt", f"{acuracia*100:.1f}%")
        col2.metric("Total Analisado", len(df_golden))
        
        st.markdown("**Resultados Detalhados**")
        st.dataframe(df_resultados, use_container_width=True)
        