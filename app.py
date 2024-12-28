import ofxparse
import pandas as pd
import os
from os import path
from datetime import datetime

STATAMENTS_FOLDER = 'statements'

df = pd.DataFrame()

for statement in os.listdir(STATAMENTS_FOLDER):
    if 'ofx' in statement:
        with open(path.join(STATAMENTS_FOLDER, statement), encoding='ISO-8859-1') as ofx_file:
            ofx = ofxparse.OfxParser.parse(ofx_file)

    transaction_data = []
    for account in ofx.accounts:
        for transaction in account.statement.transactions:
            transaction_data.append({
                'date': transaction.date,
                'value': transaction.amount,
                'description': transaction.memo,
                'id': transaction.id
            })

    df_temp = pd.DataFrame(transaction_data)
    df_temp['value'] = df_temp['value'].astype(float)
    df_temp['date'] = df_temp['date'].apply(lambda x: x.date())
    df = pd.concat([df, df_temp])

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from openai import OpenAI

template = """
Você é um analista de dados, trabalhando em um projeto de limpeza de dados.
Seu trabalho é escolher uma categoria adequada para cada item de lançamento financeiro que vou te enviar, para te auxiliar, te fornecerei uma lista de categorias e ao lado, separado por um -, deicharei a descrçao do que se encaixaria naquela categoria.

Para cada item que eu te enviar ecolha uma dentre as seguinte categorias da lista de categorias abaixo:

       CATEGORIAS                            DESCRIÇÃO                                                                                                   
Receitas                      - Tudo que for entrada e/ou transferencia recebida.                                                                             
Investimentos                 - Tudo que for deposito em corretoras finaceiras.                                                                           
Esportes                      - Tudo que for compra e/ou transferências para lojas de artigos de esportes.                                                        
Educação                      - Tudo que for compra e/ou transferências para lojas de artigos de educação/estudos.                                        
Lazer                         - Tudo que for compra e/ou transferências para lojas de artigos de lazer.                                                   
Outros                        - Tudo que não tiver categoria adequada.                                                                                    
Restaurante                   - Tudo que for compra e/ou transferências para lojas de consumo de alimentos.                                               
Saude                         - Tudo que for compra e/ou transferências para farmacias ou hospitais.                                                      
Seguro                        - Tudo que for compra e/ou transferências para seguroas de celular, carro e/ou vida.                                        
Serviços                      - Tudo que for compra e/ou transferências para prestação de serviço, seja diarista, cabelereiro, barbeiro, pedreiro e etc.  
Streaming                     - Tudo que for compra e/ou transferências para serviços de streaming como netflix, amazon prime entre outros.               
Supermercado                  - Tudo que for compra e/ou transferências para mercados e supermercados.                                                    
Transporte                    - Tudo que for compra e/ou transferências para bilhete unico, uber, 99 ou para abastecimento de carro em postos de gasolina.
Vestuario                     - Tudo que for compra e/ou transferências para lojas de artigos de roupas.                                                  
Moradia                       - Valores referentes a alugel, conta de luz, agua e etc.                                                                    
Telefone                      - Tudo que for compra e/ou transferências para colocar creditos em chip de celular, como Tim e Claro.                       
Transferencia para terceiros  - Tudo que for transferência para outras pessoas fisicas.                                                                   
Categoria Desconhecida        - Tudo que não for possivel categorizar.                                                                                    

Agora vamos iniciar a categorização dos itens de lançamento financeiro.
De acordo com a lista de categorias e sua descrição, qual é a melhor categoria que se adequa ao item abaixo?
{item}

Observações:
Lembrando, você só pode escolher uma categoria da lista fornecida acima. Categorias que não estiverem na lista, não serão aceitas.
Responda apenas com a categoria e nada alem do que apenas a categoria.
"""

prompt = PromptTemplate.from_template(template=template)
url = "http://195.29.196.251:41027/v1"
client = OpenAI(api_key='not-needed', base_url=url)
model = client.models.list().data[0].id

chat = ChatOpenAI(
    temperature=0.0,
    model=model,
    base_url=url,
    api_key='not-needed'
)

chain = prompt | chat | StrOutputParser()
categorias = chain.batch(list(df['description'].values))
df['category'] = categorias
df.to_csv('financas.csv', index=False)