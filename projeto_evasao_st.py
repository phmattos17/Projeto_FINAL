import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

import statsmodels.api as sm


# funções necessárias para o algoritmo

def colunas_evasao(serie):
    colunas_modelo = ['Localizacao', 'Sigla_da_UF', 'MEDIA_INSE**']
    for valores in ['_promo', '_repet', '_evasao']:
        colunas_modelo.append(serie + valores)

    return colunas_modelo


def filtro_dataframe(df, estado, ano, colunas):
    df_temp = df[colunas]
    df_filtrado = df_temp[(df_temp['Sigla_da_UF'] == estado) & (~df_temp[ano + '_evasao'].isna())]
    for i in df_filtrado.columns[2:]:
        try:
            df_filtrado[i] = df_filtrado[i].astype('string')
            df_filtrado[i] = pd.to_numeric(df_filtrado[i], errors='coerce')

        except:
            pass

    return df_filtrado


def lin_reg(df, target, estado, serie, colunas,predict_values):
    df_filtrado = filtro_dataframe(df, estado, serie, colunas)

    df_no_dummy = pd.get_dummies(df_filtrado, columns=['Localizacao'], drop_first=True)

    # df_no_dummy = df_no_dummy.apply(pd.to_numeric,errors='coerce')
    df_no_dummy['intercept'] = 1

    # split no dataset
    X = df_no_dummy.drop([target, 'Sigla_da_UF'], axis=1)
    y = df_no_dummy[[target]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_test.median())

    regressao = sm.OLS(y_train, X_train).fit()

    #descomentar linha abaixo caso queira ver o sumário da  regressão
    #st.write(regressao.summary())

    st.write(f'A taxa de evasão esperada é de {float(regressao.predict(predict_values)):.2f} %')



df = pd.read_csv('df_projeto_evasao.csv')

st.title('Projeto Estimativa de Taxas de Evasão - TERA')

# Requisitar ao usuario inputs para o modelo:
# UF onde a escola localiza-se
# Localização da Escola - Rural ou Urbana
# Serie da turma
# taxa de promoção de alunos
# taxa de repetência de alunos
# valor do INSE


with st.form(key='my_form'):
    serie = st.selectbox(
        'Por favor, selecione a série ou ano desejado',
        ('1o_ano', '2o_ano', '3o_ano', '4o_ano', '5o_ano', '6o_ano',
         '7o_ano', '8o_ano', '9o_ano', '1a_serie', '2a_serie', '3a_serie'
         ), key='serie')

    localizacao = st.selectbox(
        'Indique se sua escola é rural ou urbana',
        ('Urbana', 'Rural'
         ), key='localizacao')

    estado = st.selectbox(
        'Por favor, selecione o seu estado da sua escola',
        ('RO', 'AC', 'AM', 'RR', 'PA', 'AP', 'TO', 'MA', 'PI', 'CE', 'RN', 'PB',
         'PE', 'AL', 'SE', 'BA', 'MG', 'ES', 'RJ', 'SP', 'PR', 'SC', 'RS', 'MS',
         'MT', 'GO', 'DF'
         ), key='estado')

    promocao = st.number_input('Por favor, indique a taxa de promoção em sua escola', key='promocao')

    repetencia = st.number_input('Por favor, indique a taxa de repetência em sua escola', key='repet')

    inse = st.number_input('Por favor, indique o valor do INSE em sua escola', key='inse')

    botao_calcular = st.form_submit_button('Calcular')

    if botao_calcular:
        target = str(serie) + '_evasao'

        colunas_modelo = colunas_evasao(st.session_state['serie'])

        df_modelo = filtro_dataframe(df, st.session_state['estado'], st.session_state['serie'], colunas_modelo)


        predict_values = np.array([st.session_state['inse'],st.session_state['promocao'],st.session_state['repet'],
                          int(st.session_state['localizacao']=='Urbana'),1]).reshape(1,-1)

        st.write(lin_reg(df_modelo, target, st.session_state['estado'], st.session_state['serie'], colunas_modelo,predict_values))

        st.balloons()
