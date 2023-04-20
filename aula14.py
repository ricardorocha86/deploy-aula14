import streamlit as st
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from pycaret.regression import load_model, predict_model 

modelo1 = load_model('meu-modelo-para-charges')
modelo2 = load_model('meu-modelo-para-smoker')

def smap(x):  
	y = 'male' if x == 'Masculino' else 'female' 
	return y

def rmap(x):
	if x == 'Sudeste':
		return 'southeast'
	elif x == 'Noroeste':
		return 'northwest'
	elif x == 'Sudoeste':
		return 'southwest' 
	else:
		return 'northeast'

def fmap(x):  
	y = 'yes' if x == 'Sim' else 'no' 
	return y

def classificador(modelo, dados):
	pred = predict_model(estimator = modelo, data = dados) 
	return pred

opcoes = ['Boas-vindas', 
		  'Dashboard',
		  'Custos de Seguro', 
		  'Probabilidade de Fraude']

pagina = st.sidebar.selectbox('Navegue pelo menu:', opcoes)

if pagina == 'Boas-vindas':
	st.title('**Meu Primeiro Data App 🎈**') 
	st.header('Seja bem-vindo! 😀')
	



if pagina == 'Dashboard':
	st.header('Seja bem-vindo ao nosso Dashboard')

	url = 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv'
	dados = pd.read_csv(url)
	#st.experimental_data_editor(dados.sample(3))

	col1, col2, col3 = st.columns(3)

	st.markdown('---')

	regiao = col1.selectbox("Região em que mora", dados['region'].unique()) 
	sexo = col2.selectbox("Sexo", ['Masculino', 'Feminino']) 
	criancas = col3.selectbox("Quantidade de dependentes", [0, 1, 2, 3, 4, 5])

	sexo = 'male' if sexo == 'Masculino' else 'female'

	filtro_regiao = dados['region'] == regiao 
	filtro_sexo = dados['sex'] == sexo
	filtro_criancas = dados['children'] == criancas
	
	filtro_dados = dados.loc[filtro_regiao & filtro_sexo & filtro_criancas] 
	#st.table(filtro_dados)

	col1, col2 = st.columns([1,3])

	col1.metric('Idade Média', round(filtro_dados['age'].mean(), 1))
	col1.metric('IMC Médio', round(filtro_dados['bmi'].mean(), 1))
	col1.metric('Custos Médio', round(filtro_dados['charges'].mean(), 1))
	col1.metric('Fumantes', '{:.2%}'.format(filtro_dados['smoker'].value_counts(normalize = True)['yes']))
	
	fig = sns.scatterplot(data = filtro_dados, x = 'bmi', y = 'charges', hue = 'smoker')
	plt.xlabel('Indice de Massa Corporal')
	plt.ylabel('Custos de Seguro')
	plt.title('Relação IMC x Custos para Fumantes e Não Fumantes')
	col2.pyplot(fig.get_figure())

	st.markdown('---')









if pagina == 'Custos de Seguro': 

	st.header('Modelagem de valor do seguro')

	st.markdown('Nessa seção é feito o deploy do modelo para cotar o valor do seguro para um indivíduo.\
			Entre com os dados e clique em APLICAR O MODELO para obter as predições.')

	st.markdown('---')

	col1, col2, col3= st.columns(3)

	idade = col1.number_input('Idade', 18, 65, 30)
	sexo = col1.selectbox("Sexo", ['Masculino', 'Feminino'])
	imc = col2.number_input('Índice de Massa Corporal', 15, 54, 24)
	criancas = col2.selectbox("Quantidade de dependentes", [0, 1, 2, 3, 4, 5])
	fumante = col3.selectbox("É fumante?", ['Sim', 'Não'])
	regiao = col3.selectbox("Região em que mora", 
								  ['Sudeste', 'Noroeste', 'Sudoeste', 'Nordeste'])
 

	dados_dicio = {'age': [idade], 
				   'sex': [smap(sexo)], 
				   'bmi': [imc], 
				   'children': [criancas], 
				   'region': [rmap(regiao)], 
				   'smoker': [fmap(fumante)]}
		
	dados = pd.DataFrame(dados_dicio)
	#st.table(dados)

	st.markdown('---')

	if st.button('APLICAR O MODELO'): 

		saida = classificador(modelo1, dados)
		pred = float(saida['prediction_label'].round(2))  
		s1 = 'Custo Estimado do Seguro: ${:.2f}'.format(pred) 
		st.markdown('## **' + s1 + '**')  






if pagina == 'Probabilidade de Fraude': 

	st.header('Detectar probabilidade de fraude')

	st.markdown('Nessa seção é feito o deploy do modelo para detectar probabilidade de fraude na \
		     variável "fumante". Entre com os dados do indivíduo\
		      em análise e clique em APLICAR O MODELO para obter as predições.')

	st.markdown('---')

	col1, col2, col3= st.columns(3)

	idade = col1.number_input('Idade', 18, 65, 30)
	sexo = col1.selectbox("Sexo", ['Masculino', 'Feminino'])
	imc = col2.number_input('Índice de Massa Corporal', 15, 54, 24)
	criancas = col2.selectbox("Quantidade de filhos", [0, 1, 2, 3, 4, 5]) 
	regiao = col3.selectbox("Região em que mora", 
								  ['Sudeste', 'Noroeste', 'Sudoeste', 'Nordeste'])
	custos = col3.number_input('Custos da pessoa', 1000, 64000, 10000)
 
	dados_dicio = {'age': [idade], 
				   'sex': [smap(sexo)], 
				   'bmi': [imc], 
				   'children': [criancas], 
				   'region': [rmap(regiao)], 
				   'charges': [custos]}
		
	dados = pd.DataFrame(dados_dicio)

	st.markdown('---')

	if st.button('APLICAR O MODELO'):
		saida = classificador(modelo2, dados) 
		resp = 'NÃO' if saida['prediction_label'][0] == 'no' else 'SIM' 
		prob = saida['prediction_score'][0]  
		s = '{}, com propensão {:.2f}%.'.format(resp, 100*prob)
		st.markdown('## **' + s + '**') 