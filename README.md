# Regressão Linear em preços de carros
Previsão de preço de carros usados utilizando Regressão Linear

## Objetivo
O intuito foi utilizar um modelo supervisionado de Machine Learning para prever o preço de carros usados através de suas características, como o modelo, ano, tipo de câmbio, milhas percorridas (dados oriundos de carros do Reino Unido), tipo de combustível, imposto do carro, milhas por galão de combustível e cilindrada do motor.

## Obstáculo
A parte de transformação dos dados foi trabalhosa. Foi preciso transformar as features objects "model", "transmission" e "fuelType" em inteiros. 
Para isso, criei algumas linhas de código que pudessem transformar cada elemento dessas features em números inteiros (do arange numpy), em vez de substitui esses valores um por um.

~~~
lista_numeros = np.arange(start=1, stop=len(df['model'].unique())+1) # concertando erro: index 22 is out of bounds for axis 0 with size 22
indice=0

for modelo in df['model'].unique():
    df.model.replace({modelo:lista_numeros[indice]}, inplace=True)
    indice+=1
~~~

## Melhores features
Para escolher as melhores features que pudessem treinar o modelo utilizei o SlectKBest, que comparou o efeito de cada feature na label price (que é o preço dos carros).

~~~
k_best_features = SelectKBest(k='all')
k_best_features.fit_transform(features, label)
k_best_features_score = k_best_features.scores_
~~~

Depois organizei os scores de cada feature com o nome da feature.

~~~
raw_pairs = zip(features_list[0:], k_best_features_score)
ordered_pairs = list(reversed(sorted(raw_pairs, key = lambda x: x[0])))
k_best_features_final = dict(ordered_pairs[:20])
best_features = k_best_features_final.keys()
print('')
print('Melhores features:')
print(k_best_features_final)
~~~

## Normalização
Normalizei os dados para minimizar as discrepâncias dos valores das features e assim não confundir o modelo. 

~~~
scaler = MinMaxScaler().fit(features)
features_scale = scaler.transform(features)

print('Features: ', features_scale.shape)
print(features_scale)
~~~

## Resultado
O modelo conseguiu fazer uma boa aproximção dos reais preços dos carros, determinando se o carro vai ter um proço mais alto ou mais baixo.


