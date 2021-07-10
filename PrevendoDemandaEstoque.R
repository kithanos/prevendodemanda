# Prevendo Demanda de Estoque com Base em Vendas
# Autor: Lucas Kitano
# Notas: Projeto com Feedback 2 do curso Big Data Analytics com R e Microsoft Azure Machine Learning da
# formação Cientista de Dados da Data Science Academy "https://www.datascienceacademy.com.br/"


# Carrega os Pacotes Necessários
#library(data.table)
library(dplyr)
library("stringr")
library("ggplot2")
library(xgboost)
library(caret)


# Carga dos Dados

# Só execute o trecho abaixo caso queira trabalhar com todos os dados.

# Define o diretório de datasets
#arquivo = "datasets/train.csv"

# O dataset de treino possui 74.180.464, para fluidez do processo, vamos construir 
# uma amostra de treimamento aleatória com 1.000.000.

# Gera amostra de treinamento com 1.000.000 registros.
#library(sqldf)
#train_sample <- read.csv.sql(arquivo, header = TRUE, sep = ",",
#                             sql = "select * from file order by random() limit 1000000", eol = "\n")

#base::closeAllConnections()

# Salva a amostra de treinamento:
#?write.csv
#write.csv(train_sample, "datasets/train_sample.csv", row.names = FALSE)

# Lê o arquivo da amostra de treinamento:
train_sample <- read.csv("datasets/train_sample.csv")

# Verifica o formato dos dados
str(train_sample)
View(train_sample)

# Colunas do dataset:

#Entenda-se "nesta semana" a semana de referência da coluna "Semana".

#Semana — Número que identifica a semana (de Quinta para Quarta);
#Agencia_ID — ID do depósito de vendas;
#Canal_ID — ID do canal de vendas;
#Ruta_SAK — ID da rota (Várias rotas = Depósito de vendas);
#Cliente_ID — ID do Cliente;
#Producto_ID — Product ID;
#Venta_uni_hoy — Vendas unitárias (nesta semana);
#Venta_hoy — Vendas nesta semana (unidade: pesos);
#Dev_uni_proxima — Retorno por unidade nesta semana;
#Dev_proxima — Retorno na próxima semana (unidade: pesos);
#Demanda_uni_equil — Demanda Ajustada (Variável Target);

# Apesar do dataset de treino conter as variáveis numericas relacionadas as vendas
# estas não poderão ser utilizadas no modelo, pois na competição do Kaggle, apenas
# as variáveis de ID estão no dataset de teste, de fato a previsão deve ocorrer
# apenas nas variáveis citadas, uma vez que a demanda ajustada (variável alvo) é dada pelas
# colunas "Venta_uni_hoy" - "Dev_uni_proxima".


# Análise Exploratória

# Conforme análise preliminar dos arquivos disponíveis, o dataset de treinamento contém o 
# registro das semanas 3 a 9, e o dataset de teste o registro das semanas 10 e 11 - 
# que devem ter a demanda ajustada preditas pelo algoritimo. 

# Incialmente, vamos a obter a quantidade de valores unicos para cada variável ID considerando 
# a nossa amostra de treinamento:

length(unique(train_sample$Agencia_ID))
length(unique(train_sample$Canal_ID))
length(unique(train_sample$Ruta_SAK))
length(unique(train_sample$Cliente_ID))
length(unique(train_sample$Producto_ID))

# Baseado em nossa amostra de treinamento que possui 1.000.000 registros, a variável "Cliente_ID",
# possui 458.457 registros únicos, enquanto a variável "Producto_ID" possui 1.302
# registros, podemos começar a análise por estas variáveis.

# Vamos carregar o dataset "Cliente_tabla" que possui a descrição dos clientes identificados pela chave "Cliente_ID":

cliente <- read.csv("datasets/cliente_tabla.csv", stringsAsFactors = FALSE)
str(cliente)
View(cliente)

nrow(cliente)
length(unique(cliente$Cliente_ID))
length(unique(cliente$NombreCliente))

# O dataset possui 935.362 regsitros, no entanto temos 930.500 valores unicos de ID, isto indica
# que pode haver valores duplicados de ID no dataset, vamos remove-los

cliente <- cliente %>% distinct(Cliente_ID, .keep_all= TRUE)

nrow(cliente)
length(unique(cliente$Cliente_ID))
length(unique(cliente$NombreCliente))

# Agora sim, temos a mesma quantidade de valores únicos de ID, e de números de linhas na tabela, 
# entretanto, para o nome do cliente, temos 307.009 registros, isto pode indicar que o mesmo nome de cliente
# possa estar sendo identificado por 2 IDs diferentes:

head(cliente)

# Por exemplo, o nome de cliente sem nome "SIN NOMBRE" está com os ID's 0 e 2, outros casos semelhantes
# devem estar ocorrendo neste dataset, vamos obter uma tabela de frequencias com os nomes de 
# clientes


cliente %>%
  group_by(NombreCliente) %>%
  summarise(n = n()) %>%
  mutate(freq = 100*n / sum(n)) %>%
  arrange(desc(n)) %>%
  head(20)

# O nome de cliente "NO INDENTIFICADO", ocorre em mais de 30% no dataset, outros nomes comuns
# são nomes coloquiais como Lupita, Mary, Rosy, Alex... Outros nomes de clientes contém artigos
# definidos em espanhol como "EL" e "LA". A grande questão é que estes nomes podem ser agrupados,
# por exemplo, "TIENDITA" em português significa "pequena loja", vamos pesquisar quantos registros
# possui este termo:

cliente %>% 
  filter(grepl('TIENDITA', NombreCliente)) %>%
  group_by(NombreCliente) %>%
  summarise(n = n()) %>%
  mutate(freq = 100*n / sum(n)) %>%
  arrange(desc(n)) 

# O termo TIENDITA possui 441 ocorrências, destas 36,5% são referidas como "LA TIENDITA" e 
# 5,2% como "MI TIENDITA".
# Há outros termos no dataset que referem se a estabelecimentos em geral, por exemplo, "oxxo" 
# é uma rede de lojas de conveniência no México, "abarrotes" significa Mercearia, "super"
# é uma palavra que pode se referir a um supermercado, há até palavras mais comuns
# como farmácia:

cliente %>% 
  filter(grepl('OXXO', NombreCliente)) %>%
  group_by(NombreCliente) %>%
  summarise(n = n()) %>%
  mutate(freq = 100*n / sum(n)) %>%
  arrange(desc(n))

cliente %>% 
  filter(grepl('ABARROTES', NombreCliente)) %>%
  group_by(NombreCliente) %>%
  summarise(n = n()) %>%
  mutate(freq = 100*n / sum(n)) %>%
  arrange(desc(n))

cliente %>% 
  filter(grepl('SUPER', NombreCliente)) %>%
  group_by(NombreCliente) %>%
  summarise(n = n()) %>%
  mutate(freq = 100*n / sum(n)) %>%
  arrange(desc(n))

# Vamos criar agrupamentos com estes termos, estes agrupamentos foram baseados no competidor Kaggle
# ("https://www.kaggle.com/abbysobh/classifying-client-type-using-client-names") com algumas modificações pontuais.

mercado <- c("ABARROTES", "TIENDITA", "COMERCIAL", "BODEGA", "DEPOSITO", "MERCADO", "CAMBIO", "MARKET",
             "MARKET", "MART", "MINI", "PLAZA", "MISC", "MINI", "PLAZA", "MISC", "ELEVEN", "EXP",
             "SNACK", "PAPELERIA", "CARNICERIA", "LOCAL", "COMODIN", "PROVIDENCIA")
escola <- c("ESCOLA", "COLEG", "UNIV", "ESCU", "INSTI", "PREPAR", "INSTITUTO", "C E S U", "CESU")
restaurante <- c("CASA", "CAFE", "CREMERIA", "DULCERIA", "REST", "BURGER", "TACO", "TORTA", "TAQUER", "HOTDOG", "COMEDOR", "ERIA", "BURGU", "DON")
saude <- c("FARMA", "HOSPITAL", "CLINI")
fresco <-c("VERDU", "FRUT")
hotel <- c("HOTEL", "MOTEL")
super_mercados <- c("WALL MART", "SAMS CLUB", "SUPER")
pequenos <- c('LA','EL','DE','LOS','DEL','Y', 'SAN', 'SANTA', 
              'AG','LAS','MI','MA', 'II')

governo <- c('POLICIA','CONASUPO')

todos <- c(mercado, escola, restaurante, saude, fresco, hotel, pequenos, governo, "POSTO", "OXXO",
           "REMISION", "BIMBO")

cliente_new <- cliente %>% 
  mutate(local_grup = NombreCliente) %>%
  
  mutate(local_grup = replace(local_grup, grepl(paste(escola, collapse = "|"), local_grup), "EDUCACAO")) %>%
  mutate(local_grup = replace(local_grup, grepl("POSTO", local_grup, ignore.case = FALSE), "POSTO")) %>%
  mutate(local_grup = replace(local_grup, grepl(paste(saude, collapse = "|"), local_grup), "HOSPITAL")) %>%
  mutate(local_grup = replace(local_grup, grepl(paste(restaurante, collapse = "|"), local_grup), "RESTAURANTE")) %>%
  mutate(local_grup = replace(local_grup, grepl(paste(mercado, collapse = "|"), local_grup), "MERCADOS/COMERCIOS GERAIS")) %>%
  mutate(local_grup = replace(local_grup, grepl(paste(super_mercados, collapse = "|"), local_grup), "SUPER MERCADO")) %>%
  mutate(local_grup = replace(local_grup, grepl(paste(fresco, collapse = "|"), local_grup), "MERCADO FRESCO")) %>%
  mutate(local_grup = replace(local_grup, grepl(paste(hotel, collapse = "|"), local_grup), "SERVICOS")) %>%
  mutate(local_grup = replace(local_grup, grepl("OXXO", local_grup, ignore.case = FALSE), "LOJA OXXO")) %>%
  mutate(local_grup = replace(local_grup, grepl("REMISION", local_grup, ignore.case = FALSE), "CORREIOS")) %>%
  mutate(local_grup = replace(local_grup, grepl(paste(governo, collapse = "|"), local_grup), "GOVERNO")) %>%
  mutate(local_grup = replace(local_grup, grepl("BIMBO", local_grup, ignore.case = FALSE), "LOJA BIMBO")) %>%
  
  mutate(local_grup = replace(local_grup, grepl(paste(pequenos, collapse = "|"), local_grup), "FRANQUIA PEQUENA")) %>%
  mutate(local_grup = ifelse(str_detect(NombreCliente, paste(todos, collapse = "|"), negate = TRUE), "SEM IDENTIFICACAO", local_grup)) %>%
  group_by(NombreCliente, local_grup)


View(cliente_new)
length(unique(cliente_new$local_grup))

cliente_new %>%
  group_by(local_grup) %>%
  summarise(n = n()) %>%
  mutate(freq = 100*n / sum(n)) %>%
  arrange(desc(n)) %>%
  head(15)

# Agora temos 14 categorias, a única ressalva fica pela alta concentração da chamada Pequenas Franquias,
# que concentra agora 64% dos dados, esta categoria junto da Sem Identificação está concentrando a maior
# parte dos clientes individuais (com um só nome), não é o ideal, mas já está bem melhor do que ter mais
# de 300 mil registros unicos para o nome do cliente.

# Vamos estender a análise agora para a variável "Producto_ID", para a nossa amostra de treinamento ela possui
# mais de 1000 valores unicos, além disso o produto pode estar diretamente relacionado com a necessidade 
# de demanda, seja por consumo, seja por armazenamento. Alguns produtos possuem um tempo limitado de 
# armazenamento em estoque, por isto está variável pode ser relevante para o algoritmo.

# Vamos carregar inicialmente o dataset fornecido que identifica os produtos:

produtos <- read.csv("datasets/producto_tabla.csv", stringsAsFactors = FALSE)
str(produtos)
View(produtos)

# A variável "NombreProducto" segue um padrão de nomenclatura (Nome do Produto, Quantidade, Peso, Marca).
# Quem percebeu esta divisão foi o competidor Kaggle:
# ("https://www.kaggle.com/vykhand/exploring-products")
# Vamos dividir estes campos em 3 novos campos representando (Nome do Produto, Quantidade, Peso):

produtos_new <- produtos %>%
  mutate(short_product_name = str_extract(NombreProducto, regex("^\\D*"))) %>%
  mutate(pieces = as.numeric(gsub("p","",str_extract(NombreProducto, regex("(\\d+)p "))))) %>%
  mutate(weights = str_extract(NombreProducto, regex("(\\d+)(Kg|g) "))) %>%
  mutate(weight = ifelse(str_detect(weights, "Kg"), 1000*as.numeric(gsub("Kg", "", weights)), as.numeric(gsub("g", "", weights))))

str(produtos_new)
View(produtos_new)  

# Agora vamos juntar estas informações obtidas com o dataset principal (chave: Producto_ID)

train_sample <- train_sample %>%
  left_join(select(produtos_new, short_product_name, pieces, weight, Producto_ID), by = c("Producto_ID" = "Producto_ID"))

View(train_sample)


# E vamos aproveitar para trazer também as informações do local que obtemos anteriormente (chave: cliente_ID)

train_sample <- train_sample %>%
  left_join(select(cliente_new, local_grup, Cliente_ID), by = c("Cliente_ID" = "Cliente_ID"))

View(train_sample)


# Com informações mais consolidadas dos produtos, estabelecimentos, podemos fazer algums análises,
# como por exemplo quais produtos possuem maior demanda:

train_sample %>%
  group_by(short_product_name) %>%
  summarise(sum_Demanda_uni_equil = sum(Demanda_uni_equil)) %>%
  mutate(freq = 100*sum_Demanda_uni_equil/sum(sum_Demanda_uni_equil)) %>%
  arrange(desc(sum_Demanda_uni_equil)) %>%
  head(15)

# Os produtos com o termo "Nito" são os mais vendidos para este conjunto de dados (10,8% do total), "Nito" é uma 
# marca bastante conhecida no México, principalmente em sorvetes. Vamos analisar um pouco desta 
# marca:

train_sample %>%
  filter(str_detect(short_product_name,"Nito")) %>%
  select(Producto_ID, short_product_name, pieces, weight, local_grup) %>%
  head(20)

# Muitos destes produtos possuem as mesmas quantidades e peso, ou seja são os mesmos produtos
# vendidos em lojas diferentes, além disso a variável "Producto_ID" possui mais de uma representação
# para estes mesmos registros, de fato, este agrupamento que fizemos permitiu reduzir as informações
# que estão duplicadas.

# Vamos obter a mesma visualização, mas agora para os tipos de cliente:

local <- train_sample %>%
  group_by(local_grup) %>%
  summarise(sum_Demanda_uni_equil = sum(Demanda_uni_equil)) %>%
  mutate(freq = 100*sum_Demanda_uni_equil/sum(sum_Demanda_uni_equil)) %>%
  arrange(desc(sum_Demanda_uni_equil))

head(local,15)

# Aqui temos que as pequenas franquias representam quase a metade da demanda ajustada, muito dos 
# dados ficaram sem identificação, mas é preferível te-los agrupado em uma categoria assim,
# do que te-los em muitos grupos separados.

ggplot(local, aes(y = reorder(local_grup, freq), x = freq, fill = local_grup)) + geom_bar(stat = "identity") +
  theme(legend.position = "none") + ggtitle("Demanda por tipo de estabelecimento")

# Filtrando os dados considerando apenas o produto Nito:

nito <- train_sample %>%
  filter(str_detect(short_product_name,"Nito")) %>%
  group_by(local_grup) %>%
  summarise(sum_Demanda_uni_equil = sum(Demanda_uni_equil)) %>%
  mutate(freq = 100*sum_Demanda_uni_equil/sum(sum_Demanda_uni_equil)) %>%
  arrange(desc(sum_Demanda_uni_equil)) 

head(nito, 15)

ggplot(nito, aes(y = reorder(local_grup, freq), x = freq, fill = local_grup)) + geom_bar(stat = "identity") +
  theme(legend.position = "none") + ggtitle("Demanda por tipo de estabelecimento em produtos Nito")

# Não há muita diferença entre os produtos "Nito" e os dados no geral, no que tange o local.

# Antes de prosseguirmos, precisamos ajustar as variáveis peças e peso, pois há produtos
# que possuem mais peças, consequentemente o peso será maior, uma alternativa a isto é 
# incluir uma variável que represente o peso por peça.

train_sample <- train_sample %>%
  mutate(piece_per_weight = round(weight/pieces,1)) %>%
  arrange(Semana)

# Já que o dataset se trata de dados temporais (identificados pela variável Semana), vamos obter a média da
# demanda por semana, para verificar que faixa de valor ela ocupa no dataset de amostra.

semana <- train_sample %>%
  group_by(Semana) %>%
  summarise(mean_Demanda_uni_equil = mean(Demanda_uni_equil))

head(semana)

ggplot(semana, aes(y = mean_Demanda_uni_equil, x = Semana)) + geom_line(stat = "identity") +
  theme(legend.position = "none") + ggtitle("Média da demanda por semana") +
  ylim(6,8)

# A média pouco varia nas semanas do dataset de amostra.

# Uma última tabela fornecida foi a tabela com o local dos produtos, vamos investigá-la:

cidades <- read.csv("datasets/town_state.csv", stringsAsFactors = FALSE)
head(cidades)

length(unique(cidades$State))

# A variável estado possui 33 valores únicos, vamos traze-la ao nosso dataset de treinamento,
# pela chave Agencia_ID:

train_sample <- train_sample %>%
  left_join(select(cidades, State, Agencia_ID), by = c("Agencia_ID" = "Agencia_ID"))

state_ranking <- train_sample %>%
  group_by(State) %>%
  summarise(sum_Demanda_uni_equil = sum(Demanda_uni_equil)) %>%
  mutate(freq = 100*sum_Demanda_uni_equil/sum(sum_Demanda_uni_equil)) %>%
  arrange(desc(sum_Demanda_uni_equil))

head(state_ranking)

ggplot(state_ranking, aes(y = reorder(State, freq), x = freq, fill = State)) + geom_bar(stat = "identity") +
  theme(legend.position = "none") + ggtitle("Demanda por Estado")

# A demanda é maioria no Estado do Mexico, que é o estado mais populoso do México, e em México D.F
# que é aonde está a sede do governo, interessante pois no geral estados mais populosos possuem
# uma demana maior, o que de fato aparece em nossa amostra de dados. Vamos deixar está variável no
# nosso modelo.

# Vamos criar um dataset de treino apenas com as variáveis que serão usadas nesta primeira versão do
# modelo, naturalmente iremos excluir as variáveis de ID.

train <- train_sample %>%
  select(Semana, short_product_name, local_grup, pieces, weight, piece_per_weight, State, Demanda_uni_equil)

head(train)

apply(train, 2, function(x) any(is.na(x)))

# O nosso dataset de treino ainda possui alguns valores NA nas colunas "pieces", "weight" e 
# "piece_per_weight", para a coluna pieces vamos substituir os valores NA por 1. Para representar
# pelo menos que os produtos que tiveram a sua quantidade omitida tinham no minimo uma peça.

train$pieces[is.na(train$pieces)] <- 1

# Para a coluna "weight", vamos substituir o peso pela média de pesos do grupo de produtos (short_product_name)
# E desde que a coluna "piece_per_weight" é dada pelas colunas "weight" e "pieces", basta calcularmos de novo no dataset.

# Função para substituir valores NA pela média.
impute.mean <- function(x) replace(x, is.na(x), mean(x, na.rm = TRUE))

sum(is.na(train$weight))

train <- train %>%
  group_by(short_product_name) %>%
  mutate(weight = round(impute.mean(weight),1)) %>%
  mutate(piece_per_weight = round(weight/pieces,1)) 

sum(is.na(train$weight))

# Ainda sim sobram 6332 valores NA para a coluna "weights" (eram 9495 anteriormente), para estes valores
# vamos substituir pela média do tipo de estabelecimento.

train <- train %>%
  group_by(local_grup) %>%
  mutate(weight = round(impute.mean(weight),1)) %>%
  mutate(piece_per_weight = round(weight/pieces,1)) 

sum(is.na(train$weight))

apply(train, 2, function(x) any(is.na(x)))

# Sem valores NA no nosso dataset de treino, podemos seguir adiante.

# Uma última ánalise que podemos fazer é a análise de correlação para as variáveis númericas.

require("corrplot")

train_num <- as.data.frame(train) %>%
  select(pieces, weight, piece_per_weight, Demanda_uni_equil)

str(train_num)

cor <- cor(train_num)

corrplot(cor, method="color")

# No mapa de correlação, não há nenhuma relação entre as variáveis numericas do dataset e a
# variável alvo como esperado, se houvesse, o problema seria menos complexo.


# Criação do modelo

# Vamos tratar o dataset e treinamento (converte-lo para dataframe), transformar as variáveis 
#"char" em categóricas.

train <- as.data.frame(train)

train$Semana <- as.factor(train$Semana)
train$short_product_name <- as.factor(train$short_product_name)
train$local_grup <- as.factor(train$local_grup)
train$State <- as.factor(train$State)

str(train)


#Vamos usar o algoritmo Extreme Gradient Boosting (XGBoost)

# Separando a variável alvo das variáveis independentes:
x_train <- data.matrix(train[, -8])
y_train <- data.matrix(train[, 8])

# Criando a matriz xgb 
xgb_train = xgb.DMatrix(data = x_train, label = y_train)

# Treinando o modelo
xgbc1 = xgboost(data = xgb_train, max.depth = 2, nrounds = 50)
print(xgbc1)

# Usando como parâmetro o RMSE do treinamento, o valor é alto 18.6791. Antes de tentar otimizar o modelo,
# vamos dar uma olhadinha no R^2.

pred_y = predict(xgbc1, xgb_train)

y_train_mean = mean(y_train)

# Cálculo do R^2:
tss =  sum((y_train - y_train_mean)^2 )

residuals = y_train - pred_y

rss =  sum(residuals^2)

rsq  =  1 - (rss/tss)
rsq

# O valor de R^2 obtido foi de 0,1883, está muito longe de 1 (ideal), vamos tentar otimizar o modelo
# para aumenta-lo e diminuir o RMSE.


# Otimização do Modelo

# Vamos aumentar a profundidade da árvore por trás do XGBoost
#?xgboost

xgbc2 = xgboost(data = xgb_train, max.depth = 4, nrounds = 50)
print(xgbc2)

# Dobrando a profundidade, o RMSE foi para 17.3523, podemos extrapolar e aumentar muito a profundidade:

xgbc3 = xgboost(data = xgb_train, max.depth = 32, nrounds = 50)
print(xgbc3)

# Desta vez o RMSE diminuiu bem - 12.26, vamos verificar o R^2:

pred_y = predict(xgbc3, xgb_train)

tss =  sum((y_train - y_train_mean)^2 )

residuals = y_train - pred_y

rss =  sum(residuals^2)

rsq  =  1 - (rss/tss)
rsq


# O R^2 foi para 0.65, considerando que em alguns outros modelos (que foram omtidos) neste
# documento o maior R^2 estava sendo por volta de 10, e na nossa primeira versão foi de 0,1883
# não é algo ruim, podemos tentar aumentar ainda mais a dimensão da rede e verificar se há 
# algum ganho real no RMSE ou no R^2 (ao custo sempre de aumentar o tempo de treinamento).
# Vamos aproveitar e aumentar o número de iterações do algorimo.

xgbc4 = xgboost(data = xgb_train, max.depth = 128, nrounds = 200)
print(xgbc4)

pred_x = predict(xgbc4, xgb_train)

tss =  sum((y_train - y_train_mean)^2 )

residuals = y_train - pred_x

rss =  sum(residuals^2)

rsq  =  1 - (rss/tss)
rsq

# Não houve um ganho real no RMSE e no R^2, portanto vamos manter os parâmetros do modelo 3.

final <- train %>%
  select(Semana, Demanda_uni_equil)

final <- cbind(final, pred_train = round(pred_y,1))

View(final)
str(final)

final_med <- final %>%
  group_by(Semana) %>%
  summarise_at(vars(Demanda_uni_equil,pred_train), mean)

head(final_med)

# Conclusão:
# As médias das demandas ficaram identicas entre as reais e previstas, tudo bem que estes são dados de treinamento, e os
# dados de teste disponibilizados não possuem os valores da variável alvo para comparação, assim
# vamos terminar a nossa análise por aqui. O Valor do RMSE final de 12.26 é muito alto, principalmente
# considerando que são dados de treinamento, mas parece ser o minimo valor que obteremos com as 
# transformações realizadas, o valor de R^2 também está longe do ideal, mas apresentou uma significativa
# melhora em relação a primeira versão do modelo.

# É provável que para se melhorar este número, sejam necessárias novas mudanças nos dados, e uma
# análise exploratória mais profunda, analisando os trabalhos dos competidores do Kaggle, vejo que 
# como Cientista de Dado tenho muito o que melhorar ainda, mas este é o caminho. Há de se destacar
# a complexidade do problema que basicamente fornece variáveis de ID ou categóricas para regressão,
# entretanto, vale destacar que este é um problema real, proposto por uma empresa real, e naturalmente
# os problemas "reais" são tão ou mais dificeis quanto este.