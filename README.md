# prevendodemanda
Prevendo demanda em estoque com base em vendas

Projeto: Demanda de Estoque com Base em Vendas
Autor: Lucas Kitano
GitHub: https://github.com/kithanos

Este projeto é parte integrante do curso Big Data Analytics com R e Microsoft Azure Machine Learning
da Formação Cientista de Dados da DataScience Academy (https://www.datascienceacademy.com.br)

O arquivo "PrevendoDemandaEstoque.R" contém o script em R, já o arquivo "PrevendoDemandaEstoque.Rmd" é
o R MarkDown para geração do relatório PDF "PrevendoDemandaEstoque.Rmd", para a sua reprodução é necessário 
possuir o MikTex instalado: https://miktex.org/

Além disso, para a reprodução do script em R, são necessárias as seguites bibliotecas:

#library(data.table) - Apenas se quiser usar todo o dataset de treinamento.
library(dplyr)
library("stringr")
library("ggplot2")
library(xgboost)
library(caret)
library("corrplot")

O dataset completo original (~3,1 GB) pode ser baixado na página do Kaggle (https://www.kaggle.com/c/grupo-bimbo-inventory-demand/data)
arquivo "train.csv". O dataset de amostra (~40 MB) e os outros datasets utilizados são fornecidos junto aos arquivos do projeto. 
Descompacte o arquivo "datasets.rar" em uma pasta /datasets no diretório principal, e descompacte os arquivos "cliente_table.rar" e 
"train_sample.rar".
