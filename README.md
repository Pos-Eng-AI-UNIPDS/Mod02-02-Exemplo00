# 🧠 Primeira Rede Neural com TensorFlow.js

![JavaScript](https://img.shields.io/badge/JavaScript-ES%20Modules-F7DF1E?logo=javascript&logoColor=black)
![Node.js](https://img.shields.io/badge/Node.js-Node%20Runtime-339933?logo=nodedotjs&logoColor=white)
![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.22.0-FF6F00?logo=tensorflow&logoColor=white)
![Licença](https://img.shields.io/badge/License-ISC-blue)

Projeto didático do **Módulo 02** da pós-graduação em Engenharia de Inteligência Artificial, dedicado à construção e ao treinamento de redes neurais artificiais com **JavaScript**, **Node.js** e **TensorFlow.js**.

O repositório contém dois exemplos independentes de **classificação supervisionada**, utilizando conjuntos de dados pequenos para demonstrar normalização, *one-hot encoding*, camadas densas, treinamento e predição.

## 📚 Exemplos disponíveis

| Arquivo | Objetivo | Classes de saída | Atributos de entrada |
| --- | --- | --- | --- |
| [`index.js`](./index.js) | Classificar pessoas por características de exemplo | `premium`, `medium`, `basic` | Idade normalizada, cor e localização |
| [`dogPig.js`](./dogPig.js) | Classificar animais como cães ou porcos | `pig`, `dog` | Expectativa de vida normalizada, cor, rabo curto e tamanho grande |

> **Nota:** As categorias de pessoas são rótulos fictícios usados para fins de ensino, não critérios confiáveis de avaliação de indivíduos. Ambos os conjuntos de treinamento são muito pequenos; as previsões ilustram o funcionamento do modelo e não devem ser interpretadas como classificações confiáveis no mundo real.

## 🛠️ Tecnologias

- **JavaScript (ES Modules):** implementação dos exemplos.
- **Node.js:** execução dos scripts pelo terminal.
- **TensorFlow.js (`@tensorflow/tfjs`):** criação, treinamento e inferência das redes neurais.
- **npm:** instalação e gerenciamento de dependências.

## 📁 Estrutura do repositório

```text
Mod02-02-Exemplo00/
├── index.js            # Rede neural para classificação de pessoas
├── dogPig.js           # Rede neural para classificação de animais
├── package.json        # Dependências e scripts npm
├── package-lock.json   # Versões resolvidas das dependências
├── .gitignore          # Arquivos e pastas ignorados pelo Git
└── README.md           # Documentação do projeto
```

## 🚀 Como executar

### Pré-requisitos

Tenha o **Node.js** e o **npm** instalados. Uma versão LTS recente do Node.js é recomendada.

Verifique sua instalação:

```bash
node --version
npm --version
```

### 1. Clone o repositório

```bash
git clone https://github.com/Pos-Eng-AI-UNIPDS/Mod02-02-Exemplo00.git
cd Mod02-02-Exemplo00
```

### 2. Instale as dependências

```bash
npm install
```

### 3. Execute um exemplo

**Classificação de pessoas:**

```bash
npm start
```

O script `start` do projeto executa `node --no-warnings --watch index.js`, portanto fica observando alterações no arquivo. Para uma execução única, use:

```bash
node index.js
```

**Classificação de animais:**

```bash
node dogPig.js
```

Os dois programas treinam sua respectiva rede neural e mostram no terminal as classes previstas com probabilidades formatadas como percentuais. Como o treinamento não define uma semente aleatória, os valores podem variar entre execuções.

## 🧩 Como as redes neurais funcionam

Ambos os modelos são construídos com `tf.sequential()` e possuem a seguinte arquitetura:

| Configuração | Pessoas (`index.js`) | Animais (`dogPig.js`) |
| --- | --- | --- |
| Entradas | 7 atributos | 7 atributos |
| Camada oculta | 80 neurônios, ativação ReLU | 80 neurônios, ativação ReLU |
| Camada de saída | 3 neurônios, ativação Softmax | 2 neurônios, ativação Softmax |
| Otimizador | Adam | Adam |
| Função de perda | `categoricalCrossentropy` | `categoricalCrossentropy` |
| Métrica | `accuracy` | `accuracy` |
| Treinamento | 100 épocas | 100 épocas |

**Fluxo de processamento:**

```text
Dados de exemplo
      ↓
Normalização + one-hot encoding
      ↓
Tensores de entrada (xs) e rótulos (ys)
      ↓
Rede neural: Dense(80, ReLU)
      ↓
Camada de saída: Dense(3 ou 2, Softmax)
      ↓
Treinamento com Adam + categoricalCrossentropy
      ↓
Predição para uma nova entrada
      ↓
Probabilidades ordenadas no terminal
```

### Exemplo 1 — Pessoas

O vetor de entrada é organizado nesta ordem:

```js
[idadeNormalizada, azul, vermelho, verde, saoPaulo, rio, curitiba]
```

Os rótulos são representados por vetores *one-hot*:

```js
premium = [1, 0, 0]
medium  = [0, 1, 0]
basic   = [0, 0, 1]
```

O exemplo treina com três pessoas e classifica uma nova entrada. No script, a idade é normalizada utilizando os limites de 25 e 40 anos.

### Exemplo 2 — Animais

O vetor de entrada segue a ordem:

```js
[expectativaVidaNormalizada, branco, avermelhado, malhado, preto, raboCurto, tamanhoGrande]
```

Os rótulos de saída são:

```js
pig = [1, 0]
dog = [0, 1]
```

O exemplo utiliza seis animais (três porcos e três cães). A expectativa de vida é normalizada com os limites de 10 e 17 anos. As demais características são codificadas numericamente.

## 🎯 Conceitos praticados

- **Aprendizado supervisionado:** treino com entradas e rótulos conhecidos.
- **Normalização:** adequação de valores numéricos a uma escala comum.
- **One-hot encoding:** transformação de atributos categóricos em vetores numéricos.
- **Camadas densas:** conexões entre neurônios de camadas sucessivas.
- **ReLU:** função de ativação da camada oculta.
- **Softmax:** conversão das saídas em probabilidades relativas às classes.
- **Adam:** algoritmo de otimização dos pesos.
- **Entropia cruzada categórica:** função de perda para classificação multiclasse.
- **Épocas:** ciclos de treinamento sobre os dados.

## ⚠️ Limitações e próximos passos

Este é um experimento educacional. Para avançar, vale considerar conjuntos de dados maiores e mais variados, divisão entre treino e teste, métricas de validação, repetibilidade com sementes aleatórias e descarte adequado de tensores para reduzir consumo de memória. Também é importante evitar usar características pessoais ou sensíveis como base para decisões reais sobre indivíduos.

## 🔗 Referências

- [Documentação do TensorFlow.js](https://www.tensorflow.org/js)
- [API TensorFlow.js](https://js.tensorflow.org/api/latest/)
- [Documentação do Node.js](https://nodejs.org/docs/latest/api/)

## 📄 Licença

O projeto declara a licença **ISC** no arquivo `package.json`. Consulte as condições aplicáveis antes de redistribuir o código.

---

**Repositório:** [Pos-Eng-AI-UNIPDS/Mod02-02-Exemplo00](https://github.com/Pos-Eng-AI-UNIPDS/Mod02-02-Exemplo00)
