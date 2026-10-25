import tf from '@tensorflow/tfjs';

async function trainModel(inputXs, outputYs) {
    const model = tf.sequential()

    // Primeira camada da rede:
    // entrada de 7 posições (idade normalizada + 3 cores + 3 localizacoes)
    // 80 neuronios
    // A ReLU age como um filtro
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }))

    // Saída: 2 neuronios
    // um para cada categoria (pig, dog)
    model.add(tf.layers.dense({ units: 2, activation: 'softmax' }))


    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    })

    // Treinamento do modelo
    await model.fit(
        inputXs,
        outputYs,
        {
            verbose: 0,
            epochs: 100,
            shuffle: true,
            callbacks: {
                // onEpochEnd: (epoch, log) => console.log(
                //     `Epoch: ${epoch}: loss = ${log.loss}`
                // )
            }
        }
    )

    return model
}

async function predict(model, pessoa) {
    // transformar o array js para o tensor (tfjs)
    const tfInput = tf.tensor2d(pessoa)

    // Faz a predição (output será um vetor de 3 probabilidades)
    const pred = model.predict(tfInput)
    const predArray = await pred.array()
    return predArray[0].map((prob, index) => ({ prob, index }))
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
const animais = [
    { nome: "porco1", expVida: 15, cor: "branco", raboCurto: 0, tamanhoGrande: 1 },
    { nome: "porco2", expVida: 17, cor: "avermelhado", raboCurto: 1, tamanhoGrande: 1 },
    { nome: "porco3", expVida: 14, cor: "malhado", raboCurto: 1, tamanhoGrande: 0 },
    { nome: "cachorro1", expVida: 12, cor: "malhado", raboCurto: 0, tamanhoGrande: 1 },
    { nome: "cachorro2", expVida: 15, cor: "branco", raboCurto: 1, tamanhoGrande: 0 },
    { nome: "cachorro3", expVida: 10, cor: "preto", raboCurto: 0, tamanhoGrande: 1 }
];

// Função de normalização da expectativa de vida
const expVidaNormalizadas = animais.map(animal =>
    (animal.expVida - 10) / (17 - 10)
);


// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorAnimaisNormalizado = [
    [expVidaNormalizadas[0], 1, 0, 0, 0, 0, 1],    // porco1
    [expVidaNormalizadas[1], 0, 1, 0, 0, 1, 1],    // porco2
    [expVidaNormalizadas[2], 0, 0, 1, 0, 1, 0],    // porco3
    [expVidaNormalizadas[3], 0, 0, 1, 0, 0, 1],    // cachorro1
    [expVidaNormalizadas[4], 1, 0, 0, 0, 1, 0],    // cachorro2
    [expVidaNormalizadas[5], 0, 0, 0, 1, 0, 1]     // cachorro3
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["pig", "dog"]; // Ordem dos labels
const tensorLabels = [
    [1, 0], // porco1
    [1, 0], // porco2
    [1, 0],  // porco3
    [0, 1],  // cachorro1
    [0, 1],  // cachorro2
    [0, 1]  // cachorro3
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorAnimaisNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

// quanto mais dado melhor!
// assim o algoritmo consegue entender melhor os padrões complexos
// dos dados
const model = await trainModel(inputXs, outputYs)


const animalTensorNormalizado = [
    [
        0.6, // idade normalizada
        1,    // cor branco
        0,    // cor avermelhado
        0,    // cor malhado
        0,    // cor preto
        1,    // raboCurto
        1     // tamanhoGrande
    ]
]

const predictions = await predict(model, animalTensorNormalizado)

const results = predictions
    .sort((a, b) => b.prob - a.prob)
    .map(p => `${labelsNomes[p.index]} (${(p.prob * 100).toFixed(2)}%)`)
    .join('\n')
console.log(results)